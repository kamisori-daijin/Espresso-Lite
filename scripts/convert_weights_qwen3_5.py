#!/usr/bin/env python3
"""Convert HuggingFace Qwen3.5 hybrid text weights to Espresso BLOBFILE layout.

Requires:
    pip install torch transformers

Usage:
    ./scripts/convert_weights_qwen3_5.py --model Qwen/Qwen3.5-0.8B --output /tmp/qwen3_5 --max-seq 4096
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MAX_BLOBFILE_DATA_SIZE = 0xFFFF_FFFF


def make_blob_header(data_size: int) -> bytes:
    header = bytearray(128)
    header[0] = 0x01
    header[4] = 0x02
    header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    header[68] = 0x01
    struct.pack_into("<I", header, 72, data_size)
    struct.pack_into("<I", header, 80, 128)
    return bytes(header)


def write_blob(tensor: torch.Tensor, path: Path) -> None:
    payload = tensor.detach().cpu().float().to(torch.float16).numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(make_blob_header(len(payload)))
        handle.write(payload)


def max_supported_mask_sequence_length() -> int:
    return int(math.isqrt(MAX_BLOBFILE_DATA_SIZE // 2))


def write_causal_masks(output_dir: Path, max_seq: int) -> None:
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    size = 1
    while size <= max_seq:
        mask = torch.full((size, size), 0.0, dtype=torch.float16)
        mask = torch.triu(mask.fill_(-1e4), diagonal=1) + torch.tril(torch.zeros_like(mask))
        payload = mask.numpy().tobytes()
        with (mask_dir / f"causal_{size}.bin").open("wb") as handle:
            handle.write(make_blob_header(len(payload)))
            handle.write(payload)
        size *= 2


def main() -> None:
    print("=== Qwen3.5 Weight Converter Started ===", flush=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HuggingFace model name or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--max-seq",
        type=int,
        help="Override exported context length. Required when the source model context exceeds the BLOBFILE mask limit or the target runtime context.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    
    print(f"[*] Target Model : {args.model}", flush=True)
    print(f"[*] Output Dir   : {output_dir.resolve()}", flush=True)
    print("[*] Downloading/Loading model from Hugging Face... Please wait.", flush=True)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
    except Exception as e:
        print(f"\n[!] ERROR: Failed to load model. Details: {e}", file=sys.stderr, flush=True)
        return

    print("[+] Model loaded successfully. Extracting weights...", flush=True)
    state = model.state_dict()
    
    config = getattr(model.config, "text_config", model.config)

    hidden_dim = getattr(config, "intermediate_size")
    head_dim = getattr(config, "linear_key_head_dim", getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    n_kv_head = getattr(config, "linear_num_key_heads", getattr(config, "num_key_value_heads", config.num_attention_heads))
    
    # Safeguard sequence dimensions
    supported_mask_limit = max_supported_mask_sequence_length()
    if args.max_seq is not None:
        exported_max_seq = args.max_seq
        if exported_max_seq <= 0:
            raise ValueError("--max-seq must be > 0")
        if exported_max_seq > config.max_position_embeddings:
            raise ValueError(
                f"--max-seq {exported_max_seq} exceeds source model context {config.max_position_embeddings}"
            )
        if exported_max_seq > supported_mask_limit:
            raise ValueError(
                f"--max-seq {exported_max_seq} exceeds BLOBFILE mask limit {supported_mask_limit}"
            )
    else:
        if config.max_position_embeddings > supported_mask_limit:
            exported_max_seq = min(4096, supported_mask_limit)
            print(f"[!] Warning: Model original context ({config.max_position_embeddings}) exceeds BLOBFILE mask limit.")
            print(f"    Automatically fallback and capping exported context length to: {exported_max_seq}", flush=True)
        else:
            exported_max_seq = config.max_position_embeddings

    metadata = {
        "name": getattr(model.config, "_name_or_path", args.model).split("/")[-1],
        "nLayer": config.num_hidden_layers,
        "nHead": config.num_attention_heads,
        "nKVHead": n_kv_head,
        "dModel": config.hidden_size,
        "headDim": head_dim,
        "hiddenDim": hidden_dim,
        "vocab": config.vocab_size,
        "maxSeq": exported_max_seq,
        "normEps": config.rms_norm_eps,
        "architecture": "qwen3_5_hybrid",
    }
    if hasattr(config, "eos_token_id") and config.eos_token_id is not None:
        metadata["eosToken"] = config.eos_token_id if isinstance(config.eos_token_id, int) else config.eos_token_id[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("[*] Exporting embeddings and shared tensors...", flush=True)
    write_blob(state["model.embed_tokens.weight"], output_dir / "embeddings" / "token.bin")
    write_blob(state["model.norm.weight"], output_dir / "final_norm.bin")
    lm_head = state["lm_head.weight"] if "lm_head.weight" in state else state["model.embed_tokens.weight"]
    write_blob(lm_head, output_dir / "lm_head.bin")

    print(f"[*] Exporting layers (Total: {config.num_hidden_layers})...", flush=True)
    for layer in range(config.num_hidden_layers):
        if layer % 5 == 0 or layer == config.num_hidden_layers - 1:
            print(f"    -> Processing layer {layer}/{config.num_hidden_layers}...", flush=True)
            
        layer_dir = output_dir / "layers" / str(layer)
        prefix = f"model.layers.{layer}"

        # 1. Base Layer Norm components
        write_blob(state[f"{prefix}.input_layernorm.weight"], layer_dir / "rms_att.bin")
        write_blob(state[f"{prefix}.post_attention_layernorm.weight"], layer_dir / "rms_ffn.bin")

        # 2. Attention / Linear Dynamics mapping block
        linear_qkv_key = f"{prefix}.linear_attn.in_proj_qkv.weight"
        self_q_key = f"{prefix}.self_attn.q_proj.weight"

        if linear_qkv_key in state:
            # Layout Mapping for GatedDeltaNet Linear Attention Layers
            in_proj_qkv = state[linear_qkv_key]
            key_dim = head_dim * n_kv_head
            value_dim = head_dim * n_kv_head
            
            # Slice the composite fused projection into Q, K, V segments explicitly
            q_weight, k_weight, v_weight = torch.split(in_proj_qkv, [key_dim, key_dim, value_dim], dim=0)
            
            write_blob(q_weight, layer_dir / "q_proj.bin")
            write_blob(k_weight, layer_dir / "k_proj.bin")
            write_blob(v_weight, layer_dir / "v_proj.bin")
            write_blob(state[f"{prefix}.linear_attn.out_proj.weight"], layer_dir / "out_proj.bin")
            
            # Dump internal extra RNN states/parameters required for GatedDeltaNet execution
            write_blob(state[f"{prefix}.linear_attn.A_log"], layer_dir / "A_log.bin")
            write_blob(state[f"{prefix}.linear_attn.conv1d.weight"], layer_dir / "conv1d_weight.bin")
            write_blob(state[f"{prefix}.linear_attn.dt_bias"], layer_dir / "dt_bias.bin")
            write_blob(state[f"{prefix}.linear_attn.in_proj_a.weight"], layer_dir / "in_proj_a.bin")
            write_blob(state[f"{prefix}.linear_attn.in_proj_b.weight"], layer_dir / "in_proj_b.bin")
            write_blob(state[f"{prefix}.linear_attn.in_proj_z.weight"], layer_dir / "in_proj_z.bin")
            write_blob(state[f"{prefix}.linear_attn.norm.weight"], layer_dir / "attn_norm.bin")

        elif self_q_key in state:
            # Layout Mapping for Standard Full Self-Attention Layers
            write_blob(state[self_q_key], layer_dir / "q_proj.bin")
            write_blob(state[f"{prefix}.self_attn.k_proj.weight"], layer_dir / "k_proj.bin")
            write_blob(state[f"{prefix}.self_attn.v_proj.weight"], layer_dir / "v_proj.bin")
            write_blob(state[f"{prefix}.self_attn.o_proj.weight"], layer_dir / "out_proj.bin")
            
            # Save layer normalization factors unique to the attention head
            write_blob(state[f"{prefix}.self_attn.q_norm.weight"], layer_dir / "q_norm.bin")
            write_blob(state[f"{prefix}.self_attn.k_norm.weight"], layer_dir / "k_norm.bin")

        # 3. Standard MLP (SwiGLU Block) weights
        write_blob(state[f"{prefix}.mlp.gate_proj.weight"], layer_dir / "w1.bin")
        write_blob(state[f"{prefix}.mlp.down_proj.weight"], layer_dir / "w2.bin")
        write_blob(state[f"{prefix}.mlp.up_proj.weight"], layer_dir / "w3.bin")

    print("[*] Generating causal masks...", flush=True)
    write_causal_masks(output_dir, exported_max_seq)
    print("=== Conversion Completed Successfully! ===", flush=True)


if __name__ == "__main__":
    main()