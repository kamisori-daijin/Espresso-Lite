#!/usr/bin/env python3
"""Load Espresso BLOBFILE Qwen3.5-family weights into a Torch/HF-compatible shape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BLOBFILE_HEADER_BYTES = 128


@dataclass(frozen=True)
class EspressoQwen3_5Metadata:
    name: str
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    head_dim: int
    hidden_dim: int
    vocab: int
    max_seq: int
    norm_eps: float
    rope_theta: float
    eos_token: int | None


def load_espresso_metadata(weights_dir: Path) -> EspressoQwen3_5Metadata:
    payload = json.loads((weights_dir / "metadata.json").read_text(encoding="utf-8"))
    return EspressoQwen3_5Metadata(
        name=payload["name"],
        n_layer=int(payload["nLayer"]),
        n_head=int(payload["nHead"]),
        n_kv_head=int(payload.get("nKVHead", payload["nHead"])),
        d_model=int(payload["dModel"]),
        head_dim=int(payload["headDim"]),
        hidden_dim=int(payload["hiddenDim"]),
        vocab=int(payload["vocab"]),
        max_seq=int(payload["maxSeq"]),
        norm_eps=float(payload["normEps"]),
        rope_theta=float(payload.get("ropeTheta", 10_000.0)),
        eos_token=(int(payload["eosToken"]) if "eosToken" in payload else None),
    )


def read_blobfile_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    with path.open("rb") as handle:
        handle.seek(BLOBFILE_HEADER_BYTES)
        array = np.fromfile(handle, dtype=np.float16, count=count)
        trailing = handle.read(1)
    if array.size != count:
        raise ValueError(f"truncated BLOBFILE payload at {path}")
    if trailing:
        raise ValueError(f"unexpected trailing payload at {path}")
    return array.reshape(shape)


def load_espresso_qwen3_5_state_dict(
    weights_dir: Path,
    metadata: EspressoQwen3_5Metadata | None = None,
) -> dict[str, np.ndarray]:
    weights_dir = weights_dir.expanduser().resolve()
    metadata = metadata or load_espresso_metadata(weights_dir)

    #Calculate the internal dimensions of Qwen3_5GatedDeltaNet
    #Allocation: in_proj_qkv = key_dim * 2 + value_dim
    key_dim = metadata.head_dim * metadata.n_kv_head
    value_dim = metadata.head_dim * metadata.n_head  # linear_num_value_heads is usually n_head
    qkv_dim = key_dim * 2 + value_dim

    state_dict: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": read_blobfile_array(
            weights_dir / "embeddings" / "token.bin",
            (metadata.vocab, metadata.d_model),
        ),
        "model.norm.weight": read_blobfile_array(
            weights_dir / "final_norm.bin",
            (metadata.d_model,),
        ),
        "lm_head.weight": read_blobfile_array(
            weights_dir / "lm_head.bin",
            (metadata.vocab, metadata.d_model),
        ),
    }

    for layer_index in range(metadata.n_layer):
        layer_dir = weights_dir / "layers" / str(layer_index)
        prefix = f"model.layers.{layer_index}"
        
        # 1. Layer Norms (RMSNorm)
        state_dict[f"{prefix}.input_layernorm.weight"] = read_blobfile_array(
            layer_dir / "rms_att.bin",
            (metadata.d_model,),
        )
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = read_blobfile_array(
            layer_dir / "rms_ffn.bin",
            (metadata.d_model,),
        )
        
        # 2. GatedDeltaNet (Alternative to Attention Layer) Internal Parameter Mapping
        # Supports Qwen 3.5 Model Definition (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, conv1d, norm, out_proj)
        state_dict[f"{prefix}.self_attn.in_proj_qkv.weight"] = read_blobfile_array(
            layer_dir / "in_proj_qkv.bin",
            (qkv_dim, metadata.d_model),
        )
        state_dict[f"{prefix}.self_attn.in_proj_z.weight"] = read_blobfile_array(
            layer_dir / "in_proj_z.bin",
            (value_dim, metadata.d_model),
        )
        state_dict[f"{prefix}.self_attn.in_proj_b.weight"] = read_blobfile_array(
            layer_dir / "in_proj_b.bin",
            (metadata.n_head, metadata.d_model),  # num_v_heads 分の射影
        )
        state_dict[f"{prefix}.self_attn.in_proj_a.weight"] = read_blobfile_array(
            layer_dir / "in_proj_a.bin",
            (metadata.n_head, metadata.d_model),
        )
        
        # 1D Causal Convolution
        state_dict[f"{prefix}.self_attn.conv1d.weight"] = read_blobfile_array(
            layer_dir / "conv1d_weight.bin",
            (qkv_dim, 1, 4),  # Assumed shape with groups=conv_dim, kernel_size=4
        )
        
        # Internal parameters of the Gated Delta Net (A_log, dt_bias, RMSNorm for gates)
        state_dict[f"{prefix}.self_attn.A_log"] = read_blobfile_array(
            layer_dir / "A_log.bin",
            (metadata.n_head,),
        )
        state_dict[f"{prefix}.self_attn.dt_bias"] = read_blobfile_array(
            layer_dir / "dt_bias.bin",
            (metadata.n_head,),
        )
        state_dict[f"{prefix}.self_attn.norm.weight"] = read_blobfile_array(
            layer_dir / "attn_norm.bin",
            (metadata.head_dim,),
        )
        state_dict[f"{prefix}.self_attn.out_proj.weight"] = read_blobfile_array(
            layer_dir / "out_proj.bin",
            (metadata.d_model, value_dim),
        )

        # 3. MLP (Gated-SiLU) Parameters
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = read_blobfile_array(
            layer_dir / "w1.bin",
            (metadata.hidden_dim, metadata.d_model),
        )
        state_dict[f"{prefix}.mlp.down_proj.weight"] = read_blobfile_array(
            layer_dir / "w2.bin",
            (metadata.d_model, metadata.hidden_dim),
        )
        state_dict[f"{prefix}.mlp.up_proj.weight"] = read_blobfile_array(
            layer_dir / "w3.bin",
            (metadata.hidden_dim, metadata.d_model),
        )

    return state_dict


def qwen3_5_config_kwargs_from_metadata(metadata: EspressoQwen3_5Metadata) -> dict[str, object]:
    # Conforms to Qwen3_5TextConfig argument names
    kwargs: dict[str, object] = {
        "hidden_size": metadata.d_model,
        "intermediate_size": metadata.hidden_dim,
        "num_hidden_layers": metadata.n_layer,
        "num_attention_heads": metadata.n_head,
        "num_key_value_heads": metadata.n_kv_head,
        "vocab_size": metadata.vocab,
        "max_position_embeddings": metadata.max_seq,
        "rms_norm_eps": metadata.norm_eps,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        # Reflect configuration parameters for Qwen3_5GatedDeltaNet
        "linear_key_head_dim": metadata.head_dim,
        "linear_value_head_dim": metadata.head_dim,
        "linear_num_key_heads": metadata.n_kv_head,
        "linear_num_value_heads": metadata.n_head,
        "linear_conv_kernel_dim": 4,
    }
    # If RoPE parameters are required, match them to the initialization structure of TextConfig.
    kwargs["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": metadata.rope_theta,
    }
    
    if metadata.eos_token is not None:
        kwargs["eos_token_id"] = metadata.eos_token
    return kwargs


def load_espresso_qwen3_5_for_causal_lm(weights_dir: Path, torch_dtype=None):
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    metadata = load_espresso_metadata(weights_dir)
    # Use Qwen3_5TextConfig, which is dedicated to text.
    config = Qwen3_5TextConfig(**qwen3_5_config_kwargs_from_metadata(metadata))
    model = Qwen3_5ForCausalLM(config)
    
    state_dict = load_espresso_qwen3_5_state_dict(weights_dir, metadata)
    torch_state_dict = {
        name: torch.from_numpy(array.astype(np.float32, copy=False)).to(dtype=torch_dtype or torch.float32)
        for name, array in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(torch_state_dict, strict=False)
    if missing:
        raise ValueError(f"Missing Qwen3.5 tensors for Espresso weights: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected Espresso tensors for Qwen3.5 model: {unexpected}")
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model.eval()
    return metadata, model