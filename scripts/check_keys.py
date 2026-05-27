#!/usr/bin/env python3
"""A simple script to inspect and dump all state_dict keys of the specified model.

Usage:
    python check_keys.py --model Qwen/Qwen3.5-0.8B
"""

import argparse
import sys
from transformers import AutoModelForCausalLM

def main():
    print("=== Model Key Inspector Started ===", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HuggingFace model name or local path")
    args = parser.parse_args()
    
    print(f"[*] Loading model: {args.model}... Please wait.", flush=True)
    try:
        # Load the model directly to inspect its actual architecture layout
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True)
    except Exception as e:
        print(f"[!] ERROR: Failed to load model. {e}", file=sys.stderr, flush=True)
        return

    state = model.state_dict()
    keys = sorted(list(state.keys()))
    
    output_file = "model_keys.txt"
    print(f"[+] Successfully loaded. Writing {len(keys)} keys to '{output_file}'...", flush=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for key in keys:
            f.write(f"{key}\n")
            
    print(f"=== Done! Please check the contents of '{output_file}' ===", flush=True)

if __name__ == "__main__":
    main()