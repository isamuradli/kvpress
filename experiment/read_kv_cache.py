"""
KV Cache Reader

Reads and displays the contents of saved KV cache files (.pt format).

Usage:
    python read_kv_cache.py kv_cache_layer0_prefill.pt
    python read_kv_cache.py kv_cache_layer0_prefill.pt --show-values
    python read_kv_cache.py kv_cache_layer0_prefill.pt --head 0 --pos 10
"""

import argparse
import torch
import numpy as np


def print_separator(title="", char="=", width=80):
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def load_kv_cache(filepath):
    """Load KV cache from file."""
    data = torch.load(filepath, weights_only=False)
    return data


def print_metadata(data):
    """Print cache metadata."""
    print_separator("METADATA")
    metadata = data["metadata"]
    for key, value in metadata.items():
        print(f"  {key:25s}: {value}")
    print()


def print_tokens(data, max_display=50):
    """Print token information."""
    if "tokens" not in data:
        print("  No token information saved.")
        return

    tokens = data["tokens"]
    token_ids = tokens["token_ids"]
    token_strings = tokens["token_strings"]
    total = len(token_ids)

    print_separator("TOKENS")
    print(f"  Total tokens: {total}")
    print()

    # Print in 2-column format
    print(f"  {'Idx':>5} | {'ID':>8} | {'Token':<20}  ||  {'Idx':>5} | {'ID':>8} | {'Token':<20}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*20}--++--{'-'*5}-+-{'-'*8}-+-{'-'*20}")

    display_count = min(total, max_display)
    rows = (display_count + 1) // 2

    for row in range(rows):
        left_idx = row
        right_idx = row + rows

        # Left column
        if left_idx < display_count:
            tid = token_ids[left_idx]
            tstr = repr(token_strings[left_idx])
            if len(tstr) > 18:
                tstr = tstr[:15] + "..."
            left = f"  {left_idx:5d} | {tid:8d} | {tstr:<20}"
        else:
            left = " " * 40

        # Right column
        if right_idx < display_count:
            tid = token_ids[right_idx]
            tstr = repr(token_strings[right_idx])
            if len(tstr) > 18:
                tstr = tstr[:15] + "..."
            right = f"{right_idx:5d} | {tid:8d} | {tstr:<20}"
        else:
            right = ""

        print(f"{left}  ||  {right}")

    if total > max_display:
        print(f"\n  ... ({total - max_display} more tokens, use --max-tokens to show more)")
    print()


def print_cache_shape(data):
    """Print cache tensor shapes and statistics."""
    keys = data["keys"]
    values = data["values"]

    print_separator("CACHE TENSORS")
    print(f"  Keys shape:   {list(keys.shape)}")
    print(f"  Values shape: {list(values.shape)}")
    print(f"  Dtype:        {keys.dtype}")
    print()

    # Dimension labels
    batch, num_heads, seq_len, head_dim = keys.shape
    print(f"  Dimensions:")
    print(f"    batch_size:   {batch}")
    print(f"    num_kv_heads: {num_heads}")
    print(f"    seq_length:   {seq_len}")
    print(f"    head_dim:     {head_dim}")
    print()

    # Statistics
    print(f"  Keys statistics:")
    print(f"    min:  {keys.min().item():.6f}")
    print(f"    max:  {keys.max().item():.6f}")
    print(f"    mean: {keys.float().mean().item():.6f}")
    print(f"    std:  {keys.float().std().item():.6f}")
    print()

    print(f"  Values statistics:")
    print(f"    min:  {values.min().item():.6f}")
    print(f"    max:  {values.max().item():.6f}")
    print(f"    mean: {values.float().mean().item():.6f}")
    print(f"    std:  {values.float().std().item():.6f}")
    print()


def print_cache_values(data, head_idx=0, positions=None, num_dims=8):
    """Print actual cache values for specific head and positions."""
    keys = data["keys"]
    values = data["values"]

    batch, num_heads, seq_len, head_dim = keys.shape

    if head_idx >= num_heads:
        print(f"  Error: head_idx {head_idx} >= num_heads {num_heads}")
        return

    print_separator(f"CACHE VALUES (Head {head_idx})")

    if positions is None:
        # Show first 5 and last 5 positions
        if seq_len <= 10:
            positions = list(range(seq_len))
        else:
            positions = list(range(5)) + list(range(seq_len - 5, seq_len))

    # Get token strings if available
    token_strings = None
    if "tokens" in data:
        token_strings = data["tokens"]["token_strings"]

    print(f"  Showing {len(positions)} positions, first {num_dims} dimensions")
    print()

    for pos in positions:
        if pos >= seq_len:
            continue

        key_vec = keys[0, head_idx, pos, :num_dims]
        val_vec = values[0, head_idx, pos, :num_dims]

        # Format vectors
        key_str = ", ".join([f"{v:.4f}" for v in key_vec.tolist()])
        val_str = ", ".join([f"{v:.4f}" for v in val_vec.tolist()])

        token_info = ""
        if token_strings and pos < len(token_strings):
            token_info = f" ({repr(token_strings[pos])[:20]})"

        print(f"  Position {pos:4d}{token_info}:")
        print(f"    Key:   [{key_str}, ...]")
        print(f"    Value: [{val_str}, ...]")
        print()


def print_head_norms(data):
    """Print L2 norms per head to show head importance."""
    keys = data["keys"]
    values = data["values"]

    print_separator("HEAD NORMS (L2)")

    batch, num_heads, seq_len, head_dim = keys.shape

    print(f"  {'Head':>4} | {'Key Norm':>12} | {'Value Norm':>12} | {'Key Mean':>12} | {'Value Mean':>12}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for h in range(num_heads):
        key_norm = keys[0, h].float().norm().item()
        val_norm = values[0, h].float().norm().item()
        key_mean = keys[0, h].float().mean().item()
        val_mean = values[0, h].float().mean().item()
        print(f"  {h:4d} | {key_norm:12.4f} | {val_norm:12.4f} | {key_mean:12.6f} | {val_mean:12.6f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Read and display KV cache contents")
    parser.add_argument("filepath", help="Path to the .pt cache file")
    parser.add_argument("--show-values", action="store_true", help="Show actual cache values")
    parser.add_argument("--head", type=int, default=0, help="Head index for --show-values")
    parser.add_argument("--pos", type=int, nargs="+", help="Specific positions to show")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to display")
    parser.add_argument("--show-norms", action="store_true", help="Show per-head norms")
    parser.add_argument("--num-dims", type=int, default=8, help="Number of dimensions to show per vector")

    args = parser.parse_args()

    # Load cache
    print()
    print_separator(f"KV CACHE: {args.filepath}")
    print()

    data = load_kv_cache(args.filepath)

    # Print sections
    print_metadata(data)
    print_cache_shape(data)
    print_tokens(data, max_display=args.max_tokens)

    if args.show_norms:
        print_head_norms(data)

    if args.show_values:
        print_cache_values(data, head_idx=args.head, positions=args.pos, num_dims=args.num_dims)

    print_separator()


if __name__ == "__main__":
    main()
