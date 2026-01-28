"""
Calculate entropy of KV cache tensors saved from the experiment.
"""

import numpy as np
from scipy import stats


def calculate_entropy(tensor, bins=100):
    """
    Calculate entropy of a tensor using histogram-based estimation.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor
    bins : int
        Number of bins for histogram

    Returns
    -------
    float
        Entropy value
    """
    flat = tensor.flatten()
    hist, _ = np.histogram(flat, bins=bins, density=True)
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    # Normalize to get probabilities
    hist = hist / hist.sum()
    return stats.entropy(hist)


def calculate_per_head_entropy(tensor, bins=100):
    """
    Calculate entropy per attention head.

    tensor shape: (batch, num_heads, seq_len, head_dim)
    """
    batch, num_heads, seq_len, head_dim = tensor.shape
    entropies = []
    for head_idx in range(num_heads):
        head_data = tensor[:, head_idx, :, :]
        ent = calculate_entropy(head_data, bins)
        entropies.append(ent)
    return entropies


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate entropy of KV cache tensors")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for histogram")
    args = parser.parse_args()

    key_file = f"kv_cache_prefill_layer{args.layer}_keys.npy"
    value_file = f"kv_cache_prefill_layer{args.layer}_values.npy"

    print(f"Loading tensors from layer {args.layer}...")
    keys = np.load(key_file)
    values = np.load(value_file)

    print(f"Keys shape: {keys.shape}")
    print(f"Values shape: {values.shape}")
    print()

    # Overall entropy
    key_entropy = calculate_entropy(keys, args.bins)
    value_entropy = calculate_entropy(values, args.bins)

    print("=" * 50)
    print("Overall Entropy")
    print("=" * 50)
    print(f"Keys entropy:   {key_entropy:.4f}")
    print(f"Values entropy: {value_entropy:.4f}")
    print()

    # Per-head entropy
    print("=" * 50)
    print("Per-Head Entropy")
    print("=" * 50)
    key_head_entropies = calculate_per_head_entropy(keys, args.bins)
    value_head_entropies = calculate_per_head_entropy(values, args.bins)

    print(f"{'Head':<6} {'Key Entropy':<15} {'Value Entropy':<15}")
    print("-" * 40)
    for i, (k_ent, v_ent) in enumerate(zip(key_head_entropies, value_head_entropies)):
        print(f"{i:<6} {k_ent:<15.4f} {v_ent:<15.4f}")

    print()
    print(f"Key entropy   - mean: {np.mean(key_head_entropies):.4f}, std: {np.std(key_head_entropies):.4f}")
    print(f"Value entropy - mean: {np.mean(value_head_entropies):.4f}, std: {np.std(value_head_entropies):.4f}")


if __name__ == "__main__":
    main()
