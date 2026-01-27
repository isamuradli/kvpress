"""
Verify Per-Layer Compression During Prefilling

This script verifies that KVPress applies KV cache compression to EACH layer
of the transformer during the prefilling stage and counts compression events.

Expected behavior:
- During prefill, compression should be applied exactly ONCE per layer
- All layers should have the same compression count
- Cache size should be reduced by the compression ratio
"""

import torch
from collections import defaultdict
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import KnormPress
from kvpress.presses.base_press import BasePress


@dataclass
class CompressionCountingPress(BasePress):
    """
    A wrapper press that counts compression events per layer.
    Used to verify that compression is applied to each layer during prefilling.
    """

    base_press: BasePress = None
    verbose: bool = True

    # Track compression stats per layer
    compression_counts: dict = field(default_factory=lambda: defaultdict(int))
    cache_sizes_before: dict = field(default_factory=lambda: defaultdict(list))
    cache_sizes_after: dict = field(default_factory=lambda: defaultdict(list))

    def reset_stats(self):
        """Reset all tracking statistics."""
        self.compression_counts = defaultdict(int)
        self.cache_sizes_before = defaultdict(list)
        self.cache_sizes_after = defaultdict(list)

    def post_init_from_model(self, model):
        if self.base_press is not None:
            self.base_press.post_init_from_model(model)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """Delegate compression to the wrapped press."""
        if self.base_press is not None:
            return self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
        return keys, values

    def forward_hook(self, module, input, kwargs, output):
        """
        Forward hook that tracks compression events and cache sizes.
        """
        layer_idx = module.layer_idx
        cache = kwargs["past_key_values"]
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]
        cache_position = kwargs["cache_position"][-1].item()

        # Check if we're in prefill phase
        is_prefill = cache_position <= q_len

        # Get cache size BEFORE potential compression
        cache_size_before = cache.get_seq_length(layer_idx) if len(cache) > layer_idx else 0

        # Call the base press's forward_hook (which does the actual compression)
        if self.base_press is not None:
            result = self.base_press.forward_hook(module, input, kwargs, output)
        else:
            result = output

        # Get cache size AFTER potential compression
        cache_size_after = cache.get_seq_length(layer_idx)

        # Track statistics
        self.cache_sizes_before[layer_idx].append(cache_size_before)
        self.cache_sizes_after[layer_idx].append(cache_size_after)

        # Count compression events (when cache size changes)
        compression_happened = cache_size_before != cache_size_after
        if compression_happened:
            self.compression_counts[layer_idx] += 1

        # Print debug info
        if self.verbose:
            phase = "PREFILL" if is_prefill else "DECODE"
            status = "COMPRESSED" if compression_happened else "unchanged"
            print(f"  [{phase}] Layer {layer_idx:2d}: "
                  f"cache_position={cache_position:4d}, q_len={q_len:4d}, "
                  f"cache: {cache_size_before:5d} -> {cache_size_after:5d} ({status})")

        return result

    @property
    def compression_ratio(self):
        if hasattr(self.base_press, 'compression_ratio'):
            return self.base_press.compression_ratio
        return 0.0

    @compression_ratio.setter
    def compression_ratio(self, value):
        if hasattr(self.base_press, 'compression_ratio'):
            self.base_press.compression_ratio = value

    def print_summary(self):
        """Print a summary of compression statistics."""
        print("\n" + "=" * 80)
        print("COMPRESSION SUMMARY")
        print("=" * 80)

        if not self.compression_counts:
            print("No compression events recorded!")
            return

        print(f"\n{'Layer':<8} {'Compressions':<15} {'Before->After':<20}")
        print("-" * 50)

        for layer_idx in sorted(self.compression_counts.keys()):
            count = self.compression_counts[layer_idx]
            before = self.cache_sizes_before[layer_idx]
            after = self.cache_sizes_after[layer_idx]

            # Show first compression event sizes
            if before and after:
                size_str = f"{before[0]} -> {after[0]}"
            else:
                size_str = "N/A"

            print(f"Layer {layer_idx:2d}: {count:^15d} {size_str:^20}")

        # Verification
        print("\n" + "-" * 50)
        total_layers = len(self.compression_counts)
        counts = list(self.compression_counts.values())

        if len(set(counts)) == 1:
            print(f"✓ VERIFIED: All {total_layers} layers have {counts[0]} compression(s) each")
        else:
            print(f"✗ WARNING: Inconsistent compression counts across layers!")
            print(f"  Counts: {dict(self.compression_counts)}")


def run_verification(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda:0",
    compression_ratio: float = 0.5,
):
    """
    Run the verification test.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (use smaller model for quick testing)
    device : str
        Device to run on
    compression_ratio : float
        Compression ratio for prefill phase
    """

    print("=" * 80)
    print("VERIFYING PREFILL COMPRESSION PER LAYER")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print(f"Compression ratio: {compression_ratio}")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded. Number of layers: {num_layers}")
    print()

    # Prepare input - use a moderately long context
    context = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories
    and rumors of artificial beings endowed with intelligence or consciousness by master
    craftsmen. The seeds of modern AI were planted by philosophers who attempted to
    describe the process of human thinking as the mechanical manipulation of symbols.
    This work culminated in the invention of the programmable digital computer in the
    1940s, a machine based on the abstract essence of mathematical reasoning.

    The field of AI research was founded at a workshop held on the campus of Dartmouth
    College, USA during the summer of 1956. Those who attended would become the leaders
    of AI research for decades. Many of them predicted that a machine as intelligent as
    a human being would exist in no more than a generation.
    """

    question = "What is the history of AI?"

    # Apply chat template
    messages = [{"role": "user", "content": context + "\n\n" + question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    print(f"Input prompt length: {input_length} tokens")
    print()

    # Create counting press wrapper
    base_press = KnormPress(compression_ratio=compression_ratio)
    counting_press = CompressionCountingPress(
        base_press=base_press,
        verbose=True
    )

    # Create cache
    cache = DynamicCache()

    print("-" * 80)
    print("PREFILL PHASE - Watching compression on each layer")
    print("-" * 80)
    print()

    # Run prefill with compression
    with torch.no_grad(), counting_press(model):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            use_cache=True,
        )

    # Print cache info after prefill
    prefill_cache_size = cache.get_seq_length()
    print(f"\n>>> After prefill:")
    print(f"    Input length: {input_length} tokens")
    print(f"    Cache size: {prefill_cache_size} tokens")
    print(f"    Compression: {input_length} -> {prefill_cache_size} "
          f"({100 * (1 - prefill_cache_size/input_length):.1f}% reduced)")

    # Print summary
    counting_press.print_summary()

    # Verify per-layer cache sizes
    print("\n" + "-" * 50)
    print("PER-LAYER CACHE SIZES AFTER PREFILL:")
    print("-" * 50)

    all_same = True
    first_size = cache.get_seq_length(0)

    for layer_idx in range(num_layers):
        layer_size = cache.get_seq_length(layer_idx)
        status = "✓" if layer_size == first_size else "✗"
        print(f"  Layer {layer_idx:2d}: {layer_size} tokens {status}")
        if layer_size != first_size:
            all_same = False

    print("\n" + "=" * 80)
    if all_same:
        print(f"✓ VERIFICATION PASSED: All {num_layers} layers have cache size = {first_size}")
    else:
        print("✗ VERIFICATION FAILED: Layers have different cache sizes!")
    print("=" * 80)

    return {
        "num_layers": num_layers,
        "input_length": input_length,
        "cache_size": prefill_cache_size,
        "compression_counts": dict(counting_press.compression_counts),
        "all_layers_compressed": len(counting_press.compression_counts) == num_layers,
        "uniform_compression": len(set(counting_press.compression_counts.values())) == 1,
    }


def run_comparison_test(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda:0",
):
    """
    Run a comparison test: with and without compression.
    """
    print("=" * 80)
    print("COMPARISON TEST: WITH vs WITHOUT COMPRESSION")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model: {model_name}")
    print(f"Layers: {num_layers}")

    # Prepare input
    context = "The quick brown fox jumps over the lazy dog. " * 50
    messages = [{"role": "user", "content": context}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    print(f"Input length: {input_length} tokens\n")

    # Test 1: Without compression
    print("-" * 40)
    print("TEST 1: Without compression")
    print("-" * 40)
    cache_no_compress = DynamicCache()

    with torch.no_grad():
        model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache_no_compress,
            use_cache=True,
        )

    print(f"Cache size (no compression): {cache_no_compress.get_seq_length()} tokens")

    # Test 2: With 50% compression
    print("\n" + "-" * 40)
    print("TEST 2: With 50% compression (KnormPress)")
    print("-" * 40)

    press = KnormPress(compression_ratio=0.5)
    cache_compressed = DynamicCache()

    with torch.no_grad(), press(model):
        model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache_compressed,
            use_cache=True,
        )

    print(f"Cache size (50% compression): {cache_compressed.get_seq_length()} tokens")

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    no_compress_size = cache_no_compress.get_seq_length()
    compress_size = cache_compressed.get_seq_length()
    actual_ratio = 1 - (compress_size / no_compress_size)

    print(f"Without compression: {no_compress_size} tokens")
    print(f"With 50% compression: {compress_size} tokens")
    print(f"Actual reduction: {100 * actual_ratio:.1f}%")
    print(f"Expected reduction: ~50%")

    if 0.45 <= actual_ratio <= 0.55:
        print("\n✓ Compression ratio is within expected range!")
    else:
        print(f"\n✗ Compression ratio ({actual_ratio:.2f}) differs from expected (0.5)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify prefill compression per layer")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--compression-ratio", type=float, default=0.5,
                        help="Compression ratio (default: 0.5)")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison test instead")

    args = parser.parse_args()

    if args.compare:
        run_comparison_test(
            model_name=args.model,
            device=args.device,
        )
    else:
        run_verification(
            model_name=args.model,
            device=args.device,
            compression_ratio=args.compression_ratio,
        )
