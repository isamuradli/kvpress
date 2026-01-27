"""
KV Cache Compression Observation Experiment

This script demonstrates KV cache compression during both prefill and decode phases,
printing the cache size after each layer to observe the compression in action.

Model: Qwen/Qwen2.5-14B-Instruct
"""

import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import (
    KnormPress,
    DecodingPress,
    PrefillDecodingPress,
)
from kvpress.presses.base_press import BasePress


# =============================================================================
# STEP 1: Create a Custom Logging Press Wrapper
# =============================================================================

@dataclass
class LoggingPressWrapper(BasePress):
    """
    A wrapper press that logs KV cache sizes after each layer.
    Wraps any BasePress and adds logging functionality.
    """

    press: BasePress
    log_every_n_layers: int = 1  # Log every N layers (1 = all layers)
    verbose: bool = True

    def __post_init__(self):
        self.layer_count = 0
        self.phase = "unknown"

    def post_init_from_model(self, model):
        if self.press is not None:
            self.press.post_init_from_model(model)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        # Delegate to wrapped press
        if self.press is not None:
            return self.press.compress(module, hidden_states, keys, values, attentions, kwargs)
        return keys, values

    def forward_hook(self, module, input, kwargs, output):
        layer_idx = module.layer_idx
        cache = kwargs["past_key_values"]
        q_len = kwargs["hidden_states"].shape[1]
        cache_position = kwargs["cache_position"][-1].item()

        # Determine phase
        is_prefill = cache_position <= q_len
        self.phase = "PREFILL" if is_prefill else "DECODE"

        # Get cache size BEFORE compression
        cache_size_before = cache.get_seq_length(layer_idx) if len(cache) > layer_idx else 0

        # Call the wrapped press's forward hook
        if self.press is not None:
            result = self.press.forward_hook(module, input, kwargs, output)
        else:
            result = output

        # Get cache size AFTER compression
        cache_size_after = cache.get_seq_length(layer_idx)

        # Log if this layer should be logged
        if self.verbose and (layer_idx % self.log_every_n_layers == 0):
            compression_happened = cache_size_before != cache_size_after
            status = "COMPRESSED" if compression_happened else "unchanged"

            print(f"  [{self.phase}] Layer {layer_idx:2d}: "
                  f"KV Cache: {cache_size_before:5d} -> {cache_size_after:5d} "
                  f"({status})")

        return result

    @property
    def compression_ratio(self):
        if hasattr(self.press, 'compression_ratio'):
            return self.press.compression_ratio
        return 0.0

    @compression_ratio.setter
    def compression_ratio(self, value):
        if hasattr(self.press, 'compression_ratio'):
            self.press.compression_ratio = value


@dataclass
class LoggingDecodingPress(DecodingPress):
    """
    Extended DecodingPress that logs compression events.
    """

    verbose: bool = True
    log_every_n_layers: int = 1  # Log every N layers (1 = all layers)

    def forward_hook(self, module, input, kwargs, output):
        layer_idx = module.layer_idx
        cache = kwargs["past_key_values"]

        # Get cache size before
        cache_size_before = cache.get_seq_length(layer_idx)

        # Call parent's forward hook
        result = super().forward_hook(module, input, kwargs, output)

        # Get cache size after
        cache_size_after = cache.get_seq_length(layer_idx)

        # Log if compression happened AND layer matches logging interval
        if (self.verbose and
            cache_size_before != cache_size_after and
            layer_idx % self.log_every_n_layers == 0):
            print(f"  [DECODE] Layer {layer_idx:2d}: "
                  f"KV Cache: {cache_size_before:5d} -> {cache_size_after:5d} "
                  f"(COMPRESSED)")

        return result


# =============================================================================
# STEP 2: Main Experiment Function
# =============================================================================

def run_experiment(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    device: str = "cuda:0",
    prefill_compression_ratio: float = 0.5,
    decode_target_size: int = 512,
    decode_compression_interval: int = 256,
    max_new_tokens: int = 100,
    log_every_n_layers: int = 4,  # Log every 4 layers to reduce output
):
    """
    Run the KV cache compression experiment.

    Parameters
    ----------
    model_name : str
        HuggingFace model name
    device : str
        Device to run on
    prefill_compression_ratio : float
        Compression ratio for prefill phase (0.5 = keep 50%)
    decode_target_size : int
        Target cache size after decoding compression
    decode_compression_interval : int
        Compress every N generated tokens
    max_new_tokens : int
        Maximum tokens to generate
    log_every_n_layers : int
        Log every N layers (1 = all, 4 = every 4th layer)
    """

    print("=" * 80)
    print("KV CACHE COMPRESSION EXPERIMENT")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print(f"Prefill compression ratio: {prefill_compression_ratio}")
    print(f"Decode target size: {decode_target_size}")
    print(f"Decode compression interval: {decode_compression_interval}")
    print(f"Max new tokens: {max_new_tokens}")
    print()

    # -------------------------------------------------------------------------
    # STEP 2a: Load Model and Tokenizer
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 2b: Prepare Input
    # -------------------------------------------------------------------------
    # Create a reasonably long context
    context = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories
    and rumors of artificial beings endowed with intelligence or consciousness by master
    craftsmen. The seeds of modern AI were planted by philosophers who attempted to
    describe the process of human thinking as the mechanical manipulation of symbols.
    This work culminated in the invention of the programmable digital computer in the
    1940s, a machine based on the abstract essence of mathematical reasoning. This device
    and the ideas behind it inspired a handful of scientists to begin seriously discussing
    the possibility of building an electronic brain. Author of this context is Isa 

    The field of AI research was founded at a workshop held on the campus of Dartmouth
    College, USA during the summer of 1956. Those who attended would become the leaders
    of AI research for decades. Many of them predicted that a machine as intelligent as
    a human being would exist in no more than a generation, and they were given millions
    of dollars to make this vision come true. Also note that the author is a PhD student in Computer Science.

    Eventually, it became obvious that commercial developers and researchers had
    grossly underestimated the difficulty of the project. In 1974, in response to the
    criticism from James Lighthill and ongoing pressure from congress, the U.S. and
    British Governments stopped funding undirected research into artificial intelligence,
    and the difficult years that followed would later be known as an "AI winter".

    In the early 1980s, AI research was revived by the commercial success of expert
    systems, a form of AI program that simulated the knowledge and analytical skills
    of human experts. By 1985, the market for AI had reached over a billion dollars.
    At the same time, Japan's fifth generation computer project inspired the U.S and
    British governments to restore funding for academic research.
    """

    question = "What is the name of the author mentioned in the context and what does he study?"

    # Apply chat template
    messages = [
        {"role": "user", "content": context + "\n\n" + question}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    print(f"Input prompt length: {input_length} tokens")
    print()

    # -------------------------------------------------------------------------
    # STEP 2c: Create Presses with Logging
    # -------------------------------------------------------------------------
    print("Setting up compression presses...")

    # Prefill press with logging wrapper
    base_prefill_press = KnormPress(compression_ratio=prefill_compression_ratio)
    prefill_press = LoggingPressWrapper(
        press=base_prefill_press,
        log_every_n_layers=log_every_n_layers,
        verbose=True
    )

    # Decoding press with logging
    decoding_press = LoggingDecodingPress(
        base_press=KnormPress(),
        compression_interval=decode_compression_interval,
        target_size=decode_target_size,
        hidden_states_buffer_size=128,
        verbose=True
    )

    # Combined press
    combined_press = PrefillDecodingPress(
        prefilling_press=prefill_press,
        decoding_press=decoding_press
    )

    print(f"  Prefill press: KnormPress(compression_ratio={prefill_compression_ratio})")
    print(f"  Decode press: DecodingPress(interval={decode_compression_interval}, target={decode_target_size})")
    print()

    # -------------------------------------------------------------------------
    # STEP 2d: Run Inference with Compression
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("RUNNING INFERENCE WITH KV CACHE COMPRESSION")
    print("=" * 80)
    print()

    # Create cache
    cache = DynamicCache()

    print("-"*5)
    print(f"Cache size is {cache.get_seq_length()} \n")

    with torch.no_grad(), combined_press(model):
        print("-" * 40)
        print("PREFILL PHASE (processing context)")
        print("-" * 40)

        # Prefill: process the entire input
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            use_cache=True,
        )

        prefill_cache_size = cache.get_seq_length()
        print(f"\n>>> After prefill: Cache size = {prefill_cache_size} tokens")
        print(f">>> Compression: {input_length} -> {prefill_cache_size} "
              f"({100 * (1 - prefill_cache_size/input_length):.1f}% reduced)")
        print()

        print("-" * 40)
        print("DECODE PHASE (generating tokens)")
        print("-" * 40)

        # Decode: generate tokens one by one
        generated_ids = []
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        position_ids = torch.tensor([[prefill_cache_size]], device=device)

        for i in range(max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

            position_ids = position_ids + 1

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"\n>>> EOS token generated at step {i+1}")
                break

            # Print progress every 20 tokens
            if (i + 1) % 20 == 0:
                current_cache_size = cache.get_seq_length()
                print(f"\n>>> Generated {i+1} tokens, current cache size: {current_cache_size}")

    # -------------------------------------------------------------------------
    # STEP 2e: Print Results
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    final_cache_size = cache.get_seq_length()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\nInput length: {input_length} tokens")
    print(f"Cache size after prefill: {prefill_cache_size} tokens")
    print(f"Tokens generated: {len(generated_ids)}")
    print(f"Final cache size: {final_cache_size} tokens")
    print(f"\nGenerated text:\n{'-' * 40}")
    print(generated_text)
    print("-" * 40)

    return {
        "input_length": input_length,
        "prefill_cache_size": prefill_cache_size,
        "tokens_generated": len(generated_ids),
        "final_cache_size": final_cache_size,
        "generated_text": generated_text,
    }


# =============================================================================
# STEP 3: Alternative - Simple Observation Without Custom Wrapper
# =============================================================================

# def simple_cache_observation(
#     model_name: str = "Qwen/Qwen2.5-14B-Instruct",
#     device: str = "cuda:0",
# ):
#     """
#     A simpler version that just observes cache sizes at key points.
#     """

#     print("=" * 80)
#     print("SIMPLE KV CACHE OBSERVATION")
#     print("=" * 80)

#     # Load model
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map=device,
#         trust_remote_code=True,
#     )
#     model.eval()

#     # Prepare input
#     prompt = "Explain the theory of relativity in detail."
#     messages = [{"role": "user", "content": prompt}]
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(text, return_tensors="pt").to(device)

#     print(f"\nPrompt length: {inputs['input_ids'].shape[1]} tokens")

#     # Create press
#     press = KnormPress(compression_ratio=0.5)
#     cache = DynamicCache()

#     print("\n--- Without Compression ---")
#     with torch.no_grad():
#         outputs = model(**inputs, past_key_values=DynamicCache(), use_cache=True)
#         print(f"Cache size: {outputs.past_key_values.get_seq_length()} tokens")

#     print("\n--- With 50% Compression ---")
#     with torch.no_grad(), press(model):
#         outputs = model(**inputs, past_key_values=cache, use_cache=True)
#         print(f"Cache size: {cache.get_seq_length()} tokens")

#     # Print per-layer cache sizes
#     print("\n--- Per-Layer Cache Sizes ---")
#     for layer_idx in range(len(cache)):
#         layer_cache_size = cache.get_seq_length(layer_idx)
#         print(f"  Layer {layer_idx:2d}: {layer_cache_size} tokens")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KV Cache Compression Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Model name")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--prefill-ratio", type=float, default=0.5,
                        help="Prefill compression ratio")
    parser.add_argument("--decode-target", type=int, default=512,
                        help="Decode target cache size")
    parser.add_argument("--decode-interval", type=int, default=256,
                        help="Decode compression interval")
    parser.add_argument("--max-tokens", type=int, default=1000,
                        help="Max new tokens to generate")
    parser.add_argument("--log-layers", type=int, default=4,
                        help="Log every N layers")
    parser.add_argument("--simple", action="store_true",
                        help="Run simple observation instead")

    args = parser.parse_args()

    if args.simple:
        simple_cache_observation(
            model_name=args.model,
            device=args.device,
        )
    else:
        run_experiment(
            model_name=args.model,
            device=args.device,
            prefill_compression_ratio=args.prefill_ratio,
            decode_target_size=args.decode_target,
            decode_compression_interval=args.decode_interval,
            max_new_tokens=args.max_tokens,
            log_every_n_layers=args.log_layers,
        )
