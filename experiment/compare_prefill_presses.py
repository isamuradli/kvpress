"""
Compare 10 Different Prefill Presses

Tests multiple prefill compression methods on the same model, context, and questions.
Collects metrics and saves results to a file.
"""

import torch
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import (
    KnormPress,
    SnapKVPress,
    StreamingLLMPress,
    RandomPress,
    TOVAPress,
    ExpectedAttentionPress,
    PyramidKVPress,
    KeyDiffPress,
    AdaKVPress,
    CURPress,
)


@dataclass
class PressResult:
    """Results from testing a single press."""
    press_name: str
    compression_ratio: float
    input_tokens: int
    cache_size_after_prefill: int
    actual_compression: float
    generation_time_ms: float
    generated_text: str
    answer_correct: bool


# Sample context and question
CONTEXT = """
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

The author of this document is named Alice and she is 32 years old. She works at
a research lab in California studying machine learning algorithms.

In the early 1980s, AI research was revived by the commercial success of expert
systems, a form of AI program that simulated the knowledge and analytical skills
of human experts. By 1985, the market for AI had reached over a billion dollars.
"""

QUESTION = "How old is the author?"
EXPECTED_ANSWER = "32"


def get_presses_to_test(compression_ratio: float = 0.5):
    """Return list of (name, press) tuples to test."""
    return [
        ("KnormPress", KnormPress(compression_ratio=compression_ratio)),
        ("SnapKVPress", SnapKVPress(compression_ratio=compression_ratio, window_size=4)),
        ("StreamingLLMPress", StreamingLLMPress(compression_ratio=compression_ratio)),
        ("RandomPress", RandomPress(compression_ratio=compression_ratio)),
        ("TOVAPress", TOVAPress(compression_ratio=compression_ratio)),
        ("ExpectedAttentionPress", ExpectedAttentionPress(compression_ratio=compression_ratio)),
        ("PyramidKVPress", PyramidKVPress(compression_ratio=compression_ratio, window_size=4)),
        ("KeyDiffPress", KeyDiffPress(compression_ratio=compression_ratio)),
        ("AdaKV_SnapKV", AdaKVPress(SnapKVPress(compression_ratio=compression_ratio, window_size=4))),
        ("CURPress", CURPress(compression_ratio=compression_ratio)),
    ]


def test_single_press(
    model,
    tokenizer,
    press,
    press_name: str,
    inputs: dict,
    input_length: int,
    compression_ratio: float,
    device: str,
    max_new_tokens: int = 50,
) -> PressResult:
    """Test a single press and return results."""

    cache = DynamicCache()

    start_time = time.perf_counter()

    with torch.no_grad(), press(model):
        # Prefill phase
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            use_cache=True,
        )

        cache_size_after_prefill = cache.get_seq_length()

        # Decode phase - generate tokens
        generated_ids = []
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        position_ids = torch.tensor([[cache_size_after_prefill]], device=device)

        for _ in range(max_new_tokens - 1):
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

            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.perf_counter()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    actual_compression = 1 - (cache_size_after_prefill / input_length)
    answer_correct = EXPECTED_ANSWER in generated_text

    return PressResult(
        press_name=press_name,
        compression_ratio=compression_ratio,
        input_tokens=input_length,
        cache_size_after_prefill=cache_size_after_prefill,
        actual_compression=round(actual_compression, 3),
        generation_time_ms=round((end_time - start_time) * 1000, 2),
        generated_text=generated_text.strip(),
        answer_correct=answer_correct,
    )


def run_comparison(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda:0",
    compression_ratio: float = 0.5,
    max_new_tokens: int = 50,
    output_file: str = "press_comparison_results.json",
):
    """Run comparison of all presses."""

    print("=" * 70)
    print("PREFILL PRESS COMPARISON")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Target compression ratio: {compression_ratio}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    print()

    # Prepare input
    messages = [{"role": "user", "content": CONTEXT + "\n\n" + QUESTION}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    print(f"Input length: {input_length} tokens")
    print(f"Question: {QUESTION}")
    print(f"Expected answer contains: '{EXPECTED_ANSWER}'")
    print()

    # Test each press
    presses = get_presses_to_test(compression_ratio)
    results = []

    print("-" * 70)
    print(f"{'Press Name':<25} {'Cache Size':>10} {'Actual Comp':>12} {'Time (ms)':>10} {'Correct':>8}")
    print("-" * 70)

    for press_name, press in presses:
        try:
            result = test_single_press(
                model=model,
                tokenizer=tokenizer,
                press=press,
                press_name=press_name,
                inputs=inputs,
                input_length=input_length,
                compression_ratio=compression_ratio,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            results.append(result)

            print(f"{result.press_name:<25} {result.cache_size_after_prefill:>10} "
                  f"{result.actual_compression:>11.1%} {result.generation_time_ms:>10.1f} "
                  f"{'Yes' if result.answer_correct else 'No':>8}")

        except Exception as e:
            print(f"{press_name:<25} ERROR: {str(e)[:40]}")

    print("-" * 70)
    print()

    # Summary
    correct_count = sum(1 for r in results if r.answer_correct)
    print(f"Correct answers: {correct_count}/{len(results)}")
    print()

    # Save results
    output_data = {
        "config": {
            "model": model_name,
            "compression_ratio": compression_ratio,
            "input_tokens": input_length,
            "question": QUESTION,
            "expected_answer": EXPECTED_ANSWER,
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print generated texts
    print()
    print("=" * 70)
    print("GENERATED TEXTS")
    print("=" * 70)
    for r in results:
        status = "OK" if r.answer_correct else "WRONG"
        print(f"\n[{r.press_name}] ({status})")
        print(f"  {r.generated_text[:100]}...")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare prefill presses")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compression-ratio", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default="press_comparison_results.json")

    args = parser.parse_args()

    run_comparison(
        model_name=args.model,
        device=args.device,
        compression_ratio=args.compression_ratio,
        max_new_tokens=args.max_tokens,
        output_file=args.output,
    )
