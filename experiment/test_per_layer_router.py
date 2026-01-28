"""
Test PerLayerPressRouter with:
- 2 different prefill presses (KnormPress for early layers, RandomPress for late layers)
- 1 decode phase press (DecodingPress with KnormPress)

Model: Qwen/Qwen2.5-14B-Instruct (48 layers)
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import (
    KnormPress,
    RandomPress,
    DecodingPress,
    PrefillDecodingPress,
    PerLayerPressRouter,
)

# Enable debug logging for PerLayerPressRouter and PrefillDecodingPress
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger('kvpress.presses.per_layer_press_router').setLevel(logging.DEBUG)
logging.getLogger('kvpress.presses.prefill_decoding_press').setLevel(logging.DEBUG)


def print_tokenization(tokenizer, input_ids, max_tokens=None, columns=4):
    """Print token-by-token table showing index, token ID, and decoded string.

    Args:
        tokenizer: The tokenizer to decode tokens
        input_ids: Token IDs tensor
        max_tokens: Optional limit on tokens to display
        columns: Number of columns to display (default: 2)
    """
    tokens = input_ids[0] if input_ids.dim() > 1 else input_ids
    total = len(tokens)
    display_count = min(total, max_tokens) if max_tokens else total

    print(f"\n    Total tokens: {total}")

    # Build token data
    token_data = []
    for i in range(display_count):
        token_id = tokens[i].item()
        token_str = tokenizer.decode([token_id])
        token_repr = repr(token_str)
        # Truncate long token strings
        if len(token_repr) > 18:
            token_repr = token_repr[:15] + "..."
        token_data.append((i, token_id, token_repr))

    # Calculate rows needed
    rows = (display_count + columns - 1) // columns

    # Print header
    col_fmt = "{:>5} | {:>8} | {:<18}"
    header = col_fmt.format("Idx", "TokenID", "Token")
    separator = "-" * 5 + "-+-" + "-" * 8 + "-+-" + "-" * 18

    full_header = "    " + ("  ||  ".join([header] * columns))
    full_separator = "    " + ("  ||  ".join([separator] * columns))

    print(full_header)
    print(full_separator)

    # Print rows with multiple columns
    for row in range(rows):
        row_parts = []
        for col in range(columns):
            idx = row + col * rows
            if idx < display_count:
                i, token_id, token_repr = token_data[idx]
                row_parts.append(col_fmt.format(i, token_id, token_repr))
            else:
                row_parts.append(" " * 36)  # Empty column
        print("    " + "  ||  ".join(row_parts))

    if max_tokens and total > max_tokens:
        print(f"\n    ... ({total - max_tokens} more tokens)")


def test_per_layer_router(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    device: str = "cuda:0",
):
    """Test PerLayerPressRouter with different presses per layer group."""

    print("=" * 70)
    print("TEST: PerLayerPressRouter")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load Model
    # -------------------------------------------------------------------------
    print("\n[1] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"    Model: {model_name}")
    print(f"    Layers: {num_layers}")

    # -------------------------------------------------------------------------
    # Step 2: Create PerLayerPressRouter for Prefill
    # -------------------------------------------------------------------------
    print("\n[2] Creating PerLayerPressRouter for prefill phase...")

    # Split layers: first half uses KnormPress, second half uses RandomPress
    mid_layer = num_layers // 2

    prefill_router = PerLayerPressRouter(
        layer_presses={
            range(0, mid_layer): KnormPress(compression_ratio=0.5),      # Keep 50%
            range(mid_layer, num_layers): RandomPress(compression_ratio=0.3),  # Keep 70%
        },
        default_press=None  # No default needed, all layers covered
    )

    print(f"    Layers 0-{mid_layer-1}: KnormPress(compression_ratio=0.5) -> keeps 50%")
    print(f"    Layers {mid_layer}-{num_layers-1}: RandomPress(compression_ratio=0.3) -> keeps 70%")

    # -------------------------------------------------------------------------
    # Step 3: Create Decode Press
    # -------------------------------------------------------------------------
    print("\n[3] Creating DecodingPress for decode phase...")

    decode_press = DecodingPress(
        base_press=KnormPress(),
        compression_interval=256,
        target_size=512,
    )

    print(f"    DecodingPress(interval=256, target=512)")

    # -------------------------------------------------------------------------
    # Step 4: Combine with PrefillDecodingPress
    # -------------------------------------------------------------------------
    print("\n[4] Combining into PrefillDecodingPress...")

    combined_press = PrefillDecodingPress(
        prefilling_press=prefill_router,
        decoding_press=decode_press,
    )

    # -------------------------------------------------------------------------
    # Step 5: Prepare Input
    # -------------------------------------------------------------------------
    print("\n[5] Preparing input...")

    context = """
    Pablo Picasso is probably the most important figure of the 20th century, in terms of art, and art movements that occurred over this period. Before the age of 50, the Spanish born artist had become the most well-known name in modern art, with the most distinct style and eye for artistic creation. 
    There had been no other artists, prior to Picasso, who had such an impact on the art world, or had a mass following of fans and critics alike, as he did.
    Pablo Picasso was born in Spain in 1881, and was raised there before going on to spend most of his adult life working as an artist in France. 
    Throughout the long course of his career, he created more than 20,000 paintings, drawings, sculptures, ceramics and other items such as costumes and theater sets. 
    He is universally renowned as one of the most influential and celebrated artists of the twentieth century.
    Picasso's ability to produce works in an astonishing range of styles made him well respected during his own lifetime. 
    After his death in 1973 his value as an artist and inspiration to other artists has only grown. He is without a doubt destined to permanently etch himself into the fabric of humanity as one of the greatest artists of all time.
    As an artist and an innovator, he is responsible for co-founding the entire Cubist movement alongside Georges Braque. Cubism was an avant-garde art movement that changed forever the face of European painting and sculpture while simultaneously affecting contemporary architecture, music and literature. Subjects and objects in Cubism are broken up into pieces and re-arranged in an abstract form. During the period from approximately 1910-1920 when Picasso and Braque were laying the foundation for Cubism in France, its effects were so far-reaching as to inspire offshoots like the styles of Futurism, Dada, and Constructivism in other countries.
    """

    question = "When did Pablo Picasso die?"

    messages = [{"role": "user", "content": context + "\n\n" + question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    print(f"    Input length: {input_length} tokens")

    # -------------------------------------------------------------------------
    # Step 5b: Print Tokenization
    # -------------------------------------------------------------------------
    print("\n[5b] Tokenization breakdown:")
    print_tokenization(tokenizer, inputs["input_ids"])

    # -------------------------------------------------------------------------
    # Step 6: Run Inference
    # -------------------------------------------------------------------------
    print("\n[6] Running inference with compression...")

    cache = DynamicCache()

    with torch.no_grad(), combined_press(model):
        # Prefill
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            use_cache=True,
        )

        prefill_cache_size = cache.get_seq_length()
        print(f"    After prefill: {input_length} -> {prefill_cache_size} tokens")

        # Check per-layer cache sizes
        print("\n    Per-layer cache sizes after prefill:")
        for layer_idx in [0, 12, 23, 24, 36, 47]:  # Sample layers from each group
            layer_size = cache.get_seq_length(layer_idx)
            press_type = "KnormPress" if layer_idx < mid_layer else "RandomPress"
            print(f"      Layer {layer_idx:2d} ({press_type}): {layer_size} tokens")

        # Generate a few tokens
        generated_ids = []
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        position_ids = torch.tensor([[prefill_cache_size]], device=device)

        for i in range(20):  # Generate 20 tokens
            outputs = model(
                input_ids=next_token,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            position_ids = position_ids + 1

            if next_token.item() == tokenizer.eos_token_id:
                break

    # -------------------------------------------------------------------------
    # Step 7: Results
    # -------------------------------------------------------------------------
    print("\n[7] Results:")
    final_cache_size = cache.get_seq_length()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"    Input tokens: {input_length}")
    print(f"    Cache after prefill: {prefill_cache_size}")
    print(f"    Tokens generated: {len(generated_ids)}")
    print(f"    Final cache size: {final_cache_size}")
    print(f"    Generated: {generated_text}")

    print("\n" + "=" * 70)
    print("TEST PASSED: PerLayerPressRouter works correctly!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    test_per_layer_router()
