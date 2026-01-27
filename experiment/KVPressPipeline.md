# KVPress Pipeline Deep Dive

This document explains where each component of the pipeline comes from and what operations are executed internally.

---

## Table of Contents

1. [Pipeline Inheritance Chain](#1-pipeline-inheritance-chain)
2. [The \_\_call\_\_ Method - Where It Comes From](#2-the-__call__-method---where-it-comes-from)
3. [Complete Execution Flow](#3-complete-execution-flow)
4. [What Comes From Where](#4-what-comes-from-where)
5. [Internal Transformer Operations](#5-internal-transformer-operations)
6. [KVPress Contribution](#6-kvpress-contribution)
7. [Pipeline vs BasePress: Understanding the Difference](#7-pipeline-vs-basepress-understanding-the-difference)
8. [Experiment: Observing KV Cache Compression](#8-experiment-observing-kv-cache-compression)
9. [Function Attributes Deep Dive](#9-function-attributes-deep-dive)
10. [Per-Layer Compression: Advanced Techniques](#10-per-layer-compression-advanced-techniques)

---

## 1. Pipeline Inheritance Chain

```python
# kvpress/pipeline.py:24
class KVPressTextGenerationPipeline(Pipeline):
    ...
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INHERITANCE CHAIN                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  transformers.Pipeline (HuggingFace)                                        │
│       │                                                                     │
│       │  Provides:                                                          │
│       │  • __call__() method                                                │
│       │  • __init__() with model/tokenizer loading                          │
│       │  • Device management                                                │
│       │  • Batch processing infrastructure                                  │
│       │                                                                     │
│       ▼                                                                     │
│  KVPressTextGenerationPipeline (kvpress)                                    │
│       │                                                                     │
│       │  Overrides:                                                         │
│       │  • _sanitize_parameters()                                           │
│       │  • preprocess()                                                     │
│       │  • _forward()                                                       │
│       │  • postprocess()                                                    │
│       │                                                                     │
│       │  Adds:                                                              │
│       │  • generate_answer()                                                │
│       │  • _remove_answer_from_cache()                                      │
│       │  • KV cache compression integration                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The \_\_call\_\_ Method - Where It Comes From

The `__call__` method is **NOT** implemented in kvpress. It comes from HuggingFace's Pipeline base class:

```python
# From transformers/pipelines/base.py (HuggingFace)
class Pipeline:
    def __call__(self, inputs, **kwargs):
        # 1. Sanitize parameters
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # 2. Preprocess (tokenization)
        model_inputs = self.preprocess(inputs, **preprocess_params)

        # 3. Forward pass (model inference)
        model_outputs = self._forward(model_inputs, **forward_params)

        # 4. Postprocess (decode outputs)
        outputs = self.postprocess(model_outputs, **postprocess_params)

        return outputs
```

> **Key Insight:** KVPress overrides the three main methods but relies on HuggingFace's orchestration.

---

## 3. Complete Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           pipeline(context="...", question="...", press=KnormPress())       │
│                                                                             │
│                    HuggingFace Pipeline.__call__()                          │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ _sanitize_    │         │   preprocess()  │         │  postprocess()  │
│ parameters()  │         │                 │         │                 │
│               │         │  [KVPRESS]      │         │  [KVPRESS]      │
│ [KVPRESS]     │         │                 │         │                 │
│               │         │  Tokenization   │         │  Format output  │
│ Split kwargs  │         │  via HuggingFace│         │  as dict        │
│ into 3 dicts  │         │  tokenizer      │         │                 │
└───────────────┘         └────────┬────────┘         └─────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │         _forward()           │
                    │                              │
                    │         [KVPRESS]            │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │ PREFILL PHASE          │  │
                    │  │                        │  │
                    │  │ with press(model):     │  │
                    │  │   self.model.model()   │  │
                    │  │        ↓               │  │
                    │  │   [HUGGINGFACE]        │  │
                    │  │   Embedding            │  │
                    │  │   Positional Encoding  │  │
                    │  │   Attention Layers     │  │
                    │  │   FFN Layers           │  │
                    │  │        ↓               │  │
                    │  │   [KVPRESS HOOKS]      │  │
                    │  │   Compress KV Cache    │  │
                    │  └────────────────────────┘  │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │ DECODE PHASE           │  │
                    │  │                        │  │
                    │  │ generate_answer()      │  │
                    │  │   self.model()         │  │
                    │  │        ↓               │  │
                    │  │   [HUGGINGFACE]        │  │
                    │  │   Full forward pass    │  │
                    │  │   + LM Head            │  │
                    │  │        ↓               │  │
                    │  │   logits.argmax()      │  │
                    │  │   [KVPRESS]            │  │
                    │  └────────────────────────┘  │
                    └──────────────────────────────┘
```

---

## 4. What Comes From Where

### Summary Table

| Operation | Source | File/Location |
|-----------|--------|---------------|
| `__call__()` orchestration | HuggingFace | `transformers/pipelines/base.py` |
| `_sanitize_parameters()` | KVPress | `kvpress/pipeline.py:39-102` |
| `preprocess()` | KVPress | `kvpress/pipeline.py:104-170` |
| `_forward()` | KVPress | `kvpress/pipeline.py:172-246` |
| `postprocess()` | KVPress | `kvpress/pipeline.py:317-320` |
| Tokenization | HuggingFace | `self.tokenizer.encode()` |
| Chat template | HuggingFace | `self.tokenizer.apply_chat_template()` |
| Token embedding | HuggingFace | `model.model.embed_tokens()` |
| Positional encoding (RoPE) | HuggingFace | `model.model.rotary_emb()` |
| Attention computation | HuggingFace | `model.model.layers[i].self_attn()` |
| FFN layers | HuggingFace | `model.model.layers[i].mlp()` |
| LM Head (logits) | HuggingFace | `model.lm_head()` |
| KV Cache management | HuggingFace | `transformers.DynamicCache` |
| KV Cache compression | KVPress | `press.compress()` via hooks |
| Greedy decoding loop | KVPress | `pipeline.generate_answer()` |
| Detokenization | HuggingFace | `self.tokenizer.decode()` |

---

## 5. Internal Transformer Operations

All core transformer operations are executed by HuggingFace's model implementation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    self.model.model(input_ids, past_key_values=cache)       │
│                              (HuggingFace)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. TOKEN EMBEDDING                                                   │   │
│  │    model.model.embed_tokens(input_ids)                              │   │
│  │    (batch, seq_len) → (batch, seq_len, hidden_dim)                  │   │
│  │                                                                      │   │
│  │    Source: transformers/models/llama/modeling_llama.py              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. ROTARY POSITIONAL EMBEDDING (RoPE)                               │   │
│  │    model.model.rotary_emb(hidden_states, position_ids)              │   │
│  │    Computes (cos, sin) for position encoding                        │   │
│  │                                                                      │   │
│  │    Source: transformers/models/llama/modeling_llama.py              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. TRANSFORMER LAYERS (x N layers)                                  │   │
│  │                                                                      │   │
│  │    for layer in model.model.layers:                                 │   │
│  │        ┌────────────────────────────────────────────────────────┐   │   │
│  │        │ a. RMSNorm (pre-attention)                             │   │   │
│  │        │    layer.input_layernorm(hidden_states)                │   │   │
│  │        └────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │        ┌────────────────────────────────────────────────────────┐   │   │
│  │        │ b. SELF-ATTENTION                                      │   │   │
│  │        │    layer.self_attn(hidden_states, past_key_values)     │   │   │
│  │        │                                                        │   │   │
│  │        │    Internally:                                         │   │   │
│  │        │    • Q = hidden_states @ W_q                           │   │   │
│  │        │    • K = hidden_states @ W_k                           │   │   │
│  │        │    • V = hidden_states @ W_v                           │   │   │
│  │        │    • Apply RoPE to Q, K                                │   │   │
│  │        │    • Attention = softmax(Q @ K^T / sqrt(d)) @ V        │   │   │
│  │        │    • Output = Attention @ W_o                          │   │   │
│  │        │    • Update KV cache                                   │   │   │
│  │        │                                                        │   │   │
│  │        │    ════════════════════════════════════════════════    │   │   │
│  │        │    ║  KVPRESS HOOK TRIGGERS HERE (after forward)  ║    │   │   │
│  │        │    ║  press.forward_hook() → compress KV cache    ║    │   │   │
│  │        │    ════════════════════════════════════════════════    │   │   │
│  │        └────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │        ┌────────────────────────────────────────────────────────┐   │   │
│  │        │ c. RESIDUAL CONNECTION                                 │   │   │
│  │        │    hidden_states = hidden_states + attn_output         │   │   │
│  │        └────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │        ┌────────────────────────────────────────────────────────┐   │   │
│  │        │ d. RMSNorm (pre-FFN)                                   │   │   │
│  │        │    layer.post_attention_layernorm(hidden_states)       │   │   │
│  │        └────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │        ┌────────────────────────────────────────────────────────┐   │   │
│  │        │ e. FEED-FORWARD NETWORK (MLP)                          │   │   │
│  │        │    layer.mlp(hidden_states)                            │   │   │
│  │        │                                                        │   │   │
│  │        │    Internally (SwiGLU for Llama):                      │   │   │
│  │        │    • gate = hidden @ W_gate                            │   │   │
│  │        │    • up = hidden @ W_up                                │   │   │
│  │        │    • output = (SiLU(gate) * up) @ W_down               │   │   │
│  │        └────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │        ┌────────────────────────────────────────────────────────┐   │   │
│  │        │ f. RESIDUAL CONNECTION                                 │   │   │
│  │        │    hidden_states = hidden_states + mlp_output          │   │   │
│  │        └────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. FINAL LAYER NORM                                                 │   │
│  │    model.model.norm(hidden_states)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                         Output: hidden_states                               │
│                         (batch, seq_len, hidden_dim)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### For Generation (with LM Head):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    self.model(input_ids, past_key_values=cache)             │
│                              (HuggingFace)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1-4. Same as above (embed → layers → norm)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 5. LM HEAD (Language Model Head)                                    │   │
│  │    logits = model.lm_head(hidden_states)                            │   │
│  │    (batch, seq_len, hidden_dim) → (batch, seq_len, vocab_size)      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                         Output: CausalLMOutputWithPast                      │
│                         • logits: (batch, seq_len, vocab_size)              │
│                         • past_key_values: updated cache                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. KVPress Contribution

KVPress contributes only the KV cache compression logic. Everything else is HuggingFace:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KVPRESS CONTRIBUTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PIPELINE CUSTOMIZATION                                                  │
│     • _sanitize_parameters() - Parse press and other kwargs                 │
│     • preprocess() - Uses HF tokenizer but adds chat template logic         │
│     • _forward() - Orchestrates prefill/decode with press context           │
│     • postprocess() - Simple dict formatting                                │
│     • generate_answer() - Custom greedy decoding loop                       │
│                                                                             │
│  2. KV CACHE COMPRESSION (the core innovation)                              │
│     • BasePress.__call__() - Context manager for hook registration          │
│     • BasePress.forward_hook() - Intercepts attention layer output          │
│     • ScorerPress.score() - Computes importance scores                      │
│     • ScorerPress.compress() - Prunes low-scoring KV pairs                  │
│                                                                             │
│  3. REGISTRATION                                                            │
│     • Registers "kv-press-text-generation" pipeline with HuggingFace        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What KVPress Does NOT Implement:

| Operation | Implemented By |
|-----------|----------------|
| Tokenization | HuggingFace Tokenizer |
| Token Embedding | HuggingFace Model (`embed_tokens`) |
| Positional Encoding (RoPE) | HuggingFace Model (`rotary_emb`) |
| Q/K/V Projections | HuggingFace Model (`q_proj`, `k_proj`, `v_proj`) |
| Attention Computation | HuggingFace Model (`self_attn`) |
| Feed-Forward Network | HuggingFace Model (`mlp`) |
| Layer Normalization | HuggingFace Model (`input_layernorm`, etc.) |
| LM Head | HuggingFace Model (`lm_head`) |
| KV Cache Data Structure | HuggingFace (`DynamicCache`, `QuantizedCache`) |
| Detokenization | HuggingFace Tokenizer |

### What KVPress DOES Implement:

| Operation | Location |
|-----------|----------|
| Hook registration on attention layers | `base_press.py:158-201` |
| KV extraction from cache | `base_press.py:142` |
| Importance scoring | `scorer_press.py:35-74` |
| Top-k selection and pruning | `scorer_press.py:76-102` |
| Cache update with compressed KV | `base_press.py:146-154` |
| Custom pipeline flow | `pipeline.py` |

### Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  pipeline(context, question, press=KnormPress(0.5))                         │
│       │                                                                     │
│       │  ┌──────────────────────────────────────────────────────────────┐  │
│       │  │                    HUGGINGFACE                               │  │
│       │  │                                                              │  │
│       │  │  • Pipeline.__call__() orchestration                         │  │
│       │  │  • Tokenizer (encode/decode)                                 │  │
│       │  │  • Model architecture (LlamaForCausalLM, etc.)               │  │
│       │  │  • All transformer operations:                               │  │
│       │  │    - Embedding, RoPE, Attention, FFN, LayerNorm, LM Head     │  │
│       │  │  • KV Cache data structure (DynamicCache)                    │  │
│       │  │                                                              │  │
│       │  └──────────────────────────────────────────────────────────────┘  │
│       │                              +                                      │
│       │  ┌──────────────────────────────────────────────────────────────┐  │
│       │  │                      KVPRESS                                 │  │
│       │  │                                                              │  │
│       │  │  • Custom pipeline methods (preprocess, _forward, etc.)      │  │
│       │  │  • Forward hooks on attention layers                         │  │
│       │  │  • KV cache compression logic:                               │  │
│       │  │    - score() → importance scores                             │  │
│       │  │    - compress() → prune low-scoring tokens                   │  │
│       │  │  • Greedy decoding loop                                      │  │
│       │  │                                                              │  │
│       │  └──────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       ▼                                                                     │
│  {"answer": "The answer based on compressed context..."}                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Key Insight:** KVPress is a thin layer on top of HuggingFace Transformers. It leverages PyTorch's forward hooks to intercept the KV cache after each attention layer and compress it in-place. All heavy lifting (tokenization, embedding, attention, FFN, etc.) is done by HuggingFace.

---

## 7. Pipeline vs BasePress: Understanding the Difference

These two files serve distinct but complementary roles:

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROLE COMPARISON                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  pipeline.py                          base_press.py                         │
│  ────────────                         ──────────────                        │
│  "WHAT to run"                        "HOW to compress"                     │
│                                                                             │
│  • Orchestrates the inference         • Defines compression interface       │
│  • Handles input/output               • Registers hooks on model            │
│  • Manages the generation loop        • Intercepts & compresses KV cache    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Comparison

| Aspect | `pipeline.py` | `base_press.py` |
|--------|---------------|-----------------|
| Purpose | End-to-end inference orchestration | KV cache compression mechanism |
| Inherits From | HuggingFace Pipeline | None (base class for presses) |
| Main Class | `KVPressTextGenerationPipeline` | `BasePress` |
| User Interaction | Called directly by user | Passed as parameter to pipeline |
| Model Interaction | Calls `model()` and `model.model()` | Attaches hooks to `model.layers` |
| Scope | Full inference (tokenize → generate → decode) | Only KV cache manipulation |

### How They Work Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  # User code                                                                │
│  pipeline = KVPressTextGenerationPipeline(model, tokenizer)                 │
│  press = KnormPress(compression_ratio=0.5)                                  │
│  result = pipeline(context, question, press=press)                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PIPELINE.PY                              BASE_PRESS.PY                     │
│  ════════════                             ══════════════                    │
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │ 1. preprocess()     │                                                    │
│  │    Tokenize input   │                                                    │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────┐         ┌─────────────────────────┐                │
│  │ 2. _forward()       │         │                         │                │
│  │                     │         │  with press(model):     │                │
│  │  ┌───────────────┐  │ ──────► │    __call__()           │                │
│  │  │ PREFILL PHASE │  │         │    • Register hooks     │                │
│  │  │               │  │         │      on all attention   │                │
│  │  │ with press(): │  │         │      layers             │                │
│  │  │   model.model │  │         │                         │                │
│  │  │   (context)   │  │         └───────────┬─────────────┘                │
│  │  └───────────────┘  │                     │                              │
│  │         │           │                     ▼                              │
│  │         │           │         ┌─────────────────────────┐                │
│  │         │           │         │  forward_hook()         │                │
│  │         │           │         │  (called after each     │                │
│  │         │           │         │   attention layer)      │                │
│  │         │           │         │                         │                │
│  │         │           │         │  • Extract KV from cache│                │
│  │         │           │         │  • Call compress()      │                │
│  │         │           │         │  • Update cache         │                │
│  │         │           │         └─────────────────────────┘                │
│  │         │           │                                                    │
│  │  ┌──────▼────────┐  │                                                    │
│  │  │ DECODE PHASE  │  │                                                    │
│  │  │               │  │                                                    │
│  │  │ generate_     │  │                                                    │
│  │  │ answer()      │  │                                                    │
│  │  │ (greedy loop) │  │                                                    │
│  │  └───────────────┘  │                                                    │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────┐                                                    │
│  │ 3. postprocess()    │                                                    │
│  │    Return answer    │                                                    │
│  └─────────────────────┘                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Code Sections

**pipeline.py - Calls the model:**

```python
# pipeline.py:215-220 - Orchestrates WHAT runs
with press(self.model) if perform_prefill_compression else contextlib.nullcontext():
    self.model.model(          # Calls HuggingFace model
        input_ids=context_ids,
        past_key_values=cache,
    )
```

**base_press.py - Hooks into the model:**

```python
# base_press.py:158-201 - Defines HOW compression happens
@contextmanager
def __call__(self, model):
    # Register hooks on every attention layer
    for layer in model.model.layers:
        hooks.append(
            layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True)
        )
    yield  # Model runs here, hooks trigger
    # Remove hooks when done

# base_press.py:95-156 - The hook that compresses
def forward_hook(self, module, input, kwargs, output):
    # Extract KV from cache
    keys, values = extract_keys_and_values(cache, module.layer_idx)

    # Compress (implemented by subclasses like KnormPress)
    keys, values = self.compress(module, hidden_states, keys, values, ...)

    # Update cache with compressed KV
    cache_layer.keys = keys
    cache_layer.values = values
```

### Analogy

| File | Role | Analogy |
|------|------|---------|
| `pipeline.py` | Driver - orchestrates the full inference flow | The car driver (decides where to go) |
| `base_press.py` | Modifier - intercepts and compresses KV cache | The mechanic (modifies how engine works) |

### Summary Table

| Question | `pipeline.py` | `base_press.py` |
|----------|---------------|-----------------|
| What does it control? | Full inference flow | KV cache compression only |
| When is it used? | Called by user to run inference | Activated via `with press(model):` |
| What does it call? | HuggingFace model | Nothing - it's called by PyTorch hooks |
| What does it return? | Generated answers | Compressed `(keys, values)` |
| Can it work alone? | Yes (without compression) | No (needs to be passed to pipeline/model) |

---

## 8. Experiment: Observing KV Cache Compression

This section explains how to run an experiment to observe KV cache compression during both prefill and decode phases.

### 8.1 Setup: Combined Prefill + Decode Compression

```python
from transformers import pipeline
from kvpress import (
    KnormPress,
    DecodingPress,
    PrefillDecodingPress,
)

# Initialize pipeline
device = "cuda:0"
model = "Qwen/Qwen2.5-7B-Instruct"
pipe = pipeline("kv-press-text-generation", model=model, device=device, torch_dtype="auto")

# 1. Prefill press - compresses the initial context (50% compression)
prefill_press = KnormPress(compression_ratio=0.5)

# 2. Decoding press - compresses during token generation
decoding_press = DecodingPress(
    base_press=KnormPress(),
    compression_interval=256,    # Compress every 256 generated tokens
    target_size=512,             # Keep 512 tokens after each compression
    hidden_states_buffer_size=128
)

# 3. Combine both
combined_press = PrefillDecodingPress(
    prefilling_press=prefill_press,
    decoding_press=decoding_press
)

# Run inference
response = pipe(context, question=question, press=combined_press)["answer"]
```

### 8.2 Observing Cache Sizes After Each Layer

To observe KV cache sizes, you can create a custom logging wrapper:

```python
from dataclasses import dataclass
from kvpress.presses.base_press import BasePress

@dataclass
class LoggingPressWrapper(BasePress):
    """Wraps any press and logs cache sizes after each layer."""

    press: BasePress
    verbose: bool = True

    def forward_hook(self, module, input, kwargs, output):
        layer_idx = module.layer_idx
        cache = kwargs["past_key_values"]

        # Get cache size BEFORE compression
        cache_before = cache.get_seq_length(layer_idx)

        # Call wrapped press
        result = self.press.forward_hook(module, input, kwargs, output)

        # Get cache size AFTER compression
        cache_after = cache.get_seq_length(layer_idx)

        if self.verbose:
            status = "COMPRESSED" if cache_before != cache_after else "unchanged"
            print(f"Layer {layer_idx:2d}: {cache_before:5d} -> {cache_after:5d} ({status})")

        return result
```

### 8.3 Running the Experiment

A complete experiment script is provided at `experiments/kv_cache_observation_experiment.py`.

**Run with default settings:**

```bash
cd /home/isa-grc/kvpress
python experiments/kv_cache_observation_experiment.py
```

**Run with custom parameters:**

```bash
python experiments/kv_cache_observation_experiment.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --device "cuda:0" \
    --prefill-ratio 0.5 \
    --decode-target 512 \
    --decode-interval 256 \
    --max-tokens 100 \
    --log-layers 4
```

**Run simple observation:**

```bash
python experiments/kv_cache_observation_experiment.py --simple
```

### 8.3.1 CLI Arguments Explained

#### `--model`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"Qwen/Qwen2.5-14B-Instruct"` |
| **Description** | HuggingFace model identifier. The model will be downloaded and loaded. |

**Examples:**
```bash
--model "Qwen/Qwen2.5-1.5B-Instruct"   # Small, fast, ~3GB VRAM
--model "Qwen/Qwen2.5-7B-Instruct"     # Medium, ~14GB VRAM
--model "Qwen/Qwen2.5-14B-Instruct"    # Large, ~28GB VRAM
--model "meta-llama/Llama-3.1-8B"      # Alternative model
```

> **Impact:** Larger models = better quality but more memory and slower inference

---

#### `--device`

| Property | Value |
|----------|-------|
| **Type** | string |
| **Default** | `"cuda:0"` |
| **Description** | Which GPU (or CPU) to run the model on. |

**Examples:**
```bash
--device "cuda:0"    # First GPU
--device "cuda:1"    # Second GPU (multi-GPU systems)
--device "cpu"       # CPU only (very slow, not recommended)
```

**Check available GPUs:**
```bash
python -c "import torch; print(torch.cuda.device_count())"
```

---

#### `--prefill-ratio`

| Property | Value |
|----------|-------|
| **Type** | float (0.0 to 1.0) |
| **Default** | `0.5` |
| **Description** | Compression ratio for the PREFILL phase. This is the fraction of tokens to **REMOVE** from the context. |

**Formula:**
```
tokens_kept = input_length × (1 - prefill_ratio)
```

**Examples:**
```bash
--prefill-ratio 0.0    # Keep 100% (no compression)
--prefill-ratio 0.3    # Keep 70% of tokens
--prefill-ratio 0.5    # Keep 50% of tokens (default)
--prefill-ratio 0.7    # Keep 30% of tokens (aggressive)
--prefill-ratio 0.9    # Keep 10% of tokens (very aggressive)
```

**Example calculation:**
- Input: 431 tokens, `--prefill-ratio 0.5`
- Result: `431 × (1 - 0.5) = 215 tokens kept`

> **Trade-off:** Higher ratio → More memory savings, potential quality loss

---

#### `--decode-target`

| Property | Value |
|----------|-------|
| **Type** | integer (positive) |
| **Default** | `512` |
| **Description** | Target cache size **AFTER** each decode compression event. Sets the "floor" for how small the cache will be compressed to. |

**How it works:**
1. Cache grows during generation (one token per step)
2. When compression triggers, cache is pruned to `target_size`
3. Cache grows again until next compression interval

**Examples:**
```bash
--decode-target 100    # Very aggressive, keep only 100 tokens
--decode-target 256    # Aggressive compression
--decode-target 512    # Moderate compression (default)
--decode-target 1024   # Light compression
--decode-target 2048   # Minimal compression
```

**Cache oscillation pattern** (with `--decode-interval 50`):
- `--decode-target 150`: Cache oscillates between 150 ↔ 200
- `--decode-target 512`: Cache oscillates between 512 ↔ 562

> ⚠️ **IMPORTANT:** If `cache_size <= target_size`, NO compression happens!

---

#### `--decode-interval`

| Property | Value |
|----------|-------|
| **Type** | integer (positive) |
| **Default** | `256` |
| **Description** | Number of decode steps **BETWEEN** compression events. Compression triggers every N generated tokens. |

**How it works:**
- Internal counter tracks steps per layer
- When `counter >= interval`, compression triggers
- Counter resets to 0 after compression

**Examples:**
```bash
--decode-interval 50     # Compress frequently (every 50 tokens)
--decode-interval 100    # Moderate frequency
--decode-interval 256    # Less frequent (default)
--decode-interval 512    # Infrequent compression
```

**Compression events with `--max-tokens 300`:**

| `--decode-interval` | Compressions |
|---------------------|--------------|
| 50 | 6 (at 50, 100, 150, 200, 250, 300) |
| 100 | 3 (at 100, 200, 300) |
| 256 | 1 (at 256) |
| 500 | 0 (never reaches interval!) |

> ⚠️ **IMPORTANT:** If `max_tokens < decode_interval`, decode compression **NEVER** triggers!

---

#### `--max-tokens`

| Property | Value |
|----------|-------|
| **Type** | integer (positive) |
| **Default** | `100` |
| **Description** | Maximum number of **NEW** tokens to generate during decode phase. Controls the maximum length of the model's response. |

**How it works:**
- Generation loop runs for up to `max_tokens` iterations
- Stops early if EOS (end-of-sequence) token is generated
- Each iteration generates one token

**Examples:**
```bash
--max-tokens 50      # Short responses
--max-tokens 100     # Medium responses (default)
--max-tokens 300     # Long responses
--max-tokens 1000    # Very long responses
```

**Impact on decode compression:**

| Settings | Result |
|----------|--------|
| `--max-tokens 50, --decode-interval 256` | 50 < 256, decode compression **NEVER** triggers |
| `--max-tokens 300, --decode-interval 50` | 300 > 50, decode compression triggers **6 times** |

**Memory impact:**
- Without decode compression: `cache = prefill_cache + max_tokens`
- With decode compression: `cache ≈ target_size` (bounded!)

---

#### `--log-layers`

| Property | Value |
|----------|-------|
| **Type** | integer (positive) |
| **Default** | `4` |
| **Description** | Controls logging verbosity - prints compression info every N layers. |

**How it works:**
- Logs layer if: `layer_idx % log_layers == 0`
- Layer 0 is always logged

**Examples:**

| Flag | Output |
|------|--------|
| `--log-layers 1` | Layer 0, 1, 2, 3, 4, 5, ... (ALL layers) |
| `--log-layers 4` | Layer 0, 4, 8, 12, 16, 20, 24, ... |
| `--log-layers 8` | Layer 0, 8, 16, 24, ... |
| `--log-layers 28` | Layer 0 only (minimal, for 28-layer model) |

> **Tip:** Use `--log-layers 1` initially to see all layers, then increase to reduce output.

---

#### `--simple`

| Property | Value |
|----------|-------|
| **Type** | flag (no value needed) |
| **Default** | `False` |
| **Description** | Run a simpler observation mode instead of the full experiment. |

**What it does:**
1. Runs model **WITHOUT** compression, shows cache size
2. Runs model **WITH** 50% compression, shows cache size
3. Prints per-layer cache sizes

**Usage:**
```bash
python kv_cache_experiment.py --simple
python kv_cache_experiment.py --simple --model "Qwen/Qwen2.5-1.5B"
```

---

### 8.3.2 Common Usage Patterns

#### Example 1: See decode compression in action

```bash
# Low interval + high max_tokens = multiple decode compression events
python kv_cache_experiment.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --prefill-ratio 0.5 \
    --decode-interval 50 \
    --decode-target 150 \
    --max-tokens 300 \
    --log-layers 8
```

**Expected:** Cache oscillates between 150-200, compresses ~6 times

---

#### Example 2: No decode compression (for comparison)

```bash
# High interval + low max_tokens = no decode compression
python kv_cache_experiment.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --prefill-ratio 0.5 \
    --decode-interval 256 \
    --decode-target 512 \
    --max-tokens 50 \
    --log-layers 4
```

**Expected:** Only prefill compression, cache grows linearly during decode

---

#### Example 3: No compression at all

```bash
python kv_cache_experiment.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --prefill-ratio 0.0 \
    --decode-interval 9999 \
    --max-tokens 100
```

**Expected:** No compression, `cache = input_length + max_tokens`

---

#### Example 4: Aggressive compression (memory constrained)

```bash
python kv_cache_experiment.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --prefill-ratio 0.7 \
    --decode-interval 30 \
    --decode-target 100 \
    --max-tokens 500
```

**Expected:** Heavy compression, cache bounded at ~130 tokens, quality may degrade

---

### 8.3.3 Parameter Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HOW PARAMETERS INTERACT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT CONTEXT (e.g., 431 tokens)                                           │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ --prefill-ratio 0.5                                                  │   │
│  │                                                                      │   │
│  │ 431 × (1 - 0.5) = 215 tokens                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       │ Cache starts at: 215 tokens                                         │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DECODE LOOP (runs --max-tokens times)                                │   │
│  │                                                                      │   │
│  │   for step in range(max_tokens):     # e.g., 300                     │   │
│  │       generate_one_token()                                           │   │
│  │       cache_size += 1                                                │   │
│  │                                                                      │   │
│  │       if step_count >= decode_interval:   # e.g., 50                 │   │
│  │           compress_cache_to(decode_target)  # e.g., 150              │   │
│  │           step_count = 0                                             │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  FINAL CACHE SIZE:                                                          │
│    • With decode compression: ~decode_target + (step_count % interval)      │
│    • Without compression: prefill_cache + max_tokens                        │
│                                                                             │
│  MEMORY COMPARISON (prefill=215, max_tokens=300):                           │
│    • No compression:     215 + 300 = 515 tokens                             │
│    • With compression:   ~150-200 tokens (bounded!)                         │
│    • Memory savings:     ~60-70%                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 8.4 Expected Output

```
================================================================================
RUNNING INFERENCE WITH KV CACHE COMPRESSION
================================================================================

----------------------------------------
PREFILL PHASE (processing context)
----------------------------------------
  [PREFILL] Layer  0: KV Cache:   850 ->   425 (COMPRESSED)
  [PREFILL] Layer  4: KV Cache:   850 ->   425 (COMPRESSED)
  [PREFILL] Layer  8: KV Cache:   850 ->   425 (COMPRESSED)
  ...

>>> After prefill: Cache size = 425 tokens
>>> Compression: 850 -> 425 (50.0% reduced)

----------------------------------------
DECODE PHASE (generating tokens)
----------------------------------------
  [DECODE] Layer  0: KV Cache:   681 ->   512 (COMPRESSED)
  [DECODE] Layer  4: KV Cache:   681 ->   512 (COMPRESSED)
  ...

>>> Generated 20 tokens, current cache size: 532
```

---

### 8.5 Understanding the Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT EXECUTION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: 850 tokens                                                          │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PREFILL PHASE                                                        │   │
│  │                                                                      │   │
│  │ For each layer (0 to N):                                             │   │
│  │   1. Attention computes KV for all 850 tokens                        │   │
│  │   2. forward_hook triggers                                           │   │
│  │   3. LoggingPressWrapper logs: "Layer X: 850 -> ?"                   │   │
│  │   4. KnormPress.compress() keeps top 50% by key norm                 │   │
│  │   5. LoggingPressWrapper logs: "Layer X: 850 -> 425 (COMPRESSED)"    │   │
│  │   6. Cache updated with 425 tokens                                   │   │
│  │                                                                      │   │
│  │ Result: Cache = 425 tokens per layer                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DECODE PHASE (generate 100 tokens)                                   │   │
│  │                                                                      │   │
│  │ Token 1-255: Cache grows (425 -> 426 -> ... -> 680)                  │   │
│  │   - forward_hook triggers but DecodingPress skips (not at interval)  │   │
│  │                                                                      │   │
│  │ Token 256: compression_interval reached!                             │   │
│  │   1. forward_hook triggers                                           │   │
│  │   2. DecodingPress.compress() activates                              │   │
│  │   3. Computes scores for all 681 tokens                              │   │
│  │   4. Keeps top 512 (target_size)                                     │   │
│  │   5. Logs: "Layer X: 681 -> 512 (COMPRESSED)"                        │   │
│  │                                                                      │   │
│  │ Token 257-511: Cache grows again (512 -> 513 -> ... -> 767)          │   │
│  │                                                                      │   │
│  │ Token 512: compression_interval reached again!                       │   │
│  │   - Compress 768 -> 512                                              │   │
│  │                                                                      │   │
│  │ ... and so on                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  OUTPUT: Generated text with bounded memory usage                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 8.6 Key Parameters

| Parameter | Phase | Effect |
|-----------|-------|--------|
| `compression_ratio` | Prefill | Fraction of KV pairs to remove (0.5 = keep 50%) |
| `compression_interval` | Decode | Compress every N generated tokens |
| `target_size` | Decode | Number of tokens to keep after compression |
| `hidden_states_buffer_size` | Decode | Buffer for scoring recent tokens |

---

## 9. Function Attributes Deep Dive

This section provides a comprehensive explanation of all function parameters and what they control.

### 9.1 KnormPress (Prefill Compression)

```python
KnormPress(compression_ratio=0.5)
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `compression_ratio` | float (0.0 to 1.0) | 0.0 | Fraction of KV pairs to **REMOVE** from the cache during prefill |

**How it works:**
- `0.0` = Keep 100% of tokens (no compression)
- `0.5` = Remove 50%, keep 50% of tokens
- `0.9` = Remove 90%, keep only 10% of tokens

**Example:**
- Input: 431 tokens, `compression_ratio=0.5`
- Output: `431 × (1 - 0.5) = 215 tokens kept`

> **Trade-off:** Higher ratio → More memory savings, but potential information loss

---

### 9.2 DecodingPress (Decode Phase Compression)

```python
DecodingPress(
    base_press=KnormPress(),
    compression_interval=256,
    target_size=512,
    hidden_states_buffer_size=128
)
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_press` | ScorerPress | *Required* | The scoring algorithm used to determine which tokens are important |
| `compression_interval` | int | 512 | Number of decode steps between compression events |
| `target_size` | int | 2048 | Target number of tokens to keep after each compression |
| `hidden_states_buffer_size` | int | 256 | Maximum number of recent hidden states to buffer for scoring |

#### `base_press`

The scoring algorithm used to determine which tokens are important. DecodingPress delegates the actual scoring/compression to this press.

**Common choices:**
- `KnormPress()` - Score by key vector norm (fast, simple)
- `SnapKVPress()` - Score by attention patterns (more accurate)
- `ExpectedAttentionPress()` - Score by expected attention weights

#### `compression_interval`

Compression triggers every N generated tokens.

**Example:** `compression_interval=50`, generating 200 tokens:
- Token 50: First compression
- Token 100: Second compression
- Token 150: Third compression
- Token 200: Fourth compression

#### `target_size`

Sets the "floor" for cache size during generation.

**Cache oscillation pattern:** (`target_size=150`, `interval=50`)
```
150 → 200 (grow) → 150 (compress) → 200 (grow) → 150 (compress) ...
```

#### `hidden_states_buffer_size`

Provides context when computing importance scores during decode.

> **Special case:** Set to 0 for presses that don't need hidden states (e.g., KnormPress)

---

### 9.3 PrefillDecodingPress (Combined Press)

```python
PrefillDecodingPress(
    prefilling_press=KnormPress(compression_ratio=0.5),
    decoding_press=DecodingPress(base_press=KnormPress(), ...)
)
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefilling_press` | BasePress | None | Press to use during the **PREFILL** phase |
| `decoding_press` | DecodingPress | None | Press to use during the **DECODE** phase |

**When `prefilling_press` activates:**
- Condition: `cache_position[-1] <= q_len` (still processing input)

**When `decoding_press` activates:**
- Condition: `cache_position[-1] > q_len` (generating beyond input)

---

### 9.4 Experiment Function Parameters

```python
run_experiment(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda:0",
    prefill_compression_ratio=0.5,
    decode_target_size=512,
    decode_compression_interval=256,
    max_new_tokens=100,
    log_every_n_layers=4,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"Qwen/Qwen2.5-14B-Instruct"` | HuggingFace model identifier to load |
| `device` | str | `"cuda:0"` | Which GPU (or CPU) to run the model on |
| `prefill_compression_ratio` | float | 0.5 | Compression ratio for prefill phase |
| `decode_target_size` | int | 512 | Target cache size after decode compression |
| `decode_compression_interval` | int | 256 | Compress every N generated tokens |
| `max_new_tokens` | int | 100 | Maximum tokens to generate |
| `log_every_n_layers` | int | 4 | Log every N layers |

---

### 9.5 Position IDs: Critical for Correctness

> ⚠️ **IMPORTANT:** `position_ids` must use **ORIGINAL input length**, not cache size!

| | Code | Result |
|---|------|--------|
| ❌ **WRONG** | `position_ids = prefill_cache_size` | Nonsensical output |
| ✅ **CORRECT** | `position_ids = input_length` | Coherent output |

**Why this matters:**
- RoPE (Rotary Position Embedding) encodes position information
- During prefill, token at position 100 gets RoPE for position 100
- Even if that token's KV is kept after compression, its position is **100**
- New tokens must continue from position **431**, not 215

**Visualization:**
```
Original:   [0, 1, 2, ..., 430]  (431 tokens)
Compressed: [0, 5, 12, ..., 428]  (215 tokens, positions preserved!)
New token:  431  (continues from original end)
```

---

### 9.6 Quick Reference Table

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `compression_ratio` | 0.0 | 0.3-0.5 for good quality/memory |
| `compression_interval` | 512 | Lower = more bounded, higher overhead |
| `target_size` | 2048 | Based on your GPU memory budget |
| `hidden_states_buffer_size` | 256 | 0 for KnormPress, 128-256 for others |
| `max_new_tokens` | 100 | Must be > interval to see compression |

---

## 10. Per-Layer Compression: Advanced Techniques

This section explains how `base_press.py` intercepts the transformer forward pass and how to implement intelligent per-layer compression strategies.

### 10.1 How base_press.py Intercepts the Forward Pass

The compression mechanism uses PyTorch's **forward hooks** to intercept attention layer outputs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HOOK REGISTRATION MECHANISM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Context Manager __call__ (base_press.py:158-201):                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ @contextmanager                                                 │     │
│     │ def __call__(self, model):                                      │     │
│     │     for layer in model.model.layers:                            │     │
│     │         layer.self_attn.register_forward_hook(                  │     │
│     │             self.forward_hook,    # SAME hook for ALL layers    │     │
│     │             with_kwargs=True                                    │     │
│     │         )                                                       │     │
│     │     yield  # Model runs here, hooks trigger                     │     │
│     │     # Hooks removed when exiting context                        │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. Forward Hook (base_press.py:95-156):                                    │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ def forward_hook(self, module, input, kwargs, output):          │     │
│     │     layer_idx = module.layer_idx  # ◄── CAN identify layer!     │     │
│     │     cache = kwargs["past_key_values"]                           │     │
│     │     keys, values = extract_keys_and_values(cache, layer_idx)    │     │
│     │                                                                 │     │
│     │     # Compress and update cache                                 │     │
│     │     keys, values = self.compress(module, ...)                   │     │
│     │     cache_layer.keys = keys                                     │     │
│     │     cache_layer.values = values                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. DecodingPress Per-Layer State (decoding_press.py:131-134):              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ # Already tracks state PER LAYER using layer_idx as key         │     │
│     │ self.hidden_states_buffer[layer_idx].append(hidden_states)      │     │
│     │ self.layer_step_counts[layer_idx] += 1                          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Hook Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOOK EXECUTION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  with press(model):                                                         │
│      model(input_ids)                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 0: self_attn.forward()                                        │   │
│  │     └──► forward_hook(module, input, kwargs, output)                │   │
│  │              module.layer_idx = 0                                   │   │
│  │              compress() called → cache updated                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 1: self_attn.forward()                                        │   │
│  │     └──► forward_hook(module, input, kwargs, output)                │   │
│  │              module.layer_idx = 1                                   │   │
│  │              compress() called → cache updated                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│         ...                                                                 │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Layer N: self_attn.forward()                                        │   │
│  │     └──► forward_hook(module, input, kwargs, output)                │   │
│  │              module.layer_idx = N                                   │   │
│  │              compress() called → cache updated                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Current Limitation: Uniform Compression

The current implementation applies **identical** compression settings to ALL layers:

```python
# decoding_press.py:137 - Same interval for all layers
if (self.layer_step_counts[layer_idx] >= self.compression_interval):
    ...
    # self.target_size is also the same for all layers
```

This is suboptimal because:

| Layer Type | Characteristics | Optimal Strategy |
|------------|-----------------|------------------|
| **Early layers (0-7)** | Syntax, structure, positional info | Can tolerate aggressive compression |
| **Middle layers (8-23)** | Semantic processing, entity recognition | Moderate compression |
| **Late layers (24-47)** | Task-specific, output generation | Need more tokens, light compression |

---

### 10.4 Per-Layer Compression: Yes, It's Possible!

Since `module.layer_idx` is available in every hook, we can implement per-layer compression. Here are three approaches:

#### Approach 1: Layer Configuration Dictionary

The simplest approach - define settings per layer:

```python
from dataclasses import dataclass, field
from collections import defaultdict
from kvpress.presses.decoding_press import DecodingPress

@dataclass
class PerLayerDecodingPress(DecodingPress):
    """
    DecodingPress with per-layer compression settings.

    Example usage:
        layer_configs = {
            0:  {"target_size": 100, "interval": 20},   # Early: aggressive
            8:  {"target_size": 150, "interval": 30},
            16: {"target_size": 200, "interval": 50},   # Middle: moderate
            24: {"target_size": 250, "interval": 70},
            32: {"target_size": 300, "interval": 100},  # Late: light
        }
        press = PerLayerDecodingPress(
            base_press=KnormPress(),
            layer_configs=layer_configs
        )
    """
    layer_configs: dict = field(default_factory=dict)

    def get_layer_config(self, layer_idx: int) -> tuple[int, int]:
        """
        Get (target_size, interval) for a specific layer.
        Falls back to defaults if layer not in config.
        """
        if layer_idx in self.layer_configs:
            cfg = self.layer_configs[layer_idx]
            return (
                cfg.get("target_size", self.target_size),
                cfg.get("interval", self.compression_interval)
            )

        # Fallback to class defaults
        return self.target_size, self.compression_interval

    def forward_hook(self, module, input, kwargs, output):
        layer_idx = module.layer_idx

        # Get layer-specific settings
        layer_target_size, layer_interval = self.get_layer_config(layer_idx)

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]

        # Only operate during decoding phase
        if kwargs["cache_position"][-1] <= q_len:
            return output

        # Add hidden states to buffer
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.layer_step_counts[layer_idx] += 1

        # Use LAYER-SPECIFIC interval instead of uniform
        if self.layer_step_counts[layer_idx] >= layer_interval:
            cache_layer = cache.layers[layer_idx]
            keys, values = extract_keys_and_values(cache, layer_idx)

            # Temporarily override target_size for this layer
            original_target = self.target_size
            self.target_size = layer_target_size

            buffered_hidden_states = torch.cat(
                self.hidden_states_buffer[layer_idx], dim=1
            )
            keys, values = self.compress(
                module, buffered_hidden_states, keys, values, None, kwargs
            )

            self.target_size = original_target  # Restore

            # Update cache
            cache_layer.keys = keys
            cache_layer.values = values

            # Reset counters for this layer
            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []

        return output
```

#### Approach 2: Adaptive/Intelligent Mechanism

Dynamically compute compression settings based on layer position:

```python
import math
from dataclasses import dataclass
from kvpress.presses.decoding_press import DecodingPress

@dataclass
class AdaptiveLayerDecodingPress(DecodingPress):
    """
    Dynamically adjusts compression based on layer depth.

    Research shows different layers have different importance:
    - Early layers: capture syntax/structure → more compressible
    - Late layers: task-specific output → need more tokens

    Strategies:
    - "linear": Linear interpolation from min to max
    - "exponential": Exponential growth (aggressive early, light late)
    - "step": Discrete groups with different settings
    """

    strategy: str = "linear"
    min_target_size: int = 100    # For early layers
    max_target_size: int = 500    # For late layers
    min_interval: int = 20        # Compress frequently in early layers
    max_interval: int = 100       # Compress less in late layers
    num_layers: int = 48          # Set from model

    def post_init_from_model(self, model):
        """Called automatically when press is applied to model."""
        self.num_layers = len(model.model.layers)

    def compute_layer_settings(self, layer_idx: int) -> tuple[int, int]:
        """
        Compute (target_size, interval) for a layer based on strategy.
        """
        ratio = layer_idx / max(self.num_layers - 1, 1)

        if self.strategy == "linear":
            # Linear interpolation
            target = int(
                self.min_target_size +
                ratio * (self.max_target_size - self.min_target_size)
            )
            interval = int(
                self.min_interval +
                ratio * (self.max_interval - self.min_interval)
            )

        elif self.strategy == "exponential":
            # Exponential: much smaller early, larger late
            exp_ratio = (math.exp(ratio * 2) - 1) / (math.exp(2) - 1)
            target = int(
                self.min_target_size +
                exp_ratio * (self.max_target_size - self.min_target_size)
            )
            interval = int(
                self.min_interval +
                exp_ratio * (self.max_interval - self.min_interval)
            )

        elif self.strategy == "step":
            # Discrete groups
            if ratio < 0.25:      # Early layers (0-25%)
                target, interval = self.min_target_size, self.min_interval
            elif ratio < 0.75:    # Middle layers (25-75%)
                target = (self.min_target_size + self.max_target_size) // 2
                interval = (self.min_interval + self.max_interval) // 2
            else:                 # Late layers (75-100%)
                target, interval = self.max_target_size, self.max_interval

        else:
            # Default fallback
            target, interval = self.target_size, self.compression_interval

        return target, interval
```

#### Approach 3: Multiple Presses with Layer Routing

Register different press instances for different layer groups:

```python
from contextlib import contextmanager
from dataclasses import dataclass, field
from kvpress.presses.decoding_press import DecodingPress

@dataclass
class MultiPressManager:
    """
    Manages multiple presses, routing each layer to its designated press.

    Example:
        early_press = DecodingPress(base_press=KnormPress(), target_size=100)
        late_press = DecodingPress(base_press=KnormPress(), target_size=400)

        manager = MultiPressManager(
            layer_presses={
                range(0, 16): early_press,   # Layers 0-15
                range(16, 48): late_press,   # Layers 16-47
            },
            default_press=late_press
        )

        with manager(model):
            model(input_ids)
    """

    layer_presses: dict = field(default_factory=dict)  # {layer_idx or range: press}
    default_press: DecodingPress = None

    def get_press_for_layer(self, layer_idx: int) -> DecodingPress:
        """Find the appropriate press for a given layer."""
        # Check for exact layer match
        if layer_idx in self.layer_presses:
            return self.layer_presses[layer_idx]

        # Check for range match
        for key, press in self.layer_presses.items():
            if isinstance(key, range) and layer_idx in key:
                return press

        return self.default_press

    @contextmanager
    def __call__(self, model):
        """
        Context manager that registers layer-specific hooks.
        """
        hooks = []
        try:
            for layer in model.model.layers:
                layer_idx = layer.self_attn.layer_idx

                # Get the appropriate press for this layer
                press = self.get_press_for_layer(layer_idx)

                if press is not None:
                    # Register layer-specific hook
                    hooks.append(
                        layer.self_attn.register_forward_hook(
                            press.forward_hook,
                            with_kwargs=True
                        )
                    )
            yield
        finally:
            for hook in hooks:
                hook.remove()
```

---

### 10.5 Visualization: Per-Layer Compression Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 INTELLIGENT PER-LAYER COMPRESSION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 0-15 (Early):    Layer 16-31 (Middle):   Layer 32-47 (Late):         │
│  ─────────────────      ──────────────────      ─────────────────           │
│  • Syntax/Structure     • Semantic processing   • Task-specific             │
│  • High redundancy      • Moderate importance   • Critical for output       │
│  • Aggressive compress  • Moderate compression  • Light compression         │
│                                                                             │
│  target_size: 100       target_size: 200        target_size: 400            │
│  interval: 20           interval: 50            interval: 100               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Target Size                                                         │   │
│  │      ▲                                                               │   │
│  │  500 │                                               ┌──────────     │   │
│  │      │                                          ┌────┘               │   │
│  │  400 │                                     ┌────┘                    │   │
│  │      │                                ┌────┘                         │   │
│  │  300 │                           ┌────┘                              │   │
│  │      │                      ┌────┘                                   │   │
│  │  200 │                 ┌────┘                                        │   │
│  │      │            ┌────┘                                             │   │
│  │  100 │  ──────────┘                                                  │   │
│  │      │                                                               │   │
│  │      └──────────────────────────────────────────────────────► Layer  │   │
│  │        0    8    16    24    32    40    47                          │   │
│  │                                                                      │   │
│  │  Legend: ───── target_size per layer (linear strategy)               │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Memory Savings with Per-Layer Compression:                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Uniform (target=300):     Total cache ≈ 300 × 48 = 14,400 tokens    │   │
│  │  Per-Layer (100→400):      Total cache ≈ 250 × 48 = 12,000 tokens    │   │
│  │                                                                      │   │
│  │  Savings: ~17% memory reduction with potentially BETTER quality      │   │
│  │  (because late layers retain more information)                       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 10.6 Research-Backed Strategies

| Strategy | Description | Rationale | Best For |
|----------|-------------|-----------|----------|
| **Linear** | Linearly interpolate target_size from min→max | Simple, predictable | General use |
| **Exponential** | Exponential growth of target_size | Heavy early compression, preserve late | Memory constrained |
| **Step** | Discrete groups (early/middle/late) | Easy to tune | Quick experiments |
| **Attention Entropy** | Compress layers with high redundancy | Data-driven decisions | Quality focused |
| **Task-Aware** | Adjust based on downstream task | QA needs late layers, summarization needs early | Specific applications |

#### Research Findings on Layer Importance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER IMPORTANCE BY TASK                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Task Type          Critical Layers       Compressible Layers               │
│  ─────────────────  ──────────────────    ───────────────────               │
│  Question Answering Late (output)         Early (syntax)                    │
│  Summarization      Middle (semantic)     Late (verbatim)                   │
│  Translation        All equally           None significantly                │
│  Code Generation    Late (syntax rules)   Early (general patterns)          │
│  Reasoning          Middle + Late         Early                             │
│                                                                             │
│  General Recommendation:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Compress early layers MORE aggressively (target_size: 100-150)    │   │
│  │ • Compress middle layers MODERATELY (target_size: 200-300)          │   │
│  │ • Compress late layers LIGHTLY (target_size: 400-500)               │   │
│  │ • Always preserve the LAST 2-3 layers if possible                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 10.7 Key Implementation Points

| Point | Description |
|-------|-------------|
| **Layer Identification** | `module.layer_idx` is available in every hook → trivial to identify |
| **Per-Layer State** | `DecodingPress` already uses `layer_idx` as dictionary key for buffers |
| **Hook Registration** | Can be modified to register different hooks per layer |
| **compress() Access** | Receives `module` argument → can access `layer_idx` for conditional logic |
| **Backward Compatible** | Per-layer press can fall back to uniform settings if layer not configured |

---

### 10.8 Quick Reference: Implementing Per-Layer Compression

```python
# Option 1: Simple layer config dict
layer_configs = {
    0:  {"target_size": 100, "interval": 20},
    16: {"target_size": 200, "interval": 50},
    32: {"target_size": 400, "interval": 100},
}

# Option 2: Adaptive with strategy
press = AdaptiveLayerDecodingPress(
    strategy="linear",
    min_target_size=100,
    max_target_size=500,
)

# Option 3: Multiple press instances
manager = MultiPressManager(
    layer_presses={
        range(0, 16): aggressive_press,
        range(16, 32): moderate_press,
        range(32, 48): light_press,
    }
)
```

---

### 10.9 Per-Layer Press Type Routing

The previous sections focused on varying **compression settings** (ratio, interval, target_size) per layer while using the **same press type**. This section explores using **different press types** for different layers.

#### 10.9.1 Why Different Press Types Per Layer?

Different layers have different characteristics that may benefit from different scoring algorithms:

| Layer Group | Characteristics | Optimal Press Type |
|-------------|-----------------|-------------------|
| **Early (0-25%)** | Syntax, positional info, high redundancy | `KnormPress` - Fast, simple key-norm scoring |
| **Middle (25-75%)** | Semantic processing, entity recognition | `SnapKVPress` - Attention pattern-based |
| **Late (75-100%)** | Task-specific, output generation, critical | `ExpectedAttentionPress` - Statistical modeling |

**Key insight**: Using expensive attention-based scoring (SnapKV) on early layers wastes compute, while using simple scoring (Knorm) on late layers may hurt quality.

#### 10.9.2 Current Limitations

| Existing Class | Capability | Limitation |
|----------------|------------|------------|
| `PerLayerCompressionPress` | Different **compression ratios** per layer | Same press type for all layers |
| `ComposedPress` | Chain multiple presses **sequentially** | All presses applied to same layer |
| `PrefillDecodingPress` | Different press for prefill vs decode | Same press across all layers |

**Gap**: No mechanism to route different press **types** per layer.

#### 10.9.3 Proposed Solution: `PerLayerPressRouter`

A new class that routes different press instances to different layers:

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from contextlib import contextmanager

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class PerLayerPressRouter(BasePress):
    """
    Route different press types to different layers.

    Enables using different compression strategies for different transformer layers,
    allowing optimal press selection based on layer characteristics:
    - Early layers: syntax/structure → fast, aggressive (e.g., KnormPress)
    - Middle layers: semantics → attention-based (e.g., SnapKVPress)
    - Late layers: task output → careful/light (e.g., ExpectedAttentionPress)

    Parameters
    ----------
    layer_presses : Dict[Union[int, range], BasePress]
        Mapping of layer indices or ranges to press instances.
        Ranges are checked in order; first match wins.
    default_press : BasePress, optional
        Fallback press for layers not in layer_presses.
        If None and layer not matched, no compression is applied.

    Example
    -------
    >>> router = PerLayerPressRouter(
    ...     layer_presses={
    ...         range(0, 12): KnormPress(compression_ratio=0.5),
    ...         range(12, 36): SnapKVPress(compression_ratio=0.3),
    ...         range(36, 48): ExpectedAttentionPress(compression_ratio=0.2),
    ...     },
    ...     default_press=KnormPress(compression_ratio=0.3)
    ... )
    >>> with router(model):
    ...     outputs = model(input_ids, past_key_values=cache)
    """

    layer_presses: Dict[Union[int, range], BasePress] = field(default_factory=dict)
    default_press: Optional[BasePress] = None

    def __post_init__(self):
        self._computed_compression_ratio = 0.0

    def post_init_from_model(self, model):
        """Initialize all contained presses with model reference."""
        for press in self.layer_presses.values():
            if press is not None:
                press.post_init_from_model(model)
        if self.default_press is not None:
            self.default_press.post_init_from_model(model)

    def get_press_for_layer(self, layer_idx: int) -> Optional[BasePress]:
        """
        Find the appropriate press for a given layer.

        Checks exact layer match first, then range matches in insertion order.
        Falls back to default_press if no match found.
        """
        # Check exact layer match
        if layer_idx in self.layer_presses:
            return self.layer_presses[layer_idx]

        # Check range matches (in order)
        for key, press in self.layer_presses.items():
            if isinstance(key, range) and layer_idx in key:
                return press

        return self.default_press

    def forward_hook(
        self,
        module: nn.Module,
        input: list[torch.Tensor],
        kwargs: dict,
        output: list
    ):
        """Route to appropriate press based on layer index."""
        layer_idx = module.layer_idx
        press = self.get_press_for_layer(layer_idx)

        if press is None:
            return output  # No compression for this layer

        return press.forward_hook(module, input, kwargs, output)

    @property
    def compression_ratio(self):
        """Average compression ratio across all configured presses."""
        ratios = []
        for press in self.layer_presses.values():
            if press is not None and hasattr(press, 'compression_ratio'):
                ratios.append(press.compression_ratio)
        if self.default_press and hasattr(self.default_press, 'compression_ratio'):
            ratios.append(self.default_press.compression_ratio)
        return sum(ratios) / len(ratios) if ratios else 0.0

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(
            f"compression_ratio cannot be set directly for {type(self).__name__}. "
            "Set compression_ratio on individual layer presses instead."
        )
```

#### 10.9.4 Usage Examples

**Example 1: Different Press Types Per Layer Group**

```python
from kvpress import (
    KnormPress, SnapKVPress, ExpectedAttentionPress
)

router = PerLayerPressRouter(
    layer_presses={
        range(0, 12): KnormPress(compression_ratio=0.6),      # Fast for early
        range(12, 36): SnapKVPress(compression_ratio=0.4),    # Accurate for middle
        range(36, 48): ExpectedAttentionPress(compression_ratio=0.2),  # Light for late
    },
    default_press=KnormPress(compression_ratio=0.3)
)

with router(model):
    outputs = model(input_ids, past_key_values=cache, use_cache=True)
```

**Example 2: Specific Layer Override (Attention Sinks)**

```python
from kvpress import StreamingLLMPress, KnormPress

router = PerLayerPressRouter(
    layer_presses={
        0: StreamingLLMPress(n_recent=128),  # Keep attention sinks in first layer
        range(1, 40): KnormPress(compression_ratio=0.5),
        # Layers 40-47: No compression (use default_press=None)
    },
    default_press=None  # No compression for unconfigured layers
)
```

**Example 3: Combined with PrefillDecodingPress**

```python
from kvpress import DecodingPress, PrefillDecodingPress, KnormPress, SnapKVPress

# Per-layer routing for prefill phase
prefill_router = PerLayerPressRouter(
    layer_presses={
        range(0, 16): KnormPress(compression_ratio=0.5),
        range(16, 48): SnapKVPress(compression_ratio=0.3),
    }
)

# Uniform decode press (or create another router for decode)
decode_press = DecodingPress(
    base_press=KnormPress(),
    compression_interval=256,
    target_size=512
)

combined = PrefillDecodingPress(
    prefilling_press=prefill_router,
    decoding_press=decode_press
)

with combined(model):
    # Prefill uses per-layer routing, decode uses uniform compression
    outputs = model(input_ids, past_key_values=cache, use_cache=True)
```

#### 10.9.5 Architecture Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PerLayerPressRouter EXECUTION FLOW                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  with router(model):                                                        │
│      model(input_ids)                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 0-11: forward_hook triggers                                   │   │
│  │     └──► get_press_for_layer(0) → KnormPress                        │   │
│  │     └──► KnormPress.forward_hook() → Fast key-norm scoring          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 12-35: forward_hook triggers                                  │   │
│  │     └──► get_press_for_layer(12) → SnapKVPress                      │   │
│  │     └──► SnapKVPress.forward_hook() → Attention pattern scoring     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Layer 36-47: forward_hook triggers                                  │   │
│  │     └──► get_press_for_layer(36) → ExpectedAttentionPress           │   │
│  │     └──► ExpectedAttentionPress.forward_hook() → Statistical model  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Result: Each layer uses optimal press type for its characteristics         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 10.9.6 Available Press Types Reference

| Press Type | Scoring Method | Speed | Best For |
|------------|----------------|-------|----------|
| `KnormPress` | L2 norm of key vectors | ⚡ Fast | Early layers, high throughput |
| `RandomPress` | Random selection | ⚡ Fast | Baseline comparison |
| `SnapKVPress` | Recent token attention | 🔶 Medium | Middle layers, semantic content |
| `ExpectedAttentionPress` | Statistical future attention | 🔶 Medium | Late layers, quality focus |
| `ObservedAttentionPress` | Actual attention weights | 🐢 Slow | Maximum accuracy (requires eager attn) |
| `StreamingLLMPress` | Attention sinks + recent | ⚡ Fast | First layer, streaming scenarios |
| `TOVAPress` | Token-to-value attention | 🔶 Medium | Task-specific tuning |
| `LeverageScorePress` | Leverage-based importance | 🔶 Medium | Research applications |

#### 10.9.7 Implementation Checklist

To implement `PerLayerPressRouter` in kvpress:

1. **Create file**: `kvpress/presses/per_layer_press_router.py`
2. **Export in presses**: Add to `kvpress/presses/__init__.py`
3. **Export at package level**: Add to `kvpress/__init__.py`
4. **Test**: Verify routing works with mock/real model
