# KV Cache Tensor Structure

## What is KV Cache?

During autoregressive text generation, a transformer model generates tokens one at a time. Without caching, each new token would require recomputing the Key (K) and Value (V) projections for **all previous tokens** - an O(n²) operation that becomes prohibitively expensive for long sequences.

The KV cache stores these computed K and V tensors, so each generation step only needs to:
1. Compute K and V for the **new token only**
2. Append them to the cache
3. Use the full cached K/V for attention

This reduces generation from O(n²) to O(n).

---

## From Hidden States to KV Cache

### Step 1: Hidden States Enter Each Layer

Each transformer layer receives hidden states from the previous layer (or embeddings for layer 0):

```
hidden_states shape: (batch_size, seq_len, hidden_size)

Example for Qwen2.5-14B:
hidden_states: (2, 100, 5120)
               ↑    ↑     ↑
            2 seqs  100 tokens  5120-dim embedding per token
```

### Step 2: Linear Projections Create Q, K, V

The attention mechanism projects hidden states into Query, Key, and Value:

```python
# Conceptual (actual implementation may fuse operations)
Q = hidden_states @ W_q  # (batch, seq_len, num_query_heads * head_dim)
K = hidden_states @ W_k  # (batch, seq_len, num_kv_heads * head_dim)
V = hidden_states @ W_v  # (batch, seq_len, num_kv_heads * head_dim)
```

**Weight matrix dimensions:**
| Matrix | Shape | Description |
|--------|-------|-------------|
| W_q | (hidden_size, num_query_heads × head_dim) | Query projection |
| W_k | (hidden_size, num_kv_heads × head_dim) | Key projection |
| W_v | (hidden_size, num_kv_heads × head_dim) | Value projection |

### Step 3: Reshape into Multi-Head Format

After projection, tensors are reshaped to separate heads:

```python
# Reshape: (batch, seq_len, num_heads * head_dim) → (batch, num_heads, seq_len, head_dim)
K = K.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
V = V.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
```

**Qwen2.5-14B concrete example:**
```
hidden_states: (2, 100, 5120)
                        ↓
              Linear projection W_k: (5120, 1024)
                        ↓
K after projection: (2, 100, 1024)    # 1024 = 8 heads × 128 head_dim
                        ↓
              Reshape + transpose
                        ↓
K final shape: (2, 8, 100, 128)
               ↑  ↑   ↑    ↑
            batch heads tokens head_dim
```

---

## KV Cache Tensor Shape

### Canonical Shape: `(batch, num_kv_heads, seq_len, head_dim)`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SINGLE LAYER KV CACHE                                                      │
│                                                                             │
│  Keys tensor: (batch=2, num_kv_heads=8, seq_len=100, head_dim=128)          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Batch item 0 (first sequence)                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Head 0    Head 1    Head 2    ...    Head 7                │    │    │
│  │  │  ┌─────┐   ┌─────┐   ┌─────┐         ┌─────┐                │    │    │
│  │  │  │ 100 │   │ 100 │   │ 100 │   ...   │ 100 │  ← seq_len     │    │    │
│  │  │  │ × 128│   │ × 128│   │ × 128│         │ × 128│  (tokens)    │    │    │
│  │  │  └─────┘   └─────┘   └─────┘         └─────┘                │    │    │
│  │  │                                        ↑                    │    │    │
│  │  │                                   head_dim=128              │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Batch item 1 (second sequence)                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Head 0    Head 1    Head 2    ...    Head 7                │    │    │
│  │  │  ┌─────┐   ┌─────┐   ┌─────┐         ┌─────┐                │    │    │
│  │  │  │ 100 │   │ 100 │   │ 100 │   ...   │ 100 │                │    │    │
│  │  │  │ × 128│   │ × 128│   │ × 128│         │ × 128│                │    │    │
│  │  │  └─────┘   └─────┘   └─────┘         └─────┘                │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Values tensor: identical shape (batch=2, num_kv_heads=8, seq_len=100, 128) │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dimension Reference Table

| Dimension | Typical Name | Description | Example (Qwen2.5-14B) |
|-----------|--------------|-------------|----------------------|
| **dim 0** | batch_size | Number of sequences processed in parallel | 1-64 (varies) |
| **dim 1** | num_kv_heads | Number of Key-Value attention heads | 8 |
| **dim 2** | seq_len | Number of tokens currently cached | grows during generation |
| **dim 3** | head_dim | Embedding dimension per head | 128 |

---

## Multi-Layer Structure

The complete KV cache contains K and V tensors for **every layer** in the model:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  FULL MODEL KV CACHE (e.g., 48 layers)                                   │
│                                                                          │
│  past_key_values = tuple of 48 layer caches                              │
│                                                                          │
│  Layer 0:  (K: [batch, heads, seq, dim], V: [batch, heads, seq, dim])    │
│  Layer 1:  (K: [batch, heads, seq, dim], V: [batch, heads, seq, dim])    │
│  Layer 2:  (K: [batch, heads, seq, dim], V: [batch, heads, seq, dim])    │
│     ...                                                                  │
│  Layer 47: (K: [batch, heads, seq, dim], V: [batch, heads, seq, dim])    │
│                                                                          │
│  Total: 48 × 2 = 96 tensors                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

### HuggingFace DynamicCache Structure

```python
from transformers import DynamicCache

cache = DynamicCache()

# Access layer i:
keys_layer_i = cache.key_cache[i]    # shape: (batch, num_kv_heads, seq_len, head_dim)
vals_layer_i = cache.value_cache[i]  # shape: (batch, num_kv_heads, seq_len, head_dim)

# Iterate all layers:
for layer_idx, (K, V) in enumerate(zip(cache.key_cache, cache.value_cache)):
    print(f"Layer {layer_idx}: K={K.shape}, V={V.shape}")
```

### Visual: 3D Cache Across Layers

```
                          seq_len (tokens)
                              ↓
                    ┌─────────────────┐
                   ╱│                 │
       head_dim   ╱ │                 │
            ↓    ╱  │                 │
                ╱   │     Layer 0     │
               ╱    │    (K tensor)   │
              │     │                 │
   num_kv_heads     └─────────────────┘
       →      │    ╱│                 │
              │   ╱ │                 │
              │  ╱  │     Layer 1     │
              │ ╱   │    (K tensor)   │
              │╱    │                 │
              └─────└─────────────────┘
              │    ╱│                 │
              │   ╱ │                 │
              │  ╱  │     Layer 2     │
    layers    │ ╱   │    (K tensor)   │
      ↓       │╱    │                 │
              └─────└─────────────────┘
              │         ...           │
              │                       │
              └─────┬─────────────────┘
                    │                 │
                    │    Layer 47     │
                    │    (K tensor)   │
                    │                 │
                    └─────────────────┘

Same structure exists for V tensors (96 total tensors for 48 layers)
```

---

## Grouped Query Attention (GQA)

Modern models like Llama 2/3, Qwen2, and Mistral use **Grouped Query Attention** to reduce KV cache memory:

```
┌────────────────────────────────────────────────────────────────────────┐
│  MULTI-HEAD ATTENTION (MHA)           GROUPED QUERY ATTENTION (GQA)    │
│                                                                        │
│  Query heads: 40                      Query heads: 40                  │
│  KV heads:    40  (1:1 ratio)         KV heads:    8   (5:1 ratio)     │
│                                                                        │
│  Q0 → KV0                             Q0 ─┐                            │
│  Q1 → KV1                             Q1 ─┤                            │
│  Q2 → KV2                             Q2 ─┼→ KV0  (5 Q heads share 1)  │
│  ...                                  Q3 ─┤                            │
│  Q39 → KV39                           Q4 ─┘                            │
│                                       Q5 ─┐                            │
│                                       Q6 ─┤                            │
│                                       Q7 ─┼→ KV1                       │
│                                       Q8 ─┤                            │
│                                       Q9 ─┘                            │
│                                       ...                              │
│                                                                        │
│  KV cache: 40 × head_dim              KV cache: 8 × head_dim           │
│  = 40 × 128 = 5120                    = 8 × 128 = 1024                 │
│                                       (5× smaller!)                    │
└────────────────────────────────────────────────────────────────────────┘
```

### GQA Parameter Relationships

```python
# Model config relationships:
num_attention_heads = 40      # Query heads (also called num_heads)
num_key_value_heads = 8       # KV heads (also called num_kv_heads)
hidden_size = 5120            # Model embedding dimension
head_dim = hidden_size // num_attention_heads  # = 128

# GQA group size:
num_key_value_groups = num_attention_heads // num_key_value_heads  # = 5
# Each KV head is shared by 5 query heads
```

---

## Batch Processing Details

### Variable Sequence Lengths in a Batch

When batching sequences of different lengths, padding or attention masks handle the differences:

```
Batch with 3 sequences of different lengths:
┌─────────────────────────────────────────────────────────────────┐
│  Sequence 0: "Hello world"        → 2 tokens                    │
│  Sequence 1: "How are you today"  → 4 tokens                    │
│  Sequence 2: "Hi"                 → 1 token                     │
│                                                                 │
│  With left-padding to max_len=4:                                │
│                                                                 │
│  KV Cache shape: (batch=3, heads=8, seq_len=4, head_dim=128)    │
│                                                                 │
│  Batch 0: [PAD] [PAD] [Hello] [world]                           │
│  Batch 1: [How] [are] [you]   [today]                           │
│  Batch 2: [PAD] [PAD] [PAD]   [Hi]                              │
│                                                                 │
│  Attention mask ensures PAD tokens are ignored in attention     │
└─────────────────────────────────────────────────────────────────┘
```

### Accessing Specific Elements

```python
# KV cache tensor: (batch, num_kv_heads, seq_len, head_dim)
cache_k = keys  # shape: (3, 8, 4, 128)

# Get all heads for batch item 1, token position 2:
cache_k[1, :, 2, :]  # shape: (8, 128) - all 8 heads for "you" token

# Get head 0 for all batches, all positions:
cache_k[:, 0, :, :]  # shape: (3, 4, 128)

# Get the full key vector for batch 2, head 5, token 3 ("Hi"):
cache_k[2, 5, 3, :]  # shape: (128,) - single 128-dim vector
```

---

## Memory Calculations

### Per-Layer Memory

```
Keys:   batch × num_kv_heads × seq_len × head_dim × bytes_per_element
Values: batch × num_kv_heads × seq_len × head_dim × bytes_per_element
Total per layer = 2 × above
```

### Qwen2.5-14B Example

```
Model parameters:
- num_hidden_layers = 48
- num_kv_heads = 8
- head_dim = 128
- dtype = float16 (2 bytes) or bfloat16 (2 bytes)

For batch=1, seq_len=4096:

Per layer:
  K: 1 × 8 × 4096 × 128 × 2 bytes = 8,388,608 bytes = 8 MB
  V: 1 × 8 × 4096 × 128 × 2 bytes = 8,388,608 bytes = 8 MB
  Total: 16 MB per layer

Full model:
  48 layers × 16 MB = 768 MB for KV cache alone

Scaling with batch size:
  batch=8:  768 MB × 8 = 6.1 GB
  batch=32: 768 MB × 32 = 24.6 GB
```

### Memory Formula

```python
def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # float16/bfloat16
) -> int:
    """Calculate total KV cache memory in bytes."""
    per_layer = 2 * batch_size * num_kv_heads * seq_len * head_dim * dtype_bytes
    return num_layers * per_layer
```

---

## KV Cache During Generation

### Prefill Phase (Processing the Prompt)

```
Input prompt: "The quick brown fox" (4 tokens)

After prefill, KV cache contains:
┌─────────────────────────────────────────────────────────┐
│  Layer 0: K=[1,8,4,128], V=[1,8,4,128]  ← 4 tokens      │
│  Layer 1: K=[1,8,4,128], V=[1,8,4,128]                  │
│  ...                                                    │
│  Layer 47: K=[1,8,4,128], V=[1,8,4,128]                 │
└─────────────────────────────────────────────────────────┘
```

### Decode Phase (Generating New Tokens)

```
Generate token 1: "jumps"
┌──────────────────────────────────────────────────────────┐
│  1. Compute K, V for "jumps" only (not the whole seq)    │
│  2. Append to cache: seq_len grows 4 → 5                 │
│                                                          │
│  Layer 0: K=[1,8,5,128], V=[1,8,5,128]  ← now 5 tokens   │
│  ...                                                     │
└──────────────────────────────────────────────────────────┘

Generate token 2: "over"
┌──────────────────────────────────────────────────────────┐
│  1. Compute K, V for "over" only                         │
│  2. Append to cache: seq_len grows 5 → 6                 │
│                                                          │
│  Layer 0: K=[1,8,6,128], V=[1,8,6,128]  ← now 6 tokens   │
│  ...                                                     │
└──────────────────────────────────────────────────────────┘

... continues until generation complete or max_length reached
```

---

## Common Model Configurations

| Model | Layers | Query Heads | KV Heads | Head Dim | Hidden Size | GQA Ratio |
|-------|--------|-------------|----------|----------|-------------|-----------|
| Llama-2-7B | 32 | 32 | 32 | 128 | 4096 | 1:1 (MHA) |
| Llama-2-70B | 80 | 64 | 8 | 128 | 8192 | 8:1 |
| Llama-3-8B | 32 | 32 | 8 | 128 | 4096 | 4:1 |
| Llama-3-70B | 80 | 64 | 8 | 128 | 8192 | 8:1 |
| Qwen2.5-7B | 28 | 28 | 4 | 128 | 3584 | 7:1 |
| Qwen2.5-14B | 48 | 40 | 8 | 128 | 5120 | 5:1 |
| Qwen2.5-72B | 80 | 64 | 8 | 128 | 8192 | 8:1 |
| Mistral-7B | 32 | 32 | 8 | 128 | 4096 | 4:1 |

---

## Relationship Between Model Config and KV Cache

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-14B")

# These config values determine KV cache shape:
num_layers = config.num_hidden_layers        # 48 → number of cache entries
num_kv_heads = config.num_key_value_heads    # 8  → dim 1 of each tensor
head_dim = config.hidden_size // config.num_attention_heads  # 128 → dim 3

# KV cache shape per layer:
# Keys:   (batch_size, num_kv_heads, seq_len, head_dim)
# Values: (batch_size, num_kv_heads, seq_len, head_dim)
#         (variable,   8,            variable, 128)
```

---

## Value Ranges (Typical)

KV cache values are **not normalized** and can vary significantly across layers:

| Tensor | Typical Range | Notes |
|--------|---------------|-------|
| Keys | -30 to +30 | Wider range, includes positional encoding effects |
| Values | -5 to +5 | Narrower range than keys |

**Per-layer variation:** Early layers often have different statistics than later layers. Layer 0 keys may have larger magnitudes due to direct embedding influence.

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KV CACHE STRUCTURE SUMMARY                                             │
│                                                                         │
│  Input: hidden_states (batch, seq_len, hidden_size)                     │
│                          ↓                                              │
│         Linear projections (W_k, W_v)                                   │
│                          ↓                                              │
│  Reshape to: (batch, seq_len, num_kv_heads, head_dim)                   │
│                          ↓                                              │
│  Transpose to: (batch, num_kv_heads, seq_len, head_dim)                 │
│                          ↓                                              │
│  Store in cache for each of N layers                                    │
│                                                                         │
│  Total tensors: N layers × 2 (K and V) = 2N tensors                     │
│  Total memory: 2 × N × batch × num_kv_heads × seq_len × head_dim × dtype│
└─────────────────────────────────────────────────────────────────────────┘
```
