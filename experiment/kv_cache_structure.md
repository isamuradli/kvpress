# KV Cache Tensor Structure

## Shape: `(batch, num_heads, seq_len, head_dim)`

Example from Qwen2.5-14B layer 0: `(1, 8, 24, 128)`

```
┌─────────────────────────────────────────────────────────────────────┐
│  batch = 1                                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  num_heads = 8 (KV heads for Grouped Query Attention)         │  │
│  │                                                               │  │
│  │  Head 0  Head 1  Head 2  ...  Head 7                          │  │
│  │  ┌────┐  ┌────┐  ┌────┐      ┌────┐                           │  │
│  │  │    │  │    │  │    │      │    │                           │  │
│  │  │ 24 │  │ 24 │  │ 24 │ ...  │ 24 │  ← seq_len = 24 tokens    │  │
│  │  │rows│  │rows│  │rows│      │rows│                           │  │
│  │  │    │  │    │  │    │      │    │                           │  │
│  │  └────┘  └────┘  └────┘      └────┘                           │  │
│  │   128     128     128         128   ← head_dim = 128 floats   │  │
│  │   cols    cols    cols        cols                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Dimension Meanings

| Dimension | Value | Meaning |
|-----------|-------|---------|
| **batch** | 1 | Number of sequences processed in parallel. You have 1 input prompt. |
| **num_heads** | 8 | Number of KV attention heads. Qwen2.5-14B uses GQA with 8 KV heads (shared across 40 query heads). |
| **seq_len** | 24 | Number of tokens stored. Originally 49 tokens, compressed to 24 (50%). |
| **head_dim** | 128 | Size of each key/value vector per token per head. |

## Accessing Individual Tokens

```python
# For token 0, head 0:
keys[0, 0, 0, :]   # → 128-dimensional key vector
values[0, 0, 0, :] # → 128-dimensional value vector
```

## During Attention

1. Query (current token) computes dot product with all 24 keys → attention scores
2. Attention scores weight the 24 value vectors → output

## Memory Calculation

```
1 × 8 × 24 × 128 × 4 bytes (float32) = 98,304 bytes ≈ 96 KB per layer
× 48 layers = ~4.6 MB total KV cache (after compression)
```

## Value Ranges (Layer 0)

| Tensor | Min | Max | Mean |
|--------|-----|-----|------|
| Keys | -23.3 | 21.8 | 0.12 |
| Values | -2.9 | 3.2 | 0.006 |

Keys have a wider range compared to values.
