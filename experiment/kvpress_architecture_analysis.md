# KVPress Codebase Analysis

## Native KVPress Files (Committed)

### Core Files:
| File | Purpose |
|------|---------|
| `kvpress/__init__.py` | Main package init, imports all presses, patches attention functions |
| `kvpress/pipeline.py` | `KVPressTextGenerationPipeline` - handles context compression and text generation |
| `kvpress/attention_patch.py` | Patches transformer attention functions for head-wise compression support |
| `kvpress/utils.py` | Utility functions for extracting keys/values from cache |
| `kvpress/presses/base_press.py` | `BasePress` - foundation class for all compression methods |
| `kvpress/presses/scorer_press.py` | `ScorerPress` - base class for score-based compression |

### Press Implementations (in `kvpress/presses/`):
- `snapkv_press.py` - SnapKV attention-based compression
- `adakv_press.py` - Adaptive KV compression
- `knorm_press.py` - Key norm-based compression
- `expected_attention_press.py` - Expected attention scoring
- `observed_attention_press.py` - Observed attention scoring
- `per_layer_compression_press.py` - Per-layer compression wrapper
- `pyramidkv_press.py` - Pyramid-style compression
- `streaming_llm_press.py` - Streaming LLM approach
- And 20+ more press implementations

---

## KEY FINDING: Compression DOES Apply to Each Attention Layer During Prefill

### Yes, compression applies to the KV cache in EACH attention layer during the prefill stage.

### Evidence from Code:

#### 1. Hook Registration on Every Layer (`base_press.py:190-197`):
```python
for layer in language_model.layers:
    # ...
    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
```
The context manager iterates through **ALL transformer layers** and registers a `forward_hook` on each `layer.self_attn`.

#### 2. Per-Layer Forward Hook Execution (`base_press.py:95-156`):
```python
def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
    cache = kwargs["past_key_values"]
    cache_layer = cache.layers[module.layer_idx]  # <-- Layer-specific cache

    # Don't compress after pre-filling
    if kwargs["cache_position"][-1] > q_len:
        return output

    keys, values = extract_keys_and_values(cache, module.layer_idx)
    keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

    # Update the layer's cache with compressed values
    cache_layer.keys = keys
    cache_layer.values = values
```

Each layer:
- Has its own cache accessed via `cache.layers[module.layer_idx]`
- Gets compressed independently after its forward pass
- Only compresses during prefill (not during generation)

#### 3. `PerLayerCompressionPress` Confirms Architecture (`per_layer_compression_press.py:56-61`):
```python
def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
    self.press.compression_ratio = self.compression_ratios[module.layer_idx]  # <-- Per-layer ratio
    output = self.press.forward_hook(module, input, kwargs, output)
    return output
```
This wrapper explicitly allows **different compression ratios per layer**, demonstrating the system is designed for layer-wise compression.

---

## Compression Flow Diagram

```
Prefill Stage (with press context manager active):

   Input Tokens
        |
        v
   [Layer 0] --> forward_hook --> compress() --> cache.layers[0].keys/values updated
        |
        v
   [Layer 1] --> forward_hook --> compress() --> cache.layers[1].keys/values updated
        |
        v
       ...
        |
        v
   [Layer N] --> forward_hook --> compress() --> cache.layers[N].keys/values updated
        |
        v
   Compressed KV Cache Ready for Generation
```

---

## Summary

| Question | Answer |
|----------|--------|
| Does compression apply during prefill? | **YES** - hooks only fire during prefill (checked via `cache_position`) |
| Does compression apply to each layer? | **YES** - a hook is registered on every `layer.self_attn` |
| Is compression layer-specific? | **YES** - each layer's cache (`cache.layers[layer_idx]`) is updated independently |
| Can different layers have different compression? | **YES** - `PerLayerCompressionPress` supports per-layer ratios |
