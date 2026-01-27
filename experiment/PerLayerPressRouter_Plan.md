# Implementation Plan: PerLayerPressRouter

## Overview

Create a new `PerLayerPressRouter` class that enables using **different press types** for different transformer layers during KV cache compression.

---

## Goal

Enable configurations like:
```python
router = PerLayerPressRouter(
    layer_presses={
        range(0, 12): KnormPress(compression_ratio=0.5),      # Fast for early
        range(12, 36): SnapKVPress(compression_ratio=0.3),    # Accurate for middle
        range(36, 48): ExpectedAttentionPress(compression_ratio=0.2),  # Light for late
    },
    default_press=KnormPress(compression_ratio=0.3)
)
```

---

## Current State Analysis

### Existing Classes

| Class | Capability | Limitation |
|-------|------------|------------|
| `PerLayerCompressionPress` | Different **compression ratios** per layer | Same press type for all layers |
| `ComposedPress` | Chain multiple presses **sequentially** | All presses applied to same layer |
| `PrefillDecodingPress` | Different press for prefill vs decode | Same press across all layers |

### Gap Identified

**No mechanism exists to route different press TYPES per layer.**

---

## Design Decisions

| Decision | Choice |
|----------|--------|
| Phase support | Both prefill and decode |
| Default behavior | Use `default_press` for unconfigured layers |
| `None` handling | No compression if `default_press` is None |

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `kvpress/presses/per_layer_press_router.py` | **CREATE** | New router class |
| `kvpress/presses/__init__.py` | **EDIT** | Export new class |
| `kvpress/__init__.py` | **EDIT** | Package-level export |

---

## Implementation

### File: `kvpress/presses/per_layer_press_router.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

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

### Updates to `kvpress/presses/__init__.py`

Add import:
```python
from kvpress.presses.per_layer_press_router import PerLayerPressRouter
```

Add to `__all__`:
```python
"PerLayerPressRouter",
```

### Updates to `kvpress/__init__.py`

Add to imports and `__all__`:
```python
from kvpress.presses.per_layer_press_router import PerLayerPressRouter
```

---

## Usage Examples

### Example 1: Different Press Types Per Layer Group

```python
from kvpress import (
    KnormPress, SnapKVPress, ExpectedAttentionPress,
    PerLayerPressRouter
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

### Example 2: Specific Layer Override (Attention Sinks)

```python
from kvpress import StreamingLLMPress, KnormPress, PerLayerPressRouter

router = PerLayerPressRouter(
    layer_presses={
        0: StreamingLLMPress(n_recent=128),  # Keep attention sinks in first layer
        range(1, 40): KnormPress(compression_ratio=0.5),
        # Layers 40-47: No compression (use default_press=None)
    },
    default_press=None  # No compression for unconfigured layers
)
```

### Example 3: Combined with PrefillDecodingPress

```python
from kvpress import DecodingPress, PrefillDecodingPress, KnormPress, SnapKVPress, PerLayerPressRouter

# Per-layer routing for prefill phase
prefill_router = PerLayerPressRouter(
    layer_presses={
        range(0, 16): KnormPress(compression_ratio=0.5),
        range(16, 48): SnapKVPress(compression_ratio=0.3),
    }
)

# Uniform decode press
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
    outputs = model(input_ids, past_key_values=cache, use_cache=True)
```

---

## Verification Steps

1. **Syntax check**:
   ```bash
   python -c "from kvpress import PerLayerPressRouter; print('OK')"
   ```

2. **Unit test**: Create test verifying correct press is routed per layer

3. **Integration test**: Run with Qwen model, observe different compression per layer group

---

## Available Press Types Reference

| Press Type | Scoring Method | Speed | Best For |
|------------|----------------|-------|----------|
| `KnormPress` | L2 norm of key vectors | ⚡ Fast | Early layers |
| `RandomPress` | Random selection | ⚡ Fast | Baseline |
| `SnapKVPress` | Recent token attention | 🔶 Medium | Middle layers |
| `ExpectedAttentionPress` | Statistical future attention | 🔶 Medium | Late layers |
| `ObservedAttentionPress` | Actual attention weights | 🐢 Slow | Max accuracy |
| `StreamingLLMPress` | Attention sinks + recent | ⚡ Fast | First layer |
| `TOVAPress` | Token-to-value attention | 🔶 Medium | Task-specific |
