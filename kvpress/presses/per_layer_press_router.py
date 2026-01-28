# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from torch import nn

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class PerLayerPressRouter(BasePress):
    """
    Route different press types to different layers.

    Enables using different compression strategies for different transformer layers,
    allowing optimal press selection based on layer characteristics:
    - Early layers: syntax/structure -> fast, aggressive (e.g., KnormPress)
    - Middle layers: semantics -> attention-based (e.g., SnapKVPress)
    - Late layers: task output -> careful/light (e.g., ExpectedAttentionPress)

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

        # Determine phase from input sequence length
        hidden_states = output[0]
        q_len = hidden_states.shape[1]
        phase = "PREFILL" if q_len > 1 else "DECODE"

        # Get cache info before compression
        cache = kwargs.get("past_key_values")
        cache_len_before = cache.get_seq_length(layer_idx) if cache else 0

        logger.debug(
            f"[{phase}] Layer {layer_idx}: q_len={q_len}, cache_before={cache_len_before}"
        )

        if press is None:
            logger.debug(f"[{phase}] Layer {layer_idx}: No press configured, skipping compression")
            return output  # No compression for this layer

        logger.debug(f"[{phase}] Layer {layer_idx}: Routing to {type(press).__name__}")
        result = press.forward_hook(module, input, kwargs, output)

        # Get cache info after compression
        cache_len_after = cache.get_seq_length(layer_idx) if cache else 0
        if cache_len_before != cache_len_after:
            logger.debug(
                f"[{phase}] Layer {layer_idx}: Compressed {cache_len_before} -> {cache_len_after} tokens"
            )

        return result

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate compression to the appropriate press for this layer."""
        layer_idx = module.layer_idx
        press = self.get_press_for_layer(layer_idx)

        if press is None:
            return keys, values  # No compression

        return press.compress(module, hidden_states, keys, values, attentions, kwargs)

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
