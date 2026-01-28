# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress

logger = logging.getLogger(__name__)

# Configure file handler for logging
_file_handler = logging.FileHandler("prefill_decoding_press.log")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_file_handler)
logger.setLevel(logging.DEBUG)


@dataclass
class PrefillDecodingPress(BasePress):
    """
    A wrapper press that combines separate prefilling and decoding compression strategies.

    This press acts as a single press interface but internally delegates to different
    presses based on the current phase (prefilling vs decoding). During prefilling,
    it uses the prefilling_press. During decoding, it uses the decoding_press.

    Parameters
    ----------
    prefilling_press : BasePress, optional
        Press to use during the prefilling phase. If None, no compression is applied during prefilling.
    decoding_press : DecodingPress, optional
        Press to use during the decoding phase. If None, no compression is applied during decoding.
    """

    prefilling_press: Optional[BasePress] = None
    decoding_press: Optional[DecodingPress] = None

    def post_init_from_model(self, model):
        if self.prefilling_press is not None:
            self.prefilling_press.post_init_from_model(model)
        if self.decoding_press is not None:
            self.decoding_press.post_init_from_model(model)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = hidden_states.shape[1]

        # Determine if we're in prefilling or decoding phase
        if kwargs["cache_position"][-1] <= q_len and self.prefilling_press is not None:
            return self.prefilling_press.compress(module, hidden_states, keys, values, attentions, kwargs)
        elif self.decoding_press is not None:
            return self.decoding_press.compress(module, hidden_states, keys, values, attentions, kwargs)

        # No compression applied
        logger.warning("No compression applied during prefill or decoding phase")

        return keys, values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Forward hook that delegates to the appropriate press based on current phase.
        """
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx
        cache_pos = kwargs["cache_position"][-1].item() if hasattr(kwargs["cache_position"][-1], 'item') else kwargs["cache_position"][-1]

        # Get cache info
        cache = kwargs.get("past_key_values")
        cache_len = cache.get_seq_length(layer_idx) if cache else 0

        # Determine if we're in prefilling or decoding phase
        is_prefill = cache_pos <= q_len
        phase = "PREFILL" if is_prefill else "DECODE"

        logger.debug(
            f"[{phase}] Layer {layer_idx}: q_len={q_len}, cache_pos={cache_pos}, cache_len={cache_len}"
        )

        if is_prefill and self.prefilling_press is not None:
            logger.debug(f"[{phase}] Layer {layer_idx}: Routing to {type(self.prefilling_press).__name__}")
            return self.prefilling_press.forward_hook(module, input, kwargs, output)
        elif self.decoding_press is not None:
            logger.debug(f"[{phase}] Layer {layer_idx}: Routing to {type(self.decoding_press).__name__}")
            return self.decoding_press.forward_hook(module, input, kwargs, output)

        # No hook applied
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        try:
            with super().__call__(model):
                yield
        finally:
            # Reset decoding press if it exists
            if self.decoding_press is not None:
                self.decoding_press.reset()
