"""Layer-depth ablation (local addition): hide the visual tokens from decoder layer L on.

From layer ``L`` (zero-based) to the last layer, no token can read a visual token (the history
video patches and the current wrist image); layers below ``L`` run unchanged and positions never
move. The Qwen3.5 text decoder alternates two token mixers, so "cannot read" has two forms:

* ``full_attention`` layers (every 4th: 3, 7, ..., 31): the visual tokens' key columns are closed
  for every query, on the prompt prefill and on every decode step.
* ``linear_attention`` layers (gated delta rule, a recurrence with a 4-tap causal convolution): the
  layer runs on the sequence with the visual positions removed, so they neither write to the
  recurrent state, nor decay it, nor enter the convolution of the tokens after them.

The DiT action head reads only the last-layer states of the generated sub-task tokens (plus
proprioception), so nothing else needs hiding. ``L = num_layers`` hides nothing; nothing is
installed unless a blinder is constructed, and an installed blinder with no visual mask set is a
pass-through.
"""
from __future__ import annotations

import torch


class VisualBlinder:
    def __init__(self, language_model, layer_start: int):
        layers = language_model.layers
        self.num_layers = len(layers)
        self.layer_start = int(layer_start)
        if not 0 <= self.layer_start <= self.num_layers:
            raise ValueError(f"blind layer must be in [0, {self.num_layers}], got {layer_start}")
        self._visual: torch.Tensor | None = None  # (batch, prompt_len) bool, True = visual token
        self._keep_idx: torch.Tensor | None = None  # (batch, n_keep) prompt positions that stay
        self.calls = {"full_attention": 0, "linear_attention": 0}
        self._wrapped = [layers[i] for i in range(self.layer_start, self.num_layers)]
        for layer in self._wrapped:
            self._wrap(layer)

    def remove(self) -> None:
        """Restore the layers' own forward (used by the checks)."""
        for layer in self._wrapped:
            layer.__dict__.pop("forward", None)
        self._wrapped = []
        self.clear()

    # -- per-decision state -------------------------------------------------------------------
    def set_visual(self, visual: torch.Tensor) -> None:
        """``visual``: (batch, prompt_len) bool over the (left-padded) prompt of this decision."""
        visual = visual.bool()
        counts = (~visual).sum(-1)
        if not bool((counts == counts[0]).all()):
            raise RuntimeError("rows keep different numbers of tokens; cannot compact the batch")
        self._visual = visual
        self._keep_idx = (~visual).nonzero(as_tuple=True)[1].view(visual.shape[0], -1)

    def clear(self) -> None:
        self._visual = None
        self._keep_idx = None

    # -- hooks --------------------------------------------------------------------------------
    def _wrap(self, layer) -> None:
        orig = layer.forward
        kind = layer.block_type

        def forward(hidden_states, *args, **kwargs):
            if self._visual is None:
                return orig(hidden_states, *args, **kwargs)
            if args:
                raise RuntimeError("layer-blind hook expects keyword arguments after hidden_states")
            self.calls[kind] += 1
            if kind == "full_attention":
                kwargs["attention_mask"] = self._close_columns(
                    kwargs.get("attention_mask"), hidden_states, layer, kwargs.get("past_key_values"))
                return orig(hidden_states, **kwargs)
            return self._skip_visual(orig, hidden_states, kwargs)

        layer.forward = forward

    def _close_columns(self, mask, hidden_states, layer, cache):
        batch, q_len, _ = hidden_states.shape
        prompt_len = self._visual.shape[1]
        if mask is not None:
            k_len = mask.shape[-1]
        elif q_len == prompt_len:
            k_len = prompt_len
        else:
            k_len = cache.layers[layer.self_attn.layer_idx].keys.shape[-2] + q_len
        if k_len < prompt_len:
            raise RuntimeError(f"attention over {k_len} keys but the prompt has {prompt_len}")
        closed = torch.zeros(batch, k_len, dtype=torch.bool, device=hidden_states.device)
        closed[:, :prompt_len] = self._visual
        closed = closed[:, None, None, :]
        if mask is None:
            if q_len == 1:
                base = torch.ones(1, 1, 1, k_len, dtype=torch.bool, device=hidden_states.device)
            elif q_len == k_len:
                base = torch.ones(q_len, k_len, dtype=torch.bool, device=hidden_states.device).tril()[None, None]
            else:
                raise RuntimeError("cannot rebuild a causal mask for a chunked continuation")
            return base & ~closed
        if mask.dim() != 4:
            raise RuntimeError(f"expected a 4D attention mask, got {tuple(mask.shape)}")
        if mask.dtype == torch.bool:
            return mask & ~closed
        return mask.masked_fill(closed, torch.finfo(mask.dtype).min)

    def _skip_visual(self, orig, hidden_states, kwargs):
        q_len = hidden_states.shape[1]
        if q_len == 1:  # a decode step: the new token is text, the states already skip the visuals
            return orig(hidden_states, **kwargs)
        if q_len != self._visual.shape[1]:
            raise RuntimeError("layer-blind hook supports a full-prompt prefill and 1-token decode steps")
        idx = self._keep_idx
        gather = idx[:, :, None].expand(-1, -1, hidden_states.shape[-1])
        mask = kwargs.get("attention_mask")
        if mask is not None:
            if mask.dim() != 2 or mask.shape[1] != q_len:
                raise RuntimeError(f"expected a (batch, seq) recurrent mask, got {tuple(mask.shape)}")
            kwargs["attention_mask"] = mask.gather(1, idx)
        kwargs["position_embeddings"] = None  # unused by the recurrent mixer; lengths would not match
        kwargs["position_ids"] = None
        out = orig(hidden_states.gather(1, gather), **kwargs)
        return hidden_states.scatter(1, gather, out)  # visual rows keep their input; nothing reads them
