"""
Odysseuskv: Adaptive KV Cache with Attention Probing.
Minimum MVP — replaces OmniKV's fixed filter layers with per-step probe-based decision.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, rotate_half, CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple

from modeling.omnikv.omnikv import select_tokens_by_attn_universal
from modeling.odysseuskv.cache import OdysseusCache, get_cache_cls


class OdysseusLayer(LlamaDecoderLayer):
    """单层 — 继承 LlamaDecoderLayer，加入探针决策逻辑"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.prefill_len = None
        self.config = config

    def compute_probe(self, hidden_states, position_ids, cache):
        """Quest: Σ_d max(Q_d * min_b[d], Q_d * max_b[d])"""
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        bsz, q_len, _ = hidden_states.shape
        query_states = self.self_attn.q_proj(hidden_states)
        key_states = self.self_attn.k_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
        cos, sin = self.self_attn.rotary_emb(key_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        bmin, bmax = cache.get_block_metadata(self.layer_idx, num_heads=self.self_attn.num_heads)
        q = query_states.squeeze(2)  # (1, h, d)
        s_min = (q.unsqueeze(1) * bmin.unsqueeze(0)).sum(dim=-1)  # (1, nb, h)
        s_max = (q.unsqueeze(1) * bmax.unsqueeze(0)).sum(dim=-1)
        return torch.max(s_min, s_max)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[OdysseusCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # ── Prefill: 记录长度 ──
        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]

        if past_key_value:
            assert isinstance(past_key_value, OdysseusCache)

        # ── Decode: 探针决策 ──
        if hidden_states.shape[1] == 1 and past_key_value:
            probe = None
            if past_key_value.do_select_layers is not None:
                is_full = past_key_value.should_do_full(self.layer_idx, None)
            else:
                probe = self.compute_probe(hidden_states, position_ids, past_key_value)
                is_full = past_key_value.should_do_full(self.layer_idx, probe)

            if is_full:
                consider_len = self.prefill_len or hidden_states.shape[1]
                num_selected_tokens = self.config.get("num_of_selected_tokens", 4096)
                if isinstance(num_selected_tokens, float):
                    num_selected_tokens = max(1, int(num_selected_tokens * consider_len))

                idx = select_tokens_by_attn_universal(
                    self.self_attn,
                    hidden_states,
                    position_ids,
                    past_key_value,
                    num_selected_tokens,
                    consider_len,
                    self.layer_idx,
                    self.config.get("selector_cls", "last"),
                )
                past_key_value.set_selected_idx(idx, self.layer_idx)

            past_key_value.set_layer_mode(self.layer_idx, is_full)
            if past_key_value.do_select_layers is None and is_full:
                past_key_value.last_full_probe = probe

        # ── Self-Attention ──
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # ── MLP ──
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class OdysseusModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            OdysseusLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.post_init()


class OdysseusLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        if (fac := config.get("rope_factor", -1)) > 0:
            config.rope_scaling = {"type": "dynamic", "factor": fac}
        config.rope_scaling_ = config.rope_scaling
        config.rope_scaling = None
        super().__init__(config)
        self.model = OdysseusModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_context_len = config.get("max_context_len", 50_000)
        self.cache_cls = get_cache_cls(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        n = input_ids.shape[1]

        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if not isinstance(past_key_values, self.cache_cls):
            kwargs = {
                "num_hidden_layers": self.config.num_hidden_layers,
                "probe_threshold": self.config.get("probe_threshold", 0.9),
                "block_size": self.config.get("block_size", 16),
                "do_select_layers": [int(i) for i in self.config.get("do_select_layers", "").split(",")] if self.config.get("do_select_layers", "") else None,
            }
            past_key_values = self.cache_cls.from_dynamic_cache(past_key_values, **kwargs)

        past_key_values.stage = "decoding" if n == 1 else "prefill"

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        self._last_cache = past_key_values
        hidden_states = outputs[0][:, -1:]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
