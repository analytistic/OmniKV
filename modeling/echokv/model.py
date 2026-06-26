"""
EchoLM: OmniKV 架构 + 稀疏层 echo full 层的 KV（而非 RQ 重建）
"""
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from modeling.omnikv.omnikv import select_tokens_by_attn_universal
from modeling.echokv.cache import EchoCache


class EchoLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.prefill_len = None
        self.config = config
        self.do_select_layers = [int(i) for i in config.get("do_select_layers", "").split(",")] if config.get("do_select_layers", "") else [0]

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]

        # token selection（同 OmniKV / ArachneKV）
        if hidden_states.shape[1] == 1 and past_key_value:
            if self.layer_idx in self.do_select_layers:
                consider_len = self.prefill_len or hidden_states.shape[1]
                num_tok = self.config.get("num_of_selected_tokens", 4096)
                if isinstance(num_tok, float):
                    num_tok = max(1, int(num_tok * consider_len))
                idx = select_tokens_by_attn_universal(
                    self.self_attn, hidden_states, position_ids,
                    past_key_value, num_tok, consider_len,
                    self.layer_idx, self.config.get("selector_cls", "last"),
                )
                past_key_value.set_selected_idx(idx, self.layer_idx)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

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


class EchoModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            EchoLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.post_init()


class EchoLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = EchoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_context_len = config.get("max_context_len", 50_000)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, cache_position=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        n = input_ids.shape[1]

        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if not isinstance(past_key_values, EchoCache):
            do_sel = self.config.get("do_select_layers", "")
            do_sel_layers = [int(i) for i in do_sel.split(",")] if do_sel else []
            full_layers = (
                list(range(0, do_sel_layers[0])) + do_sel_layers
                + [self.config.num_hidden_layers]
            ) if do_sel_layers else list(range(self.config.num_hidden_layers))
            past_key_values = EchoCache.from_dynamic_cache(
                past_key_values,
                full_attn_layers=full_layers,
                num_hidden_layers=self.config.num_hidden_layers,
                num_wait_load_layers=self.config.get("num_wait_load_layers", 1),
                echo_mode=self.config.get("echo_mode", "kv"),
                store_every=self.config.get("store_every", 0),
            )

        past_key_values.stage = "decoding" if n == 1 else "prefill"

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position,
        )

        self._last_cache = past_key_values
        hidden_states = outputs[0][:, -1:]
        logits = self.lm_head(hidden_states).float()

        if not return_dict:
            return (logits,) + outputs[1:]
        return CausalLMOutputWithPast(
            loss=None, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
