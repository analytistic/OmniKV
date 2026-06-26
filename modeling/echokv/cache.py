"""
EchoCache: 稀疏层 echo full 层的 KV

四种 echo_mode:
  - "kv": K 和 V 都从来源层借
  - "k":  K 从来源层借，V 用自己的
  - "v":  V 从来源层借，K 用自己的
  - None: 不借，用自己的 KV

store_every: 稀疏层中间隔多少层存一份 KV（0=全借不存, 1=全存, N=每N层存）
  存的层成为附近层的 echo 来源。
"""
import torch
import transformers


class EchoCache(transformers.cache_utils.DynamicCache):
    def __init__(self, full_attn_layers, num_hidden_layers,
                 num_wait_load_layers=1, echo_mode="kv", store_every=0):
        super().__init__()
        self.full_attn_layers = full_attn_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_wait_layers = num_wait_load_layers
        self.echo_mode = echo_mode if echo_mode != "" else None
        self.store_every = store_every
        self.stage = "prefill"
        self.selected_idx = None
        self.tail_k = {}
        self.tail_v = {}
        self.selected_idx_map = {}  # anchor_layer → selected indices, 供可视化

        # 构建 layer_state 和 kv_source（存储来源映射）
        self.layer_state = {}    # (is_sparse, group_anchor)
        self.kv_source = {}      # layer_idx → 去哪个层找 prefill KV
        for _l in full_attn_layers:
            _r = num_hidden_layers
            for i in range(_l + 1, num_hidden_layers):
                if i in full_attn_layers:
                    _r = i
                    break
            # full / wait 层
            for i in range(_l, min(_r, _l + self.num_wait_layers + 1)):
                self.layer_state[i] = (False, _l)
                self.kv_source[i] = i
            # sparse 层
            first_sparse = min(_r, _l + self.num_wait_layers + 1)
            last_owner = _l + self.num_wait_layers  # 最后一个存 KV 的层
            for i in range(first_sparse, _r):
                self.layer_state[i] = (True, _l)
                if store_every and (i - first_sparse) % store_every == 0:
                    # 这层自己存 KV
                    self.kv_source[i] = i
                    last_owner = i
                else:
                    # 从最近的 owner 借
                    self.kv_source[i] = last_owner

    def set_selected_idx(self, idx, _layer_idx):
        self.selected_idx = idx.view(-1)
        self.selected_idx_map[_layer_idx] = idx.view(-1).cpu()

    def _append(self, cache_list, layer_idx, tensor):
        if len(cache_list) <= layer_idx:
            cache_list.append(tensor)
        else:
            cache_list[layer_idx] = torch.cat([cache_list[layer_idx], tensor], dim=-2)

    def _sel(self, tensor, idx):
        if idx is not None and idx.numel() > 0:
            return tensor[:, :, idx]
        return tensor

    def _kv_source_layer(self, layer_idx):
        """返回该层 prefill KV 的来源层和自己的 decode 存储行为"""
        is_sparse, _ = self.layer_state[layer_idx]
        src = self.kv_source.get(layer_idx, layer_idx)
        stores_own = not is_sparse or src == layer_idx  # full 或 sparse_store
        return src, stores_own

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_sparse, _ = self.layer_state[layer_idx]
        src, stores_own = self._kv_source_layer(layer_idx)

        # prefill: 所有层都正常存 KV
        if self.stage == "prefill":
            self._append(self.key_cache, layer_idx, key_states)
            self._append(self.value_cache, layer_idx, value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # decode: 需要自己存 KV 的层
        if stores_own:
            self._append(self.key_cache, layer_idx, key_states)
            self._append(self.value_cache, layer_idx, value_states)

        # 累积 tail（所有 sparse 层都需要）
        if is_sparse:
            if layer_idx not in self.tail_k:
                self.tail_k[layer_idx] = key_states
                self.tail_v[layer_idx] = value_states
            else:
                self.tail_k[layer_idx] = torch.cat([self.tail_k[layer_idx], key_states], dim=-2)
                self.tail_v[layer_idx] = torch.cat([self.tail_v[layer_idx], value_states], dim=-2)

        # ── 决定返回什么 KV ──
        if not is_sparse or stores_own:
            # full 层 或 sparse_store: 返回自己的 KV
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # sparse echo: 从 kv_source 取 KV
        sel = self.selected_idx
        e = self.echo_mode

        if e is None:
            k_prefill = self._sel(self.key_cache[layer_idx], sel)
            v_prefill = self._sel(self.value_cache[layer_idx], sel)
        elif e == "kv":
            k_prefill = self._sel(self.key_cache[src], sel)
            v_prefill = self._sel(self.value_cache[src], sel)
        elif e == "k":
            k_prefill = self._sel(self.key_cache[src], sel)
            v_prefill = self._sel(self.value_cache[layer_idx], sel)
        elif e == "v":
            k_prefill = self._sel(self.key_cache[layer_idx], sel)
            v_prefill = self._sel(self.value_cache[src], sel)

        return torch.cat([k_prefill, self.tail_k[layer_idx]], dim=-2), \
               torch.cat([v_prefill, self.tail_v[layer_idx]], dim=-2)

    @staticmethod
    def from_dynamic_cache(cache, full_attn_layers, num_hidden_layers,
                           num_wait_load_layers=1, echo_mode="kv", store_every=0):
        c = EchoCache(full_attn_layers, num_hidden_layers, num_wait_load_layers, echo_mode, store_every)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c
