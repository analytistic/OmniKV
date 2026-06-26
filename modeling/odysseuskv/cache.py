"""
Odysseuskv Cache: Quest-style block min/max probe + KV 管理
"""
import torch
import torch.nn.functional as F
import transformers
from typing import List, Optional


class OdysseusCache(transformers.cache_utils.DynamicCache):
    """
    用 Quest 的 per-block min/max of K 做探针。
    - Prefill: 存 KV，按 block_size 切块，存每块的 K_min / K_max
    - Decode: Q × (min, max) → upper bound score → probe 分布
    - 层间 cosine 比较决定 full/sparse
    """

    def __init__(self, num_hidden_layers, probe_threshold=0.9, block_size=16, do_select_layers=None):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.threshold = probe_threshold
        self.block_size = block_size
        self.do_select_layers = do_select_layers
        self.stage = "prefill"

        # 探针
        self.block_min = {}           # [layer]: (n_blocks, n_kv_heads, head_dim)
        self.block_max = {}           # [layer]: (n_blocks, n_kv_heads, head_dim)
        self.last_full_probe = None   # 上一个 full 层的 probe
        self.selected_idx = None
        self.layer_mode = {}
        self.tail_k = {}
        self.tail_v = {}
        self._metadata_ready = False
        # 统计
        self.stats_full = 0
        self.stats_sparse = 0
        self.step_full_per_layer = []

    # ── 探针 ──

    def get_block_metadata(self, layer_idx, num_heads=None):
        if not self._metadata_ready:
            self._build_metadata()
        bmin = self.block_min[layer_idx]  # (n_blocks, n_kv_heads, head_dim)
        bmax = self.block_max[layer_idx]
        if num_heads is not None and num_heads > bmin.shape[1]:
            groups = num_heads // bmin.shape[1]
            def rep(t):
                return t.unsqueeze(2).expand(-1, -1, groups, -1).reshape(t.shape[0], num_heads, t.shape[-1])
            bmin, bmax = rep(bmin), rep(bmax)
        return bmin, bmax

    def should_do_full(self, layer_idx, probe):
        if self.do_select_layers is not None:
            if layer_idx == 0 or layer_idx in self.do_select_layers:
                return True
            if layer_idx < self.do_select_layers[0]:
                return True
            return False
        if layer_idx == 0 or self.last_full_probe is None:
            return True
        cos = F.cosine_similarity(probe.reshape(-1), self.last_full_probe.reshape(-1), dim=0)
        is_full = abs(cos.item()) < self.threshold
        if is_full:
            self.stats_full += 1
        else:
            self.stats_sparse += 1
        return is_full

    def set_selected_idx(self, idx, layer_idx):
        self.selected_idx = idx.view(-1)

    def set_layer_mode(self, layer_idx, is_full):
        self.layer_mode[layer_idx] = 'full' if is_full else 'sparse'

    # ── Block metadata ──

    def _build_metadata(self):
        for l in range(self.num_hidden_layers):
            k = self.key_cache[l][0]  # (n_kv_heads, seq_len, head_dim)
            seq_len = k.shape[1]
            nb = seq_len // self.block_size
            trimmed = k[:, :nb * self.block_size]  # (n_kv, nb*bs, hd)
            blocks = trimmed.reshape(k.shape[0], nb, self.block_size, k.shape[2])
            self.block_min[l] = blocks.amin(dim=2).transpose(0, 1).contiguous()   # (nb, n_kv, hd)
            self.block_max[l] = blocks.amax(dim=2).transpose(0, 1).contiguous()
        self._metadata_ready = True

    # ── KV 管理 ──

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.stage == "prefill" or layer_idx not in self.layer_mode:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        mode = self.layer_mode.get(layer_idx, 'full')

        if mode == 'full':
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        else:  # sparse
            if layer_idx not in self.tail_k:
                self.tail_k[layer_idx] = key_states
                self.tail_v[layer_idx] = value_states
            else:
                self.tail_k[layer_idx] = torch.cat([self.tail_k[layer_idx], key_states], dim=-2)
                self.tail_v[layer_idx] = torch.cat([self.tail_v[layer_idx], value_states], dim=-2)

            full_k = self.key_cache[layer_idx]
            full_v = self.value_cache[layer_idx]
            effective_len = full_k.shape[-2]
            valid_idx = self.selected_idx[self.selected_idx < effective_len]
            if valid_idx.numel() == 0:
                return full_k, full_v

            sel_k = torch.index_select(full_k, -2, valid_idx)
            sel_v = torch.index_select(full_v, -2, valid_idx)
            return torch.cat([sel_k, self.tail_k[layer_idx]], dim=-2), \
                   torch.cat([sel_v, self.tail_v[layer_idx]], dim=-2)

    @staticmethod
    def from_dynamic_cache(cache, num_hidden_layers, probe_threshold=0.9, block_size=16, do_select_layers=None):
        c = OdysseusCache(num_hidden_layers, probe_threshold, block_size, do_select_layers)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c

    def print_stats(self):
        total = self.stats_full + self.stats_sparse
        if total == 0:
            return
        print(f"\n[Odysseus] Decode stats: full={self.stats_full}/{total} "
              f"({100*self.stats_full//total}%), "
              f"sparse={self.stats_sparse}/{total} ({100*self.stats_sparse//total}%)")


def get_cache_cls(config):
    return OdysseusCache
