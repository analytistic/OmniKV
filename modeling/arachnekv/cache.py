"""
ArachneCache: OmniKV 架构 + KV 存储压缩（RQ 码本 + residual）
"""
import torch
import transformers
from .quant import rq_encode, rq_reconstruct


class ArachneCache(transformers.cache_utils.DynamicCache):
    """
    同 OmniKV:
      - full_attn_layers → filter + wait 层
      - 其余层 → sparse（复用 filter 层选的 idx）

    区别：sparse 层的 KV 不直接从 key_cache 读，而是从 RQ 码本重建。
    """

    def __init__(self, full_attn_layers, num_hidden_layers,
                 num_wait_load_layers=1,
                 k_rq_levels=3, k_rq_cenyroid=10, k_residual_keep_ratio=1.0,
                 v_rq_levels=3, v_rq_cenyroid=10, v_residual_keep_ratio=1.0):
        super().__init__()
        self.full_attn_layers = full_attn_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_wait_layers = num_wait_load_layers
        self.k_rq_levels = k_rq_levels
        self.k_rq_cenyroid = k_rq_cenyroid
        self.k_residual_keep_ratio = k_residual_keep_ratio
        self.v_rq_levels = v_rq_levels
        self.v_rq_cenyroid = v_rq_cenyroid
        self.v_residual_keep_ratio = v_residual_keep_ratio
        self.stage = "prefill"
        self.layer_state = {}
        self.selected_idx = None
        self.tail_k = {}
        self.tail_v = {}

        # 层状态：跟 OmniKV 一样
        for _l in full_attn_layers:
            _r = num_hidden_layers
            for i in range(_l + 1, num_hidden_layers):
                if i in full_attn_layers:
                    _r = i
                    break
            for i in range(_l, min(_r, _l + self.num_wait_layers + 1)):
                self.layer_state[i] = (False, _l, _r)  # full
            for i in range(min(_r, _l + self.num_wait_layers + 1), _r):
                self.layer_state[i] = (True, _l, _r)   # sparse (offload)

        # 压缩存储
        self.k_centroids = {}   # [layer]: list of [n_levels] × (n_kv, n_clusters, hd)
        self.v_centroids = {}
        self.k_indices = {}     # [layer]: list of [n_levels] × (n_kv, seq)
        self.v_indices = {}
        self.k_residual = {}    # [layer]: (n_kv, seq, hd) — 全量残差
        self.v_residual = {}
        self.selected_idx_map = {}  # anchor_layer → selected indices, 供可视化
        self._compressed = False

    # ── 压缩 ──

    def compress(self):
        """Prefill 后对所有层的 KV 做 RQ 量化"""
        for l in range(self.num_hidden_layers):
            if not self.layer_state[l][0]:
                continue  # full 层不压缩
            k = self.key_cache[l][0]   # (n_kv, seq, hd)
            v = self.value_cache[l][0]
            n_kv, seq, hd = k.shape
            # K 量化
            nck = seq // self.k_rq_cenyroid
            k_cent, k_idx, k_res = zip(*[
                rq_encode(k[h], self.k_rq_levels, nck) for h in range(n_kv)
            ])
            self.k_centroids[l] = [torch.stack([c[li] for c in k_cent]) for li in range(self.k_rq_levels)]
            self.k_indices[l] = [torch.stack([c[li] for c in k_idx]) for li in range(self.k_rq_levels)]
            self.k_residual[l] = torch.stack(list(k_res))
            # K 残差截断
            krr = self.k_residual_keep_ratio
            if krr < 1.0:
                keep = int(self.k_residual[l].shape[-1] * krr)
                if keep <= 0: self.k_residual[l].zero_()
                else:
                    th = self.k_residual[l].abs().topk(keep, dim=-1).values[:, :, -1:]
                    self.k_residual[l][self.k_residual[l].abs() < th] = 0
            # V 量化
            ncv = seq // self.v_rq_cenyroid
            v_cent, v_idx, v_res = zip(*[
                rq_encode(v[h], self.v_rq_levels, ncv) for h in range(n_kv)
            ])
            self.v_centroids[l] = [torch.stack([c[li] for c in v_cent]) for li in range(self.v_rq_levels)]
            self.v_indices[l] = [torch.stack([c[li] for c in v_idx]) for li in range(self.v_rq_levels)]
            self.v_residual[l] = torch.stack(list(v_res))
            # V 残差截断
            vrr = self.v_residual_keep_ratio
            if vrr < 1.0:
                keep = int(self.v_residual[l].shape[-1] * vrr)
                if keep <= 0: self.v_residual[l].zero_()
                else:
                    th = self.v_residual[l].abs().topk(keep, dim=-1).values[:, :, -1:]
                    self.v_residual[l][self.v_residual[l].abs() < th] = 0
        self._compressed = True

    def reconstruct(self, layer_idx, indices):
        """从码本重建指定位置（indices）的 KV"""
        k_recon, v_recon = 0, 0
        for li in range(self.k_rq_levels):
            cent_k = self.k_centroids[layer_idx][li]  # (n_kv, n_c, hd)
            k_recon = k_recon + torch.stack([cent_k[h][self.k_indices[layer_idx][li][h, indices]] for h in range(cent_k.shape[0])])
        k_recon = k_recon + self.k_residual[layer_idx][:, indices]

        for li in range(self.v_rq_levels):
            cent_v = self.v_centroids[layer_idx][li]
            v_recon = v_recon + torch.stack([cent_v[h][self.v_indices[layer_idx][li][h, indices]] for h in range(cent_v.shape[0])])
        v_recon = v_recon + self.v_residual[layer_idx][:, indices]

        return k_recon.unsqueeze(0), v_recon.unsqueeze(0)

    # ── KV 管理 ──

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_sparse, _l, _r = self.layer_state[layer_idx]

        if self.stage == "prefill":
            # 所有层 prefill 时正常追加
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # Decode
        if not is_sparse:
            # full / wait 层：正常 concat
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        else:
            # sparse 层：从码本重建
            if layer_idx not in self.tail_k:
                self.tail_k[layer_idx] = key_states
                self.tail_v[layer_idx] = value_states
            else:
                self.tail_k[layer_idx] = torch.cat([self.tail_k[layer_idx], key_states], dim=-2)
                self.tail_v[layer_idx] = torch.cat([self.tail_v[layer_idx], value_states], dim=-2)

            if self.selected_idx is None or self.selected_idx.numel() == 0:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            recon_k, recon_v = self.reconstruct(layer_idx, self.selected_idx)
            return torch.cat([recon_k, self.tail_k[layer_idx]], dim=-2), \
                   torch.cat([recon_v, self.tail_v[layer_idx]], dim=-2)

    def set_selected_idx(self, idx, _layer_idx):
        self.selected_idx = idx.view(-1)
        self.selected_idx_map[_layer_idx] = idx.view(-1).cpu()

    @staticmethod
    def from_dynamic_cache(cache, full_attn_layers, num_hidden_layers, num_wait_load_layers=1,
                           k_rq_levels=3, k_rq_cenyroid=10, k_residual_keep_ratio=1.0,
                           v_rq_levels=3, v_rq_cenyroid=10, v_residual_keep_ratio=1.0):
        c = ArachneCache(full_attn_layers, num_hidden_layers, num_wait_load_layers,
                         k_rq_levels, k_rq_cenyroid, k_residual_keep_ratio,
                         v_rq_levels, v_rq_cenyroid, v_residual_keep_ratio)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c
