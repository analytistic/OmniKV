"""
RQ 量化工具：k-means → residual → 重建
"""
import torch
import torch.nn.functional as F


def kmeans_simple(x, k, iters=20):
    n, d = x.shape
    if n <= k:
        return F.pad(x, (0, 0, 0, k - n))
    c = x[torch.randperm(n)[:k]]
    for _ in range(iters):
        labels = torch.cdist(x, c).argmin(dim=1)
        new_c = torch.stack([
            x[labels == j].mean(dim=0) if (labels == j).sum() > 0 else c[j]
            for j in range(k)
        ])
        if (c - new_c).norm() < 1e-6:
            break
        c = new_c
    return c


def rq_encode(x, n_levels, n_centroids):
    """
    多级残差量化。
    x: (n, d)
    返回:
      centroids: [ (k, d) ] × n_levels
      indices:   [ (n,) ] × n_levels
      residual:  (n, d) — 最终残差
    """
    centroids_list, indices_list = [], []
    residual = x
    for _ in range(n_levels):
        cent = kmeans_simple(residual, n_centroids)
        idx = torch.cdist(residual, cent).argmin(dim=1)
        centroids_list.append(cent)
        indices_list.append(idx)
        residual = residual - cent[idx]
    return centroids_list, indices_list, residual


def rq_reconstruct(centroids_list, indices_list, residual=None):
    """
    x_recon = Σ centroids[l][indices[l]] + residual
    """
    recon = None
    for cent, idx in zip(centroids_list, indices_list):
        term = cent[idx]
        recon = term if recon is None else recon + term
    if residual is not None:
        recon = recon + residual
    return recon
