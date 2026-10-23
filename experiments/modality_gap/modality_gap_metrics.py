"""Modality-gap metrics of "Closing the Modality Gap via Simplex-Constrained Representations" (Interspeech 2026).

``compute_modality_gap`` is the function behind the gap columns of Tables 1-2 and of ``results/*.csv``
(``gap/centroid_l2``, ``gap/relative_gap``, ``gap/silhouette``, ...). ``modality_gap`` is a small wrapper for the
common case of two embedding matrices without item identifiers.

Metrics (A = text, B = audio or image; lower centroid_l2 / relative_gap / silhouette = smaller gap):
* centroid_l2        ||mean(A) - mean(B)||_2
* centroid_cos_dist  1 - cos(mean(A), mean(B))
* relative_gap       centroid_l2 / sqrt(0.5 (E||a - mean(A)||^2 + E||b - mean(B)||^2))      (RelGap, Eq. 1)
* alignment          mean similarity of matched pairs: dot product, or minus the total-variation distance on the simplex
* silhouette         cosine silhouette score of the two modalities as two clusters (1,000 samples each, seed 42)
* anisotropy_*       mean dot product of random same-modality pairs; uniformity_*: Wang & Isola (2020) log-mean-exp
* on the simplex (use_tvd=True) also: centroid Hellinger and Jensen-Shannon distances, Bhattacharyya anisotropy and
  Hellinger uniformity per modality
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_modality_gap(
    text_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    text_labels: np.ndarray,
    audio_labels: np.ndarray,
    n_pairs: int = 3000,
    sil_cap: int = 1000,
    use_tvd: bool = False,
) -> Dict[str, float]:
    """Gap metrics between the text embeddings and the audio/image embeddings.

    ``text_labels`` / ``audio_labels`` identify the item each row belongs to; rows sharing a label across the two
    modalities form the matched pairs of the alignment metric. ``use_tvd=True`` treats the rows as probability
    vectors (softmax adapters) and adds the simplex-native metrics.
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(42)
    te = torch.as_tensor(text_embs).float()
    ae = torch.as_tensor(audio_embs).float()

    mu_t = te.mean(0)
    mu_a = ae.mean(0)
    centroid_l2 = float(torch.norm(mu_t - mu_a))
    t_cn = F.normalize(mu_t.unsqueeze(0), dim=-1).squeeze()
    a_cn = F.normalize(mu_a.unsqueeze(0), dim=-1).squeeze()
    centroid_cos_dist = float(1.0 - (t_cn * a_cn).sum())

    var_t = float(((te - mu_t) ** 2).sum(-1).mean())
    var_a = float(((ae - mu_a) ** 2).sum(-1).mean())
    relative_gap = centroid_l2 / (float(torch.sqrt(torch.tensor(0.5 * (var_t + var_a)))) + 1e-8)

    shared = np.intersect1d(text_labels, audio_labels)
    if len(shared):
        ti = [int(rng.choice(np.where(text_labels == l)[0])) for l in shared]
        ai = [int(rng.choice(np.where(audio_labels == l)[0])) for l in shared]
        if use_tvd:
            P = te[ti].numpy()
            Q = ae[ai].numpy()
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
            Q = Q / (Q.sum(axis=1, keepdims=True) + 1e-12)
            alignment = float(-0.5 * np.abs(P - Q).sum(axis=-1).mean())
        else:
            alignment = float((te[ti] * ae[ai]).sum(dim=-1).mean())
    else:
        alignment = float("nan")

    ts = rng.choice(len(te), min(sil_cap, len(te)), replace=False)
    as_ = rng.choice(len(ae), min(sil_cap, len(ae)), replace=False)
    all_e = np.vstack([te[ts].numpy(), ae[as_].numpy()])
    lbl_sil = np.array([0] * len(ts) + [1] * len(as_))
    try:
        sil = float(silhouette_score(all_e, lbl_sil, metric="cosine"))
    except Exception:
        sil = float("nan")

    def anisotropy(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 2:
            return float("nan")
        i1 = rng.integers(0, n, n_pairs)
        i2 = rng.integers(0, n, n_pairs)
        same = i1 == i2
        i2[same] = (i2[same] + 1) % n
        v1 = torch.tensor(arr[i1]).float()
        v2 = torch.tensor(arr[i2]).float()
        return float((v1 * v2).sum(-1).mean())

    def uniformity(arr: np.ndarray) -> float:
        n = len(arr)
        i1 = rng.integers(0, n, n_pairs)
        i2 = rng.integers(0, n, n_pairs)
        v1 = torch.tensor(arr[i1]).float()
        v2 = torch.tensor(arr[i2]).float()
        return float(torch.log(torch.exp(-2.0 * ((v1 - v2) ** 2).sum(-1)).mean()))

    def bhattacharyya_coeff(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 2:
            return float("nan")
        i1 = rng.integers(0, n, n_pairs)
        i2 = rng.integers(0, n, n_pairs)
        same = i1 == i2
        i2[same] = (i2[same] + 1) % n
        p = torch.tensor(arr[i1]).float().clamp(min=1e-12)
        q = torch.tensor(arr[i2]).float().clamp(min=1e-12)
        return float(torch.sqrt(p * q).sum(-1).mean())

    def uniformity_hellinger(arr: np.ndarray) -> float:
        n = len(arr)
        i1 = rng.integers(0, n, n_pairs)
        i2 = rng.integers(0, n, n_pairs)
        p = torch.tensor(arr[i1]).float().clamp(min=1e-12)
        q = torch.tensor(arr[i2]).float().clamp(min=1e-12)
        bc = torch.sqrt(p * q).sum(-1)
        h2 = 2.0 * (1.0 - bc)
        return float(torch.log(torch.exp(-2.0 * h2).mean()))

    results = {
        "centroid_l2": centroid_l2,
        "centroid_cos_dist": centroid_cos_dist,
        "relative_gap": relative_gap,
        "alignment": alignment,
        "silhouette": sil,
        "anisotropy_text": anisotropy(te.numpy()),
        "anisotropy_audio": anisotropy(ae.numpy()),
        "uniformity_text": uniformity(te.numpy()),
        "uniformity_audio": uniformity(ae.numpy()),
    }

    if use_tvd:
        te_np, ae_np = te.numpy(), ae.numpy()
        t_mean = np.maximum(te_np.mean(axis=0), 1e-12)
        a_mean = np.maximum(ae_np.mean(axis=0), 1e-12)
        bc_centroid = float(np.sqrt(t_mean * a_mean).sum())
        results["centroid_hellinger"] = float(np.sqrt(2.0 * max(0.0, 1.0 - bc_centroid)))
        m = 0.5 * (t_mean + a_mean)
        kl_tm = float(np.sum(t_mean * np.log(t_mean / (m + 1e-12) + 1e-12)))
        kl_am = float(np.sum(a_mean * np.log(a_mean / (m + 1e-12) + 1e-12)))
        results["centroid_jsd"] = 0.5 * (kl_tm + kl_am)
        results["anisotropy_text_bc"] = bhattacharyya_coeff(te_np)
        results["anisotropy_audio_bc"] = bhattacharyya_coeff(ae_np)
        results["uniformity_text_hellinger"] = uniformity_hellinger(te_np)
        results["uniformity_audio_hellinger"] = uniformity_hellinger(ae_np)

    return results


def modality_gap(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    ids_a: Optional[np.ndarray] = None,
    ids_b: Optional[np.ndarray] = None,
    simplex: bool = False,
) -> Dict[str, float]:
    """``compute_modality_gap`` for two matrices; without identifiers, row i of A is paired with row i of B."""
    if ids_a is None:
        ids_a = np.arange(len(emb_a))
    if ids_b is None:
        ids_b = np.arange(len(emb_b))
    return compute_modality_gap(emb_a, emb_b, np.asarray(ids_a), np.asarray(ids_b), use_tvd=simplex)
