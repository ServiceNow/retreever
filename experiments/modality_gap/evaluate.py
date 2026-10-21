#!/usr/bin/env python
"""Evaluate trained adapters (and the frozen encoders) on the validation split: retrieval in both directions and the
modality-gap metrics of "Closing the Modality Gap via Simplex-Constrained Representations".

    python -m experiments.modality_gap.evaluate --dataset audiocaps \
        --runs baseline runs/modality_gap/audiocaps_linear_softmax_clap-fused-ta runs/modality_gap/audiocaps_mlp_200k \
        --output experiments/modality_gap/results/my_eval.csv

Each run directory is one written by scripts/train.py (config.yaml + checkpoint-*/pytorch_model.bin). ``baseline`` is
the frozen encoder's own joint space (CLAP projection heads, or FLAVA CLS features). Rows are written with the same
columns as results/modality_gap_results.csv (level == "full"). A run can be named with ``name=path``.

Protocol (identical to the paper): the validation split is de-duplicated into unique audio/image items and unique
captions; clotho_augmented keeps the first ``--n_source_uids`` source clips (captions are shared by the augmentations
of a source clip, so retrieval is scored at the source level). Softmax adapters are scored with the total-variation
similarity, everything else with cosine.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import datasets as hf_datasets
from local_paths import DATA_PATHS
from retreever.models.encoders import get_encoders
from retreever.models.retreever import load_from_ckpt
from retreever.utils.toolkit_paths import PATH_HF_CACHE_RW
from scripts.train import TEXT_AUDIO_DATASETS, TEXT_IMAGE_DATASETS, preprocessed_audio_dir
from experiments.modality_gap.modality_gap_metrics import compute_modality_gap

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("modality_gap.evaluate")

# clotho_augmented carries a source_uid (the un-augmented clip) shared by all augmentations of a clip
HAS_SOURCE_UID = {"clotho_augmented": True}
DEFAULT_ENCODER = {**{d: "clap-fused-ta" for d in TEXT_AUDIO_DATASETS}, **{d: "flava" for d in TEXT_IMAGE_DATASETS}}


def dataset_cfg(name: str) -> dict:
    root = Path(DATA_PATHS[name])
    if name in TEXT_IMAGE_DATASETS:
        return {"hf_path": root / TEXT_IMAGE_DATASETS[name], "images_base": root, "modality": "image",
                "has_source_uid": False, "encoder_type": DEFAULT_ENCODER[name]}
    subdir, source_sr = TEXT_AUDIO_DATASETS[name]
    return {"hf_path": root / subdir, "audio_base": root, "modality": "audio", "source_sr": source_sr, "target_sr": 48000,
            "has_source_uid": HAS_SOURCE_UID.get(name, False), "encoder_type": DEFAULT_ENCODER[name]}


# ─── validation split ─────────────────────────────────────────────────────────────────────────────────────────────
def load_and_select_val(name: str, cfg: dict, n_source_uids: Optional[int]) -> hf_datasets.Dataset:
    ds = hf_datasets.load_from_disk(str(cfg["hf_path"]))["val"]
    if cfg["has_source_uid"] and n_source_uids is not None:
        seen: Dict[int, int] = {}
        indices = []
        for i in range(len(ds)):
            src = ds[i]["source_uid"]
            if src not in seen:
                seen[src] = len(seen)
            if seen[src] < n_source_uids:
                indices.append(i)
        ds = ds.select(indices)
        logger.info(f"{name}: {len(ds)} rows from {n_source_uids} source_uids")
    else:
        logger.info(f"{name}: {len(ds)} val rows")
    return ds


def build_unique_rows(ds: hf_datasets.Dataset, has_src: bool):
    """Unique contexts (one per context_uid) and unique captions (one per (source_uid, caption))."""
    seen_ctx, seen_text = set(), set()
    ctx_rows, ctx_uids, ctx_src = [], [], []
    text_rows, text_src = [], []
    for i in range(len(ds)):
        row = ds[i]
        uid = row["context_uid"]
        src = row["source_uid"] if has_src else uid
        if uid not in seen_ctx:
            seen_ctx.add(uid); ctx_rows.append(row); ctx_uids.append(uid); ctx_src.append(src)
        key = (src, row["question"])
        if key not in seen_text:
            seen_text.add(key); text_rows.append(row); text_src.append(src)
    logger.info(f"  unique contexts: {len(ctx_rows)} | unique captions: {len(text_rows)}")
    return ctx_rows, np.array(ctx_uids), np.array(ctx_src), text_rows, np.array(text_src)


# ─── loading contexts ──────────────────────────────────────────────────────────────────────────────────────────────
def _load_preprocessed(uid: int, preproc_dir: Path):
    pt = preproc_dir / f"{uid}.pt"
    if pt.exists():
        d = torch.load(pt, map_location="cpu", weights_only=True)
        return d["input_features"], d["is_longer"].bool()
    return None, None


def _load_waveform(audio_path: str, audio_base: Path, tgt_sr: int, max_dur: float = 30.0) -> torch.Tensor:
    import torchaudio
    full = Path(audio_base) / audio_path
    try:
        wav, sr = torchaudio.load(str(full))
    except Exception as exc:
        logger.warning(f"Cannot load {full}: {exc}")
        return torch.zeros(1, int(tgt_sr * max_dur))
    if sr != tgt_sr:
        wav = torchaudio.transforms.Resample(sr, tgt_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    max_s = int(tgt_sr * max_dur)
    n = wav.shape[1]
    if n > max_s:
        start = (n - max_s) // 2
        wav = wav[:, start:start + max_s]
    elif n < max_s:
        wav = F.pad(wav, (0, max_s - n))
    return wav


def _load_images(rows: list, images_base: Path):
    from PIL import Image
    images = []
    for row in rows:
        try:
            images.append(Image.open(Path(images_base) / row["context"]).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (224, 224), "white"))
    return images


def _context_batch(rows: list, cfg: dict, preproc_dir: Optional[Path], processor, device: str):
    """Returns (tensor to feed the context encoder, is_longer or None)."""
    if cfg["modality"] == "image":
        pixel_values = processor(images=_load_images(rows, cfg["images_base"]), return_tensors="pt")["pixel_values"]
        return pixel_values.to(device), None
    feats, longer, wavs = [], [], []
    for row in rows:
        if preproc_dir is not None:
            f, l = _load_preprocessed(row["context_uid"], preproc_dir)
            if f is not None:
                feats.append(f); longer.append(l); continue
        wavs.append(_load_waveform(row["context"], cfg["audio_base"], cfg["target_sr"]).squeeze(0))
    if feats and not wavs:
        return torch.stack(feats).to(device), torch.stack(longer).to(device)
    if feats and wavs:
        raise RuntimeError("Mixed pre-computed and raw audio in one batch; preprocess the whole split or none of it.")
    return torch.stack(wavs).to(device), None


# ─── embeddings ────────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_with_model(model, ctx_rows, text_rows, cfg, preproc_dir, device, batch_size, normalize: bool):
    tokenizer = model.query_encoder.tokenizer
    text_embs = []
    for s in tqdm(range(0, len(text_rows), batch_size), desc="  text", leave=False):
        batch = text_rows[s:s + batch_size]
        toks = tokenizer([r["question"] for r in batch], return_tensors="pt", padding=True, truncation=True)
        q_ids, q_mask = toks["input_ids"].to(device), toks["attention_mask"].to(device)
        raw = model.query_encoder(q_ids, attention_mask=q_mask)
        emb = model._apply_projection(raw, model.query_projection, q_mask) if model.query_projection is not None else raw
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        text_embs.append(emb.float().cpu())
    ctx_embs = []
    for s in tqdm(range(0, len(ctx_rows), batch_size), desc="  context", leave=False):
        batch = ctx_rows[s:s + batch_size]
        c_ids, _ = _context_batch(batch, cfg, preproc_dir, model.context_encoder.processor, device)
        raw = model.context_encoder(c_ids, attention_mask=None)
        mask = torch.ones(raw.shape[0], raw.shape[1] if raw.dim() == 3 else 1, device=device)
        emb = model._apply_projection(raw, model.context_projection, mask) if model.context_projection is not None else raw
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        ctx_embs.append(emb.float().cpu())
    te, ce = torch.cat(text_embs), torch.cat(ctx_embs)
    if normalize:
        te, ce = F.normalize(te, dim=-1), F.normalize(ce, dim=-1)
    return te, ce


@torch.no_grad()
def embed_baseline(ctx_rows, text_rows, cfg, preproc_dir, device, batch_size):
    """The frozen encoder's own joint space: CLAP text/audio projection heads (512-d), or FLAVA CLS features (768-d)."""
    enc_type = cfg["encoder_type"]
    q_enc, c_enc = get_encoders(enc_type, cache_dir=PATH_HF_CACHE_RW, token_level=False, normalize=False)
    q_enc, c_enc = q_enc.to(device).eval(), c_enc.to(device).eval()
    hf = q_enc.model
    tokenizer = q_enc.tokenizer
    text_embs = []
    for s in tqdm(range(0, len(text_rows), batch_size), desc="  baseline text", leave=False):
        batch = text_rows[s:s + batch_size]
        toks = tokenizer([r["question"] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=77)
        ids, mask = toks["input_ids"].to(device), toks["attention_mask"].to(device)
        if enc_type == "flava":
            feats = hf.get_text_features(input_ids=ids, attention_mask=mask)[:, 0, :]
        else:
            feats = hf.get_text_features(input_ids=ids, attention_mask=mask)
        text_embs.append(feats.float().cpu())
    ctx_embs = []
    for s in tqdm(range(0, len(ctx_rows), batch_size), desc="  baseline context", leave=False):
        batch = ctx_rows[s:s + batch_size]
        c_ids, is_longer = _context_batch(batch, cfg, preproc_dir, c_enc.processor, device)
        if enc_type == "flava":
            feats = hf.get_image_features(pixel_values=c_ids)[:, 0, :]
        elif is_longer is not None:
            feats = hf.get_audio_features(input_features=c_ids, is_longer=is_longer)
        else:
            feats = hf.audio_projection(c_enc(c_ids, attention_mask=None))
        ctx_embs.append(feats.float().cpu())
    return F.normalize(torch.cat(text_embs), dim=-1), F.normalize(torch.cat(ctx_embs), dim=-1)


# ─── retrieval ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _tvd_scores(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    Q = Q / (Q.sum(axis=1, keepdims=True) + 1e-12)
    scores = np.empty((len(P), len(Q)), dtype=np.float32)
    for i in range(len(P)):
        scores[i] = -0.5 * np.abs(Q - P[i]).sum(axis=-1)
    return scores


def _retrieval_metrics(scores: np.ndarray, query_labels: np.ndarray, pool_labels: np.ndarray, ks: List[int]) -> Dict[str, float]:
    acc = defaultdict(float)
    n_valid = 0
    for i in range(len(query_labels)):
        pos = set(np.where(pool_labels == query_labels[i])[0].tolist())
        if not pos:
            continue
        n_valid += 1
        ranked = np.argsort(-scores[i])
        for k in ks:
            top = ranked[:k]
            acc[f"Hit@{k}"] += float(bool(set(top.tolist()) & pos))
            dcg = sum(1.0 / np.log2(r + 2) for r, idx in enumerate(top) if idx in pos)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(pos))))
            acc[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0
            hits, ap = 0, 0.0
            for r, idx in enumerate(top):
                if idx in pos:
                    hits += 1
                    ap += hits / (r + 1)
            acc[f"mAP@{k}"] += ap / min(k, len(pos))
    return {k: v / n_valid for k, v in acc.items()} if n_valid else {}


def run_retrieval(te, ce, text_labels, ctx_labels, ks, use_tvd) -> Dict[str, Dict[str, float]]:
    te, ce = te.numpy().astype(np.float32), ce.numpy().astype(np.float32)
    if use_tvd:
        s_tc, s_ct = _tvd_scores(te, ce), _tvd_scores(ce, te)
    else:
        s_tc, s_ct = te @ ce.T, ce @ te.T
    return {"T2A": _retrieval_metrics(s_tc, text_labels, ctx_labels, ks),
            "A2T": _retrieval_metrics(s_ct, ctx_labels, text_labels, ks)}


# ─── one run ───────────────────────────────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                   key=lambda d: int(d.name.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* directory in {run_dir}")
    return ckpts[-1]


def evaluate_run(name: str, run_dir: Optional[Path], dataset: str, cfg: dict, ds, device, batch_size, ks) -> dict:
    has_src = cfg["has_source_uid"]
    ctx_rows, ctx_uids, ctx_src, text_rows, text_src = build_unique_rows(ds, has_src)
    ctx_labels = ctx_src if has_src else ctx_uids
    preproc = None
    if cfg["modality"] == "audio":
        p = preprocessed_audio_dir(dataset, cfg["encoder_type"])
        preproc = Path(p) if p else None
    if run_dir is None:
        te, ce = embed_baseline(ctx_rows, text_rows, cfg, preproc, device, batch_size)
        use_tvd = False
    else:
        ckpt = find_latest_checkpoint(run_dir) / "pytorch_model.bin"
        model, model_cfg = load_from_ckpt(str(ckpt), str(run_dir / "config.yaml"))
        model = model.to(device).eval()
        use_tvd = "softmax" in str(model_cfg.model.get("encoder_finetune_strategy", "")) or model_cfg.model.get("sim_measure") == "tvd"
        logger.info(f"  {name}: checkpoint {ckpt.parent.name}, strategy {model_cfg.model.get('encoder_finetune_strategy')}, "
                    f"{'TVD' if use_tvd else 'cosine'} retrieval")
        te, ce = embed_with_model(model, ctx_rows, text_rows, cfg, preproc, device, batch_size, normalize=not use_tvd)
        del model
        torch.cuda.empty_cache()
    row = {"dataset": dataset, "model": name, "level": "full"}
    for task, metrics in run_retrieval(te, ce, text_src, ctx_labels, ks, use_tvd).items():
        for m, v in metrics.items():
            row[f"{task}/{m}"] = v
    for k, v in compute_modality_gap(te, ce, text_src, ctx_labels, use_tvd=use_tvd).items():
        row[f"gap/{k}"] = v
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(list(TEXT_AUDIO_DATASETS) + list(TEXT_IMAGE_DATASETS)))
    p.add_argument("--runs", nargs="+", required=True, help="run directories, 'baseline', or name=path")
    p.add_argument("--n_source_uids", type=int, default=100, help="clotho_augmented: number of source clips kept")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="experiments/modality_gap/results/eval.csv")
    p.add_argument("--append", action="store_true", help="keep existing rows of the output, replacing same (dataset, model)")
    args = p.parse_args()

    cfg = dataset_cfg(args.dataset)
    ds = load_and_select_val(args.dataset, cfg, args.n_source_uids)
    rows = []
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            path = name
            name = "baseline" if spec == "baseline" else Path(spec).name
        run_dir = None if path == "baseline" else Path(path)
        logger.info(f"=== {args.dataset} / {name}")
        rows.append(evaluate_run(name, run_dir, args.dataset, cfg, ds, args.device, args.batch_size, args.ks))
        t2x, x2t, rg = rows[-1].get("T2A/NDCG@10"), rows[-1].get("A2T/NDCG@10"), rows[-1]["gap/relative_gap"]
        logger.info(f"    NDCG@10 T2X={t2x:.3f} X2T={x2t:.3f}  centroid_l2={rows[-1]['gap/centroid_l2']:.3f}  RelGap={rg:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if args.append and out.exists():
        old = pd.read_csv(out)
        keys = set(zip(df.dataset, df.model))
        old = old[~old.apply(lambda r: (r["dataset"], r["model"]) in keys, axis=1)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    logger.info(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
