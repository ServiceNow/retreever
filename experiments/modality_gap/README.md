# Closing the Modality Gap via Simplex-Constrained Representations — experiments

Code and results of the Interspeech 2026 paper
([PDF](../../docs/modality_gap/Closing_the_Modality_Gap_Interspeech2026.pdf)) for the adapter rows of Tables 1–2:
Linear, MLP, Lin.+Sfx, MLP+Sfx and XAttn, on the paper's five datasets.


## 1. Install

```bash
pip install -r requirements.txt && pip install -e .
pip install -r experiments/modality_gap/requirements-extra.txt
cp local_paths.py.example local_paths.py   # then add the dataset directories of step 2
```

## 2. Data

Every dataset is a directory holding the audio or image files plus a Hugging Face `DatasetDict` (`train` / `val`
/ `test`; columns `question` = caption, `context` = file path relative to the directory, `context_uid` = item id).
Scripts are in `scripts/data_prep/modality_gap/`; run them in this order and register each directory in
`local_paths.py` under the dataset's name.

| dataset | commands |
|---|---|
| `clotho` | `clotho_download.py --output_dir D` · `prepare_clotho.py --data_dir D` |
| `clotho_augmented` (the paper's Clotho rows: 9 waveform augmentations per clip) | after `clotho`: `clotho_augment.py --data_dir D` · `prepare_clotho_augmented.py --data_dir D` (register `D` under both names) |
| `sounddescs` | `sounddescs_download.py --output_dir D` · `sounddescs_fix_metadata.py --data_dir D` · `prepare_sounddescs.py --data_dir D` |
| `audiocaps` | `audiocaps_download.py --output_dir D` · `prepare_audiocaps.py --data_dir D` |
| `coco` | `prepare_coco.py --coco_data_dir <COCO 2014 images and captions> --output_dir D` |
| `flickr30k` | `prepare_flickr30k.py --output_dir D` (downloads from the Hugging Face hub) |

Then cache the CLAP features of the audio datasets once (training and evaluation pick them up automatically from
`<D>_clap-fused-ta` next to `D`):

```bash
python scripts/data_prep/modality_gap/preprocess_clap_features.py --data_dir D_clotho --dataset_subdir dataset_augmented --name clotho_augmented
python scripts/data_prep/modality_gap/preprocess_clap_features.py --data_dir D_sounddescs
python scripts/data_prep/modality_gap/preprocess_clap_features.py --data_dir D_audiocaps --source_sample_rate 16000
```

## 3. Train

```bash
experiments/modality_gap/run_rows.sh DATASET ROW [VARIANT]
experiments/modality_gap/run_rows.sh audiocaps linear_softmax          # Lin.+Sfx as published
experiments/modality_gap/run_rows.sh audiocaps linear_softmax nol2_ln  # arXiv revision: raw input + LayerNorm on the logits
experiments/modality_gap/run_rows.sh flickr30k xattn                   # XAttn, cosine
```

`ROW` is `linear`, `mlp`, `linear_softmax`, `mlp_softmax`, `xattn` or `xattn_softmax`; `DATASET` is
`clotho_augmented`, `sounddescs`, `audiocaps` (CLAP) or `coco`, `flickr30k` (FLAVA). 

Recipe: 200K steps, batch 16 per GPU on 4 GPUs (set `LAUNCH` to your multi-GPU
launcher; by default the single-process entry point runs), lr 4e-4 with linear decay and 5% warm-up, weight decay
0.05, fp16, learnable temperature, encoders frozen, one seed.

Rows and their configuration: Linear / MLP use `shared_linear_zero_init_norm` / `shared_mlp_zero_init_norm` with
`sim_measure=cos_sim`; the softmax rows use the `_softmax` strategies with `sim_measure=tvd`; XAttn uses
`shared_qformer_pooling[_softmax]` over token-level features with 99 learned queries.

`VARIANT` selects the normalization variants:
`l2_ln`, `nol2_none`, `nol2_ln`, `nol2_rms`, `nol2_scale`, `nol2_lnna` (adapter input L2-normalised or raw, and a
norm on the logits before the softmax: `model.encoder_normalize`, `model.logit_norm`); `bn_l2_none`, `bn_l2_ln`,
`bn_nol2_none`, `bn_nol2_ln` (LayerNorms inside the adapter's residual branch, `model.branch_norm`); `ft2` (last two
blocks of each tower fine-tuned at their own learning rate, `model.unfreeze_last_n_layers`,
`train.encoder_learning_rate`); `noln` (cross-attention pooling without its LayerNorm).

## 4. Evaluate

```bash
python -m experiments.modality_gap.evaluate --dataset audiocaps \
    --runs baseline runs/modality_gap/audiocaps_linear_softmax_paper_clap-fused-ta \
    --output experiments/modality_gap/results/eval.csv --append
```

Embeds the de-duplicated validation split with a run's latest checkpoint (`baseline` = the frozen encoder's own joint
space), scores retrieval in both directions (cosine, or TVD for softmax adapters) and the gap metrics of
`modality_gap_metrics.py`, and writes rows with the columns of `results/modality_gap_results.csv`. .


