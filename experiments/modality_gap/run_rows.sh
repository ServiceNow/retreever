#!/bin/bash
# Train one configuration of "Closing the Modality Gap via Simplex-Constrained Representations" with the paper's recipe.
#
#   experiments/modality_gap/run_rows.sh DATASET ROW [VARIANT]
#
#   DATASET  clotho_augmented | sounddescs | audiocaps  (encoder clap-fused-ta)     coco | flickr30k  (encoder flava)
#   ROW      linear | mlp | linear_softmax | mlp_softmax | xattn | xattn_softmax
#   VARIANT  (softmax rows only; default "paper")
#     paper        L2-normalised adapter input, no norm on the logits                 -> Tables 1-2 as published
#     l2_ln        L2 input + LayerNorm on the logits
#     nol2_none    raw adapter input, no norm                                        -> Table 2 of the arXiv revision
#     nol2_ln      raw input + LayerNorm on the logits                               -> Table 1 of the arXiv revision
#     nol2_rms     raw input + RMSNorm         nol2_scale   raw input + ScaleNorm     nol2_lnna   raw input + LayerNorm without affine
#     bn_l2_none   LayerNorms inside the residual branch, L2 input     bn_l2_ln     ... + logit LayerNorm
#     bn_nol2_none ... raw input                                       bn_nol2_ln   ... raw input + logit LayerNorm
#     ft2          last two blocks of each tower fine-tuned at ENC_LR (also valid for the cosine rows linear | mlp)
#     noln         (xattn rows only) cross-attention pooling without its post-attention LayerNorm
#
# Recipe: 200K steps, batch 16 per GPU, 4 GPUs (the paper), lr 4e-4, linear schedule, 5% warm-up, weight decay 0.05,
# fp16, learnable temperature (max 100), encoders frozen unless VARIANT=ft2. Set LAUNCH to your multi-GPU launcher, e.g.
#   LAUNCH="deepspeed --num_gpus 4 hydra_entrypoint.py --deepspeed=scripts/config/deepspeed.json"
# Environment: STEPS (200000), ENC_LR (4e-5), LOG_EVERY (50000, also the checkpoint interval), SAVE_LIMIT (keep all),
# PREC (fp16 | bf16 | fp32), OUT (runs/modality_gap), DRYRUN=1 prints the command only.
set -euo pipefail
dataset=${1:?dataset}; row=${2:?row}; variant=${3:-paper}
LAUNCH=${LAUNCH:-"python -m scripts.train"}
STEPS=${STEPS:-200000}; ENC_LR=${ENC_LR:-4e-5}; LOG_EVERY=${LOG_EVERY:-50000}; SAVE_LIMIT=${SAVE_LIMIT:-null}; PREC=${PREC:-fp16}; OUT=${OUT:-runs/modality_gap}
case $dataset in
  coco|flickr30k) encoder=flava;;
  clotho_augmented|clotho|sounddescs|audiocaps) encoder=clap-fused-ta;;
  *) echo "unknown dataset $dataset"; exit 1;;
esac
case $PREC in fp16) prec="train.fp16=true train.bf16=false";; bf16) prec="train.fp16=false train.bf16=true";; fp32) prec="train.fp16=false train.bf16=false";; *) echo "PREC must be fp16|bf16|fp32"; exit 1;; esac
common="dataset=${dataset} model.encoder_type=${encoder} model.encoder_dim=768 model.freeze_encoder=false
        model.loss_type=contrastive model.tree_type=identity_tree train.hierarchical=false train.optimize_prior_levels_too=true
        train.train_batch_size=16 train.steps=${STEPS} train.max_tmp=100 ${prec}
        logging.log_every=${LOG_EVERY} train.save_total_limit=${SAVE_LIMIT} logging.factor_val_irrelevant_ctxs=0"
norm_in=true; logit_norm=none; branch_norm=false; unfreeze=0; xattn_ln=""
case $variant in
  paper) ;;
  l2_ln)        logit_norm=layernorm;;
  nol2_none)    norm_in=false;;
  nol2_ln)      norm_in=false; logit_norm=layernorm;;
  nol2_rms)     norm_in=false; logit_norm=rmsnorm;;
  nol2_scale)   norm_in=false; logit_norm=scalenorm;;
  nol2_lnna)    norm_in=false; logit_norm=layernorm_noaffine;;
  bn_l2_none)   branch_norm=true;;
  bn_l2_ln)     branch_norm=true; logit_norm=layernorm;;
  bn_nol2_none) branch_norm=true; norm_in=false;;
  bn_nol2_ln)   branch_norm=true; norm_in=false; logit_norm=layernorm;;
  ft2)          unfreeze=2;;
  noln)         xattn_ln="_noln";;
  *) echo "unknown variant $variant"; exit 1;;
esac
case $row in
  linear)         strategy=shared_linear_zero_init_norm;    sim=cos_sim; token=false;;
  mlp)            strategy=shared_mlp_zero_init_norm;       sim=cos_sim; token=false;;
  linear_softmax) strategy=shared_linear_zero_init_softmax; sim=tvd;     token=false;;
  mlp_softmax)    strategy=shared_mlp_zero_init_softmax;    sim=tvd;     token=false;;
  xattn)          strategy=shared_qformer_pooling${xattn_ln};         sim=cos_sim; token=true;;
  xattn_softmax)  strategy=shared_qformer_pooling_softmax${xattn_ln}; sim=tvd;     token=true;;
  *) echo "unknown row $row"; exit 1;;
esac
extra="model.encoder_finetune_strategy=${strategy} model.sim_measure=${sim} model.encoder_token_level=${token}
       model.encoder_normalize=${norm_in} model.logit_norm=${logit_norm} model.branch_norm=${branch_norm}
       model.unfreeze_last_n_layers=${unfreeze} model.num_cross_attn_queries=99"
[ "$unfreeze" != "0" ] && extra="$extra train.encoder_learning_rate=${ENC_LR}"
savedir="${OUT}/${dataset}_${row}_${variant}_${encoder}"
cmd="$LAUNCH $common $extra savedir=$savedir logging.wandb_run=$(basename $savedir)"
echo "$cmd" | tr -s ' \n' ' '; echo
[ "${DRYRUN:-0}" = "1" ] && exit 0
exec $cmd
