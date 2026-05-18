#!/bin/bash

source $(dirname $(realpath $0))/_base.sh
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export TORCH_HOME=${TORCH_HOME:-/tmp/torch_cache}

export OUTPUT_DIR=$DEFAULT_OUTPUT_DIR/test/
export MODEL_NAME=${MODEL_NAME:-"models/Wan2.1-T2V-1.3B"}
export EVAL_CKPT=${EVAL_CKPT:-"models/composia/composia-transformer.pt"}
export VAE_PATH=${VAE_PATH:-"models/vae/composia-vae.pkl"}

EXTRA_ARGS=(
    --config_path="config/composia_unified_i2v_eval.yaml"
    --pretrained_model_name_or_path="${MODEL_NAME}"
    --output_dir="${OUTPUT_DIR}"
    --vae_path="${VAE_PATH}"
    --transformer_path="${EVAL_CKPT}"
)

set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch -- \
    composia/eval.py \
    "${DEFAULT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    $@
