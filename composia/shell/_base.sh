# shared base settings for eval

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=".:$PYTHONPATH"

# Output root (AIDI vs local)
if [ ! -d "/job_data" ]; then
    export DEFAULT_OUTPUT_DIR="./logs/"
else
    export DEFAULT_OUTPUT_DIR="/job_data/logs/"
fi

# Install horizon data SDK
pip3 install horizon_driving_dataset -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple
pip3 install yapf -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple

RUNTIME_ARGS=(
    --train_mode="unified"
    --mixed_precision="bf16"
    --seed=42
    --crossview_attn_type="full"
)

export DEFAULT_ARGS=(
    ${RUNTIME_ARGS[@]}
    --low_vram
)
