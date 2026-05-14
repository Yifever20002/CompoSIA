# Model Files

This repository does not include pretrained weights or generated checkpoints. Before running evaluation, place the required files under `models/` with the following layout:

```text
models/
├── Wan2.1-T2V-1.3B/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/
│       └── umt5-xxl/
│           ├── special_tokens_map.json
│           ├── spiece.model
│           ├── tokenizer.json
│           └── tokenizer_config.json
├── composia/
│   └── mp_rank_00_model_states.pt
└── vae/
    └── dcae_td_47000.pkl
```

The default evaluation script reads these paths:

```bash
MODEL_NAME=models/Wan2.1-T2V-1.3B
EVAL_CKPT=models/composia/mp_rank_00_model_states.pt
VAE_PATH=models/vae/dcae_td_47000.pkl
```

You can override any of them when launching `run_eval.sh`, for example:

```bash
MODEL_NAME=/path/to/Wan2.1-T2V-1.3B \
EVAL_CKPT=/path/to/mp_rank_00_model_states.pt \
VAE_PATH=/path/to/dcae_td_47000.pkl \
bash run_eval.sh
```
