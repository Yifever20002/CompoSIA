# <img src="https://github.com/Yifever20002/CompoSIA/blob/main/images/icon.png" width="38"> CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation

<div align="center">

<strong>
<a href="https://yifever20002.github.io/yifanzhan.github.io/">Yifan Zhan</a><sup>1*</sup>, 
<a href="https://scholar.google.com/citations?user=XDFkDD4AAAAJ&hl=zh-CN">Zhengqing Chen</a><sup>2*‡</sup>, 
Qingjie Wang<sup>2*</sup>, 
Zhuo He<sup>3</sup>, 
<a href="https://myniuuu.github.io/">Muyao Niu</a><sup>1</sup>, 
<a href="https://scholar.google.com/citations?user=CrK4w4UAAAAJ&hl=en">Xiaoyang Guo</a><sup>2</sup>
</strong><br>

<strong>
<a href="https://scholar.google.com/citations?user=ZIf_rtcAAAAJ&hl=en">Wei Yin</a><sup>2</sup>, 
Weiqiang Ren<sup>2</sup>, 
Qian Zhang<sup>2</sup>, 
<a href="https://scholar.google.com/citations?user=JD-5DKcAAAAJ&hl=zh-CN">Yinqiang Zheng</a><sup>1†</sup>
</strong>

<sup>1</sup>The University of Tokyo    <sup>2</sup>Horizon Robotics    <sup>3</sup>University of Glasgow

<sup>*</sup>Equal Contribution    <sup>‡</sup>Project Lead    <sup>†</sup>Corresponding Author

<p align="center">
  <img src="images/logo_utokyo.png" alt="The University of Tokyo" height="48">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/logo_horizonrobotics.jpg" alt="Horizon Robotics" height="48">
</p>

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2603.12864-b31b1b.svg)](https://arxiv.org/abs/2603.12864)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yifever20002.github.io/CompoSIA/)

</div>

<p align="center">
    <img src="https://github.com/Yifever20002/CompoSIA/blob/main/images/logo.png" width="95%">
  </a>
</p>

---

## 📌 TODO

- [x] Paper release
- [ ] Inference code (before June, 2026)
- [ ] Training code

---

## 🌍 Overview

**CompoSIA** is a compositional driving video simulator designed for **fine-grained adversarial scenario generation** through **disentangled control** of:

* **Structure** 🚗
  
  element layout placement;

* **Identity** 🎨
  
  element appearance editing from a single reference image;

* **Action** 🎮
  
  ego-motion and controllable traffic dynamics.

---

## ✨ Key Features

* **Disentangled compositional control**
* **Noise-level identity injection for pose-agnostic editing**
* **Hierarchical dual-branch action control**
* **Adversarial scenario synthesis for planner stress-testing**


---

## 🛠️ Installation

Create a Python environment and install the project dependencies:

```bash
conda create -n composia python=3.10 -y
conda activate composia

cd CompoSIA
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch 2.7.1 with CUDA 12.8 wheels by default. If your CUDA driver stack is different, install the matching PyTorch build first, then install the remaining dependencies.

The default evaluation path does not require the optional metrics packages. Install them only if you enable the corresponding metric:

```bash
# Required only when validation_kwargs.eval_metrics contains "met3r"
pip install git+https://github.com/mohammadasim98/met3r

# Required only when validation_kwargs.eval_metrics contains VBench-related evaluation
pip install vbench
```

## 📦 Model Weights

CompoSIA uses the public Wan2.1 T2V 1.3B checkpoint as the base model and the released CompoSIA transformer/VAE weights.

Expected layout:

```text
models/
├── Wan2.1-T2V-1.3B/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.1_VAE.pth
│   └── google/
│       └── umt5-xxl/
│           ├── special_tokens_map.json
│           ├── spiece.model
│           ├── tokenizer.json
│           └── tokenizer_config.json
├── composia/
│   └── composia-transformer.pt
└── vae/
    └── composia-vae.pkl
```

Download the base model from Hugging Face:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir models/Wan2.1-T2V-1.3B
```

Download the released CompoSIA weights[https://huggingface.co/SUDOKISUI/CompoSIA]:

```bash
huggingface-cli download SUDOKISUI/CompoSIA \
  transformer/composia-transformer.pt \
  vae/composia-vae.pkl \
  --local-dir models/composia-release

mkdir -p models/composia models/vae
mv models/composia-release/transformer/composia-transformer.pt models/composia/
mv models/composia-release/vae/composia-vae.pkl models/vae/
```

You can also keep the files anywhere and pass explicit paths when running evaluation:

```bash
MODEL_NAME=/path/to/Wan2.1-T2V-1.3B \
EVAL_CKPT=/path/to/composia-transformer.pt \
VAE_PATH=/path/to/composia-vae.pkl \
bash run_eval.sh
```

## 🗂️ nuScenes Data

The released metadata files are hosted in `SUDOKISUI/CompoSIA`:

```bash
mkdir -p nuScenes-metadata-full/nuscenes_mmdet3d-12Hz

huggingface-cli download SUDOKISUI/CompoSIA \
  nuscenes_interp_12Hz_infos_val_with_bid.pkl \
  --local-dir nuScenes-metadata-full/nuscenes_mmdet3d-12Hz
```

For images, download nuScenes from the official nuScenes website and unpack it so the sample images are available under:

```text
nuScenes/origin/
└── samples/
    └── CAM_FRONT/
        └── ...
```

The default config reads:

```yaml
samples_path: "./nuScenes/origin"
ann_path: "./nuScenes-metadata-full/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"
```

If your nuScenes or metadata files are stored elsewhere, update these two paths in `config/wan_unified.yaml`.

## 🚀 Evaluation

Run the default evaluation script after preparing weights and data:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_eval.sh
```

The script uses:

```bash
MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
EVAL_CKPT=${EVAL_CKPT:-models/composia/mp_rank_00_model_states.pt}
VAE_PATH=${VAE_PATH:-models/vae/dcae_td_47000.pkl}
```

For the Hugging Face release filenames, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_NAME=models/Wan2.1-T2V-1.3B \
EVAL_CKPT=models/composia/composia-transformer.pt \
VAE_PATH=models/vae/composia-vae.pkl \
bash run_eval.sh
```

Generated videos and logs are written under `logs/test/validation_res_final/`.

The evaluation modes are configured in `config/composia_unified_i2v_eval.yaml`. By default, this file enables several action, bbox, and identity-editing modes. To run a smaller smoke test, reduce `validation_kwargs.max_validation_samples` or keep only one entry under `validation_kwargs.val_modes`.

---


## 🧪 Citation
If you find our work useful, please cite it as
```bibtex
@article{zhan2026composing,
  title={Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation},
  author={Zhan, Yifan and Chen, Zhengqing and Wang, Qingjie and He, Zhuo and Niu, Muyao and Guo, Xiaoyang and Yin, Wei and Ren, Weiqiang and Zhang, Qian and Zheng, Yinqiang},
  journal={arXiv preprint arXiv:2603.12864},
  year={2026}
}
```
