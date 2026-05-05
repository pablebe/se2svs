# se2svs — Teaching Speech Enhancement Models to Sing

**Domain Adaptation from Speech Enhancement to Singing Voice Separation**

> Paul A. Bereuter, Mark D. Plumbley, Alois Sontacchi
<!-- > *To be presented  2026* -->

🎧 **[Demo page with audio examples](https://pablebe.github.io/se2svs-webpage/)**

---

## Overview

State-of-the-art speech enhancement (SE) models benefit from large-scale labeled datasets, whereas singing voice separation (SVS) models suffer from limited available training data. This repository frames singing voice separation as **domain adaptation** from speech enhancement to SVS and investigates two fine-tuning strategies on two model families:

| Strategy | Description |
|---|---|
| **Full fine-tuning** | All model weights updated on SVS data |
| **LoRA fine-tuning** | Only low-rank adapter weights trained (~6–12% extra params) |

Both strategies are applied to:
- **SGM** — a generative score-based diffusion model (SGMSE+, `ncsnpp_48k` backbone)
- **BSRNN** — a discriminative band-split RNN model

Models fine-tuned from a pretrained SE checkpoint outperform the same architectures trained from scratch by **0.2–1.8 dB SDR**. LoRA fine-tuning preserves the original speech enhancement capability while achieving competitive SVS performance, whereas full fine-tuning leads to catastrophic forgetting on SE tasks.

### Pretrained checkpoints

Fine-tuned model checkpoints (all except base models) are available on Hugging Face: **[pablebe/se2svs-model-checkpoints](https://hf.co/collections/pablebe/se2svs-model-checkpoints)**

Run `python download_checkpoints.py` to download them automatically into `checkpoints/se2svs/`.

| File | Source | Description |
|---|---|---|
| `sgmse_pretrained/ears_wham.ckpt` | [Google Drive](https://drive.google.com/drive/folders/1Tn6pVwjxUAy1DJ8167JCg3enuSi0hiw5) | Base SGM model pretrained on EARS-WHAM speech enhancement |
| `bsrnn_pretrained/bsrnn.ckpt` | [URGENT 2026 Challenge — pretrained models](https://github.com/urgent-challenge/urgent2026_challenge_track1) | Base BSRNN model pretrained on speech enhancement |

### Test sets

| Folder | Dataset | Source |
|---|---|---|
| `test_sets/ears_wham_v2_test_5s/` | EARS-WHAM (speech enhancement) | [EARS Benchmark](https://github.com/sp-uhh/ears_benchmark) |
| `test_sets/gensvs_eval_audio/` | GenSVS (singing voice separation) | [Zenodo](https://zenodo.org/records/15911723) |
| `test_sets/MSRBench_Vocals/` | MSRBench (singing voice restoration) | [Hugging Face](https://huggingface.co/datasets/yongyizang/MSRBench) |

---

## Getting started

## Environment setup

### Conda environment (CUDA + Python)

Create and activate the Conda environment:

```bash
conda env create -f ./env_info/se2svs_env.yml 
conda activate se2svs_env
```

This installs:
- Python runtime
- CUDA toolkit and nvcc (required for GPU support and compilation)

---

### Install uv (Python package manager)

Install uv inside the environment:

```bash
pip install uv
```

---

### Install Python dependencies

Use the lock file for fully reproducible installation:

```bash
uv pip install --override env_info/overrides.txt -r env_info/requirements.lock
```

The `overrides.txt` file resolves version conflicts between `espnet`/`gensvs` metadata and the actual packages used during development (`sentencepiece==0.2.1`, `setuptools==80.10.2`, `numpy==2.3.5`).

This ensures all Python dependencies are installed at the exact versions used during development.

---

### (Optional) Update dependencies

If requirements.in changes, regenerate the lock file:

```bash
uv pip compile ./env_info/requirements.in -o ./env_info/requirements.lock
uv pip sync ./env_info/requirements.lock
```

---

## Training from scratch

### SGM (score-based diffusion model)

```bash
python train_sgmsvs.py \
    --backbone ncsnpp_48k \
    --sde ouve \
    --base_dir /path/to/MSS_datasets \
    --format MSS \
    --dataset_str musdb \
    --target_str vocals
```

### BSRNN

```bash
python train_bsrnnsvs.py --config configs/bsrnn_from_scratch.toml
```

---

## Fine-tuning

All fine-tuning scripts accept a TOML config file via `--config`. Example configs are provided in `configs/`.

### SGM — LoRA fine-tuning

```bash
python finetune_sgmse.py --config configs/lora_finetune.toml
```

Key config options (`configs/lora_finetune.toml`):

| Option | Description |
|---|---|
| `lora_pretrained_checkpoint` | Path to base SE checkpoint to initialize from |
| `lora_r` | LoRA rank (e.g. 8, 16, 32, 128) |
| `lora_alpha` | LoRA scaling factor (typically `2 × lora_r`) |
| `lora_target_modules` | Which layer types to apply LoRA to |

### SGM — full / naive fine-tuning

```bash
python finetune_sgmse.py --config configs/naive_finetune.toml
```

### BSRNN — fine-tuning (LoRA or full)

```bash
python finetune_bsrnnse.py --config configs/bsrnn_lora_finetune.toml
```

Available BSRNN configs:

| Config | Description |
|---|---|
| `bsrnn_naive_finetune.toml` | Full fine-tuning of all BSRNN weights |
| `bsrnn_lora_finetune.toml` | LoRA fine-tuning (rank 16 default) |
| `bsrnn_lora_finetune_r18/r24/r32/r128.toml` | LoRA fine-tuning at specific ranks |

---

## Inference

### SGM — base pretrained checkpoint

```bash
python inference_sgmsvs.py \
    --test_dir test_sets/gensvs_eval_audio/mixture \
    --enhanced_dir test_sets/gensvs_eval_audio/sgm_base \
    --ckpt checkpoints/sgmse_pretrained/ears_wham.ckpt \
    --N 45 --snr 0.7
```

### SGM — fine-tuned checkpoint (LoRA or full)

```bash
python infer_finetuned_models.py \
    --ckpt logs/<run_id>/epoch=XXX-sdr=X.XX.ckpt \
    --test-dir test_sets/gensvs_eval_audio/mixture \
    --out-dir test_sets/gensvs_eval_audio/sgm_lora_r16
```

Pass `--no-lora` to disable LoRA adapters and evaluate the base model through the fine-tuned checkpoint.

### BSRNN — fine-tuned checkpoint

```bash
python infer_bsrnn_finetuned_models.py \
    --ckpt logs/<run_id>/epoch=XXX-sdr=X.XX.ckpt \
    --test-dir test_sets/gensvs_eval_audio/mixture \
    --out-dir test_sets/gensvs_eval_audio/bsrnn_lora_r16
```

---

## Evaluation

### Separation metrics (SDR, SI-SDR, MR-STFT loss, MERT-MSE)

```bash
python evaluate_separation.py \
    --separated-dir test_sets/gensvs_eval_audio/sgm_lora_r16 \
    --target-dir test_sets/gensvs_eval_audio/target \
    --output-csv test_sets/gensvs_eval_audio/sgm_lora_r16/results.csv
```

### Speech enhancement metrics (SI-SDR, PESQ, STOI, DNSMOS, DistillMOS)

```bash
python evaluate_speech.py \
    --separated-dir test_sets/ears_wham_v2_test_5s/sgm_lora_r16 \
    --target-dir test_sets/ears_wham_v2_test_5s/target \
    --output-csv test_sets/ears_wham_v2_test_5s/sgm_lora_r16/results.csv
```

### Aggregate and plot results

Aggregate results across all models and datasets and generate violin plots:

```bash
python aggregate_and_plot_results.py --base-dir .
```

Compute summary statistics across evaluation iterations:

```bash
python analyze_results_stats.py --base-dir test_sets/gensvs_eval_audio/sgm_lora_r16
```

Aggregated result summaries are stored in `aggregated_results/`.

---

## Repository structure

```
configs/            TOML configs for all training/fine-tuning runs
models/             Model code (SGM, BSRNN, data module, SDE, backbones)
  sgmse/            Third-party code from https://github.com/sp-uhh/sgmse (MIT license)
sgmse_checkpoints/  Pretrained SE checkpoints
bsrnn_checkpoint/   Pretrained BSRNN checkpoint
test_sets/          Evaluation datasets and per-model output folders
aggregated_results/ Aggregated metric CSVs and plots
env_info/           Conda environment and pip requirements files
```

---

## Citation

If you use this code, please cite:

```bibtex
@INPROCEEDINGS{bereuter2026se2svs,
  author={Bereuter, Paul A. and Plumbley, Mark D. and Sontacchi, Alois},
  booktitle={},
  title={Teaching Speech Enhancement Models to Sing: Domain Adaptation from Speech Enhancement to Singing Voice Separation},
  year={2026},
  doi={}
}
```

If you use the **SGM base model** (`ears_wham.ckpt`), please also cite:

```bibtex
@inproceedings{richter2024ears,
  title={{EARS}: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation},
  author={Richter, Julius and Wu, Yi-Chiao and Krenn, Steven and Welker, Simon and Lay, Bunlong and Watanabe, Shinjii and Richard, Alexander and Gerkmann, Timo},
  booktitle={ISCA Interspeech},
  pages={4873--4877},
  year={2024}
}
```

If you use the **BSRNN base model** (`bsrnn.ckpt`), please also cite:

```bibtex
@article{liLessMoreData2025,
  title={Less is {More}: {Data} {Curation} {Matters} in {Scaling} {Speech} {Enhancement}},
  url={http://arxiv.org/abs/2506.23859},
  doi={10.48550/arXiv.2506.23859},
  publisher={arXiv},
  author={Li, Chenda and Zhang, Wangyou and Wang, Wei and Scheibler, Robin and Saijo, Kohei and Cornell, Samuele and Fu, Yihui and Sach, Marvin and Ni, Zhaoheng and Kumar, Anurag and Fingscheidt, Tim and Watanabe, Shinji and Qian, Yanmin},
  year={2025},
  note={arXiv:2506.23859 [eess]}
}
```
