# se2svs — Teaching Speech Enhancement Models to Sing

**Domain Adaptation from Speech Enhancement to Singing Voice Separation**

> Paul A. Bereuter, Mark D. Plumbley, Alois Sontacchi
<!-- > *To be presented  2026* -->

🎧 **[Demo page with audio examples](https://pablebe.github.io/se2svs-webpage/)**

📦 **[Test-set audio + result CSVs (Zenodo)](<dummy-url>)**

📄 **[arXiv preprint](<arxiv-url>)**

---

## Overview

State-of-the-art speech enhancement (SE) models benefit from large-scale labeled datasets, whereas singing voice separation (SVS) models suffer from limited available training data. This repository frames singing voice separation as **domain adaptation** from SE to SVS and investigates two fine-tuning strategies on two model families:

| Strategy | Description |
|---|---|
| **Full fine-tuning** | All model weights updated on SVS data |
| **LoRA fine-tuning** | Only low-rank adapter weights trained |

Both strategies are applied to:
- **BSRNN** — a discriminative band-split RNN model
- **SGM** — a generative score-based diffusion model (SGMSE+, `ncsnpp_48k` backbone)

Models fine-tuned from a pretrained SE checkpoint outperform the same architectures trained from scratch by **0.2–1.8 dB SDR**. LoRA fine-tuning preserves the original speech enhancement capability while achieving competitive SVS performance, whereas full fine-tuning leads to catastrophic forgetting on SE tasks.

## Model checkpoints

### Fine-tuned and from scratch checkpoints
Fine-tuned model checkpoints (all except base models) are available on Hugging Face: **[pablebe/se2svs-model-checkpoints](https://hf.co/collections/pablebe/se2svs-model-checkpoints)**

Run `python download_checkpoints.py` to download them automatically into `checkpoints/se2svs/`.

### Base speech enhancement checkpoints
Base model checkpoints are not included in that script and must be downloaded separately using the links in the table below.

| File | Source | Description |
|---|---|---|
| `./bsrnn_pretrained/bsrnn.ckpt` | [Hugging Face — bsrnn.ckpt](https://huggingface.co/lichenda/icassp_2026_urgent_baseline/blob/main/bsrnn.ckpt) | Base BSRNN model pretrained on speech enhancement |
| `./sgmse_pretrained/ears_wham.ckpt` | [Google Drive](https://drive.google.com/drive/folders/1Tn6pVwjxUAy1DJ8167JCg3enuSi0hiw5) | Base SGM model pretrained on EARS-WHAM speech enhancement |

## Test sets
To reproduce the evaluation on the three test-sets you can download the mixture audio and target vocals from:
| Folder | Dataset | Source | Details
|---|---|---|
| `./test_sets/ears_wham/` | EARS-WHAM (speech enhancement) | [EARS Benchmark](https://github.com/sp-uhh/ears_benchmark) | only first 5s were used
| `./test_sets/gensvs_eval_audio/` | GenSVS (singing voice separation) | [Zenodo](https://zenodo.org/records/15911723) | 5s subset of MUSDB18-HQ test set 
| `./test_sets/MSRBench_Vocals/` | MSRBench (singing voice restoration) | [Hugging Face](https://huggingface.co/datasets/yongyizang/MSRBench) | 10s audio samples from MSR challenge test set

The results plots and tables of our paper are located in `./aggregated_results` 

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
uv pip install --override ./env_info/overrides.txt -r ./env_info/requirements.lock
```

The `overrides.txt` file resolves version conflicts between `espnet`/`gensvs` metadata and the actual packages used during development (`sentencepiece==0.2.1`, `setuptools==80.10.2`, `numpy==2.3.5`).

This ensures all Python dependencies are installed at the exact versions used during development.


## Training from scratch

All training commands below use `--config` and load parameters from TOML files in `./configs/`.
All arguments defined in these TOML files are also parseable as CLI flags.

### BSRNN

```bash
python train_bsrnn_from_scratch.py --config ./configs/bsrnn_scratch.toml
```

### SGM (score-based diffusion model)

```bash
python finetune_sgm.py --config ./configs/sgm_scratch.toml
```

---
## Fine-tuning of speech enhancement models for singing voice separation

All fine-tuning scripts accept a TOML config file via `--config`. Example configs are provided in `./configs/`.
All arguments defined in these TOML files are also parseable as CLI flags.

### BSRNN — fine-tuning (LoRA or full)

```bash
python finetune_bsrnn.py --config ./configs/bsrnn_lora_r16.toml
```

Available BSRNN configs:

| Config | Description |
|---|---|
| `./configs/bsrnn_full.toml` | Full fine-tuning of all BSRNN weights |
| `./configs/bsrnn_lora_r16.toml` | LoRA fine-tuning (rank 16) |
| `./configs/bsrnn_lora_r32.toml` | LoRA fine-tuning (rank 32) |
| `./configs/bsrnn_lora_r128.toml` | LoRA fine-tuning (rank 128) |
| `./configs/bsrnn_scratch.toml` | BSRNN training from scratch (only usable with train_bsrnn_from_scratch.py) |

### SGM — fine-tuning (LoRA or full)

```bash
python finetune_sgm.py --config ./configs/sgm_lora_r16.toml
```

Available SGM configs:
| Config | Description |
|---|---|
| `./configs/sgm_full.toml` | Full fine-tuning of all SGM weights |
| `./configs/sgm_lora_r16.toml` | LoRA fine-tuning (rank 16) |
| `./configs/sgm_scratch.toml` | SGM training from scratch (can be used finetuning script finetune_sgm.py) |

### LoRA finetuning key parameters

| Option | Description |
|---|---|
| `lora_pretrained_checkpoint` | Path to base SE checkpoint to initialize from |
| `lora_r` | LoRA rank (e.g. 8, 16, 32, 128) |
| `lora_alpha` | LoRA scaling factor (typically `2 × lora_r`) |
| `lora_target_modules` | Which layer types to apply LoRA to |

---

## Inference

### BSRNN — fine-tuned checkpoint

```bash
python infer_bsrnn_finetuned_models.py \
  --ckpt ./logs/<run_id>/epoch=XXX-sdr=X.XX.ckpt \
  --test-dir ./test_sets/gensvs_eval_audio/mixture \
  --out-dir ./test_sets/gensvs_eval_audio/bsrnn_lora_r16
```

### SGM — inference (base, from-scratch, full finetuned, or LoRA)

```bash
python inference_sgm.py \
    --ckpt ./checkpoints/sgmse_pretrained/ears_wham.ckpt \
    --test-dir ./test_sets/gensvs_eval_audio/mixture \
    --out-dir ./test_sets/gensvs_eval_audio/sgm_base
```

Default settings in `inference_sgm.py`: `--sampler pc`, `--corrector ald`, `--corrector-steps 2`, `--snr 0.5`, and `--N 45`.

### SGM — fine-tuned checkpoint (LoRA or full)

```bash
python inference_sgm.py \
    --ckpt ./logs/<run_id>/epoch=XXX-sdr=X.XX.ckpt \
    --test-dir ./test_sets/gensvs_eval_audio/mixture \
    --out-dir ./test_sets/gensvs_eval_audio/sgm_lora_r16
```

Pass `--no-lora` to disable LoRA adapters and evaluate the base model through the fine-tuned checkpoint.

---

## Evaluation

### Separation metrics (SDR, MR-STFT loss, MERT-MSE)

```bash
python evaluate_separation.py \
    --separated-dir ./test_sets/gensvs_eval_audio/sgm_lora_r16 \
    --target-dir ./test_sets/gensvs_eval_audio/target \
    --output-csv ./test_sets/gensvs_eval_audio/sgm_lora_r16/results.csv
```
Compute summary statistics across evaluation iterations:

```bash
python evaluate_separation.py --separated-dir ./test_sets/gensvs_eval_audio/sgm_lora_r16 --target-dir ./test_sets/gensvs_eval_audio/target
```

For SGM outputs with multiple enhanced signal realizations stored in subdirectories, evaluate all runs with:

```bash
python evaluate_separation.py \
  --multi-run-dir ./test_sets/gensvs_eval_audio/sgm_lora_r16 \
  --target-dir test_sets/gensvs_eval_audio/target
```

This scans each subdirectory under `--multi-run-dir` and writes a `results.csv` file into each run folder.

### Speech enhancement metrics (SI-SDR, PESQ, STOI, DNSMOS, DistillMOS)

```bash
python evaluate_speech.py \
  --separated-dir ./test_sets/ears_wham/sgm_lora_r16 \
  --target-dir ./test_sets/ears_wham/clean \
  --output-csv ./test_sets/ears_wham/sgm_lora_r16/results.csv
```

### Aggregate and plot results

Reproduce results tables and plots of the paper with:

```bash
python aggregate_and_plot_results.py --base-dir ./
```

To reproduce the paper results, the test-set audio and CSV result files in `./test_sets/` can be downloaded from Zenodo: <dummy-url>.

Aggregated result summaries are stored in `./aggregated_results/`.

---

## Repository structure

```
configs/            TOML configs for all training/fine-tuning runs
models/             Model code (BSRNN, SGM, data module, SDE, backbones)
  sgmse/            Third-party code from https://github.com/sp-uhh/sgmse (MIT license)
checkpoints/        Model checkpoints [download required; for instructions see section "Model checkpoints"]
  bsrnn_pretrained/ Pretrained BSRNN checkpoint 
  sgmse_pretrained/ Pretrained SE checkpoint 
  se2svs/           Fine-tuned and from-scratch model checkpoints 
test_sets/          Evaluation datasets and per-model output folders [download required; for instructions see section "Test sets"]
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
