#!/usr/bin/env python3
"""
Inference script for fine-tuned ScoreModel (LoRA or full fine-tuning).

Runs separation/enhancement on all .wav files in a folder and writes results to OUT_DIR.
Works for both singing voice separation and speech enhancement tasks.
The LoRA adapter can be toggled on/off with --no-lora (off = base model only).

Example
-------
# Singing voice separation with LoRA:
python infer_finetuned_models.py \
    --ckpt logs/270219_1630_SGMSE_lora_finetune_r16/epoch=533-sdr=7.63.ckpt \
    --test-dir se2svs_results_and_audio/gensvs_eval_audio/mixture \
    --out-dir se2svs_results_and_audio/gensvs_eval_audio/output

# Speech enhancement with naive fine-tuning:
python infer_finetuned_models.py \
    --ckpt checkpoints/se2svs/sgm_full/epoch=900-sdr=8.35.ckpt \
    --test-dir se2svs_results_and_audio/ears_wham_v2_test/noisy \
    --out-dir se2svs_results_and_audio/ears_wham_v2_test/enhanced

# Without LoRA adapter (base model only):
python infer_finetuned_models.py \
    --ckpt logs/270219_1630_SGMSE_lora_finetune_r16/epoch=533-sdr=7.63.ckpt \
    --test-dir se2svs_results_and_audio/gensvs_eval_audio/mixture \
    --out-dir se2svs_results_and_audio/gensvs_eval_audio/output \
    --no-lora
"""

#TODO: Reincorporate LoRA toggling and speech enhancement evaluation.

import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Limit to single GPU for inference
# ── Backward-compatible module aliases ────────────────────────────────────────
import models.data_module
import models.MSS_model
import models.sgmse
import models.sgmse.sdes
import models.sgmse.backbones
import models.sgmse.util

sys.modules['sgmse']             = models.sgmse
sys.modules['sgmse.data_module'] = models.data_module
sys.modules['sgmse.model']       = models.MSS_model
sys.modules['sgmse.sdes']        = models.sgmse.sdes
sys.modules['sgmse.backbones']   = models.sgmse.backbones
sys.modules['sgmse.util']        = models.sgmse.util

import glob
import argparse
import torch
import numpy as np
import soundfile as sf
from librosa import resample

from tqdm import tqdm
from os.path import join, basename, splitext

from models.MSS_model import ScoreModel
from models.sgmse.util.other import pad_spec

# ── Defaults ───────────────────────────────────────────────────────────────── 
DEFAULT_CKPT     = "checkpoints/se2svs/sgm_full/epoch=900-sdr=8.35.ckpt"
DEFAULT_TEST_DIR = "se2svs_results_and_audio/MSRBench_Vocals/mixture"
DEFAULT_SEED     = 42

# ── Args ───────────────────────────────────────────────────────────────────── 
parser = argparse.ArgumentParser(
    description="Infer with fine-tuned ScoreModel (LoRA or full) for separation/enhancement"
)
parser.add_argument("--ckpt",     type=str, default=DEFAULT_CKPT,
                    help="Path to model checkpoint (LoRA or full fine-tuned)")
parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR,
                    help="Folder of input .wav files (mixtures or noisy speech)")
parser.add_argument("--out-dir",  type=str, required=True,
                    help="Output directory for separated/enhanced files")
parser.add_argument("--no-lora",  action="store_true", default=False,
                    help="Disable LoRA adapter (run base model only)")
parser.add_argument("--device",   type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--N",        type=int, default=45,
                    help="Number of reverse diffusion steps (default: 45)")
parser.add_argument("--sampler",  type=str, default="pc",
                    choices=("pc", "ode"), help="Sampler type")
parser.add_argument("--corrector", type=str, default="ald",
                    choices=("ald", "langevin", "none"),
                    help="Corrector for PC sampler")
parser.add_argument("--corrector-steps", type=int, default=2)
parser.add_argument("--snr", type=float, default=0.5,
                    help="SNR for Langevin corrector")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for deterministic inference (default: None for stochastic like validation)")
parser.add_argument("--n-iterations", type=int, default=1,
                    help="Number of iterationsthat will be run and stored in separate subdirectories iter_1/, iter_2/, ...")
args = parser.parse_args()

# ── Load model ─────────────────────────────────────────────────────────────── 
print(f"Loading checkpoint: {args.ckpt}")
model = ScoreModel.load_from_checkpoint(
    args.ckpt,
    map_location=args.device,
    lora_pretrained_checkpoint=None,
    strict=True,
    weights_only=False,
)
model.to(args.device)

# Keep checkpoint-loaded weights as-is during inference. ScoreModel.eval()
# swaps in EMA weights by default; if EMA restoration failed, that can reset
# LoRA adapter params back to their initialization while leaving base weights.
model.eval(no_ema=True)

# ── Toggle LoRA adapter ────────────────────────────────────────────────────── 
# Check if model has PEFT structure: model.dnn (NCSNpp_48k_LoRA) -> model.dnn.model (PeftModel)
is_lora = hasattr(model.dnn, 'model') and 'PeftModel' in str(type(model.dnn.model))

if is_lora:
    if args.no_lora:
        # Disable adapters (use base model weights only)
        # PEFT's disable_adapter() context manager is preferred, but for global switch:
        # We can set the adapter to a non-existent one or use base model
        print("LoRA adapter: DISABLED (base model only)")
        # This method call depends on PEFT version, but generally works for inference
        try:
            model.dnn.model.disable_adapter_layers()
        except AttributeError:
             # Fallback for older PEFT versions or different wrapping
            with model.dnn.model.disable_adapter():
                print("  Note: Using context manager for disabling adapter - this might not persist outside this block if logic was different.")
                pass 
            print("  Warning: Could not permanently disable adapter via method. Using base weights might require context manager in loop.")
    else:
        # Adapters are enabled by default in PEFT, but we can make sure
        print("LoRA adapter: ENABLED")
        try:
            model.dnn.model.set_adapter("default")
        except Exception:
            pass
        try:
            model.dnn.model.enable_adapter_layers()
        except AttributeError:
            pass
else:
    if args.no_lora:
        print("Warning: --no-lora passed but model has no LoRA adapter — ignoring")
    print("Model type: full (non-LoRA)")

# ── Pad mode based on backbone ─────────────────────────────────────────────── 
pad_mode = "reflection"
target_sr = model.sr

# ── Use N from CLI/default ─────────────────────────────────────────────────── 
N = args.N
print(f"Using N={N} diffusion steps (model checkpoint default: {model.sde.N})")

# Use model's sampler_type if available (matches validation behavior)
sampler_type = model.sde.sampler_type if hasattr(model.sde, 'sampler_type') else args.sampler
if sampler_type != args.sampler:
    print(f"Note: Using model's sampler_type='{sampler_type}' (overriding --sampler='{args.sampler}')")

print(f"Sampler: {sampler_type}, Corrector: {args.corrector}, Corrector steps: {args.corrector_steps}, SNR: {args.snr}")
print(f"Seed: {args.seed if args.seed is not None else 'None (stochastic, matching validation)'}")
print("-" * 70)

# ── Gather input files ─────────────────────────────────────────────────────── 
wav_files = sorted(glob.glob(join(args.test_dir, "*.wav")) + glob.glob(join(args.test_dir, "*.flac")))

if not wav_files:
    # Try recursive search if no files found in root
    wav_files = sorted(glob.glob(join(args.test_dir, "**", "*.wav"), recursive=True))
if not wav_files:
    print(f"No .wav files found in {args.test_dir} or its subdirectories")
    sys.exit(1)
print(f"Found {len(wav_files)} file(s) in {args.test_dir}")
os.makedirs(args.out_dir, exist_ok=True)

if args.n_iterations > 1:
    print(f"Running {args.n_iterations} iterations, writing to {args.out_dir}/iter_{{1..{args.n_iterations}}}/")

# ── Inference loop ────────────────────────────────────────────────────────────

if is_lora:
    if args.no_lora:
        print("Running inference with LoRA DISABLED (Base model only)")
    else:
        print("Running inference with LoRA ENABLED")
else:
    print("Running inference with base model (no LoRA detected)")

for iteration in range(0, args.n_iterations):
    seed = args.seed + iteration
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.n_iterations > 1:
        out_dir_iter = join(args.out_dir, f"iter_{iteration:03d}_seed{seed}")
        desc = f"Iteration {iteration}/{args.n_iterations} (seed={seed})"
    else:
        out_dir_iter = args.out_dir
        desc = f"Processing (seed={seed})"
    os.makedirs(out_dir_iter, exist_ok=True)

    for wav_path in tqdm(wav_files, desc=desc):
        # Load audio
        y_np, sr = sf.read(wav_path, always_2d=True)   # (T, C)
        
        if y_np.shape[1] > 1:
            #convert to mono by only using first channel
            y_np = y_np[:, 0]
            y_np = y_np[:, None]  # (T, 1)
        
        y_np = y_np.T.astype('float32')                # (C, T)
        if sr != model.sr:
            y = resample(y_np, orig_sr=sr, target_sr=model.sr)  # Resample to model SR
        else:
            y = y_np

        y = torch.from_numpy(y)  # (C, T)

        if is_lora and args.no_lora:
            with model.dnn.model.disable_adapter():
                x_hat = model.enhance(y=y, N=N, sampler_type=args.sampler,
                                    corrector=args.corrector,
                                    corrector_steps=args.corrector_steps, snr=args.snr)
        else:
            x_hat = model.enhance(y=y, N=N, sampler_type=args.sampler,
                                corrector=args.corrector,
                                corrector_steps=args.corrector_steps, snr=args.snr)

        x_hat_out = np.asarray(x_hat)

        if sr != model.sr:
            x_hat_out = resample(x_hat_out, orig_sr=model.sr, target_sr=sr)

        rel_path = os.path.relpath(wav_path, args.test_dir)
        stem = splitext(rel_path)[0]
        data_format = wav_path.split('.')[-1]  # Preserve original format (wav or flac)
        out_path = join(out_dir_iter, f"{stem}_separated."+data_format)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, x_hat_out, sr)

print(f"\nDone. Outputs written to: {args.out_dir}" +
      (f" (iter_1/ … iter_{args.n_iterations}/)" if args.n_iterations > 1 else ""))
