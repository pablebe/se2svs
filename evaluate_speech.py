#!/usr/bin/env python3
"""
Evaluate speech enhancement performance using SI-SDR, PESQ, STOI, DNSMOS, and DistillMOS.

This script calculates the following metrics:
- SI-SDR (Scale-Invariant Signal Distortion Ratio) from torchmetrics
- PESQ (Perceptual Evaluation of Speech Quality) from torchmetrics (Wideband, 16kHz)
- STOI (Short-Time Objective Intelligibility) from torchmetrics (16kHz)
- DNSMOS (Deep Noise Suppression Mean Opinion Score) from torchmetrics (16kHz)
- DistillMOS from https://github.com/microsoft/Distill-MOS (16kHz)

Results are saved to a CSV file and a summary is printed to the console.
"""
import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as taF
from tqdm import tqdm
from pathlib import Path

# Metrics
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
import distillmos


_DNSMOS_DISABLED = False

def resample_audio_np(audio_np, orig_sr, target_sr, axis=0):
    """Resample numpy audio with torchaudio while honoring source axis."""
    if orig_sr == target_sr:
        return audio_np

    audio = np.asarray(audio_np)
    ndim = audio.ndim
    if ndim == 0:
        return audio

    if axis < 0:
        axis += ndim

    moved = np.moveaxis(audio, axis, -1)
    t = torch.from_numpy(moved.astype(np.float32, copy=False))
    out = taF.resample(t, orig_sr, target_sr).cpu().numpy()
    return np.moveaxis(out, -1, axis)

def load_audio_pair(target_path, separated_path, eval_sr=None):
    """Load a pair of target and separated audio files.
    
    Automatically resamples separated audio to match target sample rate if needed.
    Also handles length mismatches by trimming or padding to match target length.
    Converts signals to mono.
    
    Args:
        target_path: Path to target audio file
        separated_path: Path to separated audio file
        eval_sr: Optional sample rate to resample both files to
        
    Returns:
        target: target audio tensor (1, samples)
        separated: separated audio tensor (1, samples)
        sr: sample rate (target's sample rate or eval_sr)
    """
    target, sr_target = sf.read(target_path)
    separated, sr_separated = sf.read(separated_path)

    # Resample to specific evaluation rate if requested
    if eval_sr is not None:
        if sr_target != eval_sr:
            target = resample_audio_np(target, orig_sr=sr_target, target_sr=eval_sr, axis=0)
            sr_target = eval_sr
        if sr_separated != eval_sr:
            separated = resample_audio_np(separated, orig_sr=sr_separated, target_sr=eval_sr, axis=0)
            sr_separated = eval_sr

    # Resample separated audio to match target sample rate if needed
    if sr_target != sr_separated:
        separated = resample_audio_np(separated, orig_sr=sr_separated, target_sr=sr_target, axis=0)
    
    # Ensure both have the same length
    target_len = target.shape[0]
    separated_len = separated.shape[0]
    
    if separated_len > target_len:
        separated = separated[:target_len]
    elif separated_len < target_len:
        if separated.ndim == 1:
            pad_width = ((0, target_len - separated_len))
        else:
            pad_width = ((0, target_len - separated_len), (0, 0))
        separated = np.pad(separated, pad_width, mode='constant')

    # Convert to Mono
    if target.ndim > 1 and target.shape[1] > 1:
        target = target[:, 0]
    elif target.ndim > 1:
        target = target.squeeze()
        
    if separated.ndim > 1 and separated.shape[1] > 1:
        separated = separated[:, 0]
    elif separated.ndim > 1:
        separated = separated.squeeze()
            
    # Convert to torch tensors and add channel dimension for (1, samples)
    target = torch.from_numpy(target).float().unsqueeze(0)
    separated = torch.from_numpy(separated).float().unsqueeze(0)

    return target, separated, sr_target


def calculate_speech_metrics_batch(target_batch, separated_batch, sr, distillmos_model, device='cuda'):
    """Calculate speech metrics for a batch of aligned audio pairs.
    
    Args:
        target_batch: target audio tensor (B, 1, samples)
        separated_batch: separated audio tensor (B, 1, samples)
        sr: sample rate
        distillmos_model: DistillMOS model
        device: torch device
        
    Returns:
        dict of batched metric vectors (numpy arrays)
    """
    target = target_batch.to(device)
    separated = separated_batch.to(device)

    target_flat = target.squeeze(1)
    separated_flat = separated.squeeze(1)

    si_sdr = scale_invariant_signal_distortion_ratio(separated_flat, target_flat)

    target_16k = target
    separated_16k = separated
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
        target_16k = resampler(target)
        separated_16k = resampler(separated)

    target_16k_flat = target_16k.squeeze(1)
    separated_16k_flat = separated_16k.squeeze(1)

    pesq = perceptual_evaluation_speech_quality(separated_16k_flat, target_16k_flat, fs=16000, mode='wb')
    # Keep previous behavior: STOI computed at original-rate tensors with fs=48000.
    stoi = short_time_objective_intelligibility(separated_flat, target_flat, fs=48000, extended=True)

    # Returns shape (B, 4): [p808_mos, mos_sig, mos_bak, mos_ovr]
    global _DNSMOS_DISABLED
    if _DNSMOS_DISABLED:
        dnsmos = torch.full((target_flat.shape[0], 4), float('nan'), device=target_flat.device)
    else:
        try:
            dnsmos = deep_noise_suppression_mean_opinion_score(
                separated_16k_flat, fs=16000, personalized=False
            )
        except Exception as exc:
            _DNSMOS_DISABLED = True
            print(f"Warning: DNSMOS unavailable, filling NaN for DNSMOS columns. Reason: {exc}")
            dnsmos = torch.full((target_flat.shape[0], 4), float('nan'), device=target_flat.device)

    if distillmos_model is not None:
        with torch.no_grad():
            distillmos_score = distillmos_model(separated_16k_flat)
        distillmos_score = distillmos_score.squeeze(-1)
    else:
        distillmos_score = torch.full((target_flat.shape[0],), float('nan'), device=target_flat.device)

    return {
        'si_sdr': si_sdr.detach().cpu().numpy(),
        'pesq': pesq.detach().cpu().numpy(),
        'stoi': stoi.detach().cpu().numpy(),
        'dnsmos_p808': dnsmos[:, 0].detach().cpu().numpy(),
        'dnsmos_sig': dnsmos[:, 1].detach().cpu().numpy(),
        'dnsmos_bak': dnsmos[:, 2].detach().cpu().numpy(),
        'dnsmos_ovrl': dnsmos[:, 3].detach().cpu().numpy(),
        'distillmos': distillmos_score.detach().cpu().numpy(),
    }


def find_matching_files(target_dir, separated_dir):
    """Find matching audio files between target and separated directories."""
    import re
    
    target_dir = Path(target_dir)
    separated_dir = Path(separated_dir)
    
    # Supported audio extensions
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    
    # Get all audio files
    target_files = [f for f in target_dir.iterdir() 
                    if f.suffix.lower() in audio_extensions and f.is_file()]
    separated_files = [f for f in separated_dir.iterdir() 
                       if f.suffix.lower() in audio_extensions and f.is_file()]
    
    # Extract file IDs from target files
    target_dict = {}
    for f in target_files:
        match = re.search(r'fileid[_-]?(\d+)', f.stem, re.IGNORECASE)
        if match:
            target_dict[match.group(1)] = f
            continue
        match = re.match(r'^(\d+)$', f.stem)
        if match:
            target_dict[match.group(1)] = f
            continue
        match = re.search(r'_(\d+)$', f.stem)
        if match:
            target_dict[match.group(1)] = f
    
    # Extract file IDs from separated files
    separated_dict = {}
    for f in separated_files:
        match = re.search(r'fileid[_-]?(\d+)', f.stem, re.IGNORECASE)
        if match:
            separated_dict[match.group(1)] = f
            continue
        match = re.match(r'^(\d+)_', f.stem)
        if match:
            separated_dict[match.group(1)] = f
            continue
        match = re.search(r'_(\d+)(?:_separated)?$', f.stem)
        if match:
            separated_dict[match.group(1)] = f
            
    # Match files
    matches = []
    for file_id in sorted(target_dict.keys(), key=lambda x: int(x)):
        if file_id in separated_dict:
            matches.append((file_id, str(target_dict[file_id]), str(separated_dict[file_id])))
            
    return matches


def find_matching_files_recursive(target_dir, separated_dir):
    """Find matching audio files by walking subdirectories of separated_dir.

    For each leaf subdirectory under separated_dir that contains audio files,
    the function looks for a subdirectory at the same relative path under
    target_dir and delegates to find_matching_files for ID-based matching.
    The file_id encodes the relative subdir path + numeric ID so speaker info
    is preserved (e.g. ``p102/807``).
    """
    target_path = Path(target_dir)
    separated_path = Path(separated_dir)
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}

    # Collect all leaf directories that contain audio files.
    leaf_dirs = set()
    for f in separated_path.rglob('*'):
        if f.is_file() and f.suffix.lower() in audio_extensions:
            leaf_dirs.add(f.parent)

    matches = []
    for sep_subdir in sorted(leaf_dirs):
        rel_subdir = sep_subdir.relative_to(separated_path)
        tgt_subdir = target_path / rel_subdir
        if not tgt_subdir.is_dir():
            # Try flattening one level: separated has extra subdir that target lacks.
            tgt_subdir = target_path / rel_subdir.name
        if not tgt_subdir.is_dir():
            continue
        subdir_matches = find_matching_files(str(tgt_subdir), str(sep_subdir))
        for file_id, tgt_file, sep_file in subdir_matches:
            matches.append((f"{rel_subdir}/{file_id}", tgt_file, sep_file))

    return matches


def evaluate_single_directory(target_dir, separated_dir, device, distillmos_model, output_csv=None, eval_sr=None, batch_size=8):
    """Evaluate a directory of separated files against targets.

    First tries flat matching (files directly inside both dirs).  If that yields
    no matches, falls back to recursive matching by relative path so nested
    speaker/subdirectory layouts (e.g. EARS) are handled automatically.
    """
    print(f"\nFinding matching audio files in {separated_dir}...")
    matches = find_matching_files(target_dir, separated_dir)

    if not matches:
        matches = find_matching_files_recursive(target_dir, separated_dir)
        if matches:
            print(f"Using recursive file matching ({len(matches)} pairs found)")

    if not matches:
        print("Error: No matching files found!")
        return None
    
    print(f"Found {len(matches)} matching file pairs")
    if eval_sr:
        print(f"Evaluation sample rate forced to: {eval_sr} Hz")
    print(f"Batch size: {batch_size}")
    
    results = []
    print("\nEvaluating audio files...")
    
    for start_idx in tqdm(range(0, len(matches), batch_size)):
        batch_matches = matches[start_idx:start_idx + batch_size]

        batch_items = []
        for file_id, target_path, separated_path in batch_matches:
            target, separated, sr = load_audio_pair(target_path, separated_path, eval_sr=eval_sr)
            batch_items.append((file_id, target_path, separated_path, sr, target, separated))

        if not batch_items:
            continue

        srs = {item[3] for item in batch_items}
        lengths = {item[4].shape[-1] for item in batch_items}

        # True batched path when sample rate and length align.
        if len(srs) == 1 and len(lengths) == 1:
            sr = batch_items[0][3]
            target_batch = torch.stack([item[4] for item in batch_items], dim=0)
            separated_batch = torch.stack([item[5] for item in batch_items], dim=0)
            batch_metrics = calculate_speech_metrics_batch(target_batch, separated_batch, sr, distillmos_model, device)

            for i, (file_id, target_path, separated_path, _, _, _) in enumerate(batch_items):
                result = {
                    'file_id': file_id,
                    'target_file': Path(target_path).name,
                    'separated_file': Path(separated_path).name,
                    'sample_rate': sr,
                    'si_sdr': float(batch_metrics['si_sdr'][i]),
                    'pesq': float(batch_metrics['pesq'][i]),
                    'stoi': float(batch_metrics['stoi'][i]),
                    'dnsmos_p808': float(batch_metrics['dnsmos_p808'][i]),
                    'dnsmos_sig': float(batch_metrics['dnsmos_sig'][i]),
                    'dnsmos_bak': float(batch_metrics['dnsmos_bak'][i]),
                    'dnsmos_ovrl': float(batch_metrics['dnsmos_ovrl'][i]),
                    'distillmos': float(batch_metrics['distillmos'][i]),
                }
                results.append(result)
        else:
            # Fallback preserves correctness for mixed-rate or variable-length batches.
            for file_id, target_path, separated_path, sr, target, separated in batch_items:
                single_metrics = calculate_speech_metrics_batch(
                    target.unsqueeze(0), separated.unsqueeze(0), sr, distillmos_model, device
                )
                result = {
                    'file_id': file_id,
                    'target_file': Path(target_path).name,
                    'separated_file': Path(separated_path).name,
                    'sample_rate': sr,
                    'si_sdr': float(single_metrics['si_sdr'][0]),
                    'pesq': float(single_metrics['pesq'][0]),
                    'stoi': float(single_metrics['stoi'][0]),
                    'dnsmos_p808': float(single_metrics['dnsmos_p808'][0]),
                    'dnsmos_sig': float(single_metrics['dnsmos_sig'][0]),
                    'dnsmos_bak': float(single_metrics['dnsmos_bak'][0]),
                    'dnsmos_ovrl': float(single_metrics['dnsmos_ovrl'][0]),
                    'distillmos': float(single_metrics['distillmos'][0]),
                }
                results.append(result)
    
    df = pd.DataFrame(results)
    
    if output_csv:
        save_path = output_csv
    else:
        save_path = os.path.join(separated_dir, 'speech_metrics_results.csv')
        
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SPEECH EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of files: {len(df)}")
    print("\nMetric Statistics:")
    print("-"*60)
    
    # Summarize all numeric columns except file_id/sample_rate
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = ['file_id', 'sample_rate']
    
    for col in numeric_cols:
        if col not in exclude:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"\n{col.upper()}:")
            print(f"  Mean:   {mean_val:.4f}")
            print(f"  Std:    {std_val:.4f}")
            print(f"  Min:    {df[col].min():.4f}")
            print(f"  Max:    {df[col].max():.4f}")
            
    print("\n" + "="*60)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate separated audio files using Speech Enhancement metrics (SI-SDR, PESQ, STOI, DNSMOS, DistillMOS)'
    )
    parser.add_argument('--target-dir', type=str, required=True,
                       help='Directory containing target audio files')
    parser.add_argument('--separated-dir', type=str, default=None,
                       help='Directory containing separated audio files')
    parser.add_argument('--multi-run-dir', type=str, default=None,
                       help='Parent directory containing multiple verification runs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for computation')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Output CSV filename')
    parser.add_argument('--eval-sr', type=int, default=None,
                       help='Force specific sample rate for loading')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for metric inference (default: 8)')
                       
    args = parser.parse_args()
    
    if args.separated_dir is None and args.multi_run_dir is None:
        parser.error("Either --separated-dir or --multi-run-dir must be provided")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load DistillMOS model
    print("Loading DistillMOS model...")
    distillmos_model = distillmos.ConvTransformerSQAModel()
    distillmos_model.eval()
    distillmos_model.to(device)
    print("DistillMOS model loaded.")
    
    print("Using functional torchmetrics for batched metric inference")

    # Single directory mode
    if args.separated_dir:
        evaluate_single_directory(args.target_dir, args.separated_dir, device, distillmos_model,
                                 output_csv=args.output_csv, eval_sr=args.eval_sr,
                                 batch_size=args.batch_size)

    # Multi-directory mode
    if args.multi_run_dir:
        multi_root = Path(args.multi_run_dir)
        csv_filename = args.output_csv if args.output_csv else 'speech_metrics_results.csv'
        print(f"\nScanning multi-run directory: {multi_root}")
        subdirs = sorted([d for d in multi_root.iterdir() if d.is_dir()])

        for subdir in subdirs:
            results_csv_path = subdir / csv_filename
            if results_csv_path.exists():
                print(f"Skipping {subdir.name}: {csv_filename} already exists")
                continue
            print(f"\nProcessing {subdir.name}...")
            evaluate_single_directory(args.target_dir, subdir, device, distillmos_model,
                                     output_csv=str(results_csv_path), eval_sr=args.eval_sr,
                                     batch_size=args.batch_size)


if __name__ == '__main__':
    main()
