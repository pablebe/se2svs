#!/usr/bin/env python3
"""
Evaluate separated audio files against target audio using metrics from MSS_model.py validation.

This script calculates the following metrics:
- SDR (Signal Distortion Ratio)
- SI-SDR (Scale-Invariant Signal Distortion Ratio)
- Multi-Resolution STFT Loss
- MERT Embedding MSE (using MERT-v1-95M model)

Results are saved to a CSV file and a summary is printed to the console.
"""
#TODO RERUN evaluation on active frames only (with short-time RMS analysis) and compare to full evaluation results.
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as taF
from tqdm import tqdm
from pathlib import Path

from torchmetrics.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from auraloss.freq import MultiResolutionSTFTLoss
from utils.loudness import calculate_loudness


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


def calculate_short_time_rms(audio, block_size, hop_size):
    """Calculate short-time RMS for audio signal.
    
    Args:
        audio: Audio signal (samples,) or (samples, channels)
        block_size: Block size in samples
        hop_size: Hop size in samples
        
    Returns:
        rms_frames: Array of RMS values per frame
        frame_indices: Start index of each frame
    """
    # Handle mono and stereo
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    
    num_samples, num_channels = audio.shape
    num_frames = int(np.ceil((num_samples - block_size) / hop_size)) + 1
    
    rms_frames = []
    frame_indices = []
    
    for i in range(num_frames):
        start = i * hop_size
        end = min(start + block_size, num_samples)
        frame_indices.append(start)
        
        if end - start < block_size:
            block = np.zeros((block_size, num_channels))
            block[:end - start, :] = audio[start:end, :]
        else:
            block = audio[start:end, :]
        
        rms = np.sqrt(np.mean(block**2))
        rms_frames.append(rms)
    
    return np.array(rms_frames), np.array(frame_indices)


def extract_active_audio(audio, sr, block_size_ms=50, hop_size_ms=25, threshold_db=-40):
    """Extract only active portions of audio based on short-time RMS threshold.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        block_size_ms: Block size in milliseconds
        hop_size_ms: Hop size in milliseconds
        threshold_db: Threshold in dBFS
        
    Returns:
        active_audio: Concatenated active audio segments
        active_ratio: Ratio of active frames
        active_mask: Boolean mask of active frames
    """
    # Convert time to samples
    block_size = int(block_size_ms * sr / 1000)
    hop_size = int(hop_size_ms * sr / 1000)
    
    # Calculate short-time RMS
    rms_frames, frame_indices = calculate_short_time_rms(audio, block_size, hop_size)
    
    # Convert to dBFS
    rms_db = 20 * np.log10(rms_frames + 1e-10)
    
    # Find active frames
    active_mask = rms_db > threshold_db
    active_ratio = np.sum(active_mask) / len(active_mask) if len(active_mask) > 0 else 0
    
    # Extract active segments
    if audio.ndim == 1:
        audio_2d = audio.reshape(-1, 1)
    else:
        audio_2d = audio
    
    active_segments = []
    for i, is_active in enumerate(active_mask):
        if is_active:
            start = frame_indices[i]
            end = min(start + block_size, len(audio_2d))
            active_segments.append(audio_2d[start:end])
    
    if not active_segments:
        # Return empty audio with same shape
        if audio.ndim == 1:
            return np.array([]), active_ratio, active_mask
        else:
            return np.zeros((0, audio.shape[1])), active_ratio, active_mask
    
    active_audio = np.concatenate(active_segments, axis=0)
    
    # Convert back to original shape
    if audio.ndim == 1:
        active_audio = active_audio.squeeze()
    
    return active_audio, active_ratio, active_mask


def load_audio_pair(target_path, separated_path, eval_sr=None, normalize_loudness=False):
    """Load a pair of target and separated audio files.
    
    Automatically resamples separated audio to match target sample rate if needed.
    Also handles length mismatches by trimming or padding to match target length.
    Converts signals to mono and optionally performs loudness normalization.
    
    Args:
        target_path: Path to target audio file
        separated_path: Path to separated audio file
        eval_sr: Optional sample rate to resample both files to
        normalize_loudness: Whether to normalize separated audio to target loudness
        
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
    
    # Ensure both have the same length (handle resampling artifacts)
    # Target length takes precedence
    target_len = target.shape[0]
    separated_len = separated.shape[0]
    
    if separated_len > target_len:
        # Trim separated
        separated = separated[:target_len]
    elif separated_len < target_len:
        # Pad separated
        if separated.ndim == 1:
            pad_width = ((0, target_len - separated_len))
        else:
            pad_width = ((0, target_len - separated_len), (0, 0))
        separated = np.pad(separated, pad_width, mode='constant')

    # Convert to Mono (use first channel if multi-channel)
    if target.ndim > 1 and target.shape[1] > 1:
        target = target[:, 0]
    elif target.ndim > 1:
         # (samples, 1) -> (samples,)
        target = target.squeeze()
        
    if separated.ndim > 1 and separated.shape[1] > 1:
        separated = separated[:, 0]
    elif separated.ndim > 1:
        separated = separated.squeeze()
            
    # Loudness normalization (performed on numpy arrays)
    if normalize_loudness:
        target_lufs = calculate_loudness(target, sr_target)
        separated_lufs = calculate_loudness(separated, sr_target)
        
        # Avoid normalizing providing explicit infinity issues
        if np.isfinite(target_lufs) and np.isfinite(separated_lufs):
            gain_db = target_lufs - separated_lufs
            gain_lin = 10.0 ** (gain_db / 20.0)
            separated *= gain_lin

            
    # Convert to torch tensors and add channel dimension for (1, samples)
    target = torch.from_numpy(target).float().unsqueeze(0)
    separated = torch.from_numpy(separated).float().unsqueeze(0)

    return target, separated, sr_target


def si_sdr_with_best_shift(separated: torch.Tensor, target: torch.Tensor,
                           shift_range: int = 128) -> float:
    """Compute SI-SDR maximised over shifts in [-shift_range, shift_range].

    Applies pad-based shifting to `separated` and returns the maximum SI-SDR
    value across the sweep. Both tensors must be 1-D (samples,).
    """
    best = -float('inf')
    s = separated.cpu().numpy()
    t_t = target.cpu().unsqueeze(0)  # (1, samples), kept on CPU
    for shift in range(-shift_range, shift_range + 1):
        if shift > 0:
            s_sh = np.pad(s[shift:], (0, shift), mode='constant')
        elif shift < 0:
            a = -shift
            s_sh = np.pad(s[:-a], (a, 0), mode='constant')
        else:
            s_sh = s
        val = scale_invariant_signal_distortion_ratio(
            torch.from_numpy(s_sh.astype(np.float32)).unsqueeze(0), t_t
        ).item()
        if val > best:
            best = val
    return best


def calculate_metrics(target, separated, sr, embedding_model, device='cuda',
                      align=True, shift_range=128):
    """Calculate all validation metrics for a single audio pair.
    
    Args:
        target: target audio tensor (channels, samples)
        separated: separated audio tensor (channels, samples)
        sr: sample rate
        embedding_model: MERT embedding model
        device: torch device
        
    Returns:
        dict with metric values
    """
    metrics = {}
    
    # Initialize multi-resolution STFT loss
    multi_res_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=[256, 512, 1024, 2048, 4096],
        win_lengths=[256, 512, 1024, 2048, 4096],
        hop_sizes=[64, 128, 256, 512, 1024],
        sample_rate=sr,
        perceptual_weighting=True
    ).forward
    
    # Calculate metrics per channel and average if multi-channel
    num_channels = target.shape[0]
    
    if num_channels > 1:
        sdr_sum = 0.0
        si_sdr_sum = 0.0
        multi_res_sum = 0.0
        
        for ch in range(num_channels):
            target_ch = target[ch, :].to(device)
            separated_ch = separated[ch, :].to(device)
            
            sdr_sum += signal_distortion_ratio(separated_ch, target_ch).item()
            if align:
                si_sdr_sum += si_sdr_with_best_shift(separated_ch, target_ch, shift_range=shift_range)
            else:
                si_sdr_sum += scale_invariant_signal_distortion_ratio(separated_ch, target_ch).item()
            multi_res_sum += multi_res_loss_fn(
                separated_ch.unsqueeze(0).unsqueeze(0),
                target_ch.unsqueeze(0).unsqueeze(0)
            ).item()
        
        metrics['sdr'] = sdr_sum / num_channels
        metrics['si_sdr'] = si_sdr_sum / num_channels
        metrics['multi_res_loss'] = multi_res_sum / num_channels
    else:
        target_mono = target.to(device)
        separated_mono = separated.to(device)
        
        metrics['sdr'] = signal_distortion_ratio(separated_mono, target_mono).item()
        if align:
            metrics['si_sdr'] = si_sdr_with_best_shift(separated_mono.squeeze(0), target_mono.squeeze(0),
                                                        shift_range=shift_range)
        else:
            metrics['si_sdr'] = scale_invariant_signal_distortion_ratio(separated_mono, target_mono).item()
        metrics['multi_res_loss'] = multi_res_loss_fn(
            separated_mono.unsqueeze(0),
            target_mono.unsqueeze(0)
        ).item()
    
    # Calculate embedding MSE (once per file)
    # Resample to embedding model's sample rate before embedding extraction.
    if embedding_model is None:
        metrics['mert_mse'] = float('nan')
        return metrics

    embedding_sr = embedding_model.sr

    def _prepare_embedding_audio(audio_tensor, src_sr, dst_sr):
        audio_np = audio_tensor.cpu().numpy()
        if src_sr != dst_sr:
            audio_np = resample_audio_np(audio_np, orig_sr=src_sr, target_sr=dst_sr, axis=-1)
        return audio_np.squeeze()

    target_emb_np = _prepare_embedding_audio(target, sr, embedding_sr)
    separated_emb_np = _prepare_embedding_audio(separated, sr, embedding_sr)
    
    # Get embeddings (remove channel dimension for embedding model)
    with torch.inference_mode():
        target_embedding = embedding_model._get_embedding(target_emb_np)
        separated_embedding = embedding_model._get_embedding(separated_emb_np)
        
        # Convert to numpy if needed
        if torch.is_tensor(target_embedding):
            target_embedding = target_embedding.cpu().numpy()
        if torch.is_tensor(separated_embedding):
            separated_embedding = separated_embedding.cpu().numpy()
        
        metrics['mert_mse'] = float(np.mean((target_embedding - separated_embedding)**2))
    
    return metrics


def find_matching_files(target_dir, separated_dir):
    """Find matching audio files between target and separated directories.
    
    Intelligently matches files based on file IDs extracted from filenames.
    Handles various naming patterns like:
    - target_fileid_X.wav <-> mixture_fileid_X_separated.wav
    - target_fileid_X.wav <-> separated_vocals_fileid_X.wav
    
    Args:
        target_dir: Path to target audio directory
        separated_dir: Path to separated audio directory
        
    Returns:
        list of tuples (file_id, target_path, separated_path)
    """
    import re
    
    target_dir = Path(target_dir)
    separated_dir = Path(separated_dir)
    
    # Supported audio extensions
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    
    # Get all audio files in both directories
    target_files = [f for f in target_dir.iterdir() 
                    if f.suffix.lower() in audio_extensions and f.is_file()]
    separated_files = [f for f in separated_dir.iterdir() 
                       if f.suffix.lower() in audio_extensions and f.is_file()]
    
    # Extract file IDs from target files
    # Pattern: target_fileid_X.wav or target_X.wav or fileid_X.wav or just X.wav
    target_dict = {}
    for f in target_files:
        # Try to extract numeric file ID
        match = re.search(r'fileid[_-]?(\d+)', f.stem, re.IGNORECASE)
        if match:
            file_id = match.group(1)
            target_dict[file_id] = f
            continue
            
        # Basic number pattern (e.g. "0.flac")
        match = re.match(r'^(\d+)$', f.stem)
        if match:
            file_id = match.group(1)
            target_dict[file_id] = f
            continue

        # Fallback: try to find any trailing number
        match = re.search(r'_(\d+)$', f.stem)
        if match:
            file_id = match.group(1)
            target_dict[file_id] = f
    
    # Extract file IDs from separated files
    # Pattern: mixture_fileid_X_separated.wav, separated_vocals_fileid_X.wav, or X_DT0_separated.wav
    separated_dict = {}
    for f in separated_files:
        # Try to extract numeric file ID
        match = re.search(r'fileid[_-]?(\d+)', f.stem, re.IGNORECASE)
        if match:
            file_id = match.group(1)
            separated_dict[file_id] = f
            continue

        # Pattern starting with number (e.g. "0_DT0_separated.flac")
        match = re.match(r'^(\d+)_', f.stem)
        if match:
            file_id = match.group(1)
            separated_dict[file_id] = f
            continue

        # Pattern: separated_vocals_N_DT0 (MelRoFo / gensvs baseline naming)
        match = re.search(r'separated_vocals_(\d+)', f.stem)
        if match:
            file_id = match.group(1)
            separated_dict[file_id] = f
            continue

        # Fallback: try to find any number before '_separated' or at the end
        match = re.search(r'_(\d+)(?:_separated)?$', f.stem)
        if match:
            file_id = match.group(1)
            separated_dict[file_id] = f
    
    # Find matches based on file IDs
    matches = []
    for file_id in sorted(target_dict.keys(), key=lambda x: int(x)):
        target_path = target_dict[file_id]
        
        if file_id in separated_dict:
            separated_path = separated_dict[file_id]
            matches.append((file_id, str(target_path), str(separated_path)))
            
    # Report missing files only if verbose or if explicitly requested
    if len(matches) < len(target_dict):
        missing_ids = set(target_dict.keys()) - set(separated_dict.keys())
        # print(f"Warning: {len(missing_ids)} target files have no matching separated file")
    
    return matches


def evaluate_single_directory(target_dir, separated_dir, device, embedding_model, 
                            output_csv=None, active_eval=False,
                            block_size_ms=50, hop_size_ms=25, threshold_db=-40,
                            eval_sr=None, normalize_loudness=False,
                            align=True, shift_range=128):
    """Evaluate a single directory of separated files.
    
    Args:
        target_dir: Path to directory containing target audio
        separated_dir: Path to directory containing separated audio
        device: Torch device
        embedding_model: Embedding model for MERT metric
        output_csv: Path to save result CSV
        active_eval: Whether to perform active frame evaluation
        block_size_ms: Block size for active frame detection
        hop_size_ms: Hop size for active frame detection
        threshold_db: Threshold for active frame detection
        eval_sr: Optional sample rate to resample to
        normalize_loudness: Whether to normalize separated audio to target loudness
    """
    
    # Find matching files
    print(f"\nFinding matching audio files in {separated_dir}...")
    matches = find_matching_files(target_dir, separated_dir)
    
    if not matches:
        print("Error: No matching files found!")
        return None
    
    # Check if we have all files (assuming 50 is the expected number based on target directory)
    # But we'll just report what we found
    print(f"Found {len(matches)} matching file pairs")
    if active_eval:
        print(f"Active evaluation enabled: Threshold={threshold_db}dB, Block={block_size_ms}ms")
    if eval_sr:
        print(f"Evaluation sample rate forced to: {eval_sr} Hz")
    if normalize_loudness:
        print("Loudness normalization enabled")
    if align:
        print(f"SI-SDR alignment enabled (shift range: ±{shift_range} samples)")
    
    # Evaluate all files
    results = []
    print("\nEvaluating audio files...")
    
    for file_id, target_path, separated_path in tqdm(matches):
        # Load audio
        target, separated, sr = load_audio_pair(target_path, separated_path, eval_sr=eval_sr, normalize_loudness=normalize_loudness)
        
        # ACTIVE FRAME EVALUATION LOGIC
        active_stats = {}
        if active_eval:
            # We want to use the TARGET signal to determine active frames.
            # Convert to numpy for RMS calculation if they are tensors
            target_np = target.cpu().numpy().T if target.ndim > 1 else target.cpu().numpy()
            separated_np = separated.cpu().numpy().T if separated.ndim > 1 else separated.cpu().numpy()
            
            # Ensure correct shape for extract_active_audio (samples, channels) or (samples,)
            # load_audio_pair returns (channels, samples) for multi-channel torch tensors
            # so the transpose above handles it.
            
            # 1. Determine active mask from TARGET
            _, active_ratio, active_mask = extract_active_audio(
                target_np, sr, block_size_ms, hop_size_ms, threshold_db
            )
            
            active_stats['active_ratio'] = active_ratio
            active_stats['active_frames'] = int(np.sum(active_mask))
            
            if active_ratio > 0:
                # 2. Extract active segments from TARGET and SEPARATED using the SAME mask
                # We need to re-slice based on the mask derived from target
                
                # Re-calculate indices to match extract_active_audio logic
                block_size = int(block_size_ms * sr / 1000)
                hop_size = int(hop_size_ms * sr / 1000)
                _, frame_indices = calculate_short_time_rms(target_np, block_size, hop_size)
                
                target_segments = []
                separated_segments = []
                
                # Handle shapes for slicing
                if target_np.ndim == 1:
                    t_2d = target_np.reshape(-1, 1)
                    s_2d = separated_np.reshape(-1, 1)
                else:
                    t_2d = target_np
                    s_2d = separated_np
                    
                for i, is_active in enumerate(active_mask):
                    if is_active:
                        start = frame_indices[i]
                        end = min(start + block_size, len(t_2d))
                        target_segments.append(t_2d[start:end])
                        separated_segments.append(s_2d[start:end])
                
                if target_segments:
                    # Concatenate and convert back to torch tensors
                    target_active = np.concatenate(target_segments, axis=0)
                    separated_active = np.concatenate(separated_segments, axis=0)
                    
                    if target_np.ndim == 1:
                        target_active = target_active.squeeze()
                        separated_active = separated_active.squeeze()
                        target_t = torch.from_numpy(target_active).float().unsqueeze(0)
                        separated_t = torch.from_numpy(separated_active).float().unsqueeze(0)
                    else:
                        target_t = torch.from_numpy(target_active).float().T # Back to (C, S)
                        separated_t = torch.from_numpy(separated_active).float().T
                        
                    # Calculate metrics on active portions
                    # We reuse valid target/separated variables for the calculation call
                    target = target_t
                    separated = separated_t
                else:
                    # Should not happen if active_ratio > 0, but safety check
                    target = None
            else:
                # No active frames
                target = None

        if target is not None:
            # Calculate metrics
            metrics = calculate_metrics(target, separated, sr, embedding_model, device,
                                         align=align, shift_range=shift_range)
            
            # Store results
            result = {
                'file_id': file_id,
                'target_file': Path(target_path).name,
                'separated_file': Path(separated_path).name,
                'sample_rate': sr
            }
            result.update(metrics)
            if active_eval:
                result.update(active_stats)
            results.append(result)
        else:
            # Handle silent file case
             result = {
                'file_id': file_id,
                'target_file': Path(target_path).name,
                'separated_file': Path(separated_path).name,
                'sample_rate': sr,
                'sdr': float('nan'),
                'si_sdr': float('nan'),
                'multi_res_loss': float('nan'),
                'mert_mse': float('nan')
            }
             if active_eval:
                result.update(active_stats)
             results.append(result)

    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    if output_csv:
        save_path = output_csv
    else:
        save_path = os.path.join(separated_dir, 'results.csv')
        
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of files evaluated: {len(df)}")
    print(f"Target directory: {target_dir}")
    print(f"Separated directory: {separated_dir}")
    print("\nMetric Statistics:")
    print("-"*60)
    
    # Summary statistics for each metric
    metrics_to_summarize = ['sdr', 'si_sdr', 'multi_res_loss', 'mert_mse']
    summary = {}
    
    for metric in metrics_to_summarize:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            summary[metric] = {
                'mean': mean_val,
                'std': std_val
            }
            
            print(f"\n{metric.upper()}:")
            print(f"  Mean:   {mean_val:.4f}")
            print(f"  Std:    {std_val:.4f}")
            print(f"  Min:    {min_val:.4f}")
            print(f"  Max:    {max_val:.4f}")
    
    print("\n" + "="*60)
    
    # Display per-file results (optional, can be commented out for brevity)
    # print("\nPer-file Results:")
    # print("-"*60)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.width', None)
    # print(df.to_string(index=False))
    # print("\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate separated audio files against targets using MSS_model validation metrics'
    )
    parser.add_argument('--target-dir', type=str, required=True,
                       help='Directory containing target audio files (e.g., test_sets/gensvs_eval_audio/target)')
    parser.add_argument('--separated-dir', type=str, default=None,
                       help='Directory containing separated audio files')
    parser.add_argument('--multi-run-dir', type=str, default=None,
                       help='Parent directory containing multiple verification runs (subfolders)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for computation (cuda or cpu)')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Output CSV filename (default: results.csv in separated-dir)')
    parser.add_argument('--expected-count', type=int, default=50,
                       help='Expected number of files to process a folder in multi-run mode')
    parser.add_argument('--active-eval', action='store_true',
                       help='Enable active frame evaluation (exclude silent frames)')
    parser.add_argument('--normalize-loudness', action='store_true',
                       help='Normalize the separated signal to the loudness level of the target signal')
    parser.add_argument('--align', action='store_true', default=True,
                       help='Maximise SI-SDR over a shift sweep (default: enabled)')
    parser.add_argument('--no-align', dest='align', action='store_false',
                       help='Disable shift-based SI-SDR alignment')
    parser.add_argument('--align-shift-range', type=int, default=128,
                       help='Half-width of shift sweep in samples (default: 128)')
    parser.add_argument('--eval-sr', type=int, default=None,
                       help='Resample audio to this specific sample rate for evaluation (e.g. 44100)')
    parser.add_argument('--threshold', type=float, default=-40,
                       help='Threshold in dBFS for active frame detection (default: -40)')
    parser.add_argument('--block-size', type=float, default=50,
                       help='Block size in milliseconds (default: 50)')
    parser.add_argument('--hop-size', type=float, default=25,
                       help='Hop size in milliseconds (default: 25)')
    
    args = parser.parse_args()
    
    if args.separated_dir is None and args.multi_run_dir is None:
        parser.error("Either --separated-dir or --multi-run-dir must be provided")

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize embedding model (optional; evaluation can run without MERT if unavailable).
    embedding_model = None
    try:
        print("Loading MERT-v1-95M embedding model...")
        from gensvs import get_all_models

        models = {m.name: m for m in get_all_models()}
        embedding_model = models["MERT-v1-95M"]
        embedding_model.load_model()

        # Freeze and move to device
        for param in embedding_model.model.parameters():
            param.requires_grad = False
        embedding_model.model.to(device)
        embedding_model.model.eval()
        print(f"MERT model loaded (sample rate: {embedding_model.sr} Hz)")
    except Exception as exc:
        print(f"Warning: could not initialize MERT model; mert_mse will be NaN. Reason: {exc}")
    
    # Single directory mode
    if args.separated_dir:
        # Check if directory actually contains audio files before trying to evaluate
        # This prevents "Error: No matching files found" when pointing to a parent directory of multiple runs
        audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
        has_audio = any(f.suffix.lower() in audio_extensions for f in Path(args.separated_dir).iterdir() if f.is_file())
        
        if has_audio:
            evaluate_single_directory(args.target_dir, args.separated_dir, device, embedding_model, 
                                     output_csv=args.output_csv,
                                     active_eval=args.active_eval,
                                     block_size_ms=args.block_size,
                                     hop_size_ms=args.hop_size,
                                     threshold_db=args.threshold,
                                     eval_sr=args.eval_sr,
                                     normalize_loudness=args.normalize_loudness,
                                     align=args.align,
                                     shift_range=args.align_shift_range)
        elif not args.multi_run_dir:
             print(f"Warning: No audio files found in {args.separated_dir}")
        
    # Multi-directory mode
    if args.multi_run_dir:
        multi_root = Path(args.multi_run_dir)
        print(f"\nScanning multi-run directory: {multi_root}")
        
        subdirs = sorted([d for d in multi_root.iterdir() if d.is_dir()])
        evaluated_count = 0
        
        print(f"Found {len(subdirs)} subdirectories")
        
        for subdir in subdirs:
            # Skip if results.csv already exists (unless a specific output_csv is provided for multi-run, which we treat as an override desire?
            # Actually, standard behavior is usually per-folder output.
            # If the user provided --output-csv with multi-run, they might expect it to name the file inside each subdirectory differently?
            # OR they might expect a single big CSV? The current code structure supports per-directory CSVs.
            # Let's assume if they pass --output-csv "custom_name.csv", they want "subdir/custom_name.csv"
            
            csv_filename = args.output_csv if args.output_csv else 'results.csv'
            results_csv_path = subdir / csv_filename
            
            if results_csv_path.exists():
                print(f"Skipping {subdir.name}: {csv_filename} already exists")
                continue
            
            # Check if directory has expected number of matching files
            # Quick check first for efficiency using glob (checking all supported extensions)
            audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
            audio_files = [f for f in subdir.iterdir() if f.suffix.lower() in audio_extensions]

            if len(audio_files) < args.expected_count:
                # print(f"Skipping {subdir.name}: Only {len(audio_files)} audio files found (expected {args.expected_count})")
                continue
                
            # Perform actual matching to be sure
            matches = find_matching_files(args.target_dir, subdir)
            if len(matches) >= args.expected_count:
                print(f"\nProcessing {subdir.name} (matches: {len(matches)})...")
                # When calling for a subdir in multi-run, we pass the path explicitly constructed
                # If args.output_csv is provided (e.g. "my_results.csv"), it will be saved as "subdir/my_results.csv"
                # If not, it defaults to "subdir/results.csv" inside evaluating_single_directory logic if we pass None,
                # BUT we want to enforce the name if provided.
                
                output_csv_path = subdir / csv_filename

                evaluate_single_directory(args.target_dir, subdir, device, embedding_model, 
                                         output_csv=str(output_csv_path),
                                         active_eval=args.active_eval,
                                         block_size_ms=args.block_size,
                                         hop_size_ms=args.hop_size,
                                         threshold_db=args.threshold,
                                         eval_sr=args.eval_sr,
                                         normalize_loudness=args.normalize_loudness,
                                         align=args.align,
                                         shift_range=args.align_shift_range)
                evaluated_count += 1
            else:
                # print(f"Skipping {subdir.name}: Only {len(matches)} matching pair(s) found")
                pass
                
        print(f"\nProcessed {evaluated_count} valid run directories.")


if __name__ == '__main__':
    main()
