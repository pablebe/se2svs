#!/usr/bin/env python3
"""
Inference script for GenSVS package baselines (SGMSVS, MelRoFo(S)) and
MelBandRoformer-large (MelRoFo(L)).

SGMSVS supports multiple stochastic iterations (--n-iterations); each iteration
is written to its own iter_NNN_seedXXX subdirectory.  MelRoFo(S) and MelRoFo(L)
are deterministic and always run once.

Output layout
-------------
  <out-dir>/sgmsvs/iter_000_seed42/separated_vocals_*.wav   (SGMSVS)
  <out-dir>/melroformer_small/separated_vocals_*.wav         (MelRoFo(S))
  <out-dir>/melroformer_large/separated_vocals_*.wav         (MelRoFo(L))

Example
-------
# Run both gensvs baselines + large model, 10 SGMSVS iterations:
python inference_baselines.py \\
    --mix-dir se2svs_results_and_audio/gensvs_eval_audio/mixture \\
    --out-dir se2svs_results_and_audio/gensvs_eval_audio/baselines \\
    --model all --n-iterations 10 \\
    --melrofo-large-ckpt trained_models/melroformer_large/MelBandRoformer.ckpt

# Run only MelRoFo(L):
python inference_baselines.py \\
    --mix-dir se2svs_results_and_audio/MSRBench_Vocals/mixture \\
    --out-dir se2svs_results_and_audio/MSRBench_Vocals/baselines \\
    --model melrofo_large \\
    --melrofo-large-ckpt trained_models/melroformer_large/MelBandRoformer.ckpt
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF

DEFAULT_MIX_DIR = 'se2svs_results_and_audio/gensvs_eval_audio/mixture'
DEFAULT_OUT_DIR = 'se2svs_results_and_audio/gensvs_eval_audio/baselines'
DEFAULT_LOUDNESS_LEVEL = -18.0
DEFAULT_SEED = 42
DEFAULT_MELROFO_LARGE_CKPT = 'trained_models/melroformer_large/MelBandRoformer.ckpt'
DEFAULT_MELROFO_LARGE_CONFIG = 'configs/config_melroformer_large.yaml'
MELROFO_LARGE_SR = 44100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Infer with GenSVS baselines (SGMSVS, MelRoFo(S)) and MelRoFo(L)'
    )
    parser.add_argument('--mix-dir', type=str, default=DEFAULT_MIX_DIR,
                        help='Input folder with wav/flac mixture files')
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT_DIR,
                        help='Output folder for separated files')
    parser.add_argument('--model', type=str, default='both',
                        choices=('sgmsvs', 'melrofo', 'melrofo_large', 'both', 'all'),
                        help='Which model(s) to run: sgmsvs, melrofo (=MelRoFo(S) via gensvs), '
                             'melrofo_large (=MelRoFo(L)), both (=sgmsvs+melrofo), '
                             'all (=sgmsvs+melrofo+melrofo_large)')
    parser.add_argument('--loudness-normalize', action='store_true', default=False,
                        help='Apply loudness normalization before inference')
    parser.add_argument('--loudness-level', type=float, default=DEFAULT_LOUDNESS_LEVEL,
                        help=f'Target loudness level in LUFS (default: {DEFAULT_LOUDNESS_LEVEL})')
    parser.add_argument('--n-iterations', type=int, default=1,
                        help='Number of stochastic SGMSVS iterations (default: 1)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Base random seed for SGMSVS (default: {DEFAULT_SEED})')
    parser.add_argument('--melrofo-large-ckpt', type=str, default=DEFAULT_MELROFO_LARGE_CKPT,
                        help=f'Path to MelBandRoformer-large checkpoint '
                             f'(default: {DEFAULT_MELROFO_LARGE_CKPT})')
    parser.add_argument('--melrofo-large-config', type=str, default=DEFAULT_MELROFO_LARGE_CONFIG,
                        help=f'Path to MelBandRoformer-large config yaml '
                             f'(default: {DEFAULT_MELROFO_LARGE_CONFIG})')
    return parser.parse_args()


def _resample(audio_np: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio array (C, T) with torchaudio."""
    if orig_sr == target_sr:
        return audio_np
    t = torch.from_numpy(audio_np.astype(np.float32, copy=False))
    return taF.resample(t, orig_sr, target_sr).numpy()


def _calculate_loudness(audio: np.ndarray, sr: int) -> float:
    """Broadband loudness in LUFS (mono or stereo, shape (T,) or (T, C))."""
    try:
        from sgmsvs.loudness import calculate_loudness as _cl
        return _cl(audio, sr)
    except Exception:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        return meter.integrated_loudness(audio)


def run_melrofo_large_folder(
    mix_dir: str,
    out_dir: str,
    ckpt_path: str,
    config_path: str,
    loudness_normalize: bool = False,
    loudness_level: float = DEFAULT_LOUDNESS_LEVEL,
) -> None:
    """Run MelBandRoformer-large on all wav/flac files in mix_dir."""
    import yaml
    from ml_collections import ConfigDict
    from baseline_models.util.utils import demix_track, get_model_from_config

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'MelRoFo(L) checkpoint not found: {ckpt_path}')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'MelRoFo(L) config not found: {config_path}')

    print(f'Loading MelRoFo(L) config: {config_path}')
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model_sr = int(config.model.get('sample_rate', MELROFO_LARGE_SR))

    print(f'Building MelBandRoformer model...')
    model = get_model_from_config('mel_band_roformer', config)

    print(f'Loading checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    out_model_dir = os.path.join(out_dir, 'melroformer_large')
    os.makedirs(out_model_dir, exist_ok=True)

    audio_files = sorted(
        glob.glob(os.path.join(mix_dir, '*.wav')) +
        glob.glob(os.path.join(mix_dir, '*.flac')) +
        glob.glob(os.path.join(mix_dir, '**', '*.wav'), recursive=True) +
        glob.glob(os.path.join(mix_dir, '**', '*.flac'), recursive=True)
    )
    audio_files = sorted(set(audio_files))
    if not audio_files:
        raise RuntimeError(f'No wav/flac files found in: {mix_dir}')
    print(f'MelRoFo(L): processing {len(audio_files)} files -> {out_model_dir}/')

    first_chunk_time = None
    for path in audio_files:
        rel = os.path.relpath(path, mix_dir)
        stem = Path(path).stem
        print(f'  {rel}')

        audio_np, sr = sf.read(path, always_2d=True)  # (T, C)
        audio_ct = audio_np.T.astype(np.float32)      # (C, T)

        # Model expects stereo (C=2); duplicate mono if needed
        if audio_ct.shape[0] == 1:
            audio_ct = np.concatenate([audio_ct, audio_ct], axis=0)

        # Resample to model SR
        audio_model = _resample(audio_ct, sr, model_sr)  # (2, T')
        mix_t = torch.from_numpy(audio_model)

        res, first_chunk_time = demix_track(config, model, mix_t, device, first_chunk_time)

        # vocals shape: (C, T') numpy array at model_sr
        vocals = res['vocals']  # (C, T')

        # Output mono (first channel) at model_sr — consistent with other baselines
        out_audio = vocals[0]  # (T,)

        if loudness_normalize:
            from utils.loudness import calculate_loudness
            L = calculate_loudness(out_audio[:, np.newaxis], model_sr)
            k = 10 ** ((loudness_level - L) / 20)
            out_audio = out_audio * k

        # Preserve subdirectory structure (e.g. speaker dirs in EARS-WHAM)
        rel_dir = os.path.dirname(rel)
        out_subdir = os.path.join(out_model_dir, rel_dir) if rel_dir else out_model_dir
        os.makedirs(out_subdir, exist_ok=True)
        ext = Path(path).suffix  # preserve input format (.wav or .flac)
        out_path = os.path.join(out_subdir, f'separated_vocals_{stem}{ext}')
        subtype = 'FLOAT' if ext.lower() == '.wav' else 'PCM_24'
        sf.write(out_path, out_audio, model_sr, subtype=subtype)

    print(f'MelRoFo(L) done. Outputs written to: {out_model_dir}')


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.mix_dir):
        raise FileNotFoundError(f'Mix directory not found: {args.mix_dir}')

    run_sgmsvs = args.model in ('sgmsvs', 'both', 'all')
    run_melrofo = args.model in ('melrofo', 'both', 'all')
    run_melrofo_large = args.model in ('melrofo_large', 'all')

    if run_sgmsvs:
        from gensvs import SGMSVS
        print('Loading SGMSVS model...')
        sgmsvs_model = SGMSVS()
        print(f'Running SGMSVS: {args.n_iterations} iteration(s), '
              f'writing to {args.out_dir}/sgmsvs/iter_*_seed*/')
        for i in range(args.n_iterations):
            seed = args.seed + i
            iter_out = os.path.join(args.out_dir, 'sgmsvs', f'iter_{i:03d}_seed{seed}')
            os.makedirs(iter_out, exist_ok=True)
            print(f'SGMSVS {i + 1}/{args.n_iterations} (seed={seed}): {args.mix_dir} -> {iter_out}/')
            sgmsvs_model.run_folder(
                args.mix_dir,
                iter_out,
                loudness_normalize=args.loudness_normalize,
                loudness_level=args.loudness_level,
                output_mono=True,
                ch_by_ch_processing=False,
                random_seed=seed,
            )
        print(f'SGMSVS done. Outputs written under: {args.out_dir}')

    if run_melrofo:
        from gensvs import MelRoFoBigVGAN
        print('Loading MelRoFoBigVGAN model...')
        melrofo_model = MelRoFoBigVGAN()
        print(f'Running MelRoFo(S): {args.mix_dir} -> {args.out_dir}/melroformer_small/')
        melrofo_model.run_folder(
            args.mix_dir,
            args.out_dir,
            loudness_normalize=args.loudness_normalize,
            loudness_level=args.loudness_level,
            output_mono=True,
        )
        print(f'MelRoFo(S) done. Outputs written under: {args.out_dir}')

    if run_melrofo_large:
        run_melrofo_large_folder(
            mix_dir=args.mix_dir,
            out_dir=args.out_dir,
            ckpt_path=args.melrofo_large_ckpt,
            config_path=args.melrofo_large_config,
            loudness_normalize=args.loudness_normalize,
            loudness_level=args.loudness_level,
        )

    print(f'\nDone. All outputs written under: {args.out_dir}')


if __name__ == '__main__':
    main()
