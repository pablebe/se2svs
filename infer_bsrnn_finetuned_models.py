#!/usr/bin/env python3
"""
Inference script for fine-tuned BSRNN SVS checkpoints.

This script is for checkpoints trained with `finetune_bsrnnse.py` / `BSRNNSVSModel`.
It supports LoRA-enabled checkpoints and can optionally disable LoRA adapters.
"""

import argparse
import glob
import os
import sys
import types
from contextlib import nullcontext
from os.path import join, splitext

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF
from tqdm import tqdm

from models.bsrnn_svs_model import BSRNNSVSModel
from models.sgmse.util.other import set_torch_cuda_arch_list


def resample_audio_last_dim(audio_np: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample numpy audio along the last dimension using torchaudio."""
    if orig_sr == target_sr:
        return audio_np
    audio_t = torch.from_numpy(audio_np.astype(np.float32, copy=False))
    resampled = taF.resample(audio_t, orig_sr, target_sr)
    return resampled.cpu().numpy()


class InferenceDataModule:
    """Minimal placeholder for BSRNNSVSModel init during inference.

    BSRNNSVSModel expects a data_module_cls argument and instantiates it in
    __init__. For inference, none of its dataloader functionality is needed.
    """

    def __init__(self, *args, **kwargs):
        pass


def register_legacy_baseline_config_safe_global() -> None:
    """Allow loading legacy checkpoints that pickle baseline_code.config.Config.

    This is required for older checkpoints when torch/Lightning uses safe
    loading behavior with weights_only=True.
    """
    package_name = "baseline_code"
    module_name = "baseline_code.config"

    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package

    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)

        class LegacyConfig:
            pass

        LegacyConfig.__name__ = "Config"
        LegacyConfig.__qualname__ = "Config"
        LegacyConfig.__module__ = module_name
        module.Config = LegacyConfig
        sys.modules[module_name] = module
        setattr(package, "config", module)

    config_cls = getattr(module, "Config", None)
    if config_cls is not None:
        torch.serialization.add_safe_globals([config_cls])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with BSRNNSVSModel checkpoints")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to BSRNN checkpoint")
    parser.add_argument("--test-dir", type=str, required=True, help="Input folder with wav/flac files")
    parser.add_argument("--out-dir", type=str, required=True, help="Output folder")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument("--no-lora", action="store_true", default=False, help="Disable LoRA adapter if present")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--recursive-search",
        action="store_true",
        default=False,
        help="Search for wav/flac recursively under test-dir",
    )
    return parser.parse_args()


def get_input_files(test_dir: str, recursive_search: bool = False):
    patterns = ["*.wav", "*.flac"]
    files = []
    for p in patterns:
        files.extend(glob.glob(join(test_dir, p), recursive=False))

    files = sorted(set(files))
    if files or not recursive_search:
        return files

    recursive_patterns = ["**/*.wav", "**/*.flac"]
    recursive_files = []
    for p in recursive_patterns:
        recursive_files.extend(glob.glob(join(test_dir, p), recursive=True))
    return sorted(set(recursive_files))


def is_lora_model(model: BSRNNSVSModel) -> bool:
    dnn = model.dnn
    module_name = dnn.__class__.__module__.lower()
    return hasattr(dnn, "disable_adapter") or hasattr(dnn, "disable_adapter_layers") or "peft" in module_name


def get_lora_disable_context(model: BSRNNSVSModel):
    dnn = model.dnn
    if hasattr(dnn, "disable_adapter"):
        return dnn.disable_adapter()
    if hasattr(dnn, "disable_adapter_layers"):
        dnn.disable_adapter_layers()
        return nullcontext()
    return nullcontext()


def _extract_state_dict_for_format_probe(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict):
        return state_dict
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        return checkpoint["model_state_dict"]
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    return None


def should_use_finetune_pretrained_loader(ckpt_path: str) -> bool:
    """Use finetune-style pretrained loading for legacy/base SE checkpoints.

    finetune_bsrnnse.py initializes BSRNNSVSModel(..., pretrained_ckpt=...).
    Legacy SE checkpoints typically store weights under se_model.* keys.
    """
    checkpoint = BSRNNSVSModel._load_checkpoint_raw(ckpt_path)
    state_dict = _extract_state_dict_for_format_probe(checkpoint)
    if not isinstance(state_dict, dict):
        return False

    return any(
        k.startswith(("se_model.", "model.se_model.", "module.se_model."))
        for k in state_dict.keys()
    )


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    set_torch_cuda_arch_list()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading BSRNN checkpoint: {args.ckpt}")
    register_legacy_baseline_config_safe_global()

    # BSRNNSVSModel always instantiates SpecsDataModule in __init__, which
    # requires base_dir even though inference doesn't use dataloaders.
    base_dir = os.path.dirname(args.test_dir)
    if should_use_finetune_pretrained_loader(args.ckpt):
        # Match finetune_bsrnnse.py loading path for pretrained SE checkpoints.
        model = BSRNNSVSModel(
            data_module_cls=InferenceDataModule,
            base_dir=base_dir,
            pretrained_ckpt=args.ckpt,
            finetune_mode="none",
            nolog=True,
        )
        print("Checkpoint load mode: finetune-style pretrained_ckpt")
    else:
        model = BSRNNSVSModel.load_from_checkpoint(
            args.ckpt,
            map_location=args.device,
            data_module_cls=InferenceDataModule,
            base_dir=base_dir,
            strict=False,
        )
        print("Checkpoint load mode: Lightning load_from_checkpoint")
    model.to(args.device)
    model.eval()

    has_lora = is_lora_model(model)
    if args.no_lora and has_lora:
        print("LoRA adapter: DISABLED")
    elif has_lora:
        print("LoRA adapter: ENABLED")
    else:
        if args.no_lora:
            print("Warning: --no-lora requested, but checkpoint model has no LoRA adapter")
        print("Model type: non-LoRA")

    wav_files = get_input_files(args.test_dir, recursive_search=args.recursive_search)
    if not wav_files:
        raise RuntimeError(f"No wav/flac files found under: {args.test_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Found {len(wav_files)} files in {args.test_dir}")

    for wav_path in tqdm(wav_files, desc="Processing"):
        y_np, sr = sf.read(wav_path, always_2d=True)  # (T, C)

        # Match the previous inference behavior: use first channel when input is multi-channel.
        if y_np.shape[1] > 1:
            y_np = y_np[:, 0:1]

        y_np = y_np.T.astype("float32")  # (C, T)
        if sr != model.sr:
            y_np = resample_audio_last_dim(y_np, orig_sr=sr, target_sr=model.sr)

        mix_wav = torch.from_numpy(y_np).unsqueeze(0).to(args.device)  # (1, C, T)

        with torch.inference_mode():
            if args.no_lora and has_lora:
                with get_lora_disable_context(model):
                    est = model._separate_with_noisy_norm_and_peak_output(mix_wav)
            else:
                est = model._separate_with_noisy_norm_and_peak_output(mix_wav)

        est_np = est.detach().cpu().squeeze(0).numpy()  # (C, T) or (T,)
        if est_np.ndim == 1:
            out_audio = est_np
        else:
            out_audio = est_np.T

        if sr != model.sr:
            if out_audio.ndim == 1:
                out_audio = resample_audio_last_dim(out_audio, orig_sr=model.sr, target_sr=sr)
            else:
                # out_audio is (T, C), so transpose to (C, T), resample, then transpose back.
                out_audio = resample_audio_last_dim(out_audio.T, orig_sr=model.sr, target_sr=sr).T

        rel_path = os.path.relpath(wav_path, args.test_dir)
        stem = splitext(rel_path)[0]
        ext = wav_path.split(".")[-1]
        out_path = join(args.out_dir, f"{stem}_separated.{ext}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, out_audio, sr)

    print(f"Done. Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
