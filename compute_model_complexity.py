#!/usr/bin/env python3
"""
Compute model complexity with ptflops and export parameter counts and complexity metrics.

Notes:
- Complexity and parameter counts are both computed with ptflops.
- For SGM models, ptflops is run on model.dnn (score backbone) with a dummy
  input shape derived from the model's training data pipeline.
- For BSRNN models, ptflops is run on model.dnn with a 1-second waveform.
"""

from __future__ import annotations

import argparse
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

import models.data_module
import models.MSS_model
import models.sgmse
import models.sgmse.backbones
import models.sgmse.sdes
import models.sgmse.util
from models.bsrnn_svs_model import BSRNNSVSModel
from models.sgmse.backbones.shared import BackboneRegistry




@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    family: str
    checkpoint: str


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("SGMSVS (full finetuning)", "SGM", "checkpoints/se2svs/sgm_full/epoch=900-sdr=8.35.ckpt"),
    ModelSpec("LoRA-SGMSVS (rank 16)", "SGM", "checkpoints/se2svs/sgm_lora_r16/epoch=533-sdr=7.63.ckpt"),
    ModelSpec("SGMSVS (from scratch)", "SGM", "checkpoints/se2svs/sgm_scratch/epoch=522-sdr=7.26.ckpt"),
    ModelSpec("SGMSE  (base)", "SGM", "checkpoints/sgmse_pretrained/ears_wham.ckpt"),
    ModelSpec("BSRNNSVS (full finetuning)", "BSRNN", "checkpoints/se2svs/bsrnn_full/epoch=378-sdr=10.03.ckpt"),
    ModelSpec("LoRA-BSRNNSVS (rank 128)", "BSRNN", "checkpoints/se2svs/bsrnn_lora_r128/epoch=543-sdr=9.15.ckpt"),
    ModelSpec("LoRA-BSRNNSVS (rank 32)", "BSRNN", "checkpoints/se2svs/bsrnn_lora_r32/epoch=544-sdr=9.05.ckpt"),
    ModelSpec("LoRA-BSRNNSVS (rank 16)", "BSRNN", "checkpoints/se2svs/bsrnn_lora_r16/epoch=503-sdr=8.94.ckpt"),
    ModelSpec("BSRNNSVS (from scratch)", "BSRNN", "checkpoints/se2svs/bsrnn_scratch/epoch=480-sdr=8.25.ckpt"),
    ModelSpec("BSRNNSE  (base)", "BSRNN", "checkpoints/bsrnn_pretrained/bsrnn.ckpt"),
]


def register_legacy_checkpoint_aliases() -> None:
    """Register aliases/safe-globals needed to load legacy checkpoints."""
    sys.modules["sgmse"] = models.sgmse
    sys.modules["sgmse.data_module"] = models.data_module
    sys.modules["sgmse.model"] = models.MSS_model
    sys.modules["sgmse.sdes"] = models.sgmse.sdes
    sys.modules["sgmse.backbones"] = models.sgmse.backbones
    sys.modules["sgmse.util"] = models.sgmse.util

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


def _extract_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        return None
    state_dict = checkpoint_obj.get("state_dict")
    if isinstance(state_dict, dict):
        return state_dict
    model_state_dict = checkpoint_obj.get("model_state_dict")
    if isinstance(model_state_dict, dict):
        return model_state_dict
    if all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
        return checkpoint_obj
    return None


def _extract_dnn_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    dnn_state = {}
    for k, v in state_dict.items():
        if k.startswith("dnn."):
            dnn_state[k[len("dnn."):]] = v
    if not dnn_state:
        # Some checkpoints may already store bare backbone keys.
        dnn_state = state_dict
    return dnn_state


def _sgm_ptflops_input_constructor(input_res):
    """Create dummy inputs for SGM backbone: complex x and scalar time_cond."""
    channels, n_freq, n_time = input_res
    x = torch.randn(1, channels, n_freq, n_time) + 1j * torch.randn(1, channels, n_freq, n_time)
    time_cond = torch.randn(1)
    return {"x": x, "time_cond": time_cond}


def load_sgm_backbone_for_profile(ckpt_path: Path, root_dir: Path):
    """Instantiate SGM backbone directly from checkpoint hparams and dnn weights."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    backbone_name = hparams.get("backbone")
    if backbone_name is None:
        raise RuntimeError(f"Checkpoint missing 'backbone' in hyper_parameters: {ckpt_path}")

    # Some legacy checkpoints store stale LoRA base-checkpoint paths.
    # Normalize to the current repository layout when possible.
    lora_ckpt = hparams.get("lora_pretrained_checkpoint")
    if isinstance(lora_ckpt, str) and lora_ckpt:
        legacy = "./sgmse_checkpoints/ears_wham.ckpt"
        if lora_ckpt == legacy:
            hparams["lora_pretrained_checkpoint"] = str(root_dir / "checkpoints/sgmse_pretrained/ears_wham.ckpt")
        else:
            p = Path(lora_ckpt)
            if not p.exists():
                candidate = root_dir / lora_ckpt.lstrip("./")
                if candidate.exists():
                    hparams["lora_pretrained_checkpoint"] = str(candidate)

    dnn_cls = BackboneRegistry.get_by_name(backbone_name)

    # Backbones accept **kwargs and ignore unrelated entries.
    dnn = dnn_cls(**hparams)

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise RuntimeError(f"Could not extract state_dict from checkpoint: {ckpt_path}")
    dnn_state = _extract_dnn_state_dict(state_dict)
    dnn.load_state_dict(dnn_state, strict=False)
    dnn.eval()

    return dnn, hparams


def profile_sgm_with_ptflops(dnn, hparams: Dict) -> Tuple[float, float]:
    """Return (MACs, params) from ptflops for SGM backbone."""
    from ptflops import get_model_complexity_info

    # Derive representative STFT-like dimensions from training hparams.
    # Use same channel count as training script: concat of (x, y) -> 2 channels.
    n_fft = int(hparams.get("n_fft", 1534))
    sr = int(hparams.get("sr", 48000))
    hop = int(hparams.get("hop_length", 384))
    duration = float(hparams.get("duration", 5.0))

    n_freq = n_fft // 2 + 1
    n_time_raw = int((duration * sr) / hop)
    n_time = ((max(1, n_time_raw) + 63) // 64) * 64

    macs, params = get_model_complexity_info(
        dnn,
        input_res=(2, n_freq, n_time),
        input_constructor=_sgm_ptflops_input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return float(macs), float(params)


def load_bsrnn_model_for_profile(ckpt_path: Path, root_dir: Path) -> BSRNNSVSModel:
    """Load BSRNNSVSModel in the same robust style as inference code."""

    class InferenceDataModule:
        def __init__(self, *args, **kwargs):
            pass

    checkpoint = BSRNNSVSModel._load_checkpoint_raw(str(ckpt_path))

    # Some legacy checkpoints store stale pretrained_ckpt hparams such as
    # './bsrnn_checkpoint/bsrnn.ckpt'. Override to current repo layout.
    pretrained_override = None
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    hp_pretrained = hparams.get("pretrained_ckpt") if isinstance(hparams, dict) else None
    if isinstance(hp_pretrained, str) and hp_pretrained:
        legacy_candidates = {
            "./bsrnn_checkpoint/bsrnn.ckpt",
            "bsrnn_checkpoint/bsrnn.ckpt",
            "./checkpoints/bsrnn_checkpoint/bsrnn.ckpt",
            "checkpoints/bsrnn_checkpoint/bsrnn.ckpt",
        }
        if hp_pretrained in legacy_candidates:
            pretrained_override = str(root_dir / "checkpoints/bsrnn_pretrained/bsrnn.ckpt")
        else:
            p = Path(hp_pretrained)
            if not p.exists():
                candidate = root_dir / hp_pretrained.lstrip("./")
                if candidate.exists():
                    pretrained_override = str(candidate)

    def _extract_state_dict_for_probe(checkpoint_obj):
        if not isinstance(checkpoint_obj, dict):
            return None
        state_dict = checkpoint_obj.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        model_state_dict = checkpoint_obj.get("model_state_dict")
        if isinstance(model_state_dict, dict):
            return model_state_dict
        if all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return checkpoint_obj
        return None

    state_dict = _extract_state_dict_for_probe(checkpoint)
    use_pretrained_loader = isinstance(state_dict, dict) and any(
        k.startswith(("se_model.", "model.se_model.", "module.se_model."))
        for k in state_dict.keys()
    )

    base_dir = str(ckpt_path.parent)
    if use_pretrained_loader:
        model = BSRNNSVSModel(
            data_module_cls=InferenceDataModule,
            base_dir=base_dir,
            pretrained_ckpt=str(ckpt_path),
            finetune_mode="none",
            nolog=True,
        )
    else:
        model = BSRNNSVSModel.load_from_checkpoint(
            str(ckpt_path),
            map_location="cpu",
            data_module_cls=InferenceDataModule,
            base_dir=base_dir,
            pretrained_ckpt=pretrained_override,
            strict=False,
        )

    model.eval()
    return model


def _bsrnn_ptflops_input_constructor(input_res):
    """Create dummy args for BSRNN_SE.forward(speech_mix, speech_lengths, fs)."""
    batch, time_len = input_res
    speech_mix = torch.randn(batch, time_len)
    speech_lengths = torch.full((batch,), time_len, dtype=torch.long)
    return {"speech_mix": speech_mix, "speech_lengths": speech_lengths, "fs": 48000}


def profile_bsrnn_with_ptflops(model: BSRNNSVSModel, sample_rate: int = 48000, seconds: float = 1.0) -> Tuple[float, float]:
    """Return (MACs, params) from ptflops for BSRNN backbone."""
    from ptflops import get_model_complexity_info

    model.eval()
    time_len = int(sample_rate * seconds)

    macs, params = get_model_complexity_info(
        model.dnn,
        input_res=(1, time_len),
        input_constructor=_bsrnn_ptflops_input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return float(macs), float(params)


def build_ptflops_complexity_table(root_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        ckpt_path = root_dir / spec.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for model '{spec.display_name}': {ckpt_path}")

        print(f"Processing {spec.display_name}...")

        if spec.family == "SGM":
            dnn, hparams = load_sgm_backbone_for_profile(ckpt_path, root_dir)
            macs, _ = profile_sgm_with_ptflops(dnn, hparams)
            profile_seconds = float(hparams.get("duration", 5.0))
            # Count all parameters (frozen base + trainable LoRA adapters).
            # ptflops only counts requires_grad=True params, which misses the
            # frozen base model in LoRA models.
            params = sum(p.numel() for p in dnn.parameters())
            trainable_params = sum(p.numel() for p in dnn.parameters() if p.requires_grad)
        else:
            model = load_bsrnn_model_for_profile(ckpt_path, root_dir)
            macs, _ = profile_bsrnn_with_ptflops(model)
            profile_seconds = 1.0
            # Same fix: count all params in dnn, not just trainable ones.
            params = sum(p.numel() for p in model.dnn.parameters())
            trainable_params = sum(p.numel() for p in model.dnn.parameters() if p.requires_grad)

        rows.append(
            {
                "model": spec.display_name,
                "family": spec.family,
                "checkpoint": spec.checkpoint,
                "ptflops_macs": macs,
                "ptflops_gmacs": macs / 1e9,
                "profile_seconds": profile_seconds,
                "ptflops_gmacs_per_second": (macs / 1e9) / profile_seconds,
                "params": params,
                "params_millions": params / 1e6,
                "trainable_params": trainable_params,
                "trainable_params_millions": trainable_params / 1e6,
            }
        )

    return pd.DataFrame(rows)


def save_complexity_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """Save a human-readable complexity summary table as CSV."""
    summary = df[["model", "params_millions", "trainable_params_millions", "ptflops_gmacs_per_second"]].copy()
    summary = summary.rename(columns={
        "params_millions": "total_params_M",
        "trainable_params_millions": "trainable_params_M",
        "ptflops_gmacs_per_second": "gmacs_per_s",
    })
    summary = summary.sort_values("total_params_M").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, float_format="%.2f")
    # Also print to console
    print("\nModel Complexity Summary:")
    print(summary.to_string(index=False))




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ptflops complexity for models and export to CSV")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory")
    parser.add_argument(
        "--output-complexity-csv",
        type=Path,
        default=Path("aggregated_results/model_complexity_summary.csv"),
        help="Output CSV path for complexity summary (params, trainable params, MACs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_legacy_checkpoint_aliases()

    root_dir = args.root_dir.resolve()
    
    print("Building ptflops complexity table...")
    complexity_df = build_ptflops_complexity_table(root_dir)

    # Save complexity summary
    out_complexity_csv = (root_dir / args.output_complexity_csv).resolve()
    save_complexity_summary_table(complexity_df, out_complexity_csv)
    print(f"\nSaved complexity summary: {out_complexity_csv}")


if __name__ == "__main__":
    main()
