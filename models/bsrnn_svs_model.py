import argparse
import sys
import types
from typing import Dict, Iterable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import wandb
import soundfile
import os
import torchaudio.functional as taF
from torchmetrics.audio.sdr import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
)
from auraloss.freq import MultiResolutionSTFTLoss

from models.bsrnn.bsrnn import BSRNN_SE


def _resample_np_1d(audio_np: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D numpy audio with torchaudio."""
    if orig_sr == target_sr:
        return audio_np
    audio_t = torch.from_numpy(audio_np.astype(np.float32, copy=False))
    return taF.resample(audio_t, orig_sr, target_sr).cpu().numpy()


class BSRNNSVSModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sr", type=int, default=48000, help="Audio sample rate.")
        parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
        parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
        parser.add_argument(
            "--si-sdr-loss-weight",
            type=float,
            default=0.8,
            help="Weight of negative SI-SDR term in waveform loss.",
        )
        parser.add_argument(
            "--bsrnn-num-channel",
            type=int,
            default=192,
            help="Number of channels in BSRNN separator blocks.",
        )
        parser.add_argument(
            "--bsrnn-num-layer",
            type=int,
            default=6,
            help="Number of BSRNN separator layers.",
        )
        parser.add_argument(
            "--pretrained-ckpt",
            type=str,
            default=None,
            help="Optional BSRNN checkpoint to initialize model weights.",
        )
        parser.add_argument(
            "--finetune-mode",
            type=str,
            choices=("none", "naive", "lora"),
            default="none",
            help="Select finetune mode. 'none' is intended for from-scratch training.",
        )
        parser.add_argument(
            "--lora-r",
            type=int,
            default=8,
            help="LoRA rank.",
        )
        parser.add_argument(
            "--lora-alpha",
            type=int,
            default=16,
            help="LoRA alpha scaling.",
        )
        parser.add_argument(
            "--lora-dropout",
            type=float,
            default=0.0,
            help="LoRA dropout.",
        )
        parser.add_argument(
            "--lora-bias",
            type=str,
            default="none",
            help="LoRA bias mode: none, all, lora_only.",
        )
        parser.add_argument(
            "--lora-target-modules",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Optional PEFT target module leaf names. If omitted, script discovers "
                "reasonable defaults from Linear/Conv modules."
            ),
        )
        parser.add_argument(
            "--lora-modules-to-save",
            type=str,
            nargs="*",
            default=None,
            help="Optional extra trainable modules to save with LoRA adapters.",
        )
        parser.add_argument(
            "--print-lora-candidates-only",
            action="store_true",
            default=False,
            help="Print discovered target names and exit before training.",
        )
        parser.add_argument(
            "--audio-log-files",
            nargs="+",
            type=int,
            default=None,
            help="List of validation file ids to log for deterministic audio monitoring.",
        )
        return parser

    def __init__(
        self,
        data_module_cls,
        sr=48000,
        lr=1e-4,
        weight_decay=0.0,
        si_sdr_loss_weight=0.8,
        bsrnn_num_channel=192,
        bsrnn_num_layer=6,
        pretrained_ckpt=None,
        finetune_mode="none",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_bias="none",
        lora_target_modules=None,
        lora_modules_to_save=None,
        print_lora_candidates_only=False,
        audio_log_files=None,
        audio_log_interval=5,
        nolog=False,
        **kwargs,
    ):
        super().__init__()

        self.sr = sr
        self.lr = lr
        self.weight_decay = weight_decay
        self.si_sdr_loss_weight = si_sdr_loss_weight
        self.finetune_mode = finetune_mode
        self.valid_audio_log_files = audio_log_files
        self.audio_log_interval = audio_log_interval
        self.nolog = nolog

        if pretrained_ckpt:
            inferred_num_channel, inferred_num_layer = self.infer_bsrnn_arch_from_checkpoint(pretrained_ckpt)
            inferred_epoch = self.infer_checkpoint_epoch(pretrained_ckpt)
            if inferred_epoch is not None:
                print(f"Pretrained SE checkpoint was trained for {inferred_epoch} epochs")
            if inferred_num_channel is not None and inferred_num_channel != bsrnn_num_channel:
                print(
                    "Overriding bsrnn_num_channel from "
                    f"{bsrnn_num_channel} to {inferred_num_channel} based on checkpoint architecture"
                )
                bsrnn_num_channel = inferred_num_channel
            if inferred_num_layer is not None and inferred_num_layer != bsrnn_num_layer:
                print(
                    "Overriding bsrnn_num_layer from "
                    f"{bsrnn_num_layer} to {inferred_num_layer} based on checkpoint architecture"
                )
                bsrnn_num_layer = inferred_num_layer

        self.dnn = BSRNN_SE(num_channel=bsrnn_num_channel, num_layer=bsrnn_num_layer)

        if pretrained_ckpt:
            self._load_pretrained_bsrnn_checkpoint(pretrained_ckpt)

        self.discovered_lora_targets = self.discover_lora_target_modules(self.dnn)

        if print_lora_candidates_only:
            print("Discovered BSRNN LoRA target candidates:")
            for name in self.discovered_lora_targets:
                print(f"  - {name}")

        if finetune_mode == "lora":
            if print_lora_candidates_only:
                raise SystemExit(0)
            self._apply_lora(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
                requested_targets=lora_target_modules,
                modules_to_save=lora_modules_to_save,
            )

        # Initialize MERT embedding model for validation metrics.
        # Keep this lazy so inference can run even if optional deps are unavailable.
        self.embedding_model = None
        try:
            from gensvs import get_all_models  # local import to avoid hard import-time dependency

            models = {m.name: m for m in get_all_models()}
            embedding_model = models["MERT-v1-95M"]
            embedding_model.load_model()
            for param in embedding_model.model.parameters():
                param.requires_grad = False
            self.embedding_model = embedding_model
        except Exception as exc:
            print(f"Warning: failed to initialize MERT embedding model: {exc}")

        self.data_module = data_module_cls(sr=sr, **kwargs, gpu=torch.cuda.is_available())
        self.save_hyperparameters(ignore=["data_module_cls"])
        
        # Initialize metric accumulators for epoch-level logging
        self.val_multi_res_loss = []
        self.val_mert_mse = []

    @staticmethod
    def _is_candidate_lora_module(module: torch.nn.Module) -> bool:
        return isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))

    @classmethod
    def discover_lora_target_modules(cls, model: torch.nn.Module) -> List[str]:
        discovered = []
        for full_name, module in model.named_modules():
            if not full_name:
                continue
            if cls._is_candidate_lora_module(module):
                discovered.append(full_name)

        # Preserve order while deduplicating.
        seen = set()
        ordered = []
        for name in discovered:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _resolve_lora_targets(self, requested_targets: Optional[Iterable[str]]) -> List[str]:
        if requested_targets:
            return list(requested_targets)

        # Prefer projection-style layers if present, otherwise all discovered names.
        priority_substrings = ("linear", "proj", "out", "in")
        preferred = [
            n for n in self.discovered_lora_targets if any(tag in n.lower() for tag in priority_substrings)
        ]
        if preferred:
            return preferred
        return self.discovered_lora_targets

    @staticmethod
    def _extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        raise ValueError("Could not find a valid state_dict inside checkpoint")

    @staticmethod
    def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
        return state_dict

    @staticmethod
    def _load_checkpoint_raw(ckpt_path: str) -> Dict:
        BSRNNSVSModel._register_legacy_baseline_config_safe_global()

        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu")

    @staticmethod
    def _register_legacy_baseline_config_safe_global() -> None:
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

    @classmethod
    def _normalize_checkpoint_state_dict(cls, checkpoint: Dict) -> Dict[str, torch.Tensor]:
        state_dict = cls._extract_state_dict(checkpoint)

        # Legacy checkpoints may be nested (e.g., se_model.*, model.dnn.*, or module.model.dnn.*).
        # Repeatedly strip known wrapper prefixes until keys stabilize.
        for _ in range(4):
            previous_keys = tuple(state_dict.keys())
            for prefix in ("se_model.", "model.", "module.", "dnn."):
                state_dict = cls._strip_prefix_if_present(state_dict, prefix)
            if tuple(state_dict.keys()) == previous_keys:
                break
        return state_dict

    @classmethod
    def infer_bsrnn_arch_from_checkpoint(cls, ckpt_path: str) -> Tuple[Optional[int], Optional[int]]:
        try:
            checkpoint = cls._load_checkpoint_raw(ckpt_path)
            state_dict = cls._normalize_checkpoint_state_dict(checkpoint)
        except Exception:
            return None, None

        num_channel = None
        base_key = "bsrnn.bsrnn.rnn_time.0.weight_ih_l0"
        if base_key in state_dict and state_dict[base_key].dim() >= 2:
            num_channel = int(state_dict[base_key].shape[1])

        layer_ids = set()
        prefix = "bsrnn.bsrnn.rnn_time."
        suffix = ".weight_ih_l0"
        for key in state_dict.keys():
            if key.startswith(prefix) and key.endswith(suffix):
                layer_token = key[len(prefix) : -len(suffix)]
                if layer_token.isdigit():
                    layer_ids.add(int(layer_token))
        num_layer = max(layer_ids) + 1 if layer_ids else None

        return num_channel, num_layer

    @classmethod
    def infer_checkpoint_epoch(cls, ckpt_path: str) -> Optional[int]:
        try:
            checkpoint = cls._load_checkpoint_raw(ckpt_path)
        except Exception:
            return None

        epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        if epoch is None:
            return None

        try:
            return int(epoch)
        except (TypeError, ValueError):
            return None

    def _load_pretrained_bsrnn_checkpoint(self, ckpt_path: str) -> None:
        checkpoint = self._load_checkpoint_raw(ckpt_path)
        state_dict = self._normalize_checkpoint_state_dict(checkpoint)

        missing, unexpected = self.dnn.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained BSRNN checkpoint from: {ckpt_path}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    def _apply_lora(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_bias: str,
        requested_targets: Optional[Iterable[str]],
        modules_to_save: Optional[Iterable[str]],
    ) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("PEFT is required for LoRA finetuning. Install with `pip install peft`.") from exc

        target_modules = self._resolve_lora_targets(requested_targets)
        if not target_modules:
            raise ValueError("No candidate LoRA target modules were found for BSRNN")

        print("Applying LoRA to BSRNN with target modules:")
        for name in target_modules:
            print(f"  - {name}")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            modules_to_save=list(modules_to_save) if modules_to_save else None,
        )

        self.dnn = get_peft_model(self.dnn, lora_config)
        self.dnn.print_trainable_parameters()

    def _flatten_batched_waveform(self, wav: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)

        if wav.dim() != 3:
            raise ValueError(f"Expected waveform with shape [B, C, T] or [B, T], got {tuple(wav.shape)}")

        batch, channels, time = wav.shape
        flat = wav.reshape(batch * channels, time)
        return flat, batch, channels

    def separate(self, mixture_waveform: torch.Tensor) -> torch.Tensor:
        mixture_flat, batch, channels = self._flatten_batched_waveform(mixture_waveform)
        lengths = torch.full(
            (mixture_flat.size(0),),
            mixture_flat.size(-1),
            dtype=torch.long,
            device=mixture_flat.device,
        )

        estimate_flat, _ = self.dnn(mixture_flat, lengths, fs=self.sr)
        if estimate_flat.dim() == 1:
            estimate_flat = estimate_flat.unsqueeze(0)

        return estimate_flat.reshape(batch, channels, -1)

    def _compute_losses(self, estimate: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        min_len = min(estimate.size(-1), target.size(-1))
        estimate = estimate[..., :min_len]
        target = target[..., :min_len]

        l1 = F.l1_loss(estimate, target)
        est_flat = estimate.reshape(-1, min_len)
        tgt_flat = target.reshape(-1, min_len)
        si_sdr = scale_invariant_signal_distortion_ratio(est_flat, tgt_flat).mean()
        loss = (1.0 - self.si_sdr_loss_weight) * l1 - self.si_sdr_loss_weight * si_sdr
        return {"loss": loss, "l1": l1, "si_sdr": si_sdr}

    def _apply_mixture_least_squares_gain(
        self,
        mixture_waveform: torch.Tensor,
        estimate_waveform: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Scale estimate by least-squares gain that minimizes ||mixture - g*estimate||^2."""
        min_len = min(mixture_waveform.size(-1), estimate_waveform.size(-1))
        mix = mixture_waveform[..., :min_len]
        est = estimate_waveform[..., :min_len]

        numerator = (mix * est).sum(dim=(-1, -2), keepdim=True)
        denominator = (est * est).sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
        gain = numerator / denominator
        return estimate_waveform * gain

    def _separate_with_noisy_norm_and_peak_output(self, mixture_waveform: torch.Tensor) -> torch.Tensor:
        """Apply SGMSE-like noisy normalization, then peak-normalize estimate before de-normalization."""
        eps = 1e-8
        normfac = mixture_waveform.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(eps)
        mixture_norm = mixture_waveform / normfac

        estimate_norm = self.separate(mixture_norm)

        peak = estimate_norm.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(eps)
        estimate_peak_norm = estimate_norm / peak
        estimate = estimate_peak_norm * normfac
        return self._apply_mixture_least_squares_gain(mixture_waveform, estimate, eps=eps)

    def _log_fixed_validation_audio_samples(self) -> None:
        if self.nolog or self.valid_audio_log_files is None or wandb.run is None:
            return
        if self.current_epoch % self.audio_log_interval != 0:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        valid_dataset = self.data_module.valid_set.dataset
        for requested_file_id in self.valid_audio_log_files:
            sample = None
            resolved_file_id = str(requested_file_id)

            # Match SGMSE behavior for custom validation sets: treat values as filename IDs.
            if hasattr(valid_dataset, "file_ids"):
                requested_str = str(requested_file_id)
                if requested_str not in valid_dataset.file_ids:
                    print(
                        f"Warning: requested audio_log_file id '{requested_str}' was not found in validation file_ids"
                    )
                    continue
                sample_idx = valid_dataset.file_ids.index(requested_str)
                sample = valid_dataset[sample_idx]
                resolved_file_id = requested_str
            else:
                # Fallback for non-custom datasets: treat requested values as dataset indices.
                sample_idx = int(requested_file_id)
                if sample_idx < 0 or sample_idx >= len(valid_dataset):
                    print(f"Warning: requested audio_log_file index {sample_idx} is out of range")
                    continue
                sample = valid_dataset[sample_idx]

            if len(sample) == 4:
                y, _, _, sample_file_id = sample
                resolved_file_id = str(sample_file_id)
            else:
                y, _, _ = sample

            mix = y.to(self.device)
            if mix.dim() == 1:
                mix = mix.unsqueeze(0).unsqueeze(0)
            elif mix.dim() == 2:
                mix = mix.unsqueeze(0)
            else:
                continue

            with torch.inference_mode():
                estimate = self._separate_with_noisy_norm_and_peak_output(mix).detach().cpu().squeeze(0)

            if estimate.dim() == 1:
                audio_np = estimate.numpy()
            else:
                audio_np = estimate.transpose(0, 1).numpy()

            log_sr = self.sr
            fs_original = getattr(valid_dataset, "fs_original", None)
            if fs_original is not None and fs_original != self.sr:
                if audio_np.ndim == 1:
                    audio_np = _resample_np_1d(audio_np, orig_sr=self.sr, target_sr=fs_original)
                else:
                    audio_np = np.stack(
                        [
                            _resample_np_1d(audio_np[:, ch], orig_sr=self.sr, target_sr=fs_original)
                            for ch in range(audio_np.shape[1])
                        ],
                        axis=1,
                    )
                log_sr = fs_original

            wandb.log(
                {
                    f"file #{resolved_file_id}": wandb.Audio(
                        audio_np,
                        sample_rate=log_sr,
                        caption="separated",
                    )
                },
                step=self.global_step,
            )


    def training_step(self, batch, batch_idx):
        _, _, target_wav, mix_wav = batch
        estimate = self.separate(mix_wav)
        estimate = self._apply_mixture_least_squares_gain(mix_wav, estimate)
        losses = self._compute_losses(estimate, target_wav)

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        _, _, target_wav, mix_wav = batch

        if batch_idx == 0:
            self._log_fixed_validation_audio_samples()

        estimate = self._separate_with_noisy_norm_and_peak_output(mix_wav)
        losses = self._compute_losses(estimate, target_wav)

        min_len = min(estimate.size(-1), target_wav.size(-1))
        est_flat = estimate[..., :min_len].reshape(-1, min_len)
        tgt_flat = target_wav[..., :min_len].reshape(-1, min_len)
        sdr = signal_distortion_ratio(est_flat, tgt_flat).mean()

        self.log("valid_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("si_sdr", losses["si_sdr"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("sdr", sdr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Ensure embedding model is on correct device and in eval mode
        if hasattr(self.embedding_model, 'model'):
            if next(self.embedding_model.model.parameters()).device != self.device:
                self.embedding_model.model.to(self.device)
            self.embedding_model.model.eval()
        
        # Compute multi_res_loss for each sample in batch
        multi_res_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024, 2048, 4096],
            win_lengths=[256, 512, 1024, 2048, 4096],
            hop_sizes=[64, 128, 256, 512, 1024],
            sample_rate=self.sr,
            perceptual_weighting=True
        ).forward
        
        batch_multi_res_loss = 0.0
        batch_mert_mse = 0.0
        batch_count = 0
        
        for b in range(estimate.shape[0]):
            # Multi-resolution STFT loss
            est_b = estimate[b:b+1, :min_len].float().to(self.device)
            tgt_b = target_wav[b:b+1, :min_len].float().to(self.device)
            if est_b.dim() == 2:
                est_b = est_b.unsqueeze(0)  # [1, C, T] -> [1, 1, C, T]
            if tgt_b.dim() == 2:
                tgt_b = tgt_b.unsqueeze(0)
            batch_multi_res_loss += multi_res_loss_fn(est_b, tgt_b).item()
            
            # MERT embedding MSE with error handling
            with torch.inference_mode():
                # Convert to numpy for embedding model
                est_np = est_b.squeeze().cpu().numpy() if torch.is_tensor(est_b) else est_b.squeeze()
                tgt_np = tgt_b.squeeze().cpu().numpy() if torch.is_tensor(tgt_b) else tgt_b.squeeze()
                
                # Ensure audio is 1D (squeeze to remove channel dim if needed)
                if est_np.ndim > 1:
                    est_np = est_np.squeeze()
                if tgt_np.ndim > 1:
                    tgt_np = tgt_np.squeeze()

                # Explicitly resample to embedding model sample rate before MERT extraction.
                emb_sr = int(getattr(self.embedding_model, "sr", self.sr))
                if self.sr != emb_sr:
                    est_np = _resample_np_1d(est_np, orig_sr=self.sr, target_sr=emb_sr)
                    tgt_np = _resample_np_1d(tgt_np, orig_sr=self.sr, target_sr=emb_sr)
                
                # Get embeddings
                est_emb = self.embedding_model._get_embedding(est_np)
                tgt_emb = self.embedding_model._get_embedding(tgt_np)
                
                # Convert to numpy if tensors
                if torch.is_tensor(est_emb):
                    est_emb = est_emb.cpu().numpy()
                if torch.is_tensor(tgt_emb):
                    tgt_emb = tgt_emb.cpu().numpy()
                
                # Compute MSE with NaN safety
                mse_val = float(np.mean((est_emb - tgt_emb) ** 2))
                if not np.isnan(mse_val) and np.isfinite(mse_val):
                    batch_mert_mse += mse_val
                    batch_count += 1

        
        # Average over batch (only count successful computations)
        batch_size = max(estimate.shape[0], 1)
        avg_multi_res_loss = batch_multi_res_loss / batch_size
        avg_mert_mse = batch_mert_mse / max(batch_count, 1)
        
        # Accumulate for epoch-level aggregation
        self.val_multi_res_loss.append(avg_multi_res_loss)
        self.val_mert_mse.append(avg_mert_mse)

    def on_validation_epoch_end(self):
        """Log epoch-level metrics after validation finishes."""
        if len(self.val_multi_res_loss) > 0:
            epoch_multi_res_loss = np.mean(self.val_multi_res_loss)
            epoch_mert_mse = np.mean(self.val_mert_mse)
            
            # Filter out NaN values before logging
            valid_multi_res = [x for x in self.val_multi_res_loss if np.isfinite(x)]
            valid_mert = [x for x in self.val_mert_mse if np.isfinite(x)]
            
            if valid_multi_res:
                epoch_multi_res_loss = np.mean(valid_multi_res)
                self.log("multi_res_loss", epoch_multi_res_loss, on_step=False, on_epoch=True, sync_dist=True)
            
            if valid_mert:
                epoch_mert_mse = np.mean(valid_mert)
                self.log("mert_mse", epoch_mert_mse, on_step=False, on_epoch=True, sync_dist=True)
            
            # Reset accumulators for next epoch
            self.val_multi_res_loss = []
            self.val_mert_mse = []

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloadernv()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def on_train_epoch_start(self):
        # Keep datamodule setup explicit for parity with existing scripts.
        if not hasattr(self.data_module, "train_set"):
            self.data_module.setup(stage="fit")
        # Clear metric accumulators at start of epoch
        self.val_multi_res_loss = []
        self.val_mert_mse = []


# Keep parser helper style aligned with existing scripts.
def get_argparse_groups(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Dict[str, argparse.Namespace]:
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups
