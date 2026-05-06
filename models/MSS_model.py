import time
from math import ceil
import warnings
import re

import torch
import pytorch_lightning as pl
import torch.distributed as dist
import numpy as np
import tqdm
import soundfile
import os
import wandb

from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
from torchmetrics.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from auraloss.freq import MultiResolutionSTFTLoss
from models.sgmse import sampling
from models.sgmse.sdes import SDERegistry
from models.sgmse.backbones import BackboneRegistry
from models.sgmse.util.other import pad_spec, si_sdr
from gensvs import get_all_models


def _remap_nin_to_linear_keys(state_dict, target_is_peft_wrapped=False):
    """Remap NIN parameter keys to NINLinear keys for checkpoint compatibility.

    Old NIN layout (raw Parameters):
        ...NIN_X.W   shape (in, out)
        ...NIN_X.b   shape (out,)

    New NINLinear layout (nn.Linear submodule):
        ...NIN_X.nin_linear.weight  shape (out, in)  <- transposed!
        ...NIN_X.nin_linear.bias    shape (out,)
    
    PEFT-wrapped NINLinear layout (for LoRA models):
        ...NIN_X.nin_linear.base_layer.weight  shape (out, in)
        ...NIN_X.nin_linear.base_layer.bias    shape (out,)
    
    Args:
        state_dict: Checkpoint state dict to remap
        target_is_peft_wrapped: If True, adds .base_layer for PEFT compatibility
    """
    new_sd = {}
    for k, v in state_dict.items():
        # Case 1: Old NIN format (NIN_X.W / NIN_X.b) -> NINLinear format
        m = re.match(r'^(.*NIN_\d+)\.W$', k)
        if m:
            base_key = m.group(1) + '.nin_linear'
            if target_is_peft_wrapped:
                new_sd[base_key + '.base_layer.weight'] = v.T
            else:
                new_sd[base_key + '.weight'] = v.T
            continue
        m = re.match(r'^(.*NIN_\d+)\.b$', k)
        if m:
            base_key = m.group(1) + '.nin_linear'
            if target_is_peft_wrapped:
                new_sd[base_key + '.base_layer.bias'] = v
            else:
                new_sd[base_key + '.bias'] = v
            continue
        
        # Case 2: NINLinear format -> PEFT base_layer format
        # (for checkpoints that already have NINLinear but need PEFT wrapping)
        if target_is_peft_wrapped:
            m = re.match(r'^(.*NIN_\d+\.nin_linear)\.weight$', k)
            if m:
                new_sd[m.group(1) + '.base_layer.weight'] = v
                continue
            m = re.match(r'^(.*NIN_\d+\.nin_linear)\.bias$', k)
            if m:
                new_sd[m.group(1) + '.base_layer.bias'] = v
                continue
        
        # Default: keep key as-is
        new_sd[k] = v
    return new_sd


#TODO: when logging metrics with multiple gpu training => metrics need to be logged globally! => use method validation step end (see: https://stackoverflow.com/questions/66854148/proper-way-to-log-things-when-using-pytorch-lightning-ddp)
class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema-decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t-eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num-eval-files", type=int, default=20, help="Number of files for musical source separation enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss-type", type=str, default="score_matching", help="The type of loss function to use.")
        parser.add_argument("--loss-weighting", type=str, default="sigma^2", help="The weighting of the loss function.")
        parser.add_argument("--network-scaling", type=str, default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c-in", type=str, default="1", help="The input scaling for x.")
        parser.add_argument("--c-out", type=str, default="1", help="The output scaling.")
        parser.add_argument("--c-skip", type=str, default="0", help="The skip connection scaling.")
        parser.add_argument("--sigma-data", type=float, default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1-weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--valid-sep-dir", type=str, default=None, help="The directory in which separated validation examples are stored.")
        parser.add_argument("--audio-log-files", nargs='+', type=int, default=None, help="List of audio ids of files to log during training.")
        parser.add_argument("--target-is-accompaniment", action='store_true', default=False, help="Use the accompaniment as target data to diffuse into.")
        parser.add_argument("--sr", type=int, default=48000, help="The sample rate of the audio files.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, valid_sep_dir=None, audio_log_files=None, sr=48000, data_module_cls=None, target_is_accompaniment=False, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        
        # Initialize Backbone DNN
        kwargs['sr']=sr
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        
        # DEBUG: Verify model structure after initialization
        if hasattr(self.dnn, 'base_model'):
            print(f"\n  DEBUG (MSS_model init) - LoRA model detected")
            # Check if NIN swap happened by looking for nin_linear in module names
            dnn_modules = dict(self.dnn.named_modules())
            nin_modules = {k: type(v).__name__ for k, v in dnn_modules.items() if 'NIN' in k}
            ninlinear_count = sum(1 for v in nin_modules.values() if 'NINLinear' in v)
            plain_nin_count = sum(1 for v in nin_modules.values() if v == 'NIN')
            lora_count = sum(1 for k in dnn_modules.keys() if 'lora' in k.lower())
            
            print(f"  DEBUG - NINLinear modules: {ninlinear_count}")
            print(f"  DEBUG - Plain NIN modules: {plain_nin_count}")
            print(f"  DEBUG - LoRA modules: {lora_count}")
            
            if ninlinear_count > 0:
                print(f"  ✓ NIN → NINLinear swap successful ({ninlinear_count} modules)")
            if lora_count > 0:
                print(f"  ✓ LoRA layers present ({lora_count} modules)")
            
            # Show a few sample module names
            sample_nin = [k for k in nin_modules.keys()][:3]
            if sample_nin:
                print(f"  DEBUG - Sample NIN module names:")
                for name in sample_nin:
                    print(f"    {name}: {nin_modules[name]}")
        
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        
        # Initialize embedding model for MSE evaluation BEFORE EMA
        # Set as a buffer so its parameters aren't tracked in self.parameters()
        models = {m.name: m for m in get_all_models()}
        embedding_model = models["MERT-v1-95M"]
        embedding_model.load_model()
        # Freeze embedding model and prevent parameter tracking
        for param in embedding_model.model.parameters():
            param.requires_grad = False
        self.embedding_model = embedding_model
        print("Initialized MERT-v1-95M for embedding MSE evaluation (frozen)")
        
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        # EMA tracks only trainable dnn parameters (LoRA params only if using LoRA, all dnn params otherwise)
        self.ema = ExponentialMovingAverage(self._ema_params(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.nolog = kwargs.get('nolog', False)
        self.audio_log_interv = kwargs.get('audio_log_interval', 1)
        self.sdr = signal_distortion_ratio
        self.si_sdr = scale_invariant_signal_distortion_ratio
        # multi_res_loss = MultiResolutionSTFTLoss(fft_sizes=[256, 512, 1024, 2048, 4096],
        #                                      win_lengths=[256, 512, 1024, 2048, 4096],
        #                                      hop_sizes=[64, 128,  256, 512, 1024],
        #                                      sample_rate=sr, 
        #                                      perceptual_weighting=True)
        # self.multi_res_loss = multi_res_loss.forward
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.valid_sep_dir = valid_sep_dir
        self.valid_audio_log_files = audio_log_files
        self.sr = sr
        self.accomp_target = target_is_accompaniment
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.ckpt = None
        self.valid_out_foldername = kwargs.get('valid_audio_foldername', None)

    def _ema_params(self):
        """Parameters tracked by EMA.
        For LoRA models: only the LoRA adapter params (requires_grad=True).
        For full models: ALL dnn params (matching the original sgmse convention)."""
        is_lora = hasattr(self.dnn, 'base_model')
        if is_lora:
            return [p for p in self.dnn.parameters() if p.requires_grad]
        else:
            return list(self.dnn.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """Override to prevent issues when loading LoRA checkpoints."""
        if gradient_clip_val is not None:
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self._ema_params())

    def _load_state_dict_legacy(self, state_dict, strict=True):
        """
        [DISABLED] Override load_state_dict to use strict=False when loading into LoRA model.
        No longer needed: on_load_checkpoint now patches checkpoint['state_dict'] directly.
        """
        # Check if we're in the middle of loading full model into LoRA
        print(f"DEBUG load_state_dict: _loading_full_to_lora={self._loading_full_to_lora}, strict={strict}")
        if self._loading_full_to_lora:
            # Use strict=False to allow missing/unexpected keys
            print("→ Loading with strict=False (LoRA model from full checkpoint)")
            result = super().load_state_dict(state_dict, strict=False)
            # Reset flag after loading
            self._loading_full_to_lora = False
            return result
        else:
            # Normal loading
            return super().load_state_dict(state_dict, strict=strict)

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def _on_load_checkpoint_legacy(self, checkpoint):
        ema = checkpoint.get('ema', None)

        if not(self.nolog):
            #set completed epoch for logging
            try:
                self.trainer.fit_loop.epoch_progress.current.completed=checkpoint['epoch']
            except RuntimeError as e:
                print('No Trainer found, current epoch was not set.')
        
        # ============================================================================
        # CRITICAL: Load pretrained weights into LoRA base model
        # ============================================================================
        # When using LoRA, the model structure changes:
        #   - Full model:  self.dnn.layer.weight
        #   - LoRA model:  self.dnn.base_model.model.layer.weight (PEFT wrapping)
        # 
        # If we're loading a full model checkpoint into a LoRA model, we need to:
        # 1. Detect the mismatch in structure
        # 2. Extract the base model weights from the checkpoint
        # 3. Load them into self.dnn.base_model (not self.dnn directly)
        # ============================================================================
        
        is_lora_model = hasattr(self.dnn, 'base_model')
        
        if is_lora_model and 'state_dict' in checkpoint:
            # Get checkpoint keys to detect if it's from a full (non-LoRA) model
            checkpoint_keys = list(checkpoint['state_dict'].keys())
            dnn_keys = [k for k in checkpoint_keys if k.startswith('dnn.')]
            
            # Check if checkpoint is from a full model (no 'base_model' in keys)
            is_full_model_ckpt = any(k.startswith('dnn.') for k in dnn_keys) and \
                                 not any('base_model' in k for k in dnn_keys)
            
            if is_full_model_ckpt:
                # Set flag to tell load_state_dict to use strict=False
                self._loading_full_to_lora = True
                print(f"DEBUG: Set _loading_full_to_lora = True")
                
                print("\n" + "="*70)
                print("LOADING PRETRAINED WEIGHTS INTO LoRA MODEL")
                print("="*70)
                
                # Extract dnn weights from checkpoint and strip 'dnn.' prefix
                dnn_state_dict = {}
                for key in dnn_keys:
                    # Remove 'dnn.' prefix: 'dnn.layer.weight' -> 'layer.weight'
                    new_key = key[4:]
                    dnn_state_dict[new_key] = checkpoint['state_dict'][key]
                
                # PEFT wraps the model, so we need to load into the wrapped model's base_model
                # The structure is: self.dnn.model (PeftModel) contains the actual base model
                # Try to access it via get_base_model() method or .model attribute
                if hasattr(self.dnn.model, 'get_base_model'):
                    target_model = self.dnn.model.get_base_model()
                elif hasattr(self.dnn.model, 'model'):
                    target_model = self.dnn.model.model
                else:
                    # Fallback to base_model attribute
                    target_model = self.dnn.base_model
                
                print(f"  Loading into: {type(target_model).__name__}")
                
                # Debug: Check a sample key to see structure mismatch
                sample_checkpoint_key = 'all_modules.4.Conv_0.weight'
                if sample_checkpoint_key in dnn_state_dict:
                    print(f"\n  DEBUG: Sample checkpoint key: '{sample_checkpoint_key}'")
                    target_state = target_model.state_dict()
                    # Check if this exact key exists
                    if sample_checkpoint_key in target_state:
                        print(f"  ✓ Key exists in target model")
                    else:
                        print(f"  ✗ Key NOT in target model")
                        # Look for similar keys
                        matching = [k for k in target_state.keys() if 'all_modules.4.Conv_0.weight' in k]
                        if matching:
                            print(f"  Similar keys in target: {matching}")
                        else:
                            # Show first few keys of target model
                            first_keys = list(target_state.keys())[:5]
                            print(f"  First keys in target model: {first_keys}")
                    print()
                
                # Load into the actual base model (strict=False allows for missing LoRA parameters)
                missing_keys, unexpected_keys = target_model.load_state_dict(
                    dnn_state_dict, strict=False
                )
                
                print(f"✓ Loaded {len(dnn_state_dict)} pretrained weights into base model")
                if missing_keys:
                    print(f"  Missing keys: {len(missing_keys)} (expected for LoRA wrappers)")
                if unexpected_keys:
                    print(f"  Warning - Unexpected keys: {unexpected_keys[:5]}")
                
                # Verify only LoRA parameters are trainable
                trainable = sum(p.numel() for p in self.dnn.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.dnn.parameters())
                print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
                
                # Note: We don't delete dnn.* keys from checkpoint['state_dict'] anymore
                # since we're loading manually, not using PyTorch Lightning's ckpt_path
                
                # CRITICAL: Remove ALL optimizer and scheduler states from checkpoint
                # The optimizer state is for the full model's parameters, but LoRA
                # only trains a subset. Loading it would cause parameter count mismatch.
                
                # Check what keys exist in checkpoint
                print(f"  DEBUG - Checkpoint keys: {list(checkpoint.keys())}")
                
                # Remove ALL keys that might contain optimizer states
                keys_to_remove = []
                for key in checkpoint.keys():
                    if 'optimizer' in key.lower() or 'scheduler' in key.lower():
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    print(f"  Removing checkpoint key: {key}")
                    del checkpoint[key]
                
                # Also check state_dict for optimizer keys
                if 'state_dict' in checkpoint:
                    state_keys_to_remove = []
                    for key in checkpoint['state_dict'].keys():
                        if 'optimizer' in key.lower():
                            state_keys_to_remove.append(key)
                    for key in state_keys_to_remove:
                        print(f"  Removing state_dict key: {key}")
                        del checkpoint['state_dict'][key]
                
                # CRITICAL: Modify checkpoint metadata to indicate this is a weights-only load
                # This prevents PyTorch Lightning from trying to restore optimizer state
                # Reset epoch and global step to start fresh training
                if 'epoch' in checkpoint:
                    print(f"  Resetting epoch from {checkpoint['epoch']} to 0 (starting fresh LoRA training)")
                    checkpoint['epoch'] = 0
                if 'global_step' in checkpoint:
                    print(f"  Resetting global_step from {checkpoint['global_step']} to 0")
                    checkpoint['global_step'] = 0
                
                # Remove any connector or loop information that might trigger optimizer loading
                keys_to_clear = ['loops', 'callbacks', 'datamodule_hyper_parameters']
                for key in keys_to_clear:
                    if key in checkpoint:
                        print(f"  Removing {key}")
                        del checkpoint[key]
                
                print("  ✓ Removed dnn.* keys and all optimizer/scheduler states")
                print("  ✓ Reset training state (epoch, global_step) to start fresh")
                print("="*70 + "\n")
            else:
                # Reset flag if not loading full model into LoRA
                self._loading_full_to_lora = False
        else:
            # Reset flag if not LoRA model
            self._loading_full_to_lora = False
        
        # ============================================================================
        # Handle EMA loading with intelligent parameter remapping
        # ============================================================================
        if ema is not None:
            try:
                self.ema.load_state_dict(checkpoint['ema'])
                print("✓ EMA weights loaded successfully")
                
                # CRITICAL: Swap state_dict and collected_params if needed
                # When checkpoint is saved during validation:
                #   - checkpoint['state_dict'] contains EMA weights (what was in model during save)
                #   - checkpoint['ema']['collected_params'] contains training weights (stored by ema.store())
                # We need to swap them so PyTorch Lightning loads training weights into the model
                # (PyTorch Lightning loads state_dict AFTER this hook returns)
                if self.ema.collected_params is not None:
                    print("   ✓ Swapping state_dict (EMA) with collected_params (training weights)")
                    print("      (Checkpoint was saved during validation)")
                    
                    # Build mapping of parameter names to collected_params indices
                    # Only swap dnn parameters (not sde or other model components)
                    dnn_param_names = []
                    for name, param in self.named_parameters():
                        if name.startswith('dnn.'):
                            dnn_param_names.append(name)
                    
                    # Create new collected_params list to store EMA weights (from state_dict)
                    new_collected_params = []
                    
                    # Swap: put training weights in state_dict, EMA weights in collected_params
                    param_idx = 0
                    for name in dnn_param_names:
                        # Store current state_dict value (EMA weights) for later
                        ema_weight = checkpoint['state_dict'][name].clone()
                        # Replace state_dict with training weights from collected_params
                        checkpoint['state_dict'][name] = self.ema.collected_params[param_idx].clone()
                        # Save EMA weights to new collected_params
                        new_collected_params.append(ema_weight)
                        param_idx += 1
                    
                    # Update collected_params to contain EMA weights (for later eval mode switch)
                    self.ema.collected_params = new_collected_params
                    
                    print(f"      Swapped {param_idx} parameter tensors")
                    print("      → state_dict now has TRAINING weights (for PyTorch Lightning to load)")
                    print("      → collected_params now has EMA weights (for validation)")
                else:
                    print("   ⚠️  No collected_params in checkpoint (checkpoint saved during training)")
                
                # CRITICAL: Verify EMA loaded correctly for LoRA models
                if is_lora_model and not is_full_model_ckpt:
                    # Resuming LoRA training - verify parameter count matches
                    num_model_params = len(list(self.dnn.parameters()))
                    num_ema_shadow_params = len(self.ema.shadow_params)
                    
                    if num_model_params != num_ema_shadow_params:
                        # Checkpoint saved with old code (EMA tracked self.parameters() including embedding_model)
                        # Need to extract only dnn-related shadow params
                        print(f"\n⚠️  EMA parameter count mismatch detected")
                        print(f"   DNN parameters: {num_model_params}")
                        print(f"   Checkpoint EMA shadow params: {num_ema_shadow_params}")
                        print(f"   → Checkpoint likely saved with old code (EMA included embedding model)")
                        print(f"   → Extracting only DNN-related EMA weights...")
                        
                        # Old checkpoint had EMA for self.parameters() which includes:
                        # 1. self.dnn.parameters() (what we want)
                        # 2. self.sde.parameters() (if any)
                        # 3. self.embedding_model.parameters() (95M params)
                        # 
                        # Extract only the first num_model_params shadow params (the dnn ones)
                        old_shadow_params = checkpoint['ema']['shadow_params']
                        
                        if len(old_shadow_params) >= num_model_params:
                            # Take only the first num_model_params (corresponding to dnn)
                            new_shadow_params = [old_shadow_params[i].clone() for i in range(num_model_params)]
                            
                            # Update EMA with extracted shadow params
                            self.ema.shadow_params = new_shadow_params
                            print(f"   ✓ Extracted {len(new_shadow_params)} DNN EMA weights from old checkpoint")
                            print(f"     (Discarded {num_ema_shadow_params - num_model_params} non-DNN EMA params)")
                        else:
                            raise ValueError(
                                f"Cannot extract DNN EMA: checkpoint has {num_ema_shadow_params} params "
                                f"but need at least {num_model_params}"
                            )
                    else:
                        print(f"   ✓ EMA parameter count verified: {num_model_params} parameters")
                
                # Set flag to indicate EMA loaded successfully
                self._error_loading_ema = False
                
                # Set debug flag for first eval after loading checkpoint
                if is_lora_model and not is_full_model_ckpt:
                    self._first_eval_after_load = True
            except (ValueError, RuntimeError) as e:
                if is_lora_model and 'state_dict' in checkpoint and is_full_model_ckpt:
                    # When loading full model checkpoint into LoRA model:
                    # The EMA has shadow parameters for the old structure, but we can map them!
                    print("\nRemapping EMA weights for LoRA model...")
                    print(f"  EMA structure mismatch: {str(e)[:80]}...")
                    
                    # Map checkpoint EMA shadow params to LoRA model structure
                    checkpoint_ema = checkpoint['ema']
                    checkpoint_shadows = checkpoint_ema['shadow_params']
                    
                    # Build name mapping: checkpoint param names -> current param names
                    # Get parameter names from checkpoint state_dict
                    checkpoint_param_names = [n for n in checkpoint['state_dict'].keys() 
                                             if not n.startswith('ema.')]
                    
                    # Build mapping for dnn parameters
                    name_to_shadow = {}
                    shadow_idx = 0
                    for ckpt_name in checkpoint_param_names:
                        # Each checkpoint param corresponds to one shadow param
                        name_to_shadow[ckpt_name] = checkpoint_shadows[shadow_idx]
                        shadow_idx += 1
                    
                    # Now map to current model parameters
                    # For LoRA: 'dnn.X' maps to 'dnn.base_model.X' or 'dnn.base_model.model.X'
                    current_param_dict = dict(self.named_parameters())
                    
                    # Reinitialize EMA with current structure (dnn only)
                    self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
                    
                    # Copy shadow params where names match
                    loaded_count = 0
                    for ckpt_name, shadow_value in name_to_shadow.items():
                        if ckpt_name.startswith('dnn.'):
                            # Try to find corresponding parameter in current model
                            # Pattern: 'dnn.X' -> 'dnn.base_model.X' or 'dnn.base_model.model.X'
                            base_name = ckpt_name[4:]  # Remove 'dnn.' prefix
                            
                            # Try different mappings
                            possible_names = [
                                f'dnn.base_model.model.{base_name}',
                                f'dnn.base_model.{base_name}',
                                f'dnn.model.{base_name}'
                            ]
                            
                            for new_name in possible_names:
                                if new_name in current_param_dict:
                                    # Find index of this parameter in current model
                                    current_param_list = list(self.named_parameters())
                                    for idx, (pname, _) in enumerate(current_param_list):
                                        if pname == new_name:
                                            # Copy shadow parameter
                                            self.ema.shadow_params[idx] = shadow_value.clone()
                                            loaded_count += 1
                                            break
                                    break
                    
                    print(f"  ✓ Remapped {loaded_count} EMA shadow parameters to LoRA structure")
                    print(f"     (Loaded pretrained EMA weights into base model)")
                    self._error_loading_ema = False
                else:
                    warnings.warn(
                        f"Could not load EMA from checkpoint: {str(e)}\n"
                        f"EMA will be reinitialized with current model state."
                    )
                    self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
                    self._error_loading_ema = False
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")


    def on_load_checkpoint_busted(self, checkpoint):
        ema = checkpoint.get('ema', None)

        if not(self.nolog):
            try:
                self.trainer.fit_loop.epoch_progress.current.completed = checkpoint['epoch']
            except RuntimeError:
                print('No Trainer found, current epoch was not set.')

        is_lora_model = hasattr(self.dnn, 'base_model')

        if is_lora_model and 'state_dict' in checkpoint:
            # Detect whether this is a full (non-LoRA) checkpoint
            dnn_keys = [k for k in checkpoint['state_dict'] if k.startswith('dnn.')]
            is_full_model_ckpt = bool(dnn_keys) and not any('base_model' in k for k in dnn_keys)

            if is_full_model_ckpt:
                # Strip 'dnn.' prefix and load into the actual backbone through PEFT wrapping
                dnn_state_dict = {k[4:]: v for k, v in checkpoint['state_dict'].items()
                                  if k.startswith('dnn.')}
                
                # Remap NIN.W/NIN.b -> NINLinear format (with PEFT base_layer wrapping)
                dnn_state_dict = _remap_nin_to_linear_keys(dnn_state_dict, target_is_peft_wrapped=True)

                if hasattr(self.dnn, 'model') and hasattr(self.dnn.model, 'get_base_model'):
                    target_model = self.dnn.model.get_base_model()
                elif hasattr(self.dnn, 'model') and hasattr(self.dnn.model, 'model'):
                    target_model = self.dnn.model.model
                else:
                    target_model = self.dnn.base_model

                missing, unexpected = target_model.load_state_dict(dnn_state_dict, strict=False)
                
                # Verify successful loading
                num_params = sum(v.numel() for v in dnn_state_dict.values())
                if not unexpected:
                    print(f"✓ Successfully loaded pretrained checkpoint into LoRA model "
                          f"({len(dnn_state_dict)} tensors, {num_params/1e6:.2f}M parameters)")
                else:
                    print(f"⚠️  Loaded checkpoint with {len(unexpected)} unexpected keys")

                # Patch checkpoint state_dict with current model state so PL loads cleanly
                checkpoint['state_dict'] = self.state_dict()

                # Remove optimizer/scheduler states — incompatible with the LoRA parameter set
                for key in [k for k in list(checkpoint.keys())
                            if 'optimizer' in k.lower() or 'scheduler' in k.lower()]:
                    del checkpoint[key]
                # Inject empty optimizer/scheduler lists so PL's restore_optimizers_and_schedulers()
                # doesn't raise KeyError when the source checkpoint is weights-only (no optimizer saved)
                checkpoint['optimizer_states'] = []
                checkpoint['lr_schedulers'] = []

                # Reset training counters to start fresh LoRA training
                checkpoint['epoch'] = 0
                checkpoint['global_step'] = 0
                for key in ['loops', 'callbacks']:
                    checkpoint.pop(key, None)

                # Initialize a fresh EMA for LoRA params — nothing to restore from a full model ckpt
                self.ema = ExponentialMovingAverage(self._ema_params(), decay=self.ema_decay)
                self._error_loading_ema = False

                return  # Skip EMA loading — nothing to restore for a fresh LoRA run

        # Normal path: resume training (non-LoRA, or LoRA-to-LoRA resume)
        if ema is not None:
            try:
                self.ema.load_state_dict(ema)
                self._error_loading_ema = False
                print("\u2713 EMA weights loaded successfully")
                
                # Detect if we're loading for inference (no trainer) vs resuming training
                has_trainer = False
                try:
                    has_trainer = self.trainer is not None
                except (RuntimeError, AttributeError):
                    # No trainer attached (inference mode)
                    has_trainer = False
                
                # CRITICAL: Restore collected_params ONLY for training resume
                # When loading for inference, state_dict already has EMA weights (they match shadow_params).
                # Restoring collected_params would overwrite EMA with training weights!
                # The extra swap is unnecessary and causes ~1dB performance degradation.
                ema_collected_params = checkpoint.get('ema_collected_params', None)
                
                # DEBUG: Check which weights are in state_dict  
                if not(self.nolog):
                    try:
                        # Simple check using first dnn weight
                        sample_key = [k for k in checkpoint['state_dict'].keys() if k.startswith('dnn.') and 'weight' in k and 'norm' not in k][0]
                        state_weight = checkpoint['state_dict'][sample_key]
                        # Find corresponding EMA index (0 is typically the first dnn parameter)
                        shadow_weight = self.ema.shadow_params[0]
                        collected_weight = self.ema.collected_params[0] if self.ema.collected_params else None
                        
                        state_is_ema = (state_weight.flatten()[:5] - shadow_weight.flatten()[:5]).abs().max() < 1e-6
                        state_is_collected = collected_weight is not None and (state_weight.flatten()[:5] - collected_weight.flatten()[:5]).abs().max() < 1e-6
                        
                        print(f"  → Checkpoint state_dict contains: {'EMA weights' if state_is_ema else 'training weights' if state_is_collected else 'UNKNOWN'}")
                        print(f"  → collected_params present in ema: {self.ema.collected_params is not None} ({len(self.ema.collected_params) if self.ema.collected_params else 0} params)")
                    except Exception as e:
                        print(f"  → [DEBUG] Could not verify weights: {e}")
                
                if not has_trainer:
                    # Inference mode: Keep EMA weights that are already in the model from state_dict
                    print("  → Inference mode detected (no trainer): Keeping EMA weights in model")
                    print("     (NOT restoring training weights from collected_params)")
                    # Clear collected_params so train(False) doesn't do unnecessary swaps
                    self.ema.collected_params = None
                elif ema_collected_params is not None:
                    # Training resume with explicit collected_params in checkpoint
                    print("  → [TRAINING RESUME] Found collected_params in checkpoint (saved during validation)")
                    print("  → [TRAINING RESUME] Restoring training weights from collected_params")
                    # Manually set collected_params and restore
                    self.ema.collected_params = ema_collected_params
                    self.ema.restore(self._ema_params())
                    # Clear collected_params after restore
                    self.ema.collected_params = None
                    if not(self.nolog):
                        # Verify weights changed
                        restored_weight = self.state_dict()[sample_key]
                        restored_is_collected = (restored_weight - collected_weight).abs().max() < 1e-6 if collected_weight is not None else False
                        print(f"  → [TRAINING RESUME] After restore, model has: {'training weights ✓' if restored_is_collected else 'EMA weights (ERROR!)'}")
                elif self.ema.collected_params is not None:
                    # Training resume with collected_params in ema.state_dict  
                    print("  → [TRAINING RESUME] Restoring training weights from EMA's collected_params")
                    self.ema.restore(self._ema_params())
                    self.ema.collected_params = None
                    if not(self.nolog):
                        # Verify weights changed
                        restored_weight = self.state_dict()[sample_key]
                        restored_is_collected = (restored_weight - collected_weight).abs().max() < 1e-6 if collected_weight is not None else False
                        print(f"  → [TRAINING RESUME] After restore, model has: {'training weights ✓' if restored_is_collected else 'EMA weights (ERROR!)'}")
                elif has_trainer:
                    # Training resume but no collected_params - this is unexpected
                    print("  → [WARNING] Training resume but no collected_params found!")
                    print("     Model will continue training with EMA weights (may cause issues)")
                
                # Set flag for debug output on first validation after resume
                if is_lora_model:
                    self._first_eval_after_load = True
                    
            except ValueError:
                # Parameter count mismatch (e.g. loading an ears_wham.ckpt whose EMA
                # tracked a different parameter set).  Try to apply shadow params
                # positionally to self.dnn.parameters() — this works when the backbone
                # architecture is identical but the top-level param list differs.
                shadow_params = ema.get('shadow_params', [])
                dnn_params = list(self.dnn.parameters())
                if len(shadow_params) == len(dnn_params):
                    with torch.no_grad():
                        for p, shadow in zip(dnn_params, shadow_params):
                            p.copy_(shadow.to(p.device))
                    # Reinitialize EMA from the freshly-loaded weights
                    self.ema = ExponentialMovingAverage(self._ema_params(), decay=self.ema_decay)
                    self._error_loading_ema = False
                    print("\u2713 EMA shadow params applied directly to dnn parameters "
                          f"({len(shadow_params)} tensors)")
                else:
                    warnings.warn(
                        f"EMA shadow param count ({len(shadow_params)}) != "
                        f"dnn param count ({len(dnn_params)}) — cannot apply EMA weights. "
                        "Keeping state_dict weights."
                    )
                    self.ema = ExponentialMovingAverage(self._ema_params(), decay=self.ema_decay)
                    self._error_loading_ema = False
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint_busted(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()
        
        # CRITICAL: Save collected_params separately if checkpoint is saved during validation
        # collected_params contains training weights (not saved in state_dict())
        if hasattr(self.ema, 'collected_params') and self.ema.collected_params is not None:
            checkpoint['ema_collected_params'] = [p.clone() for p in self.ema.collected_params]
        else:
            checkpoint['ema_collected_params'] = None
        
        # MERT embedding model is only for validation metrics — strip it from saved checkpoints
        checkpoint['state_dict'] = {
            k: v for k, v in checkpoint['state_dict'].items()
            if not k.startswith('embedding_model.')
        }

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)

        if not(self.nolog):
            #set completed epoch for logging
            try:
                self.trainer.fit_loop.epoch_progress.current.completed=checkpoint['epoch']
            except RuntimeError as e:
                print('No Trainer found, current epoch was not set.')

        # ----------------------------------------------------------------
        # Fix optimizer state saved by old code that passed self.parameters()
        # (including requires_grad=False params) to Adam, whereas the current
        # code passes only filter(requires_grad, self.parameters()).
        #
        # Strategy: identify params that were tracked by the old optimizer but
        # have no Adam state (never received gradients → were frozen).
        # Remove them from param_groups and renumber the remaining state keys
        # so they match the current 0-based param list.  Adam buffers for all
        # real trainable params are preserved exactly.
        # ----------------------------------------------------------------
        if 'optimizer_states' in checkpoint and checkpoint['optimizer_states']:
            current_trainable = sum(1 for p in self.parameters() if p.requires_grad)
            try:
                ckpt_opt = checkpoint['optimizer_states'][0]
                ckpt_params = ckpt_opt['param_groups'][0]['params']  # [0,1,2,...,N-1]
                ckpt_state  = ckpt_opt['state']                       # {idx: {exp_avg,...}}
                ckpt_param_count = len(ckpt_params)

                if ckpt_param_count != current_trainable:
                    # Params that have no Adam state were frozen in the old run
                    # (e.g. GaussianFourierProjection.W included via self.parameters())
                    frozen_in_ckpt = [p for p in ckpt_params if p not in ckpt_state]

                    if frozen_in_ckpt and ckpt_param_count - len(frozen_in_ckpt) == current_trainable:
                        # Build old-index → new-index mapping for updated params only
                        old_to_new = {}
                        new_idx = 0
                        for old_idx in ckpt_params:
                            if old_idx in ckpt_state:
                                old_to_new[old_idx] = new_idx
                                new_idx += 1

                        ckpt_opt['state'] = {old_to_new[k]: v for k, v in ckpt_state.items()}
                        ckpt_opt['param_groups'][0]['params'] = list(old_to_new.values())

                        print(f"\n✓ Optimizer state patched: removed {len(frozen_in_ckpt)} frozen "
                              f"param(s) that were tracked by old code (e.g. GaussianFourierProjection.W).")
                        print(f"  Adam buffers for all {current_trainable} trainable params preserved.")
                    else:
                        # Mismatch can't be explained by frozen params alone — strip to be safe
                        print(f"\n⚠️  Optimizer state mismatch ({ckpt_param_count} → {current_trainable} params).")
                        print(f"   Cannot reconcile automatically → stripping optimizer state.")
                        checkpoint['optimizer_states'] = []
                        checkpoint['lr_schedulers'] = []
            except (KeyError, IndexError, TypeError) as e:
                print(f"   (Optimizer patch failed: {e})")

        if ema is not None:
            try:
                self.ema.load_state_dict(checkpoint['ema'])
            except ValueError:
                # Old checkpoint had EMA initialized with self.parameters() (all params
                # including sde, embedding_model etc.), but current EMA only tracks
                # self.dnn.parameters(). Extract the first N shadow params (dnn comes
                # first in self.parameters() ordering).
                ckpt_shadows = checkpoint['ema']['shadow_params']
                current_n = len(self.ema.shadow_params)
                if len(ckpt_shadows) >= current_n:
                    print(f"\n⚠️  EMA shadow_params count mismatch: "
                          f"checkpoint={len(ckpt_shadows)}, current={current_n}")
                    print(f"   Old code tracked all self.parameters(); "
                          f"extracting first {current_n} (dnn) shadow params.")
                    # Patch shadow_params in-place and reload
                    checkpoint['ema']['shadow_params'] = ckpt_shadows[:current_n]
                    if checkpoint['ema'].get('collected_params') is not None:
                        checkpoint['ema']['collected_params'] = \
                            checkpoint['ema']['collected_params'][:current_n]
                    try:
                        self.ema.load_state_dict(checkpoint['ema'])
                        print(f"   ✓ EMA loaded successfully ({current_n} shadow params).")
                    except ValueError as e:
                        warnings.warn(
                            f"EMA load still mismatched after truncation ({e}). "
                            f"Reinitializing EMA."
                        )
                        self.ema = ExponentialMovingAverage(self._ema_params(), decay=self.ema_decay)
                else:
                    warnings.warn(
                        f"EMA shadow_params mismatch and cannot recover "
                        f"(checkpoint={len(ckpt_shadows)} < current={current_n}). "
                        f"Reinitializing EMA."
                    )
                    self.ema = ExponentialMovingAverage(self._ema_params(), decay=self.ema_decay)
            self._error_loading_ema = False
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")


    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()


    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval - switch to EMA weights for validation
                # Check if this is a LoRA model to add extra verification
                is_lora = hasattr(self.dnn, 'base_model')
                
                if is_lora and hasattr(self, '_first_eval_after_load'):
                    # First time switching to eval after loading checkpoint
                    print("\n" + "="*70)
                    print("SWITCHING TO EMA WEIGHTS FOR VALIDATION (first time after resume)")
                    print("="*70)
                    # Sample a LoRA parameter to verify it changes
                    lora_params = [n for n, p in self.named_parameters() if 'lora' in n.lower()]
                    if lora_params:
                        sample_param_name = lora_params[0]
                        for n, p in self.named_parameters():
                            if n == sample_param_name:
                                before_ema = p.data.flatten()[:3].clone()
                                break
                    
                self.ema.store(self._ema_params())        # store current params in collected_params
                self.ema.copy_to(self._ema_params())      # copy EMA parameters over current params for evaluation
                
                if is_lora and hasattr(self, '_first_eval_after_load'):
                    if lora_params:
                        for n, p in self.named_parameters():
                            if n == sample_param_name:
                                after_ema = p.data.flatten()[:3]
                                print(f"  Sample LoRA param: {sample_param_name}")
                                print(f"    Before EMA: {before_ema}")
                                print(f"    After EMA:  {after_ema}")
                                params_changed = not torch.allclose(before_ema, after_ema)
                                print(f"    Weights changed: {'✓ YES' if params_changed else '✗ NO (WARNING!)'}")
                                break
                    print("="*70 + "\n")
                    delattr(self, '_first_eval_after_load')
            else:
                # train - restore training weights
                if self.ema.collected_params is not None:
                    self.ema.restore(self._ema_params())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, forward_out, x_t, z, t, mean, x):
        """
        Different loss functions can be used to train the score model, see the paper: 
        
        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753

        """

        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma + z)) # Eq. (7)
            else:
                raise ValueError("Invalid loss weighting for loss_type=score_matching: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        elif self.loss_type == "denoiser":
            score = forward_out
            D = score * sigma.pow(2) + x_t # equivalent to Eq. (10)
            losses = torch.square(torch.abs(D - mean)) # Eq. (8)
            if self.loss_weighting == "1":
                losses = losses
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma**2
            elif self.loss_weighting == "edm":
                losses = ((sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2))[:, None, None, None] * losses
            else:
                raise ValueError("Invalid loss weighting for loss_type=denoiser: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))     
        elif self.loss_type == "data_prediction":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
            losses_tf = torch.mean(0.5*torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

            # losses in the time domain (td)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = torch.mean(0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))
            loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss

    def _step(self, batch, batch_idx):
        x, y, audio_x, audio_y = batch
        
        # Check for NaN/Inf in input data
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            print(f"Warning: NaN/Inf detected in input data at batch {batch_idx}")
            return torch.tensor(0.0, device=x.device, requires_grad=True)
          
        if self.accomp_target:
            x = y-x # make accompaniment target

        #reshape => fuse channel and batch dimensions and unsqueeze so dimension fits for sde.marginal_prob()

        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3]).unsqueeze(1)
        y = y.reshape(y.shape[0]*y.shape[1], y.shape[2], y.shape[3]).unsqueeze(1)
        y = pad_spec(y, mode="reflection")              
        x = pad_spec(x, mode="reflection")

        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t)
        
        # Check for numerical issues in marginal_prob
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            print(f"Warning: NaN/Inf in marginal_prob at batch {batch_idx}, t_range: [{t.min():.4f}, {t.max():.4f}]")
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        forward_out = self(x_t, y, t)
        
        # Check for NaN/Inf in model output
        if not torch.isfinite(forward_out).all():
            print(f"Warning: NaN/Inf in model output at batch {batch_idx}")
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        loss = self._loss(forward_out, x_t, z, t, mean, x)
        return loss

    def training_step(self, batch, batch_idx):

        loss = self._step(batch, batch_idx)
        
        # Check for NaN loss
        if not torch.isfinite(loss):
            print(f"Warning: NaN or Inf loss detected at batch {batch_idx}, skipping this batch")
            return None
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self):
        """Setup embedding model once at the start of validation epoch"""
        # Get rank for DDP
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
            
        # Reinitialize/reload the embedding model on non-zero ranks
        # This ensures proper model loading across all DDP processes
        if rank != 0:
            models = {m.name: m for m in get_all_models()}
            self.embedding_model = models["MERT-v1-95M"]
            self.embedding_model.load_model()
            
        # Move embedding model to each rank's device and set to eval mode
        if hasattr(self.embedding_model, 'model'):
            current_device = next(self.embedding_model.model.parameters()).device
            if current_device != self.device:
                self.embedding_model.model.to(self.device)
            
            # Set to eval mode and disable all stochastic operations for determinism
            self.embedding_model.model.eval()
            
            # Disable dropout/batchnorm to ensure deterministic outputs
            for module in self.embedding_model.model.modules():
                if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                    module.eval()
                    module.p = 0.0
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module.eval()
                    module.track_running_stats = False

    def validation_step(self, batch, batch_idx):

        # Evaluate speech enhancement performance
        # Handle both DDP and single GPU cases
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        # Verify model is in eval mode (important for dropout/batchnorm behavior)
        if batch_idx == 0 and rank == 0:
            if self.training:
                print("⚠️  WARNING: Model is in training mode during validation!")
            
        if batch_idx == 0 and self.num_eval_files != 0:
            # Create multi_res_loss with the correct sample rate (fs_original)
            fs_original = self.data_module.valid_set.dataset.fs_original
            multi_res_loss_fn = MultiResolutionSTFTLoss(fft_sizes=[256, 512, 1024, 2048, 4096],
                                             win_lengths=[256, 512, 1024, 2048, 4096],
                                             hop_sizes=[64, 128,  256, 512, 1024],
                                             sample_rate=fs_original, 
                                             perceptual_weighting=True).forward
            # Split the evaluation files among the GPUs
            eval_files_per_gpu = self.num_eval_files // world_size

            # Select the files for this GPU     
            if rank == world_size - 1:
                first_valid_file = rank*eval_files_per_gpu
                last_valid_file = self.num_eval_files
            else:   
                first_valid_file = rank*eval_files_per_gpu
                last_valid_file = (rank+1)*eval_files_per_gpu

            np.random.seed(2*self.data_module.rand_seed)
            rand_idx = np.arange(self.data_module.valid_set.__len__())
            np.random.shuffle(rand_idx)
            rand_idx = rand_idx[:self.num_eval_files]
            np.random.seed()
            # Evaluate the performance of the model
            # Initialize sums as Python floats for accumulation (convert to tensors before all_reduce)
            sdr_sum = 0.0
            si_sdr_sum = 0.0
            multi_res_loss_sum = 0.0
            test_sisdr = 0.0
            embedding_mse_sum = 0.0

            for item_id in tqdm.tqdm(rand_idx[first_valid_file:last_valid_file],desc='Validation on GPU '+str(rank)):
                # Load the clean and noisy speech

                y, x, target_rms, file_id = self.data_module.valid_set.dataset[item_id]
                # x = x.squeeze().numpy()
                # y = y.squeeze().numpy()

                # if self.data_module.valid_set.dataset.fs != self.sr:
                #     #resample audio
                #     y = resample(y, orig_sr=self.data_module.valid_set.dataset.fs, target_sr=self.sr)
                #     x = resample(x, orig_sr=self.data_module.valid_set.dataset.fs, target_sr=self.sr)
                # y = torch.from_numpy(y)
                # x = torch.from_numpy(x)

                x_hat = self.enhance(y, N=self.sde.N)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0)

                #resample to original sr if needed for evaluation
                if self.data_module.valid_set.dataset.fs_original != self.sr:
                    x_hat = resample(x_hat.numpy(), orig_sr=self.sr, target_sr=self.data_module.valid_set.dataset.fs_original)
                    x_hat = torch.from_numpy(x_hat)
                    x = resample(x.numpy(), orig_sr=self.sr, target_sr=self.data_module.valid_set.dataset.fs_original)
                    x = torch.from_numpy(x)
                
                # Now x_hat and x are at fs_original sample rate
                # Create copies for embedding model if it uses different sample rate
                if self.embedding_model.sr != self.data_module.valid_set.dataset.fs_original:
                    x_hat_emb = resample(x_hat.numpy(), orig_sr=self.data_module.valid_set.dataset.fs_original, target_sr=self.embedding_model.sr)
                    x_hat_emb = torch.from_numpy(x_hat_emb)
                    x_emb = resample(x.numpy(), orig_sr=self.data_module.valid_set.dataset.fs_original, target_sr=self.embedding_model.sr)
                    x_emb = torch.from_numpy(x_emb)
                    emb_input_sr = self.embedding_model.sr
                else:
                    x_hat_emb = x_hat
                    x_emb = x
                    emb_input_sr = self.data_module.valid_set.dataset.fs_original
                    
                if self.accomp_target:
                    x_hat = y-x_hat
                if self.valid_sep_dir is not None:
#                    os.makedirs(os.path.join(self.valid_sep_dir,'target'), exist_ok=True)
#                    os.makedirs(os.path.join(self.valid_sep_dir,'mixture'), exist_ok=True)
                    os.makedirs(os.path.join(self.valid_sep_dir,self.valid_out_foldername), exist_ok=True)

#                    soundfile.write(os.path.join(self.valid_sep_dir,'mixture','mixture_fileid_'+str(item_id.item())+'.wav'), y.T, self.data_module.valid_set.dataset.fs_original)
#                    soundfile.write(os.path.join(self.valid_sep_dir,'target','target_fileid_'+str(item_id.item())+'.wav'), x.T, self.data_module.valid_set.dataset.fs_original)
                    soundfile.write(os.path.join(self.valid_sep_dir,self.valid_out_foldername,f'separated_vocals_fileid_{file_id}.wav'), x_hat.T, self.data_module.valid_set.dataset.fs_original)
                    os.makedirs(os.path.join(self.valid_sep_dir,self.valid_out_foldername,'convert',str(self.embedding_model.sr)), exist_ok=True)
                    soundfile.write(os.path.join(self.valid_sep_dir,self.valid_out_foldername,'convert',str(self.embedding_model.sr),f'separated_vocals_fileid_{file_id}.wav'), x_hat.T, self.data_module.valid_set.dataset.fs_original)
                    emb_tgt_path = os.path.join(self.valid_sep_dir,'target','convert',str(self.embedding_model.sr),f'target_fileid_{file_id}.wav')
                    if not os.path.exists(emb_tgt_path):
                        soundfile.write(emb_tgt_path, x_emb.T, self.embedding_model.sr)
                        

                # Compute embedding MSE once per file (not per channel) for consistency  
                # Use inference_mode for maximum determinism
                with torch.inference_mode():
                    # Convert to numpy for embedding model
                    x_emb_np = x_emb.cpu().numpy() if torch.is_tensor(x_emb) else x_emb
                    x_hat_emb_np = x_hat_emb.cpu().numpy() if torch.is_tensor(x_hat_emb) else x_hat_emb

                    # Defensive guard: ensure MERT input is at the wrapper sample rate.
                    if emb_input_sr != self.embedding_model.sr:
                        x_emb_np = resample(x_emb_np, orig_sr=emb_input_sr, target_sr=self.embedding_model.sr)
                        x_hat_emb_np = resample(x_hat_emb_np, orig_sr=emb_input_sr, target_sr=self.embedding_model.sr)
                    
                    # Squeeze to remove channel dimension if mono, keep if stereo
                    target_emb = self.embedding_model._get_embedding(x_emb_np.squeeze())
                    separated_emb = self.embedding_model._get_embedding(x_hat_emb_np.squeeze())
                    
                    # Convert embeddings to numpy if they are tensors
                    if torch.is_tensor(target_emb):
                        target_emb = target_emb.cpu().numpy()
                    if torch.is_tensor(separated_emb):
                        separated_emb = separated_emb.cpu().numpy()
                    
                    file_emb_mse = float(np.mean((target_emb - separated_emb)**2))
                    embedding_mse_sum += file_emb_mse

                if x.shape[0] > 1:
                    temp_sdr = 0.0
                    temp_sisdr = 0.0
                    temp_test_sisdr = 0.0
                    temp_multi_res = 0.0
                    for ii in range(x.shape[0]):
                        temp_sdr += self.sdr(x_hat[ii,:].to(self.device), x[ii,:].to(self.device)).item()
                        temp_sisdr += self.si_sdr(x_hat[ii,:].to(self.device), x[ii,:].to(self.device)).item()
                        temp_test_sisdr += si_sdr(x[ii,:].to(self.device), x_hat[ii,:].to(self.device)).item()
                        temp_multi_res += multi_res_loss_fn(x_hat[ii,:].to(self.device).unsqueeze(0).unsqueeze(0), x[ii,:].to(self.device).unsqueeze(0).unsqueeze(0)).item()
                    
                    sdr_sum += temp_sdr / x.shape[0]
                    si_sdr_sum += temp_sisdr / x.shape[0]
                    test_sisdr += temp_test_sisdr / x.shape[0]
                    multi_res_loss_sum += temp_multi_res / x.shape[0]
                    # Embedding MSE computed once per file above
                else:
                    sdr_sum += self.sdr(x_hat.to(self.device), x.to(self.device)).item()
                    si_sdr_sum += self.si_sdr(x_hat.to(self.device), x.to(self.device)).item()
                    
                    # soundfile.write(os.path.join('TEMP_OUT', f'separated_vocals_fileid_{file_id}.wav'), x_hat.T, self.data_module.valid_set.dataset.fs_original)
                    # soundfile.write(os.path.join('TEMP_OUT', f'target_fileid_{file_id}.wav'), x.T, self.data_module.valid_set.dataset.fs_original)
#                    test_sisdr += si_sdr(x, x_hat)
                    multi_res_loss_sum += multi_res_loss_fn(x_hat.to(self.device).unsqueeze(0), x.to(self.device).unsqueeze(0)).item()
                    

            num_files = len(rand_idx[first_valid_file:last_valid_file])
            
            # Wait for all ranks to finish processing before logging audio
            if dist.is_initialized():
                dist.barrier()
            
            # Log audio files on rank 0 after all ranks have finished processing
            # Load from disk since files may have been processed on different ranks
            if rank == 0 and not(self.nolog) and (self.valid_audio_log_files is not None) and (self.current_epoch % self.audio_log_interv)==0:
                for file_id in self.valid_audio_log_files:
                    audio_path = os.path.join(self.valid_sep_dir, self.valid_out_foldername, f'separated_vocals_fileid_{file_id}.wav')
                    if os.path.exists(audio_path):
                        audio_data, sr = soundfile.read(audio_path)
                        log_data_dict = {f"file #{file_id}": [wandb.Audio(audio_data, sr, caption='separated')]}
                        wandb.log(log_data_dict)
            
            # Aggregate sums across GPUs to compute correct global averages
            # Convert Python floats to tensors for all_reduce operation
            global_sdr_sum = torch.tensor(sdr_sum, dtype=torch.float32, device=self.device)
            global_si_sdr_sum = torch.tensor(si_sdr_sum, dtype=torch.float32, device=self.device)
            global_multi_res_sum = torch.tensor(multi_res_loss_sum, dtype=torch.float32, device=self.device)
            global_emb_mse_sum = torch.tensor(embedding_mse_sum, dtype=torch.float32, device=self.device)
            global_num_files = torch.tensor(num_files, dtype=torch.float32, device=self.device)
            
            # Only perform all_reduce if using DDP
            if dist.is_initialized() and world_size > 1:
                dist.all_reduce(global_sdr_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_si_sdr_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_multi_res_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_emb_mse_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_num_files, op=dist.ReduceOp.SUM)
            
            if global_num_files > 0:
                # Compute global averages (already aggregated across all GPUs via all_reduce)
                sdr_avg = global_sdr_sum / global_num_files
                si_sdr_avg = global_si_sdr_sum / global_num_files
                multi_res_loss_avg = global_multi_res_sum / global_num_files
                embedding_mse_avg = global_emb_mse_sum / global_num_files
                
                # sync_dist=True tells PL to sync across GPUs
                # Note: Values are already globally aggregated via all_reduce above,
                # so keep tensors on GPU device for PL's sync operation (required for all_reduce)
                self.log('sdr', sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
                self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
                self.log('multi_res_loss', multi_res_loss_avg, on_step=False, on_epoch=True, sync_dist=True)
                self.log('mert_mse', embedding_mse_avg, on_step=False, on_epoch=True, sync_dist=True)
            
        else:
            sdr_avg = None
            si_sdr_avg = None
            multi_res_loss_avg = None

        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    


    def forward(self, x_t, y, t):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates 
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').
        
        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann 
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. 

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024. 

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)
            
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat

        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([x_t, y], dim=1)            
            score = -self.dnn(dnn_input, t)
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))
    
    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=2, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max()
        y = y / norm_factor
        if y.shape[0]>1:
            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 1)
        else:
            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
  
        Y = pad_spec(Y, mode="reflection")

        # SGMSE sampling with OUVE SDE
        if self.sde.__class__.__name__ == 'OUVESDE':
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False, **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
            else:
                raise ValueError("Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
            
        # Schrödinger bridge sampling with VE SDE
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(sde=self.sde, y=Y.cuda(), sampler_type=self.sde.sampler_type)
        else:
            raise ValueError("Invalid SDE type for speech enhancement: {}".format(self.sde.__class__.__name__))

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
