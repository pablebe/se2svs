# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""
NCSN++ model with Low-Rank Adaptation (LoRA) using HuggingFace PEFT library.
This allows adapting the speech enhancement model to singing voice separation while
preserving the original model's capabilities.

Uses the production-ready PEFT library for robust LoRA implementation.
"""

from .ncsnpp_48k import NCSNpp_48k
from .ncsnpp_utils.layers import NIN, NINLinear
from .shared import BackboneRegistry
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn



def _swap_nin_to_ninlinear(model):
    """Replace every NIN instance in *model* with an equivalent NINLinear in-place.

    Called before get_peft_model when 'attn_nin' targeting is requested so that
    PEFT can see standard nn.Linear submodules ('nin_linear') to wrap with LoRA.

    Weight mapping:
        NIN.W  (in, out) -> NINLinear.nin_linear.weight (out, in)  [transposed]
        NIN.b  (out,)    -> NINLinear.nin_linear.bias   (out,)
    """
    # Collect replacements first to avoid mutating while iterating
    replacements = []  # (parent_module, attr_name, new_module)
    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if type(child) is NIN:
                in_dim, num_units = child.W.shape  # W is (in, out)
                ninlinear = NINLinear(in_dim, num_units)
                with torch.no_grad():
                    ninlinear.nin_linear.weight.copy_(child.W.data.T)  # (out, in)
                    ninlinear.nin_linear.bias.copy_(child.b.data)
                ninlinear = ninlinear.to(child.W.device)
                replacements.append((module, attr_name, ninlinear))

    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    return len(replacements)


def get_target_modules_for_lora(model, target_types):
    """
    Returns the list of PEFT target module names for the given target types.

    Supported types:
        'resnet_dense' : Dense_0 (nn.Linear, time-embedding projection in ResNet blocks)
        'resnet_conv'  : Conv_0, Conv_1, Conv_2 (nn.Conv2d in ResNet blocks)
        'attn_nin'     : nin_linear (nn.Linear inside NINLinear; the Q/K/V/O projections
                         of every AttnBlockpp, plus channel-mixing skips in ResNet blocks)

    Args:
        model: The PyTorch model (unused, kept for API consistency)
        target_types: List of target type strings (can be empty)

    Returns:
        List of leaf module attribute names for PEFT target_modules (can be empty)

    Raises:
        ValueError: If target_types contains unsupported values
    """
    if not target_types:
        return []  # Allow empty list - will be handled by caller

    valid_types = {"resnet_dense", "resnet_conv", "attn_nin"}
    invalid_types = set(target_types) - valid_types
    if invalid_types:
        raise ValueError(
            f"Invalid target_types: {invalid_types}. Valid types are: {valid_types}"
        )

    target_modules = []

    if "resnet_dense" in target_types:
        # nn.Linear: time-embedding projection in every ResNet block
        target_modules.append("Dense_0")

    if "resnet_conv" in target_types:
        # nn.Conv2d: main convolutions in ResNet blocks
        target_modules.extend(["Conv_0", "Conv_1", "Conv_2"])

    if "attn_nin" in target_types:
        # nn.Linear (inside NINLinear): Q/K/V/O projections in AttnBlockpp
        # and channel-mixing skip connections in ResNet blocks
        target_modules.append("nin_linear")

    return target_modules


@BackboneRegistry.register("ncsnpp_48k_lora")
class NCSNpp_48k_LoRA(nn.Module):
    """
    NCSN++ model with LoRA (Low-Rank Adaptation) using HuggingFace PEFT library.
    
    This is a wrapper around the base NCSNpp_48k model that applies LoRA using the
    production-ready PEFT library. Much simpler and more robust than custom implementation.
    
    The PEFT library automatically:
    - Freezes base model weights
    - Injects LoRA layers into target modules  
    - Manages trainable parameters
    - Provides save/load utilities for adapters
    """

    @staticmethod
    def add_argparse_args(parser):
        # Get base model args first
        parser = NCSNpp_48k.add_argparse_args(parser)
        
        # Add LoRA-specific args
        parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
        parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling parameter (default: 16)")
        parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout (default: 0.0)")
        parser.add_argument("--lora-target-modules", type=str, nargs='*', 
                            default=["resnet_dense", "resnet_conv"],
                            help="Which module types to apply LoRA to: resnet_dense, resnet_conv (use empty list for none)")
        parser.add_argument("--lora-attn-nin", action="store_true", default=False,
                            help="Also apply LoRA to the NIN (Q/K/V/O) projections inside attention blocks")
        parser.add_argument("--lora-bias", type=str, default="none", 
                            help="LoRA bias training mode: none, all, lora_only")
        parser.add_argument("--lora-modules-to-save", type=str, nargs='*', default=None,
                            help="Additional modules to train alongside LoRA (e.g., output_layer)")
        parser.add_argument("--lora-pretrained-checkpoint", type=str, default=None,
                            help="Path to pretrained checkpoint to load BEFORE applying LoRA (much cleaner than post-hoc loading)")
        return parser

    def __init__(self,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=["resnet_dense", "resnet_conv"],
        lora_attn_nin=False,
        lora_bias="none",
        lora_modules_to_save=None,
        lora_pretrained_checkpoint=None,
        **kwargs
    ):
        """
        Create a NCSNpp_48k model with LoRA adaptation using PEFT library.
        
        Args:
            lora_r: LoRA rank (default: 8)
            lora_alpha: LoRA alpha parameter for scaling (default: 16)
            lora_dropout: Dropout probability for LoRA layers (default: 0.0)
            lora_target_modules: List of module types to apply LoRA to (resnet_dense, resnet_conv)
            lora_attn_nin: If True, also apply LoRA to NIN Q/K/V/O projections in attention blocks
            lora_bias: How to handle bias parameters ("none", "all", "lora_only")
            lora_modules_to_save: Additional modules to keep trainable
            lora_pretrained_checkpoint: Path to checkpoint to load BEFORE applying LoRA
            **kwargs: Arguments passed to base NCSNpp_48k model
        """
        super().__init__()
        
        print(f"\n{'='*70}")
        print(f"Initializing LoRA using HuggingFace PEFT library")
        print(f"{'='*70}\n")
        
        # Create base model first (without LoRA)
        self.base_model = NCSNpp_48k(**kwargs)

        # Resolve target modules: handle None and add attn_nin if requested
        if lora_target_modules is None:
            lora_target_modules = []
        else:
            lora_target_modules = list(lora_target_modules)  # Copy to avoid mutating input
        
        if lora_attn_nin and "attn_nin" not in lora_target_modules:
            lora_target_modules.append("attn_nin")
        
        # If no target modules after resolution, skip LoRA and just use base model
        if not lora_target_modules:
            print(f"\n⚠ No LoRA target modules specified. Using base model without LoRA.\n")
            self.model = self.base_model
            self.lora_config = None
            return
        
        # Load pretrained weights BEFORE applying LoRA (if provided)
        if lora_pretrained_checkpoint:
            print(f"Loading pretrained weights from: {lora_pretrained_checkpoint}")
            checkpoint = torch.load(lora_pretrained_checkpoint, map_location='cpu', weights_only=False)
            
            # Extract model weights (handle both full checkpoint and state_dict-only)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Strip 'dnn.' prefix if present
                clean_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('dnn.'):
                        clean_key = key[4:]  # Remove 'dnn.' prefix
                    else:
                        clean_key = key
                    clean_state_dict[clean_key] = value
            else:
                clean_state_dict = checkpoint
            
            # Load into base model (before PEFT wrapping)
            missing, unexpected = self.base_model.load_state_dict(clean_state_dict, strict=False)
            
            print(f"✓ Loaded pretrained weights into base model")
            if missing:
                print(f"  Missing keys: {len(missing)} (ok if expected)")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            
            # Verify a sample weight was loaded correctly
            sample_key = 'all_modules.4.Conv_0.weight'
            if sample_key in clean_state_dict:
                checkpoint_weight = clean_state_dict[sample_key]
                model_weight = self.base_model.all_modules[4].Conv_0.weight
                match = torch.allclose(checkpoint_weight, model_weight)
                print(f"\n  Verification (sample weight: {sample_key}):")
                print(f"    Checkpoint: {checkpoint_weight.flatten()[:3]}")
                print(f"    Loaded:     {model_weight.flatten()[:3]}")
                print(f"    Match: {'✓' if match else '✗'}")
            print()
      
        # If attention NIN targeting is requested, swap NIN → NINLinear in the base
        # model so that PEFT can wrap the inner nn.Linear ('nin_linear') with LoRA.
        # The base NCSNpp_48k is always built with plain NIN; this swap is done here
        # so the base class (and its checkpoint format) remains untouched.
        if "attn_nin" in lora_target_modules:
            n_swapped = _swap_nin_to_ninlinear(self.base_model)
            print(f"  Swapped {n_swapped} NIN → NINLinear modules for LoRA targeting")

        # Get target modules for LoRA
        target_module_names = get_target_modules_for_lora(self.base_model, lora_target_modules)

        print(f"LoRA Configuration:")
        print(f"  Rank (r):           {lora_r}")
        print(f"  Alpha:              {lora_alpha}")
        print(f"  Dropout:            {lora_dropout}")
        print(f"  Target types:       {lora_target_modules}")
        print(f"  Target modules:     {target_module_names}")
        print(f"  Bias handling:      {lora_bias}")
        if lora_modules_to_save:
            print(f"  Additional modules: {lora_modules_to_save}")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_module_names,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            modules_to_save=lora_modules_to_save,
            init_lora_weights=True,
        )

        # Apply LoRA to the base model using PEFT
        self.model = get_peft_model(self.base_model, lora_config)
        
        print(f"\n✓ LoRA successfully applied using PEFT!")
        
        # Print parameter statistics
        self.model.print_trainable_parameters()
        print(f"{'='*70}\n")
        
        # Store config for later use
        self.lora_config = lora_config

    def forward(self, *args, **kwargs):
        """Forward pass - delegate to the PEFT-wrapped model."""
        return self.model(*args, **kwargs)

    def get_lora_parameters(self):
        """Get all LoRA parameters for optimizer."""
        return [p for n, p in self.model.named_parameters() if p.requires_grad]

    def save_lora_weights(self, path):
        """
        Save only the LoRA adapter weights (not the full model).
        This allows for efficient storage of task-specific adapters.
        """
        self.model.save_pretrained(path)
        print(f"✓ LoRA adapter weights saved to {path}")

    def load_lora_weights(self, path):
        """
        Load LoRA adapter weights from a saved checkpoint.
        """
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)
        print(f"✓ LoRA adapter weights loaded from {path}")

    def merge_and_unload(self):
        """
        Merge LoRA weights into the base model and remove LoRA layers.
        Returns a standard NCSNpp_48k model with the adapted weights.
        """
        merged_model = self.model.merge_and_unload()
        print("✓ LoRA weights merged into base model")
        return merged_model

    def disable_adapter(self):
        """Disable LoRA adapter (use base model only)."""
        if hasattr(self.model, 'disable_adapter'):
            self.model.disable_adapter()

    def enable_adapter(self):
        """Enable LoRA adapter."""
        if hasattr(self.model, 'enable_adapter'):
            self.model.enable_adapter()
    
    def __getattr__(self, name):
        """Delegate attribute access to the PEFT model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
