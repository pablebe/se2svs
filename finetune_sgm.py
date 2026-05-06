#NOTE for setup of environment with miniconda: 
# - create virtual environment with python=3.11: conda create -n <env_name> -r python=3.11 
# - install all necessary packages with:         pip install -r requirements.txt
# - set CUDA_HOME to path of conda environment:  conda env config vars set CUDA_HOME=/home/<username>/miniconda3/envs/<env_name>
# - install cuda in environment with:            conda install cuda -c nvidia
# - install auraloss for multi-res-stft-loss:    pip install auraloss => don't forget to cite: https://github.com/csteinmetz1/auraloss 


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
#os.system("export CPATH=/opt/conda/envs/sgmse_env/targets/x86_64-linux/include:$CPATH")
#os.system("export LD_LIBRARY_PATH=/opt/conda/envs/sgmse_env/targets/x86_64-linux/lib:$LD_LIBRARY_PATH")
#os.system("echo ${CPATH}")
#os.system("echo ${LD_LIBRARY_PATH}")

import torch
import wandb
import argparse
import sys
import toml
from argparse import ArgumentParser
from os.path import join

# IMPORTANT: Create module aliases BEFORE importing anything that might load checkpoints
# The checkpoint was saved with 'sgmse' module structure, but code now uses 'models'
import models
import models.data_module
import models.MSS_model
import models.sgmse
import models.sgmse.sdes
import models.sgmse.backbones
import models.sgmse.util

# # Create backward-compatible module names
sys.modules['sgmse'] = models.sgmse
sys.modules['sgmse.data_module'] = models.data_module
sys.modules['sgmse.model'] = models.MSS_model
sys.modules['sgmse.sdes'] = models.sgmse.sdes
sys.modules['sgmse.backbones'] = models.sgmse.backbones
sys.modules['sgmse.util'] = models.sgmse.util

# Register safe globals for PyTorch 2.6+ (allows these classes in checkpoints)
from models.data_module import SpecsDataModule
from models.MSS_model import ScoreModel
torch.serialization.add_safe_globals([SpecsDataModule, ScoreModel])

# Now import PyTorch Lightning after aliases are set up
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.plugins import TorchCheckpointIO

# Custom checkpoint plugin to load with weights_only=False for compatibility
class LegacyCheckpointIO(TorchCheckpointIO):
    """Custom checkpoint IO that loads with weights_only=False for old checkpoints."""
    
    def load_checkpoint(self, path, map_location=None, weights_only=None):
        # Override to always use weights_only=False for compatibility
        # This is safe since we trust the checkpoint source
        return super().load_checkpoint(path, map_location=map_location, weights_only=False)

#TODO implement parser convention with hyphens
#TODO continue!

# Set CUDA architecture list and float32 matmul precision high
from models.sgmse.util.other import set_torch_cuda_arch_list, pad_spec
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

# Suppress benign CUDA stream mismatch warning with DDP
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

from models.sgmse.backbones.shared import BackboneRegistry
from models.sgmse.sdes import SDERegistry

# Input constructor for ptflops MAC calculation
def input_constructor(input_res):
    """Create dummy inputs for MAC calculation with ptflops."""
    x1 = torch.randn(1, *input_res[0]) + 1j * torch.randn(*input_res[0])
    x2 = torch.randn(*input_res[1])
    return dict(x=x1, time_cond=x2)

#torch.manual_seed(0)
wandb.login()

def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--config", type=str, default=None, help="Path to TOML config file. Parameters in config file override defaults.")
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp_48k")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--audio-log-interval", type=int, default=5, help="Log audio every n epochs.")
          parser_.add_argument("--wandb-name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
          parser_.add_argument("--log-dir", type=str, default="logs", help="Directory to save logs.")
          parser_.add_argument("--save-ckpt-interval", type=int, default=50000, help="Save checkpoint interval.")
          parser_.add_argument("--start-with-validation", action='store_true', help="Start with validation before training.")
          parser_.add_argument("--run-id", type=str, default=None, help="Set run id so distributed training is logged on same run")
     temp_args, _ = base_parser.parse_known_args()
     
     # Load TOML config if provided
     config_dict = {}
     if temp_args.config:
          config_dict = toml.load(temp_args.config)
          print(f"Loaded config from {temp_args.config}")
          
          # Update temp_args with config values that affect which argument groups to add
          if 'backbone' in config_dict and '--backbone' not in sys.argv:
               temp_args.backbone = config_dict['backbone']
          if 'sde' in config_dict and '--sde' not in sys.argv:
               temp_args.sde = config_dict['sde']

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     trainer_parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max-epochs", type=int, default=550, help="Number of epochs to train.")
     
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     
     # Build argument list with TOML config values if not already provided via CLI
     argv_to_parse = sys.argv[1:]  # Exclude script name
     if config_dict:
          # Get list of args already in command line
          cli_args = set()
          i = 0
          while i < len(argv_to_parse):
               if argv_to_parse[i].startswith('--'):
                    cli_args.add(argv_to_parse[i])
               i += 1
          
          # Add TOML config values that aren't in CLI
          for key, value in config_dict.items():
               # Convert underscores to hyphens for command-line arguments
               arg_name = f"--{key.replace('_', '-')}"
               
               # Only add if not already in command line arguments
               if arg_name not in cli_args:
                    # Handle boolean values
                    if isinstance(value, bool):
                         if value:
                              argv_to_parse.append(arg_name)
                    # Handle lists
                    elif isinstance(value, list):
                         argv_to_parse.extend([arg_name] + [str(v) for v in value])
                    # Handle other values
                    else:
                         argv_to_parse.extend([arg_name, str(value)])
     
     # Parse args and separate into groups
     args = parser.parse_args(argv_to_parse)
     arg_groups = get_argparse_groups(parser)

     # kwargs = {
     #           **vars(arg_groups['ScoreModel']),
     #           **vars(arg_groups['SDE']),
     #           **vars(arg_groups['Backbone']),
     #           **vars(arg_groups['DataModule'])
     #      }
     # Initialize logger, trainer, model, datamodule

     kwargs = {'nolog': args.nolog, 'audio_log_interval': args.audio_log_interval, 'valid_audio_foldername': args.wandb_name}

     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule']),
               **kwargs
          }
     )

     # Verify pretrained weights loaded correctly (for LoRA models)
     # Only run this verification when starting fresh (not resuming from checkpoint)
     if args.backbone == "ncsnpp_48k_lora" and hasattr(args, 'lora_pretrained_checkpoint') and args.lora_pretrained_checkpoint and not args.ckpt:
          print(f"\n{'='*70}")
          print("VERIFYING PRETRAINED WEIGHTS IN LoRA MODEL")
          print(f"{'='*70}")
          
          # Load checkpoint to compare
          checkpoint = torch.load(args.lora_pretrained_checkpoint, map_location='cpu', weights_only=False)
          checkpoint_state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
          
          # Sample key to verify
          sample_key = 'dnn.all_modules.4.Conv_0.weight'
          if sample_key in checkpoint_state:
               checkpoint_weight = checkpoint_state[sample_key]
               
               # Access the PEFT base model
               if hasattr(model.dnn, 'model') and hasattr(model.dnn.model, 'get_base_model'):
                    base_model = model.dnn.model.get_base_model()
               elif hasattr(model.dnn, 'base_model'):
                    base_model = model.dnn.base_model
               else:
                    base_model = model.dnn
               
               # Get the corresponding weight from model
               model_weight = base_model.all_modules[4].Conv_0.weight
               
               print(f"Sample layer: {sample_key}")
               print(f"  Checkpoint weights (first 5): {checkpoint_weight.flatten()[:5]}")
               print(f"  LoRA model weights (first 5):  {model_weight.flatten()[:5]}")
               
               match = torch.allclose(checkpoint_weight, model_weight)
               print(f"  Weights match: {'✓ YES' if match else '✗ NO'}")
               
               if not match:
                    diff = (checkpoint_weight - model_weight).abs().max()
                    print(f"  Max difference: {diff}")
          
          print(f"{'='*70}\n")
          
          # Verify that only LoRA parameters are trainable
          print(f"\n{'='*70}")
          print("VERIFYING ONLY LoRA PARAMETERS ARE TRAINABLE")
          print(f"{'='*70}\n")
          
          trainable_params = []
          frozen_params = []
          
          for name, param in model.dnn.named_parameters():
               if param.requires_grad:
                    trainable_params.append(name)
               else:
                    frozen_params.append(name)
          
          # Check that base model parameters are frozen
          base_model_trainable = [n for n in trainable_params if 'lora' not in n.lower()]
          
          if base_model_trainable:
               print(f"⚠️  WARNING: Found {len(base_model_trainable)} base model parameters that are trainable!")
               print(f"   These should be frozen:")
               for name in base_model_trainable[:5]:
                    print(f"     - {name}")
               if len(base_model_trainable) > 5:
                    print(f"     ... and {len(base_model_trainable) - 5} more")
          else:
               print(f"✓ Base model parameters are frozen (not trainable)")
          
          # Show some examples of trainable LoRA parameters
          lora_trainable = [n for n in trainable_params if 'lora' in n.lower()]
          print(f"\n✓ LoRA parameters that are trainable: {len(lora_trainable)}")
          if lora_trainable:
               print(f"  Examples:")
               for name in lora_trainable[:5]:
                    print(f"    - {name}")
               if len(lora_trainable) > 5:
                    print(f"    ... and {len(lora_trainable) - 5} more")
          
          # Summary
          total_params = len(trainable_params) + len(frozen_params)
          trainable_count = sum(p.numel() for p in model.dnn.parameters() if p.requires_grad)
          total_count = sum(p.numel() for p in model.dnn.parameters())
          
          print(f"\nParameter Summary:")
          print(f"  Total parameters:     {total_count:,}")
          print(f"  Trainable parameters: {trainable_count:,} ({100*trainable_count/total_count:.2f}%)")
          print(f"  Frozen parameters:    {total_count - trainable_count:,} ({100*(1-trainable_count/total_count):.2f}%)")
          
          print(f"{'='*70}\n")
     
     # Calculate MAC (Multiply-Accumulate) operations using ptflops
     print(f"\n{'='*70}")
     print("CALCULATING MAC OPERATIONS (Multiply-Accumulate)")
     print(f"{'='*70}\n")
     
     import sys
     import io
     from ptflops import get_model_complexity_info
     
     # Setup data module to get sample data
     model.data_module.setup(stage='fit')
     dummy_batch = model.data_module.train_set.__getitem__(0)
     
     # Prepare dummy inputs (matching the model's forward signature)
     x_dummy = dummy_batch[0].to(device=model.device).unsqueeze(1)
     y_dummy = dummy_batch[1].to(device=model.device).unsqueeze(1)
     x_dummy = pad_spec(x_dummy, mode="reflection")
     y_dummy = pad_spec(y_dummy, mode="reflection")
     dummy_input = torch.cat([x_dummy, y_dummy], dim=1).to(device=model.device)
     dummy_input_2 = torch.randn(dummy_batch[0].shape[0]).to(device=model.device)
     
     # Suppress ptflops warnings by redirecting stderr temporarily
     # Calculate MACs (ignore parameter count from ptflops - it's unreliable for PEFT models)
     macs, _ = get_model_complexity_info(
          model.dnn,
          input_res=(dummy_input.shape[1:], dummy_input_2.shape),
          input_constructor=input_constructor,
          as_strings=False,
          print_per_layer_stat=False,  # Set to True for detailed layer-by-layer stats
          verbose=False,
     )

     # Get accurate parameter count ourselves (ptflops doesn't handle PEFT correctly)
     total_params = sum(p.numel() for p in model.dnn.parameters())
     trainable_params = sum(p.numel() for p in model.dnn.parameters() if p.requires_grad)
     
     # Get audio duration for per-second calculation
     audio_duration = dummy_batch[2].shape[-1] / args.sr  # audio samples / sample rate
     
     print(f"Model: {args.backbone}")
     print(f"Audio duration: {audio_duration:.2f} seconds")
     print(f"\nComputational Complexity:")
     print(f"  Total MACs:       {macs / 1e9:.2f} G (Giga MACs)")
     print(f"  Equivalent FLOPs: {2 * macs / 1e9:.2f} G (2 FLOPs = 1 MAC)")
     print(f"  MACs per second:  {macs / 1e9 / audio_duration:.2f} G/s")
     print(f"\nParameters:")
     print(f"  Total:            {total_params / 1e6:.2f} M")
     
     if hasattr(model.dnn, 'base_model'):  # LoRA model
          print(f"  Trainable:        {trainable_params / 1e6:.4f} M ({100*trainable_params/total_params:.4f}%)")
          print(f"  Frozen:           {(total_params - trainable_params) / 1e6:.2f} M ({100*(1-trainable_params/total_params):.2f}%)")
     
     print(f"\n{'='*70}\n")
     
     # Set up logger configuration
     if args.nolog:
          logger = None
     else:
          wandb_settings = wandb.Settings(init_timeout=120)
          logger = WandbLogger(project="finetuning-for-svs", log_model=False, save_dir="logs", name=args.wandb_name, id=args.run_id, settings=wandb_settings)
          # logger.experiment.log_code(".")
          # wandb.init(project="finetuning-for-svs", name=args.wandb_name, id=args.run_id, dir="logs", settings=wandb_settings)


     # Set up callbacks for logger
     if logger != None:
          callbacks = [ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), save_last=True, 
               filename='{epoch}-last')]
          callbacks += [ModelCheckpoint(dirpath=join(args.log_dir, f'{str(logger.version)}-{args.wandb_name}'),
               filename='{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval)]
          if args.num_eval_files:
               checkpoint_callback_sdr = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), 
                    save_top_k=1, monitor="sdr", mode="max", filename='{epoch}-{sdr:.2f}')
               checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), 
                    save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
               checkpoint_callback_multi_res_loss = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)),
                    save_top_k=1, monitor="multi_res_loss", mode="min", filename='{epoch}-{multi_res_loss:.2f}')
               checkpoint_callback_mert_mse = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)),
                    save_top_k=1, monitor="mert_mse", mode="min", filename='{epoch}-{mert_mse:.4f}')
               callbacks += [checkpoint_callback_sdr, checkpoint_callback_si_sdr, checkpoint_callback_multi_res_loss, checkpoint_callback_mert_mse]
     else:
          callbacks = None

     # Initialize the Trainer and the DataModule
     # Use custom checkpoint IO to load with weights_only=False for compatibility
     from pytorch_lightning.strategies import DDPStrategy
     ddp_strategy = DDPStrategy(checkpoint_io=LegacyCheckpointIO())
     
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy=ddp_strategy, logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks
     )

     # No manual checkpoint loading needed anymore!
     # For LoRA models, pretrained weights are loaded during model initialization
     # via the lora_pretrained_checkpoint parameter (happens BEFORE PEFT wrapping)
     
     # Run validation first if requested
     if args.start_with_validation:
          model.eval()
          trainer.validate(model, ckpt_path=args.ckpt)
     
     model.train(mode=True)
     # Train model - pass ckpt_path to properly resume training from checkpoint
     trainer.fit(model, ckpt_path=args.ckpt)
