#!/usr/bin/env python3
"""
Aggregate results from multiple CSV files and create violin plots.

This script:
1. Recursively finds all result CSV files from multi-iteration model evaluations
2. Aggregates results from all iterations for diffusion-based SGMSE models
3. Creates violin plots showing performance variation across iterations for each dataset
"""

import argparse
import os
import re
import sys
import types
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from collections import defaultdict


def find_all_result_csvs(base_dir, csv_name='results.csv', msrbench_csv_name='results_loudness_normalize.csv'):
    """
    Recursively find all result CSV files in the directory structure.
    
    Assumes structure like:
    test_sets/
        MSRBench_Vocals/
            sgm_scratch/
                iter_000_seed42/results.csv
                iter_001_seed43/results.csv
            ...
        gensvs_eval_audio/
            ...
        ears_wham_v2_test_5s/
            ...
    
    Returns list of tuples: (dataset_name, model_name, iteration, csv_path)
    """
    results = []
    
    test_sets_dir = Path(base_dir) / 'test_sets'
    
    if not test_sets_dir.exists():
        raise FileNotFoundError(f"test_sets directory not found at {test_sets_dir}")
    
    # Get all datasets
    datasets = [d for d in test_sets_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in sorted(datasets):
        dataset_name = dataset_dir.name
        dataset_csv_name = msrbench_csv_name if dataset_name == 'MSRBench_Vocals' else csv_name
        
        # Get all model directories
        model_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name

            # Always record a direct model-level CSV when present.
            # This is needed for single-run models that may also contain subfolders
            # (e.g., speaker folders in EARS-WHAM).
            direct_csv_path = model_dir / dataset_csv_name
            if direct_csv_path.exists():
                results.append((dataset_name, model_name, -1, str(direct_csv_path)))
            
            # Check if this is a multi-iteration model directory
            iteration_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            
            if iteration_dirs:
                # This is a multi-iteration model
                def extract_iteration_number(dir_name):
                    """Extract iteration number from directory name like 'iter_000_seed42' or '0'"""
                    # Try to extract from "iter_XXX_seedYY" format
                    match = re.search(r'iter[_-]?(\d+)', dir_name)
                    if match:
                        return int(match.group(1))
                    # Try simple numeric format
                    if dir_name.isdigit():
                        return int(dir_name)
                    return 1000  # Put unknown formats at the end
                
                for iter_dir in sorted(iteration_dirs, key=lambda x: extract_iteration_number(x.name)):
                    csv_path = iter_dir / dataset_csv_name
                    if csv_path.exists():
                        iteration = extract_iteration_number(iter_dir.name)
                        results.append((dataset_name, model_name, iteration, str(csv_path)))
            else:
                # Single-run model (no multi-iteration structure)
                pass
    
    return results


MODEL_DISPLAY_NAMES = {
    'sgm_scratch':          'SGMSVS\n(from scratch)',
    'sgm_lora_r16_no_lora': 'LoRA-SGMSVS\n(no LoRA)',
    'sgm_lora_r16':         'LoRA-SGMSVS\n(rank 16)',
    'sgm_lora_r16_adaptive':'LoRA-SGMSVS\n(rank 16)',
    'sgm_full':             'SGMSVS\n(full finetuning)',
    'sgm_base':             'SGMSE \n(base)',
}

BSRNN_MODEL_DISPLAY_NAMES = {
    'bsrnn_base': 'BSRNNSE \n(base)',
    'bsrnn_lora_r16': 'LoRA-BSRNNSVS\n(rank 16)',
    'bsrnn_lora_r32': 'LoRA-BSRNNSVS\n(rank 32)',
    'bsrnn_lora_r128': 'LoRA-BSRNNSVS\n(rank 128)',
    'bsrnn_lora_r16_no_lora': 'LoRA-BSRNNSVS\n(no LoRA)',
    # merged adaptive LoRA row (dataset-aware selection between w/ and no-LoRA)
    'bsrnn_lora_r16_adaptive':       'LoRA-BSRNNSVS\n(rank 16)',
    # virtual names for r32/r128 on EARS-WHAM (uses no_lora CSV, keeps rank label)
    'bsrnn_lora_r32_adaptive':   'LoRA-BSRNNSVS\n(rank 32)',
    'bsrnn_lora_r128_adaptive':  'LoRA-BSRNNSVS\n(rank 128)',
    'bsrnn_full': 'BSRNNSVS\n(full finetuning)',
    'bsrnn_scratch': 'BSRNNSVS\n(from scratch)',
}


def get_bsrnn_lora_rank(model_name):
    """Extract the LoRA rank from BSRNN model names like bsrnn_lora_r32."""
    match = re.fullmatch(r'bsrnn_lora_r(\d+)', model_name)
    if match:
        return int(match.group(1))
    if model_name == 'bsrnn_lora_r16':
        return 16
    return None

# Human-readable metric labels for LaTeX
METRIC_LABELS = {
    'sdr':             'SDR',
    'si_sdr':          'SI-SDR',
    'multi_res_loss':  'MR-Loss',
    'mert_mse':        'MERT-MSE',
    'pesq':            'PESQ',
    'stoi':            'STOI',
    'dnsmos_ovrl':     'DNSMOS',
    'distillmos':      'DistillMOS',
}

METRIC_DIRECTIONS = {
    'sdr': r'$\uparrow$',
    'si_sdr': r'$\uparrow$',
    'multi_res_loss': r'$\downarrow$',
    'mert_mse': r'$\downarrow$',
    'pesq': r'$\uparrow$',
    'stoi': r'$\uparrow$',
    'dnsmos_ovrl': r'$\uparrow$',
    'distillmos': r'$\uparrow$',
}

# Metrics where LOWER is better (used to determine bold formatting)
LOWER_IS_BETTER = {'multi_res_loss', 'mert_mse'}

# Preferred display order within a dataset
METRIC_ORDER = ['sdr', 'si_sdr', 'multi_res_loss', 'mert_mse',
                'pesq', 'stoi', 'dnsmos_ovrl', 'distillmos']

# Dataset display names for column headers
DATASET_LABELS = {
    'MSRBench_Vocals':    'MSRBench',
    'gensvs_eval_audio':  'GenSVS',
    'ears_wham_v2_test_5s': 'EARS-WHAM',
}

# Canonical model display order
MODEL_ORDER = [
    'SGMSVS\n(full finetuning)',
    'LoRA-SGMSVS\n(rank 16)',
    'SGMSVS\n(from scratch)',
    'SGMSE \n(base)',
]

BSRNN_MODEL_ORDER = [
    'BSRNNSVS\n(full finetuning)',
    'LoRA-BSRNNSVS\n(rank 128)',
    'LoRA-BSRNNSVS\n(rank 32)',
    'LoRA-BSRNNSVS\n(rank 16)',
    'BSRNNSVS\n(from scratch)',
    'BSRNNSE \n(base)',
]

SGM_LORA_W_MODELS = {'sgm_lora_r16'}
SGM_LORA_NO_MODELS = {'sgm_lora_r16_no_lora'}
BSRNN_LORA_W_MODELS = {'bsrnn_lora_r16'}
BSRNN_LORA_NO_MODELS = {'bsrnn_lora_r16_no_lora'}

SGM_LORA_MERGED_MODEL = 'sgm_lora_r16_adaptive'  # virtual key: dataset-aware w/ or no-LoRA selection
BSRNN_LORA_MERGED_MODEL = 'bsrnn_lora_r16_adaptive'

COMBINED_FAMILY_ROWSPAN = {
    'SGM': 6,
    'BSRNN': 9,
}


MODEL_CHECKPOINT_PATHS = {
    # SGM checkpoints
    'sgm_full':             'checkpoints/se2svs/sgm_full/epoch=900-sdr=8.35.ckpt',
    'sgm_lora_r16':         'checkpoints/se2svs/sgm_lora_r16/epoch=533-sdr=7.63.ckpt',
    'sgm_lora_r16_no_lora': 'checkpoints/se2svs/sgm_lora_r16/epoch=533-sdr=7.63.ckpt',
    'sgm_lora_r16_adaptive':'checkpoints/se2svs/sgm_lora_r16/epoch=533-sdr=7.63.ckpt',
    'sgm_scratch':          'checkpoints/se2svs/sgm_scratch/epoch=522-sdr=7.26.ckpt',
    'sgm_base':             'checkpoints/sgmse_pretrained/ears_wham.ckpt',
    # BSRNN checkpoints
    'bsrnn_full': 'checkpoints/se2svs/bsrnn_full/epoch=378-sdr=10.03.ckpt',
    'bsrnn_lora_r16': 'checkpoints/se2svs/bsrnn_lora_r16/epoch=503-sdr=8.94.ckpt',
    'bsrnn_lora_r16_no_lora': 'checkpoints/se2svs/bsrnn_lora_r16/epoch=503-sdr=8.94.ckpt',
    'bsrnn_lora_r16_adaptive': 'checkpoints/se2svs/bsrnn_lora_r16/epoch=503-sdr=8.94.ckpt',
    'bsrnn_lora_r32': 'checkpoints/se2svs/bsrnn_lora_r32/epoch=544-sdr=9.05.ckpt',
    'bsrnn_lora_r32_adaptive': 'checkpoints/se2svs/bsrnn_lora_r32/epoch=544-sdr=9.05.ckpt',
    'bsrnn_lora_r128': 'checkpoints/se2svs/bsrnn_lora_r128/epoch=543-sdr=9.15.ckpt',
    'bsrnn_lora_r128_adaptive': 'checkpoints/se2svs/bsrnn_lora_r128/epoch=543-sdr=9.15.ckpt',
    'bsrnn_scratch': 'checkpoints/se2svs/bsrnn_scratch/epoch=480-sdr=8.25.ckpt',
    'bsrnn_base': 'checkpoints/bsrnn_pretrained/bsrnn.ckpt',
}

# Precomputed total parameter counts used for table generation.
# Values are in absolute parameter counts (not millions).
PRECOMPUTED_PARAM_COUNTS = {
    # SGM models
    'sgm_full':             64.74e6,
    'sgm_scratch':          64.74e6,
    'sgm_base':             64.74e6,
    'sgm_lora_r16':         69.76e6,
    'sgm_lora_r16_no_lora': 69.76e6,
    # BSRNN models
    'bsrnn_full':           37.80e6,
    'bsrnn_scratch':        37.80e6,
    'bsrnn_base':           37.80e6,
    'bsrnn_lora_r16':       40.09e6,
    'bsrnn_lora_r16_no_lora': 40.09e6,
    'bsrnn_lora_r32':       42.38e6,
    'bsrnn_lora_r128':      56.13e6,
}

PRECOMPUTED_GMACS_PER_SECOND = {
    'SGMSVS (full finetuning)': 399.64,
    'LoRA-SGMSVS (rank 16)': 895.07,
    'SGMSVS (from scratch)': 399.64,
    'SGMSE (base)': 399.64,
    'BSRNNSVS (full finetuning)': 84.31,
    'LoRA-BSRNNSVS (rank 128)': 91.18,
    'LoRA-BSRNNSVS (rank 32)': 86.03,
    'LoRA-BSRNNSVS (rank 16)': 85.17,
    'BSRNNSVS (from scratch)': 84.31,
    'BSRNNSE (base)': 84.31,
}

_LEGACY_ALIASES_READY = False


def _register_legacy_checkpoint_aliases():
    """Register aliases/safe-globals needed to load legacy checkpoints."""
    global _LEGACY_ALIASES_READY
    if _LEGACY_ALIASES_READY:
        return

    import models.data_module
    import models.MSS_model
    import models.sgmse
    import models.sgmse.backbones
    import models.sgmse.sdes
    import models.sgmse.util

    sys.modules['sgmse'] = models.sgmse
    sys.modules['sgmse.data_module'] = models.data_module
    sys.modules['sgmse.model'] = models.MSS_model
    sys.modules['sgmse.sdes'] = models.sgmse.sdes
    sys.modules['sgmse.backbones'] = models.sgmse.backbones
    sys.modules['sgmse.util'] = models.sgmse.util

    package_name = 'baseline_code'
    module_name = 'baseline_code.config'

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

        LegacyConfig.__name__ = 'Config'
        LegacyConfig.__qualname__ = 'Config'
        LegacyConfig.__module__ = module_name
        module.Config = LegacyConfig
        sys.modules[module_name] = module
        setattr(package, 'config', module)

    config_cls = getattr(module, 'Config', None)
    if config_cls is not None:
        torch.serialization.add_safe_globals([config_cls])

    _LEGACY_ALIASES_READY = True


def get_model_parameter_count(raw_model_name):
    """Return total parameter count for a raw model name.

    Uses precomputed values (or family/rank heuristics) to avoid runtime model
    loading during table generation.
    """
    if raw_model_name in PRECOMPUTED_PARAM_COUNTS:
        return float(PRECOMPUTED_PARAM_COUNTS[raw_model_name])

    lower_name = raw_model_name.lower()
    if is_bsrnn_model(raw_model_name):
        if 'r128' in lower_name or 'rank_128' in lower_name:
            return 56.13e6
        if 'r32' in lower_name or 'rank_32' in lower_name:
            return 42.38e6
        if 'lora' in lower_name:
            return 40.09e6
        return 37.80e6

    if 'lora' in lower_name:
        return 69.76e6
    return 64.74e6


def get_display_name(model_name):
    """Return the display name for a model, falling back to the raw name."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def get_bsrnn_display_name(model_name):
    """Return BSRNN display name for a model, falling back to the raw name."""
    lora_rank = get_bsrnn_lora_rank(model_name)
    if lora_rank is not None:
        return f'LoRA-BSRNNSVS\n(rank {lora_rank})'
    return BSRNN_MODEL_DISPLAY_NAMES.get(model_name, model_name)


def is_diffusion_model(model_name):
    """Check if model is a diffusion-based SGMSE model."""
    return model_name.lower().startswith('sgm_')


def is_bsrnn_model(model_name):
    """Check if model is one of the BSRNN variants used in this study."""
    return model_name in BSRNN_MODEL_DISPLAY_NAMES or get_bsrnn_lora_rank(model_name) is not None


def metrics_for_dataset(dataset_name):
    """Return the metric list expected for a dataset."""
    if 'ears_wham' in dataset_name.lower():
        return ['si_sdr', 'pesq', 'stoi', 'dnsmos_ovrl', 'distillmos']
    # MSS datasets: prefer SDR and remove SI-SDR from aggregation outputs.
    return ['sdr', 'multi_res_loss', 'mert_mse']


def get_metric_label(metric):
    """Return a LaTeX metric label with an up/down direction marker."""
    base_label = METRIC_LABELS.get(metric, metric)
    direction = METRIC_DIRECTIONS.get(metric)
    if direction is None:
        return base_label
    return f'{base_label} {direction}'


def merge_lora_model_for_tables(dataset_name, model_name):
    """Return a list of model names to emit for this (dataset, model) pair.

    For GenSVS and MSRBench the w_lora variant is used; for EARS-WHAM the
    no_lora variant is used for all LoRA ranks (rank 16, 32, and 128).
    """
    prefer_with_lora = dataset_name in {'gensvs_eval_audio', 'MSRBench_Vocals'}
    prefer_no_lora = dataset_name == 'ears_wham_v2_test_5s'

    # SGM LoRA: w_lora → GenSVS/MSRBench only
    if model_name in SGM_LORA_W_MODELS:
        return [SGM_LORA_MERGED_MODEL] if prefer_with_lora else []
    # SGM LoRA: no_lora → EARS-WHAM only
    if model_name in SGM_LORA_NO_MODELS:
        return [SGM_LORA_MERGED_MODEL] if prefer_no_lora else []

    # BSRNN LoRA rank 16: w_lora → GenSVS/MSRBench only
    if model_name in BSRNN_LORA_W_MODELS:
        return [BSRNN_LORA_MERGED_MODEL] if prefer_with_lora else []
    # BSRNN LoRA rank 16: no_lora → EARS-WHAM for rank 16, 32, and 128
    if model_name in BSRNN_LORA_NO_MODELS:
        if prefer_no_lora:
            return [
                BSRNN_LORA_MERGED_MODEL,
                'bsrnn_lora_r32_adaptive',
                'bsrnn_lora_r128_adaptive',
            ]
        return []

    # BSRNN rank 32 / rank 128: only used for GenSVS/MSRBench (EARS-WHAM covered above)
    if model_name in {'bsrnn_lora_r32', 'bsrnn_lora_r128'}:
        return [] if prefer_no_lora else [model_name]

    return [model_name]


def build_table_csv_list_with_merged_lora(csv_list):
    """Apply dataset-aware LoRA merge policy to CSV entries used for table exports."""
    merged = []
    for dataset, model, iteration, csv_path in csv_list:
        for merged_model in merge_lora_model_for_tables(dataset, model):
            merged.append((dataset, merged_model, iteration, csv_path))
    return merged


def load_and_aggregate_results(csv_list):
    """
    Load all CSV files and aggregate into a single DataFrame.
    
    Args:
        csv_list: List of tuples (dataset_name, model_name, iteration, csv_path)
        
    Returns:
        Aggregated DataFrame with columns: dataset, model, iteration, and metric columns
    """
    all_data = []
    
    for dataset, model, iteration, csv_path in csv_list:
        df = pd.read_csv(csv_path)
        
        # Detect available metrics for this dataset
        metric_cols = metrics_for_dataset(dataset)
        
        available_metrics = [col for col in metric_cols if col in df.columns]
        
        if available_metrics:
            row = {'dataset': dataset, 'model': model, 'iteration': iteration}
            
            # Add mean and std for each metric
            for metric in available_metrics:
                # Filter out NaN values
                valid_values = df[metric].dropna()
                if len(valid_values) > 0:
                    row[f'{metric}_mean'] = valid_values.mean()
                    row[f'{metric}_std'] = valid_values.std()
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
            
            all_data.append(row)
    
    return pd.DataFrame(all_data)


def create_violin_plots(df, output_dir=None):
    """
    Create violin plots organized by dataset.
    
    Each plot contains subplots for all available metrics, showing variation across iterations
    and models within a single dataset. Subplot layout adapts based on number of metrics.
    Only metrics with actual data are displayed.
    
    Args:
        df: Aggregated DataFrame from load_and_aggregate_results
        output_dir: Directory to save plots (default: current directory)
    """
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter to only diffusion-based models
    df_diffusion = df[df['model'].apply(is_diffusion_model)].copy()
    
    if df_diffusion.empty:
        print("Warning: No diffusion-based models found in results")
        return
    
    # Filter to only multi-iteration models (iteration >= 0)
    df_diffusion = df_diffusion[df_diffusion['iteration'] >= 0].copy()
    
    if df_diffusion.empty:
        print("Warning: No multi-iteration diffusion models found in results")
        return
    
    datasets = sorted(df_diffusion['dataset'].unique())
    
    # Create plots for each dataset
    for dataset in datasets:
        # Filter data for this dataset
        data_for_dataset = df_diffusion[df_diffusion['dataset'] == dataset].copy()
        
        if data_for_dataset.empty:
            print(f"Skipping {dataset}: no data")
            continue
        
        # Apply display names
        data_for_dataset['model'] = data_for_dataset['model'].map(get_display_name)
        
        # Identify available metrics with actual data for this dataset
        all_metric_cols = [col.replace('_mean', '') for col in data_for_dataset.columns 
                           if col.endswith('_mean')]
        
        # Filter to only metrics that have at least some data (not all NaN)
        metric_cols = []
        for metric in all_metric_cols:
            metric_col = f'{metric}_mean'
            if metric_col in data_for_dataset.columns:
                if not data_for_dataset[metric_col].isna().all():
                    metric_cols.append(metric)
        
        if not metric_cols:
            print(f"Skipping {dataset}: no metrics with data")
            continue
        
        # Determine subplot layout based on actual number of metrics
        num_metrics = len(metric_cols)
        if num_metrics == 1:
            nrows, ncols = 1, 1
        elif num_metrics == 2:
            nrows, ncols = 1, 2
        elif num_metrics <= 4:
            nrows, ncols = 2, 2
        elif num_metrics <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 3, 3
        
        # Create subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(metric_cols):
            metric_col = f'{metric}_mean'
            ax = axes[idx]
            
            # Remove NaN values for this metric
            data_for_plot = data_for_dataset.dropna(subset=[metric_col]).copy()
            
            if data_for_plot.empty:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric)
                continue
            
            # Create violin plot
            sns.violinplot(data=data_for_plot, x='model', y=metric_col, ax=ax)
            ax.set_title(f'{metric}')
            ax.set_xlabel('')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=30)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
            ax.grid(True, axis='both')
            ax.set_axisbelow(True)
        
        # Hide unused subplots only if we have extras
        total_subplots = nrows * ncols
        for idx in range(num_metrics, total_subplots):
            axes[idx].axis('off')
        
        fig.suptitle(f'{dataset}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        # Sanitize dataset name for filename
        safe_dataset_name = dataset.replace('/', '_').replace(' ', '_')
        plot_path = os.path.join(output_dir, f'violin_plot_{safe_dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        plt.close()


def _create_column_delta_metric_plot(df, output_dir, metric, filename_stub, datasets_to_plot):
    """Create a 2-row conference-column delta-metric violin figure."""
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Keep only multi-iteration diffusion models
    df_diffusion = df[df['model'].apply(is_diffusion_model) & (df['iteration'] >= 0)].copy()
    if df_diffusion.empty:
        print(f"Warning: No multi-iteration diffusion models found for delta-{metric} violin plots")
        return

    # Use display names and fixed display order used throughout this script
    df_diffusion['model'] = df_diffusion['model'].map(get_display_name)
    metric_col = f'{metric}_mean'

    # Single-column conference figure size (approx. 3.5in width)
    nrows = len(datasets_to_plot)
    fig, axes = plt.subplots(nrows, 1, figsize=(3.5, 1.95 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    plotted_any = False
    for idx, dataset in enumerate(datasets_to_plot):
        ax = axes[idx]
        dataset_data = df_diffusion[df_diffusion['dataset'] == dataset].copy()
        if dataset_data.empty:
            ax.axis('off')
            continue
        if metric_col not in dataset_data.columns or dataset_data[metric_col].isna().all():
            ax.axis('off')
            continue

        plot_data = dataset_data.dropna(subset=[metric_col]).copy()
        if plot_data.empty:
            ax.axis('off')
            continue

        centered_col = f'{metric_col}_centered'
        medians = plot_data.groupby('model')[metric_col].transform('median')
        plot_data[centered_col] = plot_data[metric_col] - medians

        present_models = list(plot_data['model'].unique())
        model_order = [m for m in MODEL_ORDER if m in present_models]
        model_order += sorted(set(present_models) - set(model_order))

        sns.violinplot(
            data=plot_data,
            x='model',
            y=centered_col,
            order=model_order,
            inner='box',
            cut=0,
            linewidth=0.9,
            ax=ax,
        )

        dataset_label = DATASET_LABELS.get(dataset, dataset)
        metric_label = METRIC_LABELS.get(metric, metric)
        ax.set_title(f'{dataset_label}', fontsize=9, pad=4)
        ax.set_xlabel('')
        ax.set_ylabel(f'Δ {metric_label}', fontsize=8)
        ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.7)
        ax.tick_params(axis='y', labelsize=7)
        if idx < nrows - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.tick_params(axis='x', labelsize=7, rotation=28)
            plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"Warning: Could not create delta-{metric} violin plot (no valid data)")
        return

    fig.tight_layout(h_pad=0.35)

    png_path = os.path.join(output_dir, f'{filename_stub}.png')
    pdf_path = os.path.join(output_dir, f'{filename_stub}.pdf')
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved delta-{metric} violin plot: {png_path}")
    print(f"Saved delta-{metric} violin plot: {pdf_path}")
    plt.close()


def create_column_stacked_violin_plot(df, output_dir=None):
    """
    Create a conference-column figure for GenSVS and MSRBench.

    The figure visualizes iteration variance as delta SDR (median-centered per
    model) using a violin plot with an inner box representation.
    """
    _create_column_delta_metric_plot(
        df,
        output_dir,
        metric='sdr',
        filename_stub='violin_plot_iteration_variance_column',
        datasets_to_plot=['gensvs_eval_audio', 'MSRBench_Vocals'],
    )


def create_column_stacked_violin_plot_sisdr(df, output_dir=None):
    """Create the same conference-column figure style for delta SI-SDR."""
    _create_column_delta_metric_plot(
        df,
        output_dir,
        metric='si_sdr',
        filename_stub='violin_plot_iteration_variance_column_sisdr',
        datasets_to_plot=['gensvs_eval_audio', 'MSRBench_Vocals', 'ears_wham_v2_test_5s'],
    )


def create_combined_delta_sisdr_violin_plot(df, output_dir=None):
    """Create a single-axes delta SI-SDR violin plot with dataset color coding."""
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Keep only multi-iteration diffusion models
    df_diffusion = df[df['model'].apply(is_diffusion_model) & (df['iteration'] >= 0)].copy()
    if df_diffusion.empty:
        print("Warning: No multi-iteration diffusion models found for combined delta SI-SDR plot")
        return

    dataset_order = ['gensvs_eval_audio', 'MSRBench_Vocals', 'ears_wham_v2_test_5s']
    metric_col = 'si_sdr_mean'

    # Build long-form data for selected datasets
    data = df_diffusion[df_diffusion['dataset'].isin(dataset_order)].copy()
    if data.empty or metric_col not in data.columns:
        print("Warning: No SI-SDR data found for combined delta SI-SDR plot")
        return

    data = data.dropna(subset=[metric_col]).copy()
    if data.empty:
        print("Warning: SI-SDR has no valid values for combined delta SI-SDR plot")
        return

    data['model'] = data['model'].map(get_display_name)
    data['dataset_label'] = data['dataset'].map(lambda d: DATASET_LABELS.get(d, d))

    # Center each distribution around zero per (dataset, model)
    grouped_median = data.groupby(['dataset', 'model'])[metric_col].transform('median')
    data['delta_si_sdr'] = data[metric_col] - grouped_median

    # Keep deterministic display order
    present_models = list(data['model'].unique())
    model_order = [m for m in MODEL_ORDER if m in present_models]
    model_order += sorted(set(present_models) - set(model_order))
    hue_order = [DATASET_LABELS.get(d, d) for d in dataset_order if d in set(data['dataset'])]

    # Compact single-column figure
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    sns.violinplot(
        data=data,
        x='model',
        y='delta_si_sdr',
        hue='dataset_label',
        order=model_order,
        hue_order=hue_order,
        inner='box',
        cut=0,
        linewidth=0.9,
        dodge=True,
        ax=ax,
    )

    ax.set_title('All Datasets', fontsize=9, pad=4)
    ax.set_xlabel('')
    ax.set_ylabel('Δ SI-SDR', fontsize=8)
    ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.7)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7, rotation=28)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title='', fontsize=7, loc='upper right', frameon=True)

    fig.tight_layout(pad=0.4)

    png_path = os.path.join(output_dir, 'violin_plot_iteration_variance_combined_sisdr.png')
    pdf_path = os.path.join(output_dir, 'violin_plot_iteration_variance_combined_sisdr.pdf')
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved combined delta SI-SDR violin plot: {png_path}")
    print(f"Saved combined delta SI-SDR violin plot: {pdf_path}")
    plt.close()


def create_comparison_table(df, output_dir=None):
    """
    Create a summary table showing mean and std for each model and dataset.
    
    Handles different metrics for different datasets (e.g., speech metrics for ears_wham).
    
    Args:
        df: Aggregated DataFrame
        output_dir: Directory to save table (default: current directory)
    """
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter to only diffusion-based models with multiple iterations
    df_diffusion = df[df['model'].apply(is_diffusion_model) & (df['iteration'] >= 0)].copy()
    
    if df_diffusion.empty:
        print("Warning: No multi-iteration diffusion models for comparison table")
        return
    
    # Create per-dataset summary tables
    datasets = df_diffusion['dataset'].unique()
    
    for dataset in sorted(datasets):
        dataset_data = df_diffusion[df_diffusion['dataset'] == dataset].copy()
        
        # Get metrics available for this dataset
        metric_cols = [col.replace('_mean', '') for col in dataset_data.columns 
                       if col.endswith('_mean') and not dataset_data[col].isna().all()]
        
        summary_data = []
        for model in sorted(dataset_data['model'].unique()):
            subset = dataset_data[dataset_data['model'] == model]
            
            if not subset.empty:
                row = {'model': model}
                
                for metric in metric_cols:
                    metric_col = f'{metric}_mean'
                    if metric_col in subset.columns:
                        values = subset[metric_col].dropna()
                        if len(values) > 0:
                            row[f'{metric}_mean'] = f"{values.mean():.4f}"
                            row[f'{metric}_std'] = f"{values.std():.4f}"
                
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Save to CSV
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_')
            table_path = os.path.join(output_dir, f'summary_{safe_dataset_name}.csv')
            summary_df.to_csv(table_path, index=False)
            print(f"Saved summary table: {table_path}")


def export_gensvs_to_msrbench_delta_table(df, output_dir=None):
    """Export SDR and MERT-MSE deltas (MSRBench - GenSVS) for target model variants."""
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    source_dataset = 'gensvs_eval_audio'
    target_dataset = 'MSRBench_Vocals'
    required_metrics = ['sdr_mean', 'mert_mse_mean']

    # Build per-(dataset, model) averages from already aggregated rows.
    grouped_means = (
        df.groupby(['dataset', 'model'], as_index=False)[required_metrics]
        .mean(numeric_only=True)
    )

    model_specs = [
        ('Generative (SGM)', 'from scratch',    ['sgm_scratch']),
        ('Generative (SGM)', 'full finetuning', ['sgm_full']),
        ('Generative (SGM)', 'LoRA 16', [
            'sgm_lora_r16_adaptive',
            'sgm_lora_r16',
        ]),
        ('Discriminative (BSRNN)', 'from scratch',    ['bsrnn_scratch']),
        ('Discriminative (BSRNN)', 'full finetuning', ['bsrnn_full']),
        ('Discriminative (BSRNN)', 'LoRA 16', [
            'bsrnn_lora_r16_adaptive',
            'bsrnn_lora_r16',
        ]),
    ]

    rows = []
    for family, variant, aliases in model_specs:
        model_rows = grouped_means[grouped_means['model'].isin(aliases)]

        gensvs = model_rows[model_rows['dataset'] == source_dataset]
        msrbench = model_rows[model_rows['dataset'] == target_dataset]

        if gensvs.empty or msrbench.empty:
            row = {
                'family': family,
                'variant': variant,
                'gensvs_model_key': aliases[0],
                'msrbench_model_key': aliases[0],
                'sdr_gensvs_mean': np.nan,
                'sdr_msrbench_mean': np.nan,
                'sdr_delta_msrbench_minus_gensvs': np.nan,
                'mert_mse_gensvs_mean': np.nan,
                'mert_mse_msrbench_mean': np.nan,
                'mert_mse_delta_msrbench_minus_gensvs': np.nan,
            }
            rows.append(row)
            continue

        gensvs_row = gensvs.iloc[0]
        msrbench_row = msrbench.iloc[0]

        sdr_g = float(gensvs_row['sdr_mean'])
        sdr_m = float(msrbench_row['sdr_mean'])
        mert_g = float(gensvs_row['mert_mse_mean'])
        mert_m = float(msrbench_row['mert_mse_mean'])

        rows.append({
            'family': family,
            'variant': variant,
            'gensvs_model_key': str(gensvs_row['model']),
            'msrbench_model_key': str(msrbench_row['model']),
            'sdr_gensvs_mean': sdr_g,
            'sdr_msrbench_mean': sdr_m,
            'sdr_delta_msrbench_minus_gensvs': sdr_m - sdr_g,
            'mert_mse_gensvs_mean': mert_g,
            'mert_mse_msrbench_mean': mert_m,
            'mert_mse_delta_msrbench_minus_gensvs': mert_m - mert_g,
        })

    out_df = pd.DataFrame(rows)
    return out_df


def _delta_variant_to_compact_model_label(variant):
    """Map delta-table variant names to combined-table model labels."""
    mapping = {
        'full finetuning': 'full finetuning',
        'LoRA 16': 'LoRA (rank 16)',
        'from scratch': 'from scratch',
    }
    return mapping.get(variant, variant)


def create_latex_table_gensvs_to_msrbench_delta(
    delta_df,
    output_dir=None,
    output_filename='results_table_gensvs_to_msrbench_delta.tex',
):
    """Create a LaTeX table with SVS deltas from GenSVS to MSRBench."""
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if delta_df is None or delta_df.empty:
        print('Warning: No data for GenSVS->MSRBench delta LaTeX table')
        return

    family_sections = [
        ('SG Models', 'Generative (SGM)'),
        ('BSRNN Models', 'Discriminative (BSRNN)'),
    ]
    variant_order = ['full finetuning', 'LoRA 16', 'from scratch']
    metric_columns = [
        ('sdr_delta_msrbench_minus_gensvs', 'sdr', 'Delta SDR', 'max'),
        ('mert_mse_delta_msrbench_minus_gensvs', 'mert_mse', 'Delta MSE', 'min'),
    ]

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{SVS delta metrics from GenSVS to MSRBench (MSRBench - GenSVS) '
        r'for selected SGM and BSRNN model variants. Bold values indicate the '
        r'better model per row and metric ($\Delta$SDR: higher is better, '
        r'$\Delta$MSE: lower is better).}'
    )
    lines.append(r'  \resizebox{\columnwidth}{!}{%')
    lines.append(r'  \setlength{\tabcolsep}{4pt}%')
    lines.append(r'  \begin{tabular}{lrrrr}')
    lines.append(r'       & \multicolumn{2}{c}{\textbf{SGM}} & \multicolumn{2}{c}{\textbf{BSRNN}} \\')
    lines.append(r'    \cmidrule(lr){2-3} \cmidrule(lr){4-5}')
    lines.append(r'    Model variant & $\Delta$ SDR & $\Delta$ MSE & $\Delta$ SDR & $\Delta$ MSE \\')
    lines.append(r'    \midrule')

    value_lookup = {}
    for _, family_name in family_sections:
        fam_df = delta_df[delta_df['family'] == family_name]
        for _, row in fam_df.iterrows():
            variant = row['variant']
            for metric_col, _, _, _ in metric_columns:
                value_lookup[(family_name, variant, metric_col)] = row.get(metric_col, np.nan)

    for variant in variant_order:
        model_label = _delta_variant_to_compact_model_label(variant)
        cells = [f'    {model_label}']

        sg_cells = []
        bsrnn_cells = []
        for metric_col, metric_name, _, better_rule in metric_columns:
            sg_family = 'Generative (SGM)'
            bsrnn_family = 'Discriminative (BSRNN)'
            sg_val = value_lookup.get((sg_family, variant, metric_col), np.nan)
            bsrnn_val = value_lookup.get((bsrnn_family, variant, metric_col), np.nan)

            sg_fmt = '--'
            bsrnn_fmt = '--'
            if pd.notna(sg_val):
                sg_fmt = _fmt(float(sg_val), metric_name)
            if pd.notna(bsrnn_val):
                bsrnn_fmt = _fmt(float(bsrnn_val), metric_name)

            # Exactly one bold entry per row+metric. If tied, prefer SG.
            if pd.notna(sg_val) and pd.notna(bsrnn_val):
                if better_rule == 'min':
                    sg_wins = float(sg_val) <= float(bsrnn_val)
                else:
                    sg_wins = float(sg_val) >= float(bsrnn_val)

                if sg_wins:
                    sg_fmt = f'\\textbf{{{sg_fmt}}}'
                else:
                    bsrnn_fmt = f'\\textbf{{{bsrnn_fmt}}}'
            elif pd.notna(sg_val):
                sg_fmt = f'\\textbf{{{sg_fmt}}}'
            elif pd.notna(bsrnn_val):
                bsrnn_fmt = f'\\textbf{{{bsrnn_fmt}}}'

            sg_cells.append(sg_fmt)
            bsrnn_cells.append(bsrnn_fmt)

        cells.extend(sg_cells)
        cells.extend(bsrnn_cells)

        lines.append(' & '.join(cells) + r' \\[5pt]')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }%')
    lines.append(r'\end{table}')

    tex_path = os.path.join(output_dir, output_filename)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved LaTeX delta table: {tex_path}')


def _fmt(value, metric):
    """Format a numeric value: 3 dp for MR-Loss and MERT-MSE, 2 dp otherwise."""
    if metric in {'multi_res_loss', 'mert_mse'}:
        return f'{value:.3f}'
    return f'{value:.2f}'


def _rounded(value, metric):
    """Rounded numeric value used for tie-aware best-value selection."""
    if metric in {'multi_res_loss', 'mert_mse'}:
        return round(float(value), 3)
    return round(float(value), 2)


def compact_combined_model_label(display_model, family):
    """Return compact model labels for the combined-family LaTeX table."""
    name = display_model.replace('\n', ' ').strip()
    lowered = name.lower()

    if 'lora' in lowered:
        rank_match = re.search(r'rank\s*(\d+)', lowered)
        if rank_match:
            return f'LoRA (rank {rank_match.group(1)})'
        return 'LoRA'
    if 'full finetuning' in lowered:
        return 'full finetuning'
    if 'from scratch' in lowered:
        return 'from scratch'
    if 'no finetuning' in lowered or lowered == 'base' or '(base)' in lowered:
        return 'base'

    if family == 'SGM' and name.startswith('SGMSVS '):
        return name.replace('SGMSVS ', '', 1)
    if family == 'BSRNN' and (name.startswith('BSRNNSVS ') or name.startswith('BSRNNSE ')):
        return re.sub(r'^BSRNN(?:SVS|SE)\s+', '', name)
    return name


def _normalize_model_name_for_complexity_lookup(name):
    """Normalize display names to match ptflops complexity summary entries."""
    normalized = re.sub(r'\s+', ' ', name.replace('\n', ' ')).strip()
    return normalized.replace('(no finetuning)', '(base)')


def _canonical_model_name_from_compact_label(model_label, family_key):
    """Map compact combined-table labels to canonical names used in complexity tables."""
    lowered = model_label.lower().strip()

    if lowered == 'full finetuning':
        return 'SGMSVS (full finetuning)' if family_key == 'SGM' else 'BSRNNSVS (full finetuning)'
    if lowered == 'from scratch':
        return 'SGMSVS (from scratch)' if family_key == 'SGM' else 'BSRNNSVS (from scratch)'
    if lowered in {'no finetuning', 'base'}:
        return 'SGMSE (base)' if family_key == 'SGM' else 'BSRNNSE (base)'

    rank_match = re.search(r'rank\s*(\d+)', lowered)
    if rank_match:
        rank = rank_match.group(1)
        if family_key == 'SGM':
            return f'LoRA-SGMSVS (rank {rank})'
        return f'LoRA-BSRNNSVS (rank {rank})'

    return _normalize_model_name_for_complexity_lookup(model_label)


def load_model_complexity_summary(output_dir=None):
    """Load GMACs/s values from the ptflops summary CSV, with a precomputed fallback."""
    candidate_paths = []
    if output_dir is not None:
        candidate_paths.append(Path(output_dir) / 'model_complexity_summary.csv')
    candidate_paths.append(Path('aggregated_results') / 'model_complexity_summary.csv')

    for csv_path in candidate_paths:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if 'model' not in df.columns or 'gmacs_per_s' not in df.columns:
            continue

        summary = {}
        for _, row in df.iterrows():
            model_name = _normalize_model_name_for_complexity_lookup(str(row['model']))
            gmacs = pd.to_numeric(row['gmacs_per_s'], errors='coerce')
            if pd.notna(gmacs):
                summary[model_name] = float(gmacs)
        if summary:
            return summary

    return dict(PRECOMPUTED_GMACS_PER_SECOND)


def create_latex_table(
    df,
    output_dir=None,
    model_filter_fn=is_diffusion_model,
    display_name_fn=get_display_name,
    model_order=MODEL_ORDER,
    output_filename='results_table.tex',
    require_multi_iteration=True,
    dataset_order=None,
    include_all_datasets=False,
):
    """
    Export a single LaTeX table with sub-multicolumns for each dataset.
    Best value per metric/dataset is bolded. Wrapped in resizebox.
    """
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_mask = df['model'].apply(model_filter_fn)
    if require_multi_iteration:
        model_mask = model_mask & (df['iteration'] >= 0)
    df_diff = df[model_mask].copy()
    if df_diff.empty:
        print('Warning: No matching models for LaTeX table')
        return

    # Apply display names
    df_diff['model'] = df_diff['model'].map(display_name_fn)

    # Build per-dataset aggregated stats: mean of per-iteration means
    # Default display order: GenSVS, MSRBench, EARS-WHAM
    if dataset_order is None:
        dataset_order = ['gensvs_eval_audio', 'MSRBench_Vocals', 'ears_wham_v2_test_5s']
    available = set(df_diff['dataset'].unique())
    if include_all_datasets:
        datasets = list(dataset_order)
        datasets += sorted(available - set(datasets))
    else:
        datasets = [d for d in dataset_order if d in available] + \
                   sorted(available - set(dataset_order))

    # Collect which metrics exist per dataset (respecting preferred order)
    dataset_metrics = {}
    for dataset in datasets:
        sub = df_diff[df_diff['dataset'] == dataset]
        if sub.empty and include_all_datasets:
            dataset_metrics[dataset] = [m for m in METRIC_ORDER if m in metrics_for_dataset(dataset)]
        else:
            available = [m for m in METRIC_ORDER
                         if f'{m}_mean' in sub.columns and not sub[f'{m}_mean'].isna().all()]
            dataset_metrics[dataset] = available

    # Aggregate: for each (dataset, display_model) compute mean across iterations
    agg = {}
    for dataset in datasets:
        sub = df_diff[df_diff['dataset'] == dataset]
        agg[dataset] = {}
        for model in sub['model'].unique():
            msub = sub[sub['model'] == model]
            agg[dataset][model] = {}
            for metric in dataset_metrics[dataset]:
                vals = msub[f'{metric}_mean'].dropna()
                if len(vals):
                    std_ddof = 1 if require_multi_iteration else 0
                    agg[dataset][model][metric] = {
                        'mean': vals.mean(),
                        'std': vals.std(ddof=std_ddof),
                    }

    # Determine canonical model list (intersection of all datasets, ordered)
    all_models = set()
    for dataset in datasets:
        all_models |= set(agg[dataset].keys())
    # Sort by configured model order, then any remaining alphabetically
    ordered_models = [m for m in model_order if m in all_models]
    ordered_models += sorted(all_models - set(ordered_models))

    # ----- build LaTeX -----
    # Count total metric columns
    col_counts = [len(dataset_metrics[d]) for d in datasets]
    total_metric_cols = sum(col_counts)

    lines = []
    lines.append(r'\begin{table*}[ht]')
    lines.append(r'  \centering')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \setlength{\tabcolsep}{3pt}%')
    lines.append(r'  \begin{tabular}{l' + 'r' * total_metric_cols + '}')

    # Row 1: dataset multicolumn headers
    header1 = ['  Model']
    for dataset, ncols in zip(datasets, col_counts):
        label = DATASET_LABELS.get(dataset, dataset)
        header1.append(f'\\multicolumn{{{ncols}}}{{c}}{{{label}}}')
    lines.append('    ' + ' & '.join(header1) + ' \\\\')

    # Cmidrule separators under each dataset block
    col_cursor = 2  # model col is 1
    cmidrules = []
    for ncols in col_counts:
        cmidrules.append(f'\\cmidrule(lr){{{col_cursor}-{col_cursor + ncols - 1}}}')
        col_cursor += ncols
    lines.append('    ' + ' '.join(cmidrules))

    # Row 2: metric sub-headers
    header2 = ['  ']
    for dataset in datasets:
        for metric in dataset_metrics[dataset]:
            header2.append(get_metric_label(metric))
    lines.append('    ' + ' & '.join(header2) + ' \\\\')
    lines.append(r'    \midrule')

    # Find best rounded value per (dataset, metric) for bolding ties after rounding
    best = {}
    for dataset in datasets:
        best[dataset] = {}
        for metric in dataset_metrics[dataset]:
            vals = {m: _rounded(agg[dataset][m][metric]['mean'], metric)
                    for m in ordered_models
                    if m in agg[dataset] and metric in agg[dataset][m]}
            if not vals:
                continue
            if metric in LOWER_IS_BETTER:
                best[dataset][metric] = min(vals.values())
            else:
                best[dataset][metric] = max(vals.values())

    # Data rows
    for model in ordered_models:
        # Single-line model name for LaTeX (replace \n with space)
        model_tex = model.replace('\n', ' ')
        cells = [f'  {model_tex}']
        for dataset in datasets:
            for metric in dataset_metrics[dataset]:
                if model in agg[dataset] and metric in agg[dataset][model]:
                    stats = agg[dataset][model][metric]
                    mean_val = stats['mean']
                    std_val = stats['std']
                    formatted = (
                        f"{_fmt(mean_val, metric)}"
                        f"{{\\scriptsize $\\pm$ {_fmt(std_val, metric)}}}"
                    )
                    if _rounded(mean_val, metric) == best[dataset].get(metric):
                        formatted = f'{{\\boldmath {formatted}}}'
                    cells.append(formatted)
                else:
                    cells.append('--')
        lines.append('    ' + ' & '.join(cells) + ' \\\\[5pt]')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }%')
    lines.append(r'\end{table*}')

    tex = '\n'.join(lines) + '\n'

    tex_path = os.path.join(output_dir, output_filename)
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f'Saved LaTeX table: {tex_path}')


def create_latex_table_combined_families(
    csv_list,
    output_dir=None,
    output_filename='results_table_all_models.tex',
    base_dir='.',
    dataset_order=None,
    include_all_datasets=True,
):
    """Export one LaTeX table with SGM and BSRNN sections using dataset-level std."""
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def classify_family(model_name, iteration):
        if is_bsrnn_model(model_name):
            return 'BSRNN'
        if is_diffusion_model(model_name) and iteration >= 0:
            return 'SGM'
        return None

    def display_name_for_family(model_name, family):
        if family == 'BSRNN':
            return get_bsrnn_display_name(model_name)
        return get_display_name(model_name)

    filtered = [
        (dataset, model, iteration, csv_path)
        for dataset, model, iteration, csv_path in csv_list
        if classify_family(model, iteration) is not None
    ]
    if not filtered:
        print('Warning: No matching models for combined-family LaTeX table')
        return

    if dataset_order is None:
        dataset_order = ['gensvs_eval_audio', 'MSRBench_Vocals', 'ears_wham_v2_test_5s']
    available = set(dataset for dataset, _, _, _ in filtered)
    if include_all_datasets:
        datasets = list(dataset_order)
        datasets += sorted(available - set(datasets))
    else:
        datasets = [d for d in dataset_order if d in available] + \
                   sorted(available - set(dataset_order))

    dataset_metrics = {}
    for dataset in datasets:
        dataset_metrics[dataset] = [m for m in METRIC_ORDER if m in metrics_for_dataset(dataset)]

    agg = {}
    model_families = {}
    display_to_raw_model = {}
    grouped = defaultdict(list)
    for dataset, model, iteration, csv_path in filtered:
        family = classify_family(model, iteration)
        grouped[(dataset, model, family)].append((iteration, csv_path))

    for dataset in datasets:
        agg[dataset] = {}
        models_in_dataset = sorted(
            {(model, family) for (d, model, family) in grouped.keys() if d == dataset}
        )
        for raw_model, family in models_in_dataset:
            display_model = display_name_for_family(raw_model, family)
            model_families[display_model] = family
            display_to_raw_model.setdefault(display_model, raw_model)
            agg[dataset][display_model] = {}
            iter_entries = sorted(grouped[(dataset, raw_model, family)], key=lambda x: x[0])

            for metric in dataset_metrics[dataset]:
                per_iter_series = []
                for _, csv_path in iter_entries:
                    df_iter = pd.read_csv(csv_path)

                    if metric not in df_iter.columns:
                        continue

                    if 'file_id' in df_iter.columns:
                        series = pd.to_numeric(df_iter[metric], errors='coerce')
                        series.index = df_iter['file_id']
                    else:
                        series = pd.to_numeric(df_iter[metric], errors='coerce')

                    series = series.dropna()
                    if len(series) > 0:
                        per_iter_series.append(series)

                if not per_iter_series:
                    continue

                stacked = pd.concat(per_iter_series, axis=1)
                per_file_mean_over_iters = stacked.mean(axis=1, skipna=True)
                if len(per_file_mean_over_iters) == 0:
                    continue

                agg[dataset][display_model][metric] = {
                    'mean': per_file_mean_over_iters.mean(),
                    'std': per_file_mean_over_iters.std(),
                }

    family_sections = [
        ('SG Models', MODEL_ORDER, 'SGM'),
        ('BSRNN Models', BSRNN_MODEL_ORDER, 'BSRNN'),
    ]
    noisy_model_name = 'Noisy'

    family_models = {}
    all_models = set()
    for dataset in datasets:
        all_models |= set(agg[dataset].keys())

    noisy_stats = {}

    # Add dataset-level noisy baseline metrics, if available.
    # Expected location: <base_dir>/test_sets/<dataset>/noisy_metrics.csv
    for dataset in datasets:
        noisy_csv_path = Path(base_dir) / 'test_sets' / dataset / 'noisy_metrics.csv'
        if not noisy_csv_path.exists():
            continue

        try:
            noisy_df = pd.read_csv(noisy_csv_path)
        except Exception as exc:
            print(f"Warning: Failed to read noisy metrics CSV for {dataset}: {noisy_csv_path} ({exc})")
            continue

        noisy_stats[dataset] = {}
        for metric in dataset_metrics[dataset]:
            if metric not in noisy_df.columns:
                continue
            values = pd.to_numeric(noisy_df[metric], errors='coerce').dropna()
            if len(values) == 0:
                continue
            noisy_stats[dataset][metric] = {
                'mean': values.mean(),
                'std': values.std(),
            }

    for _, family_order, family_key in family_sections:
        present = {model for model in all_models if model_families.get(model) == family_key}
        ordered = [model for model in family_order if model in present]
        ordered += sorted(present - set(ordered))
        family_models[family_key] = ordered

    col_counts = [len(dataset_metrics[d]) for d in datasets]
    total_metric_cols = sum(col_counts)

    lines = []
    lines.append(r'\begin{table*}[ht]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{Combined SGM and BSRNN results across GenSVS, MSRBench, and EARS-WHAM datasets. '
        r'For the evaluation on MSRBench, predictions were loudness-normalized to match the target loudness before evaluation, '
        r'since MR-Loss is sensitive to scaling mismatches and the MSRBench dataset contains targets that are not matched with the mixture. '
        r'For the LoRA models evaluated on EARS-WHAM, the LoRA adapter was disabled to retrieve the full speech enhancement results of the base model.}'
    )
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \setlength{\tabcolsep}{3pt}%')
    lines.append(r'  \begin{tabular}{ll' + 'r' * total_metric_cols + '}')

    header1 = ['  ', '  ']
    for dataset, ncols in zip(datasets, col_counts):
        label = DATASET_LABELS.get(dataset, dataset)
        header1.append(f'\\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{label}}}}}')
    lines.append('    ' + ' & '.join(header1) + ' \\\\')

    col_cursor = 3
    cmidrules = []
    for ncols in col_counts:
        cmidrules.append(f'\\cmidrule(lr){{{col_cursor}-{col_cursor + ncols - 1}}}')
        col_cursor += ncols
    lines.append('    ' + ' '.join(cmidrules))

    header2 = ['  ', '  ']
    for dataset in datasets:
        for metric in dataset_metrics[dataset]:
            header2.append(get_metric_label(metric))
    lines.append('    ' + ' & '.join(header2) + ' \\\\')
    lines.append(r'    \midrule')

    best = {}
    for _, _, family_key in family_sections:
        best[family_key] = {}
        for dataset in datasets:
            best[family_key][dataset] = {}
            for metric in dataset_metrics[dataset]:
                vals = {
                    model: _rounded(agg[dataset][model][metric]['mean'], metric)
                    for model in family_models[family_key]
                    if model in agg[dataset] and metric in agg[dataset][model]
                }
                if not vals:
                    continue
                if metric in LOWER_IS_BETTER:
                    best[family_key][dataset][metric] = min(vals.values())
                else:
                    best[family_key][dataset][metric] = max(vals.values())

    for section_idx, (section_label, _, family_key) in enumerate(family_sections):
        section_models = family_models[family_key]
        if not section_models:
            continue

        for model_idx, model in enumerate(section_models):
            model_tex = compact_combined_model_label(model, family_key)
            if model_idx == 0:
                rowspan = max(len(section_models), COMBINED_FAMILY_ROWSPAN.get(family_key, len(section_models)))
                family_cell = (
                    f'\\multirow{{{rowspan}}}{{*}}'
                    f'{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{section_label}}}}}}}'
                )
            else:
                family_cell = ''

            cells = [family_cell, f'  {model_tex}']
            for dataset in datasets:
                for metric in dataset_metrics[dataset]:
                    if model in agg[dataset] and metric in agg[dataset][model]:
                        stats = agg[dataset][model][metric]
                        mean_val = stats['mean']
                        std_val = stats['std']
                        formatted = (
                            f"{_fmt(mean_val, metric)}"
                            f"{{\\scriptsize $\\pm$ {_fmt(std_val, metric)}}}"
                        )
                        if _rounded(mean_val, metric) == best[family_key][dataset].get(metric):
                            formatted = f'\\textbf{{{formatted}}}'
                        cells.append(formatted)
                    else:
                        cells.append('--')
            lines.append('    ' + ' & '.join(cells) + ' \\\\[5pt]')

        if section_idx < len(family_sections) - 1:
            lines.append(r'    \midrule')

    # Add a single bottom noisy row with horizontal lines above and below.
    has_noisy_data = any(noisy_stats.get(dataset) for dataset in datasets)
    if has_noisy_data:
        lines.append(r'    \hline')
        noisy_cells = ['', f'  {noisy_model_name}']
        for dataset in datasets:
            for metric in dataset_metrics[dataset]:
                if metric in noisy_stats.get(dataset, {}):
                    stats = noisy_stats[dataset][metric]
                    noisy_cells.append(
                        f"{_fmt(stats['mean'], metric)}"
                        f"{{\\scriptsize $\\pm$ {_fmt(stats['std'], metric)}}}"
                    )
                else:
                    noisy_cells.append('--')
        lines.append('    ' + ' & '.join(noisy_cells) + ' \\\\[5pt]')
        lines.append(r'    \hline')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }%')
    lines.append(r'\end{table*}')

    tex = '\n'.join(lines) + '\n'
    tex_path = os.path.join(output_dir, output_filename)
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f'Saved LaTeX table (combined families): {tex_path}')
    
    return family_sections, family_models, display_to_raw_model, model_families



def create_parameter_table(
    family_sections,
    family_models,
    display_to_raw_model,
    model_families,
    output_dir=None,
    output_filename='model_parameters.tex',
):
    """Create a LaTeX table with model names, total parameters, added parameters (%), and complexity."""
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    param_cache = {}
    complexity_lookup = load_model_complexity_summary(output_dir)

    def get_cached_params(raw_model):
        if raw_model not in param_cache:
            param_cache[raw_model] = get_model_parameter_count(raw_model)
        return param_cache[raw_model]

    # Compute total params and added params percentages
    model_params_millions = {}
    added_param_pct = {}
    model_complexity_gmacs = {}

    for _, _, family_key in family_sections:
        section_models = family_models.get(family_key, [])
        baseline_display = next(
            (m for m in section_models if 'no finetuning' in m.lower() or 'base' in m.lower()),
            None,
        )
        baseline_raw = display_to_raw_model.get(baseline_display) if baseline_display else None
        baseline_params = get_cached_params(baseline_raw) if baseline_raw else np.nan

        for display_model in section_models:
            raw_model = display_to_raw_model.get(display_model)
            model_params = get_cached_params(raw_model) if raw_model else np.nan
            if pd.notna(model_params):
                model_params_millions[display_model] = model_params / 1e6
            else:
                model_params_millions[display_model] = np.nan

            if pd.notna(model_params) and pd.notna(baseline_params) and baseline_params > 0:
                added_param_pct[display_model] = ((model_params - baseline_params) / baseline_params) * 100.0
            else:
                added_param_pct[display_model] = np.nan

            canonical_name = _canonical_model_name_from_compact_label(
                compact_combined_model_label(display_model, family_key),
                family_key,
            )
            model_complexity_gmacs[display_model] = complexity_lookup.get(canonical_name, np.nan)

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Model parameter counts and complexity: total parameters (M), relative parameters added compared to the base model within each model family, and model complexity measured with ptflops in GMACs/s.}')
    lines.append(r'  \label{tab:model_parameters}')
    lines.append(r'  \resizebox{0.8\columnwidth}{!}{%')
    lines.append(r'  \setlength{\tabcolsep}{4pt}%')
    lines.append(r'  \begin{tabular}{llrrr}')
    lines.append(r'    \toprule')
    lines.append(r'    & Model & \shortstack[c]{Total\\Params\\{[M]}} & \shortstack[c]{Add.\\Params\\{[\%]}} & \shortstack[c]{Complexity\\{[GMACs/s]}} \\')
    lines.append(r'    \midrule')

    for section_idx, (section_label, _, family_key) in enumerate(family_sections):
        section_models = family_models[family_key]
        if not section_models:
            continue

        for model_idx, model in enumerate(section_models):
            model_tex = compact_combined_model_label(model, family_key)
            total_params = model_params_millions.get(model, np.nan)
            total_params_tex = f'{total_params:.2f}' if pd.notna(total_params) else '--'
            added_params = added_param_pct.get(model, np.nan)
            added_params_tex = f'{added_params:.2f}' if pd.notna(added_params) else '--'
            complexity = model_complexity_gmacs.get(model, np.nan)
            complexity_tex = f'{complexity:.2f}' if pd.notna(complexity) else '--'

            if model_idx == 0:
                rowspan = len(section_models)
                family_cell = (
                    f'\\multirow{{{rowspan}}}{{*}}'
                    f'{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{section_label}}}}}}}'
                )
            else:
                family_cell = ''

            cells = [family_cell, f'{model_tex}', total_params_tex, added_params_tex, complexity_tex]
            lines.append('    ' + ' & '.join(cells) + ' \\\\')

        if section_idx < len(family_sections) - 1:
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }%')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines) + '\n'
    tex_path = os.path.join(output_dir, output_filename)
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f'Saved LaTeX parameter table: {tex_path}')


def create_latex_table_dataset_std(
    csv_list,
    output_dir=None,
    model_filter_fn=is_diffusion_model,
    display_name_fn=get_display_name,
    model_order=MODEL_ORDER,
    output_filename='results_table_dataset_std.tex',
    require_multi_iteration=True,
    dataset_order=None,
    include_all_datasets=False,
):
    """
    Export a second LaTeX table with the same layout, but with:
    - mean: average over iterations first (per file), then over dataset files
    - std:  standard deviation over dataset files (after per-file iteration averaging)
    """
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Keep only matching models (optionally requiring multi-iteration runs)
    filtered = [
        (dataset, model, iteration, csv_path)
        for dataset, model, iteration, csv_path in csv_list
        if model_filter_fn(model) and (iteration >= 0 if require_multi_iteration else True)
    ]
    if not filtered:
        print('Warning: No matching models for dataset-std LaTeX table')
        return

    # Dataset order
    if dataset_order is None:
        dataset_order = ['gensvs_eval_audio', 'MSRBench_Vocals', 'ears_wham_v2_test_5s']
    available_datasets = sorted(set(d for d, _, _, _ in filtered))
    if include_all_datasets:
        datasets = list(dataset_order)
        datasets += sorted(set(available_datasets) - set(datasets))
    else:
        datasets = [d for d in dataset_order if d in available_datasets] + \
                   sorted(set(available_datasets) - set(dataset_order))

    # Group CSVs by dataset+model
    grouped = defaultdict(list)
    for dataset, model, iteration, csv_path in filtered:
        grouped[(dataset, model)].append((iteration, csv_path))

    # Build stats dict: agg[dataset][display_model][metric] = {'mean': x, 'std': y}
    agg = {}
    dataset_metrics = {}
    for dataset in datasets:
        dataset_metrics[dataset] = [m for m in METRIC_ORDER if m in metrics_for_dataset(dataset)]
        agg[dataset] = {}

        models_in_dataset = sorted({m for (d, m) in grouped.keys() if d == dataset})
        for model in models_in_dataset:
            display_model = display_name_fn(model)

            iter_entries = sorted(grouped[(dataset, model)], key=lambda x: x[0])
            metric_stats = {}

            for metric in dataset_metrics[dataset]:
                per_iter_series = []

                for _, csv_path in iter_entries:
                    df_iter = pd.read_csv(csv_path)

                    if metric not in df_iter.columns:
                        continue

                    # Use file_id when available; otherwise fallback to row index.
                    if 'file_id' in df_iter.columns:
                        s = pd.to_numeric(df_iter[metric], errors='coerce')
                        s.index = df_iter['file_id']
                    else:
                        s = pd.to_numeric(df_iter[metric], errors='coerce')

                    s = s.dropna()
                    if len(s) > 0:
                        per_iter_series.append(s)

                if not per_iter_series:
                    continue

                # Average over iterations per file, then compute dataset mean/std across files.
                stacked = pd.concat(per_iter_series, axis=1)
                per_file_mean_over_iters = stacked.mean(axis=1, skipna=True)
                if len(per_file_mean_over_iters) == 0:
                    continue

                metric_stats[metric] = {
                    'mean': per_file_mean_over_iters.mean(),
                    'std': per_file_mean_over_iters.std(),
                }

            if metric_stats:
                agg[dataset][display_model] = metric_stats

    # Restrict metrics to those that actually exist in aggregated results
    # unless a fixed dataset layout is requested.
    if not include_all_datasets:
        for dataset in datasets:
            dataset_metrics[dataset] = [
                m for m in METRIC_ORDER
                if m in dataset_metrics[dataset] and any(
                    (m in agg[dataset].get(model, {})) for model in agg[dataset]
                )
            ]

    # Model order
    all_models = set()
    for dataset in datasets:
        all_models |= set(agg[dataset].keys())
    ordered_models = [m for m in model_order if m in all_models]
    ordered_models += sorted(all_models - set(ordered_models))

    # Build LaTeX table
    col_counts = [len(dataset_metrics[d]) for d in datasets]
    total_metric_cols = sum(col_counts)

    lines = []
    lines.append(r'\begin{table*}[ht]')
    lines.append(r'  \centering')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \setlength{\tabcolsep}{3pt}%')
    lines.append(r'  \begin{tabular}{l' + 'r' * total_metric_cols + '}')


    header1 = ['  Model']
    for dataset, ncols in zip(datasets, col_counts):
        label = DATASET_LABELS.get(dataset, dataset)
        header1.append(f'\\multicolumn{{{ncols}}}{{c}}{{{label}}}')
    lines.append('    ' + ' & '.join(header1) + ' \\\\')

    col_cursor = 2
    cmidrules = []
    for ncols in col_counts:
        cmidrules.append(f'\\cmidrule(lr){{{col_cursor}-{col_cursor + ncols - 1}}}')
        col_cursor += ncols
    lines.append('    ' + ' '.join(cmidrules))

    header2 = ['  ']
    for dataset in datasets:
        for metric in dataset_metrics[dataset]:
            header2.append(get_metric_label(metric))
    lines.append('    ' + ' & '.join(header2) + ' \\\\')
    lines.append(r'    \midrule')

    # Best rounded value per dataset/metric; bold all ties after rounding
    best = {}
    for dataset in datasets:
        best[dataset] = {}
        for metric in dataset_metrics[dataset]:
            vals = {
                m: _rounded(agg[dataset][m][metric]['mean'], metric)
                for m in ordered_models
                if m in agg[dataset] and metric in agg[dataset][m]
            }
            if not vals:
                continue
            if metric in LOWER_IS_BETTER:
                best[dataset][metric] = min(vals.values())
            else:
                best[dataset][metric] = max(vals.values())

    for model in ordered_models:
        model_tex = model.replace('\n', ' ')
        cells = [f'  {model_tex}']
        for dataset in datasets:
            for metric in dataset_metrics[dataset]:
                if model in agg[dataset] and metric in agg[dataset][model]:
                    stats = agg[dataset][model][metric]
                    mean_val = stats['mean']
                    std_val = stats['std']
                    formatted = (
                        f"$\\, {_fmt(mean_val, metric)}"
                        f"{{\\scriptsize \\pm {_fmt(std_val, metric)}}}$"
                    )
                    if _rounded(mean_val, metric) == best[dataset].get(metric):
                        formatted = f'{{\\boldmath {formatted}}}'
                    cells.append(formatted)
                else:
                    cells.append('--')
        lines.append('    ' + ' & '.join(cells) + ' \\\\[5pt]')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }%')
    lines.append(r'\end{table*}')

    tex = '\n'.join(lines) + '\n'
    tex_path = os.path.join(output_dir, output_filename)
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f'Saved LaTeX table (dataset std): {tex_path}')


def create_latex_table_combined_std(
    df,
    csv_list,
    output_dir=None,
    model_filter_fn=is_diffusion_model,
    display_name_fn=get_display_name,
    model_order=MODEL_ORDER,
    output_filename='results_table_combined_std.tex',
    require_multi_iteration=True,
    dataset_order=None,
    include_all_datasets=False,
):
    """
    Export LaTeX table with combined uncertainty in each cell:
      mean^{+std_iter}_{-std_dataset}
    where:
      - std_iter: std across iterations of dataset-level means
      - std_dataset: std across dataset files after averaging each file over iterations
    """
    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Build iteration-level stats from aggregated df ----
    model_mask = df['model'].apply(model_filter_fn)
    if require_multi_iteration:
        model_mask = model_mask & (df['iteration'] >= 0)
    df_diff = df[model_mask].copy()
    if df_diff.empty:
        print('Warning: No matching models for combined-std LaTeX table')
        return

    df_diff['model'] = df_diff['model'].map(display_name_fn)
    std_ddof = 1 if require_multi_iteration else 0

    if dataset_order is None:
        dataset_order = ['gensvs_eval_audio', 'MSRBench_Vocals', 'ears_wham_v2_test_5s']
    available = set(df_diff['dataset'].unique())
    if include_all_datasets:
        datasets = list(dataset_order)
        datasets += sorted(available - set(datasets))
    else:
        datasets = [d for d in dataset_order if d in available] + sorted(available - set(dataset_order))

    dataset_metrics = {}
    for dataset in datasets:
        sub = df_diff[df_diff['dataset'] == dataset]
        if sub.empty and include_all_datasets:
            dataset_metrics[dataset] = [m for m in METRIC_ORDER if m in metrics_for_dataset(dataset)]
        else:
            dataset_metrics[dataset] = [
                m for m in METRIC_ORDER
                if f'{m}_mean' in sub.columns and not sub[f'{m}_mean'].isna().all()
            ]

    # iteration-based stats
    iter_stats = {}
    for dataset in datasets:
        sub = df_diff[df_diff['dataset'] == dataset]
        iter_stats[dataset] = {}
        for model in sub['model'].unique():
            msub = sub[sub['model'] == model]
            iter_stats[dataset][model] = {}
            for metric in dataset_metrics[dataset]:
                vals = msub[f'{metric}_mean'].dropna()
                if len(vals):
                    iter_stats[dataset][model][metric] = {
                        'mean': vals.mean(),
                        'std_iter': vals.std(ddof=std_ddof),
                    }

    # ---- Build dataset-file std from raw per-iteration CSVs ----
    filtered = [
        (dataset, model, iteration, csv_path)
        for dataset, model, iteration, csv_path in csv_list
        if model_filter_fn(model) and (iteration >= 0 if require_multi_iteration else True)
    ]
    grouped = defaultdict(list)
    for dataset, model, iteration, csv_path in filtered:
        grouped[(dataset, display_name_fn(model))].append((iteration, csv_path))

    dataset_std = {}
    for dataset in datasets:
        dataset_std[dataset] = {}
        models_in_dataset = sorted({m for (d, m) in grouped.keys() if d == dataset})
        for model in models_in_dataset:
            dataset_std[dataset][model] = {}
            iter_entries = sorted(grouped[(dataset, model)], key=lambda x: x[0])
            for metric in dataset_metrics[dataset]:
                per_iter_series = []
                for _, csv_path in iter_entries:
                    df_iter = pd.read_csv(csv_path)
                    if metric not in df_iter.columns:
                        continue

                    s = pd.to_numeric(df_iter[metric], errors='coerce')
                    if 'file_id' in df_iter.columns:
                        s.index = df_iter['file_id']
                    s = s.dropna()
                    if len(s):
                        per_iter_series.append(s)

                if not per_iter_series:
                    continue

                stacked = pd.concat(per_iter_series, axis=1)
                per_file_mean_over_iters = stacked.mean(axis=1, skipna=True)
                if len(per_file_mean_over_iters):
                    dataset_std[dataset][model][metric] = per_file_mean_over_iters.std()

    # model order
    all_models = set()
    for dataset in datasets:
        all_models |= set(iter_stats[dataset].keys())
    ordered_models = [m for m in model_order if m in all_models]
    ordered_models += sorted(all_models - set(ordered_models))

    # best by rounded mean (ties bolded)
    best = {}
    for dataset in datasets:
        best[dataset] = {}
        for metric in dataset_metrics[dataset]:
            vals = {
                m: _rounded(iter_stats[dataset][m][metric]['mean'], metric)
                for m in ordered_models
                if m in iter_stats[dataset] and metric in iter_stats[dataset][m]
            }
            if not vals:
                continue
            if metric in LOWER_IS_BETTER:
                best[dataset][metric] = min(vals.values())
            else:
                best[dataset][metric] = max(vals.values())

    # ---- Build LaTeX ----
    col_counts = [len(dataset_metrics[d]) for d in datasets]
    total_metric_cols = sum(col_counts)

    lines = []
    lines.append(r'\begin{table*}[ht]')
    lines.append(r'  \centering')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \setlength{\tabcolsep}{3pt}%')
    lines.append(r'  \begin{tabular}{l' + 'r' * total_metric_cols + '}')

    header1 = ['  Model']
    for dataset, ncols in zip(datasets, col_counts):
        label = DATASET_LABELS.get(dataset, dataset)
        header1.append(f'\\multicolumn{{{ncols}}}{{c}}{{{label}}}')
    lines.append('    ' + ' & '.join(header1) + ' \\\\')

    col_cursor = 2
    cmidrules = []
    for ncols in col_counts:
        cmidrules.append(f'\\cmidrule(lr){{{col_cursor}-{col_cursor + ncols - 1}}}')
        col_cursor += ncols
    lines.append('    ' + ' '.join(cmidrules))

    header2 = ['  ']
    for dataset in datasets:
        for metric in dataset_metrics[dataset]:
            header2.append(get_metric_label(metric))
    lines.append('    ' + ' & '.join(header2) + ' \\\\')
    lines.append(r'    \midrule')

    for model in ordered_models:
        model_tex = model.replace('\n', ' ')
        cells = [f'  {model_tex}']
        for dataset in datasets:
            for metric in dataset_metrics[dataset]:
                if model in iter_stats[dataset] and metric in iter_stats[dataset][model]:
                    mean_val = iter_stats[dataset][model][metric]['mean']
                    std_iter = iter_stats[dataset][model][metric]['std_iter']
                    std_data = dataset_std.get(dataset, {}).get(model, {}).get(metric, np.nan)

                    mean_s = _fmt(mean_val, metric)
                    iter_s = _fmt(std_iter, metric)
                    data_s = _fmt(std_data, metric) if pd.notna(std_data) else '--'
                    formatted = f"$\\, {mean_s} \\pm^{{\\textstyle {iter_s}}}_{{\\textstyle {data_s}}} $"

                    if _rounded(mean_val, metric) == best[dataset].get(metric):
                        formatted = f'{{\\boldmath {formatted}}}'
                    cells.append(formatted)
                else:
                    cells.append('--')
        lines.append('    ' + ' & '.join(cells) + ' \\\\[4pt]')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }%')
    lines.append(r'\end{table*}')

    tex = '\n'.join(lines) + '\n'
    tex_path = os.path.join(output_dir, output_filename)
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f'Saved LaTeX table (combined std): {tex_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results from multi-iteration model evaluations and create violin plots'
    )
    parser.add_argument('--csv-name', type=str, default='results.csv',
                       help='Name of CSV files to aggregate (e.g., results.csv, results_loudness_normalize.csv)')
    parser.add_argument('--msrbench-csv-name', type=str, default='results_loudness_normalize.csv',
                       help='CSV name used for MSRBench_Vocals to avoid loudness-scaling bias')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Base directory containing test_sets (default: current directory)')
    parser.add_argument('--output-dir', type=str, default='./aggregated_results',
                       help='Directory to save output plots and tables (default: ./aggregated_results)')
    args = parser.parse_args()
    
    print(f"Looking for CSV files: {args.csv_name}")
    print(f"Using MSRBench CSV override: {args.msrbench_csv_name}")
    print(f"Base directory: {args.base_dir}")
    
    # Find all result CSVs
    print("\nFinding result CSV files...")
    csv_list = find_all_result_csvs(args.base_dir, args.csv_name, args.msrbench_csv_name)
    
    print(f"Found {len(csv_list)} result files from:")
    
    # Group by dataset and model for display
    grouped = defaultdict(set)
    for dataset, model, iteration, _ in csv_list:
        grouped[dataset].add(model)
    
    for dataset in sorted(grouped.keys()):
        models = sorted(grouped[dataset])
        print(f"\n  {dataset}:")
        for model in models:
            count = sum(1 for d, m, i, _ in csv_list if d == dataset and m == model)
            print(f"    - {model}: {count} iteration(s)")
    
    # Build dataset-aware merged LoRA view for all table exports.
    table_csv_list = build_table_csv_list_with_merged_lora(csv_list)
    table_df = load_and_aggregate_results(table_csv_list)
    print(f"Table view data shape (LoRA merged): {table_df.shape}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Export requested cross-dataset deltas for selected model families/variants.
    print("\nCreating GenSVS->MSRBench delta LaTeX table...")
    delta_df = export_gensvs_to_msrbench_delta_table(table_df, args.output_dir)
    create_latex_table_gensvs_to_msrbench_delta(delta_df, args.output_dir)

    print("\nCreating combined SGM/BSRNN LaTeX table...")
    family_sections, family_models, display_to_raw_model, model_families = create_latex_table_combined_families(
        table_csv_list,
        args.output_dir,
        base_dir=args.base_dir,
    )

    print("\nCreating parameter table...")
    create_parameter_table(
        family_sections,
        family_models,
        display_to_raw_model,
        model_families,
        args.output_dir,
    )

    print(f"\nAll output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
