#!/usr/bin/env python3
"""
Aggregate results from multiple CSV files.

This script:
1. Recursively finds all result CSV files from multi-iteration model evaluations
2. Aggregates results from all iterations for diffusion-based SGMSE models
3. Exports comparison CSVs and LaTeX tables per model and dataset
"""

#TODO: Simplify string parsing, couldn't all strings for tables and plots be predefined, would this not reduce the amount of lines. 

import argparse
import os
import re
import sys
import types
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from collections import defaultdict


def find_all_result_csvs(base_dir, csv_name='results.csv', msrbench_csv_name='results_loudness_normalize.csv'):
    """
    Recursively find all result CSV files in the directory structure.
    
    Assumes structure like:
    se2svs_results_and_audio/
        MSRBench_Vocals/
            sgm_scratch/
                iter_000_seed42/results.csv
                iter_001_seed43/results.csv
            ...
        gensvs_eval_audio/
            ...
        ears_wham/
            ...
    
    Returns list of tuples: (dataset_name, model_name, iteration, csv_path)
    """
    results = []
    
    se2svs_results_and_audio_dir = Path(base_dir) / 'se2svs_results_and_audio'
    
    if not se2svs_results_and_audio_dir.exists():
        raise FileNotFoundError(f"se2svs_results_and_audio directory not found at {se2svs_results_and_audio_dir}")
    
    # Get all datasets
    datasets = [d for d in se2svs_results_and_audio_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in sorted(datasets):
        dataset_name = dataset_dir.name
        dataset_csv_name = msrbench_csv_name if dataset_name == 'MSRBench_Vocals' else csv_name
        
        # Get all model directories.  The optional 'baselines/' subfolder is
        # treated as a transparent grouping dir — its children are promoted to
        # model dirs so they appear under the dataset, not under 'baselines'.
        raw_model_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        model_dirs = []
        for d in raw_model_dirs:
            if d.name == 'baselines':
                model_dirs.extend(sorted(d.iterdir()))
            else:
                model_dirs.append(d)
        model_dirs = sorted(model_dirs)

        for model_dir in model_dirs:
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
    'sgm_full':             'SGMSVS\n(full fine-tuning)',
    'sgm_base':             'SGMSE \n(base)',
}

BASELINE_MODEL_DISPLAY_NAMES = {
    'melroformer_small':  'MelRoFo (S)',
    'melroformer_large':  'MelRoFo (L)',
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
    'bsrnn_full': 'BSRNNSVS\n(full fine-tuning)',
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
    'stoi':            'ESTOI',
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
                'pesq', 'stoi', 'distillmos']

# Dataset display names for column headers
DATASET_LABELS = {
    'MSRBench_Vocals':      'MSRBench (SVR: out-of-domain)',
    'gensvs_eval_audio':    'GenSVS (SVS: adapted in-domain)',
    'ears_wham':            'EARS-WHAM (SE: source domain)',
}

# Canonical model display order
MODEL_ORDER = [
    'SGMSVS\n(full fine-tuning)',
    'LoRA-SGMSVS\n(rank 16)',
    'SGMSVS\n(from scratch)',
    'SGMSE \n(base)',
]

BSRNN_MODEL_ORDER = [
    'BSRNNSVS\n(full fine-tuning)',
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

BASELINE_MODEL_ORDER = [
    'MelRoFo (S)',
    'MelRoFo (L)',
]

COMBINED_FAMILY_ROWSPAN = {
    'SGM': 5,
    'BSRNN': 7,
    'Baselines': 2,
}

COMBINED_FAMILY_FIXUP = {
    'SGM': '0pt',
    'BSRNN': '3pt',
    'Baselines': '2.5pt',
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
    'bsrnn_lora_r128': 'checkpoints/se2svs/bsrnn_lora_r128/epoch=489-sdr=9.43.ckpt',
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
    'bsrnn_lora_r128': 56.13e6,
}

PRECOMPUTED_GMACS_PER_SECOND = {
    'SGMSVS (full fine-tuning)': 399.64,
    'LoRA-SGMSVS (rank 16)': 895.07,
    'SGMSVS (from scratch)': 399.64,
    'SGMSE (base)': 399.64,
    'BSRNNSVS (full fine-tuning)': 84.31,
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


def is_baseline_model(model_name):
    return model_name in BASELINE_MODEL_DISPLAY_NAMES


def get_baseline_display_name(model_name):
    return BASELINE_MODEL_DISPLAY_NAMES.get(model_name, model_name)


def metrics_for_dataset(dataset_name):
    """Return the metric list expected for a dataset."""
    if 'ears_wham' in dataset_name.lower():
        return ['si_sdr', 'pesq', 'distillmos']
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
    prefer_no_lora = 'ears_wham' in dataset_name.lower()

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


def extract_seed_from_path(csv_path):
    """Extract seed number from a CSV path segment like 'iter_000_seed42'."""
    match = re.search(r'seed[_-]?(\d+)', str(csv_path), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def filter_gensvs_sgm_seed_range(csv_list, min_seed=42, max_seed=51):
    """Filter only GenSVS SGM entries to a specific seed range.

    Non-GenSVS rows, non-SGM rows, and rows without an identifiable seed are
    preserved unchanged.
    """
    filtered = []
    for dataset, model, iteration, csv_path in csv_list:
        if dataset != 'gensvs_eval_audio' or not is_diffusion_model(model):
            filtered.append((dataset, model, iteration, csv_path))
            continue

        seed = extract_seed_from_path(csv_path)
        if seed is None or (min_seed <= seed <= max_seed):
            filtered.append((dataset, model, iteration, csv_path))

    return filtered


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
        ('Generative (SGM)', 'full fine-tuning', ['sgm_full']),
        ('Generative (SGM)', 'LoRA 16', [
            'sgm_lora_r16_adaptive',
            'sgm_lora_r16',
        ]),
        ('Discriminative (BSRNN)', 'from scratch',    ['bsrnn_scratch']),
        ('Discriminative (BSRNN)', 'full fine-tuning', ['bsrnn_full']),
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

    output_path = os.path.join(output_dir, 'summary_gensvs_to_msrbench_delta.csv')
    out_df.to_csv(output_path, index=False)
    print(f'Saved GenSVS->MSRBench delta table: {output_path}')

    return out_df


def _delta_variant_to_compact_model_label(variant):
    """Map delta-table variant names to combined-table model labels."""
    mapping = {
        'full finetuning': 'full fine-tuning',
        'full fine-tuning': 'full fine-tuning',
        'LoRA 16': 'LoRA 16',
        'from scratch': 'from scratch',
    }
    return mapping.get(variant, variant)


def _normalize_delta_variant_key(variant):
    """Normalize delta-variant tokens so naming differences map to one key."""
    if variant is None:
        return ''
    key = str(variant).strip().lower()
    key = key.replace('-', ' ')
    key = ' '.join(key.split())
    return key


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
        ('SGM', 'Generative (SGM)'),
        ('BSRNN', 'Discriminative (BSRNN)'),
    ]
    variant_order = ['full fine-tuning', 'LoRA 16', 'from scratch']
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
            variant = _normalize_delta_variant_key(row['variant'])
            for metric_col, _, _, _ in metric_columns:
                value_lookup[(family_name, variant, metric_col)] = row.get(metric_col, np.nan)

    for variant in variant_order:
        model_label = _delta_variant_to_compact_model_label(variant)
        cells = [f'    {model_label}']
        variant_key = _normalize_delta_variant_key(variant)

        sg_cells = []
        bsrnn_cells = []
        for metric_col, metric_name, _, better_rule in metric_columns:
            sg_family = 'Generative (SGM)'
            bsrnn_family = 'Discriminative (BSRNN)'
            sg_val = value_lookup.get((sg_family, variant_key, metric_col), np.nan)
            bsrnn_val = value_lookup.get((bsrnn_family, variant_key, metric_col), np.nan)

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
            return f'LoRA {rank_match.group(1)}'
        return 'LoRA'
    if 'full fine-tuning' in lowered or 'full fine tuning' in lowered or 'full finetuning' in lowered:
        return 'full fine-tuning'
    if 'from scratch' in lowered:
        return 'from scratch'
    if 'no finetuning' in lowered or lowered == 'base' or '(base)' in lowered:
        return 'base'

    if family == 'SGM' and name.startswith('SGMSVS '):
        return name.replace('SGMSVS ', '', 1)
    if family == 'BSRNN' and (name.startswith('BSRNNSVS ') or name.startswith('BSRNNSE ')):
        return re.sub(r'^BSRNN(?:SVS|SE)\s+', '', name)
    if family == 'Baselines':
        return name
    return name


def _normalize_model_name_for_complexity_lookup(name):
    """Normalize display names to match ptflops complexity summary entries."""
    normalized = re.sub(r'\s+', ' ', name.replace('\n', ' ')).strip()
    normalized = normalized.replace('(no finetuning)', '(base)')
    return normalized.replace('full finetuning', 'full fine-tuning')


def _canonical_model_name_from_compact_label(model_label, family_key):
    """Map compact combined-table labels to canonical names used in complexity tables."""
    lowered = model_label.lower().strip()

    if lowered in {'full finetuning', 'full fine-tuning', 'full fine tuning'}:
        return 'SGMSVS (full fine-tuning)' if family_key == 'SGM' else 'BSRNNSVS (full fine-tuning)'
    if lowered == 'from scratch':
        return 'SGMSVS (from scratch)' if family_key == 'SGM' else 'BSRNNSVS (from scratch)'
    if lowered in {'no finetuning', 'base'}:
        return 'SGMSE (base)' if family_key == 'SGM' else 'BSRNNSE (base)'

    rank_match = re.search(r'(?:rank\s*|lora\s+)(\d+)', lowered)
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
        if is_baseline_model(model_name):
            return 'Baselines'
        return None

    def display_name_for_family(model_name, family):
        if family == 'BSRNN':
            return get_bsrnn_display_name(model_name)
        if family == 'Baselines':
            return get_baseline_display_name(model_name)
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
        dataset_order = ['ears_wham', 'gensvs_eval_audio', 'MSRBench_Vocals']
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
        ('SGM', MODEL_ORDER, 'SGM'),
        ('BSRNN', BSRNN_MODEL_ORDER, 'BSRNN'),
        ('Baselines', BASELINE_MODEL_ORDER, 'Baselines'),
    ]
    noisy_model_name = 'Noisy'

    family_models = {}
    all_models = set()
    for dataset in datasets:
        all_models |= set(agg[dataset].keys())

    noisy_stats = {}

    # Add dataset-level noisy baseline metrics, if available.
    # Expected location: <base_dir>/se2svs_results_and_audio/<dataset>/noisy_metrics.csv
    for dataset in datasets:
        noisy_csv_path = Path(base_dir) / 'se2svs_results_and_audio' / dataset / 'noisy_metrics.csv'
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
        r"  \caption{Mean and standard deviation for SE (EARS-WHAM), SVS (GenSVS), and SVR (MSRBench) results. "
        r"The numbers next to ``LoRA'' indicate the rank $r$. "
        r"Bold numbers indicate the best performance per metric--dataset combination within each model family. "
        r"For context, two differently sized discriminative Mel-RoFormer models from~\cite{jensen2024melbandroformer, bereuter2025gensvs} are included. "
        r"The Noisy row corresponds to unprocessed input mixtures.}"
    )
    lines.append(r'  \label{tab:combined_results}')
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

    has_noisy_data = any(noisy_stats.get(dataset) for dataset in datasets)

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
                if family_key == 'Baselines':
                    fixup = COMBINED_FAMILY_FIXUP.get(family_key, '0pt')
                    family_cell = (
                        '\\multirow{' + str(rowspan) + '}{*}[' + fixup + ']'
                        '{\\rotatebox[origin=c]{90}'
                        '{\\shortstack{\\textbf{Ref.} \\\\ \\cite{jensen2024melbandroformer, bereuter2025gensvs}}}}'
                    )
                else:
                    fixup = COMBINED_FAMILY_FIXUP.get(family_key, '0pt')
                    family_cell = (
                        f'\\multirow{{{rowspan}}}{{*}}[{fixup}]'
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
            lines.append('    ' + ' & '.join(cells) + ' \\\\[2pt]')

        if section_idx < len(family_sections) - 1:
            lines.append(r'    \midrule')

    if has_noisy_data:
        lines.append(r'    \midrule')
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
        lines.append('    ' + ' & '.join(noisy_cells) + ' \\\\')

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

    non_baseline_sections = [(sl, mo, fk) for sl, mo, fk in family_sections if fk != 'Baselines']
    for section_idx, (section_label, _, family_key) in enumerate(non_baseline_sections):
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

        if section_idx < len(non_baseline_sections) - 1:
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


# ---------------------------------------------------------------------------
# PESQ vs MERT-MSE tradeoff plot — label placement
# Each entry maps (family, abbreviated_label) -> (dx_pts, dy_pts, ha, va).
# ---------------------------------------------------------------------------
_PESQ_MERT_LABEL_OFFSET_DEFAULT = (0, 6, 'center', 'bottom')

_PESQ_MERT_LABEL_OFFSETS_GENSVS = {
    ('SGM',   'base'):         ( 0, -6, 'center', 'top'    ),
    ('BSRNN', 'base'):         ( 0, -8, 'right',  'top'    ),
    ('SGM',   'full'):         ( 0, -6, 'center', 'top'    ),
    ('BSRNN', 'full'):         ( 0,  4, 'center', 'bottom' ),
    ('SGM',   'scratch'):      (15,  6, 'center', 'bottom' ),
    ('BSRNN', 'scratch'):      (15,  6, 'center', 'bottom' ),
    ('SGM',   'LoRA\n16'):     ( 9,  4, 'right',  'bottom' ),
    ('BSRNN', 'LoRA\n16'):     ( 5,  4, 'right',  'bottom' ),
    ('BSRNN', 'LoRA\n32/128'):( 6, -9, 'right',  'top'    ),
    ('BSRNN', 'LoRA\n128'):   (-3,  1, 'right',  'bottom' ),
    ('BSRNN', 'LoRA\n32'):    ( 6, -4, 'right',  'top'    ),
}

_PESQ_MERT_LABEL_OFFSETS_MSRBENCH = {
    ('SGM',   'base'):         ( 0, -6, 'center', 'top'    ),
    ('BSRNN', 'base'):         ( 0, -8, 'right',  'top'    ),
    ('SGM',   'full'):         ( 0, -6, 'center', 'top'    ),
    ('BSRNN', 'full'):         ( 0,  4, 'center', 'bottom' ),
    ('SGM',   'scratch'):      (15,  6, 'center', 'bottom' ),
    ('BSRNN', 'scratch'):      (15,  6, 'center', 'bottom' ),
    ('SGM',   'LoRA\n16'):     ( 9,  4, 'right',  'bottom' ),
    ('BSRNN', 'LoRA\n16'):     ( 5,  4, 'right',  'bottom' ),
#    ('BSRNN', 'LoRA\n32/128'):( 6, -6, 'right',  'top'    ),
    ('BSRNN', 'LoRA\n128'):   ( 6, -6, 'right',  'top'    ),
    ('BSRNN', 'LoRA\n32'):    ( -6, 0, 'right',  'center'    ),
}


def _get_family_from_display(display_name):
    """Return 'SGM' or 'BSRNN' based on display name, or None if unrecognised."""
    name = display_name.replace('\n', ' ').lower()
    if 'bsrnn' in name:
        return 'BSRNN'
    if 'sgm' in name:
        return 'SGM'
    return None


def _get_abbreviated_label(display_name):
    """Return a short scatter-plot annotation label from a display name."""
    name = display_name.replace('\n', ' ').lower()
    if 'full fine-tuning' in name or 'full finetuning' in name or 'full fine tuning' in name:
        return 'full'
    if 'from scratch' in name:
        return 'scratch'
    if 'base' in name:
        return 'base'
    rank_match = re.search(r'rank\s*(\d+)', name)
    if rank_match:
        return f'LoRA\n{rank_match.group(1)}'
    return display_name.replace('\n', ' ')


def create_pesq_vs_mert_tradeoff_plot(df, output_dir=None):
    """Create a stacked 2-row PESQ (EARS-WHAM) vs MERT-MSE tradeoff plot.

    Top panel: GenSVS MERT-MSE vs EARS-WHAM PESQ.
    Bottom panel: MSRBench MERT-MSE vs EARS-WHAM PESQ.
    Includes both SGM and BSRNN model families.
    Models are matched by display name so that the dataset-aware LoRA adaptive
    variant (e.g. with-LoRA on MSS, no-LoRA on EARS-WHAM) is handled correctly.
    """
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    import matplotlib as mpl
    prev_font_family = mpl.rcParams.get('font.family', 'sans-serif')
    prev_font_serif  = list(mpl.rcParams.get('font.serif', []))
    prev_mathtext    = mpl.rcParams.get('mathtext.fontset', 'dejavusans')
    mpl.rcParams['font.family']      = 'serif'
    mpl.rcParams['font.serif']       = ['Times New Roman', 'Times', 'DejaVu Serif']
    mpl.rcParams['mathtext.fontset'] = 'stix'

    if output_dir is None:
        output_dir = '.'
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Classify each row: SGM uses multi-iteration, BSRNN uses single-run
    def _classify(row):
        m = row['model']
        if is_bsrnn_model(m):
            return 'BSRNN'
        if is_diffusion_model(m) and row['iteration'] >= 0:
            return 'SGM'
        return None

    df_work = df.copy()
    df_work['family'] = df_work.apply(_classify, axis=1)
    df_work = df_work[df_work['family'].notna()].copy()

    if df_work.empty:
        print('Warning: No data for PESQ vs MERT-MSE tradeoff plot')
        mpl.rcParams['font.family']      = prev_font_family
        mpl.rcParams['font.serif']       = prev_font_serif
        mpl.rcParams['mathtext.fontset'] = prev_mathtext
        return

    # Apply display names (BSRNN uses its own mapping)
    df_work['display_name'] = df_work.apply(
        lambda r: get_bsrnn_display_name(r['model']) if r['family'] == 'BSRNN'
                  else get_display_name(r['model']),
        axis=1,
    )

    all_datasets = df_work['dataset'].unique()
    gensvs_ds   = next((d for d in all_datasets if 'gensvs'    in d.lower()), None)
    msrbench_ds = next((d for d in all_datasets if 'msrbench'  in d.lower()), None)
    earswham_ds = next((d for d in all_datasets if 'ears_wham' in d.lower()), None)

    if not (gensvs_ds and msrbench_ds and earswham_ds):
        missing = [n for n, v in [('GenSVS', gensvs_ds), ('MSRBench', msrbench_ds), ('EARS-WHAM', earswham_ds)] if not v]
        print(f'Warning: Missing datasets for PESQ vs MERT-MSE tradeoff plot: {", ".join(missing)}')
        mpl.rcParams['font.family']      = prev_font_family
        mpl.rcParams['font.serif']       = prev_font_serif
        mpl.rcParams['mathtext.fontset'] = prev_mathtext
        return

    def _mean_by_display(ds_name, metric):
        col = f'{metric}_mean'
        sub = df_work[df_work['dataset'] == ds_name]
        if col not in sub.columns:
            return {}
        return sub.groupby('display_name')[col].mean().to_dict()

    pesq          = _mean_by_display(earswham_ds,  'pesq')
    mert_gensvs   = _mean_by_display(gensvs_ds,    'mert_mse')
    mert_msrbench = _mean_by_display(msrbench_ds,  'mert_mse')

    if not pesq or not (mert_gensvs or mert_msrbench):
        print('Warning: Required metric data missing for PESQ vs MERT-MSE tradeoff plot')
        mpl.rcParams['font.family']      = prev_font_family
        mpl.rcParams['font.serif']       = prev_font_serif
        mpl.rcParams['mathtext.fontset'] = prev_mathtext
        return

    # Build per-display-name rows for plotting
    all_names = set(mert_gensvs) | set(mert_msrbench)
    rows = []
    for dn in all_names:
        if dn not in pesq:
            continue
        family = _get_family_from_display(dn)
        if family is None:
            continue
        rows.append({
            'model':           dn,
            'family':          family,
            'pesq':            pesq[dn],
            'mert_mse_gensvs':   mert_gensvs.get(dn, np.nan),
            'mert_mse_msrbench': mert_msrbench.get(dn, np.nan),
        })

    if not rows:
        print('Warning: No matched model rows for PESQ vs MERT-MSE tradeoff plot')
        mpl.rcParams['font.family']      = prev_font_family
        mpl.rcParams['font.serif']       = prev_font_serif
        mpl.rcParams['mathtext.fontset'] = prev_mathtext
        return

    plot_df = pd.DataFrame(rows).dropna(subset=['pesq'])
    if plot_df.empty:
        print('Warning: No complete rows for PESQ vs MERT-MSE tradeoff plot')
        mpl.rcParams['font.family']      = prev_font_family
        mpl.rcParams['font.serif']       = prev_font_serif
        mpl.rcParams['mathtext.fontset'] = prev_mathtext
        return

    csv_path = os.path.join(output_dir, 'pesq_vs_mert_tradeoff.csv')
    plot_df.to_csv(csv_path, index=False)
    print(f'Saved PESQ vs MERT-MSE tradeoff data: {csv_path}')

    family_colors = {'BSRNN': 'tab:blue', 'SGM': 'tab:orange'}

    def _annotate(ax, x_col, y_col, offsets, is_msrbench):
        combined_lora_drawn = False
        for _, row in plot_df.iterrows():
            family = row['family']
            label  = _get_abbreviated_label(row['model'])
            if not is_msrbench and family == 'BSRNN' and label in {'LoRA\n32', 'LoRA\n128'}:
                if combined_lora_drawn:
                    continue
                label = 'LoRA\n32/128'
                combined_lora_drawn = True
            dx, dy, ha, va = offsets.get((family, label), _PESQ_MERT_LABEL_OFFSET_DEFAULT)
            emphasize = (
                (family == 'BSRNN' and label in {'full', 'LoRA\n32/128', 'LoRA\n128'}) or
                (is_msrbench and family == 'SGM' and label == 'LoRA\n16')
            )
            ax.annotate(
                label,
                (row[x_col], row[y_col]),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=14,
                alpha=0.9,
                fontweight='bold' if emphasize else 'normal',
                ha=ha,
                va=va,
            )

    fig, axes = plt.subplots(2, 1, figsize=(4.8, 7.2), sharex=True)

    for family, fdf in plot_df.groupby('family', sort=False):
        color  = family_colors.get(family, 'tab:gray')
        marker = 'D' if family == 'SGM' else 'o'
        for ax, y_col in zip(axes, ['mert_mse_gensvs', 'mert_mse_msrbench']):
            ax.scatter(fdf['pesq'], fdf[y_col], s=60, color=color,
                       label=family, marker=marker)

    _annotate(axes[0], 'pesq', 'mert_mse_gensvs',   _PESQ_MERT_LABEL_OFFSETS_GENSVS,   is_msrbench=False)
    _annotate(axes[1], 'pesq', 'mert_mse_msrbench', _PESQ_MERT_LABEL_OFFSETS_MSRBENCH, is_msrbench=True)

    for ax, title in zip(axes, ['GenSVS/EARS-WHAM', 'MSRBench/EARS-WHAM']):
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(r'MERT-MSE $\leftarrow$', fontsize=14)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=14)

    # Fine-tune y-axis formatting to match the original
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.01))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    xmin, xmax = axes[0].get_xlim()
    axes[0].set_xlim(left=xmin - 0.03, right=xmax)
    ymin, ymax = axes[0].get_ylim()
    axes[0].set_ylim(bottom=ymin - 0.005, top=ymax)

    axes[1].set_xlabel(r'PESQ $\rightarrow$', fontsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels,
        loc='center right',
        bbox_to_anchor=(0.98, 0.60),
        ncol=1,
        frameon=True, facecolor='white', edgecolor='black', framealpha=1.0,
        fontsize=14, borderaxespad=0.0, handletextpad=0.3,
    )

    fig.tight_layout()

    png_path = os.path.join(output_dir, 'pesq_vs_mert_tradeoff.png')
    pdf_path = os.path.join(output_dir, 'pesq_vs_mert_tradeoff.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved PESQ vs MERT-MSE tradeoff plot: {png_path}')
    print(f'Saved PESQ vs MERT-MSE tradeoff plot: {pdf_path}')

    # Restore rcParams
    mpl.rcParams['font.family']      = prev_font_family
    mpl.rcParams['font.serif']       = prev_font_serif
    mpl.rcParams['mathtext.fontset'] = prev_mathtext


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results from multi-iteration model evaluations'
    )
    parser.add_argument('--csv-name', type=str, default='results.csv',
                       help='Name of CSV files to aggregate (e.g., results.csv, results_loudness_normalize.csv)')
    parser.add_argument('--msrbench-csv-name', type=str, default='results_loudness_normalize.csv',
                       help='CSV name used for MSRBench_Vocals to avoid loudness-scaling bias')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Base directory containing se2svs_results_and_audio (default: current directory)')
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
    
    # Load and aggregate results (non-merged, for comparison table)
    print("\nAggregating results...")
    df = load_and_aggregate_results(csv_list)
    print(f"Aggregated data shape: {df.shape}")

    # Build dataset-aware merged LoRA view for all table exports.
    table_csv_list = build_table_csv_list_with_merged_lora(csv_list)
    table_df = load_and_aggregate_results(table_csv_list)
    print(f"Table view data shape (LoRA merged): {table_df.shape}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create comparison table (per-dataset summary CSVs)
    print("\nCreating comparison table...")
    create_comparison_table(df, args.output_dir)

    # Export requested cross-dataset deltas for selected model families/variants.
    print("\nCreating GenSVS->MSRBench delta summary table...")
    delta_df = export_gensvs_to_msrbench_delta_table(table_df, args.output_dir)
    print("\nCreating GenSVS->MSRBench delta LaTeX table...")
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

    print("\nCreating PESQ vs MERT-MSE tradeoff plot...")
    create_pesq_vs_mert_tradeoff_plot(table_df, args.output_dir)

    print(f"\nAll output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
