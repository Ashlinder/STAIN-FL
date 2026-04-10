"""
Visualization utilities for FL experiment results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_global_metrics(
    df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    title: str = "Global Test Set - Classification Metrics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot global test metrics over FL rounds.
    
    Args:
        df: DataFrame with columns ['round', 'accuracy', 'precision', 'recall', 'f1']
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
        if metric in df.columns:
            ax.plot(df['round'], df[metric], color=colors[idx], linewidth=2)
            ax.set_xlabel('Round')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_per_client_metrics(
    df: pd.DataFrame,
    metric: str = 'accuracy',
    title: str = "Per-Client Evaluation of Aggregated Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot per-client metrics over FL rounds.
    
    Args:
        df: DataFrame with columns ['round', 'client', metric]
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Client colors
    client_colors = {
        'Client 1: SPF': '#9b59b6',
        'Client 2: ICA': '#2ecc71',
        'Client 3: LTA': '#e74c3c',
        'Client 4: NParks': '#3498db'
    }
    
    for metric, ax in zip(metrics, axes.flatten()):
        if metric in df.columns:
            for client in df['client'].unique():
                client_df = df[df['client'] == client]
                color = client_colors.get(client, '#333333')
                ax.plot(
                    client_df['round'], 
                    client_df[metric], 
                    label=client.split(': ')[1] if ': ' in client else client,
                    color=color,
                    linewidth=1.5
                )
            
            ax.set_xlabel('Round')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, 1.05)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_backdoor_accuracy(
    df: pd.DataFrame,
    attack_start: int = 0,
    attack_end: int = None,
    title: str = "Backdoor Attack Metrics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot backdoor accuracy over FL rounds.
    
    Args:
        df: DataFrame with columns ['round', 'backdoor_accuracy', 'attack_active']
        attack_start: Round when attack started
        attack_end: Round when attack ended
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot backdoor accuracy
    ax.plot(
        df['round'], 
        df['backdoor_accuracy'], 
        color='#e74c3c',
        linewidth=2,
        label='Backdoor Accuracy (BA)'
    )
    
    # Highlight attack period
    if attack_end is None:
        attack_end = df['round'].max()
    
    ax.axvspan(attack_start, attack_end, alpha=0.2, color='red', label='Attack Period')
    ax.axvline(x=attack_start, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=attack_end, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Backdoor Accuracy')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attack_comparison(
    baseline_df: pd.DataFrame,
    attack_df: pd.DataFrame,
    metric: str = 'accuracy',
    title: str = "Baseline vs Attack Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare baseline and attack experiment metrics.
    
    Args:
        baseline_df: DataFrame from baseline experiment
        attack_df: DataFrame from attack experiment
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    axes[0].plot(baseline_df['round'], baseline_df[metric], 
                 label='Baseline', color='#2ecc71', linewidth=2)
    axes[0].plot(attack_df['round'], attack_df[metric], 
                 label='Under Attack', color='#e74c3c', linewidth=2)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel(metric.capitalize())
    axes[0].set_title(f'{metric.capitalize()} Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    
    # Backdoor accuracy (if available)
    if 'backdoor_accuracy' in attack_df.columns:
        axes[1].plot(attack_df['round'], attack_df['backdoor_accuracy'],
                     color='#e74c3c', linewidth=2)
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Backdoor Accuracy')
        axes[1].set_title('Attack Success Rate (Backdoor Accuracy)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_metrics(
    df: pd.DataFrame,
    title: str = "Per-Client Training Metrics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training metrics for all clients.
    
    Args:
        df: DataFrame with columns ['round', 'client', 'loss', 'accuracy']
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    client_colors = {
        'Client 1: SPF': '#e74c3c',
        'Client 2: ICA': '#3498db',
        'Client 4: NParks': '#2ecc71',
        'Client 3: LTA': '#9b59b6'
    }
    
    for metric, ax in zip(metrics, axes.flatten()):
        if metric in df.columns:
            for client in df['client'].unique():
                client_df = df[df['client'] == client]
                color = client_colors.get(client, '#333333')
                short_name = client.split(': ')[1] if ': ' in client else client
                ax.plot(
                    client_df['round'],
                    client_df[metric],
                    label=short_name,
                    color=color,
                    linewidth=1.5
                )
            
            ax.set_xlabel('Round')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, 1.05)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_table(results: Dict) -> pd.DataFrame:
    """
    Create a summary table of experiment results.
    
    Args:
        results: Results dictionary from FLSimulator
    
    Returns:
        Summary DataFrame
    """
    summary = []
    
    # Global test final metrics
    if results['global_test']:
        final_global = results['global_test'][-1]
        summary.append({
            'Dataset': 'Global Test',
            'Final Accuracy': f"{final_global['accuracy']:.4f}",
            'Final Precision': f"{final_global['precision']:.4f}",
            'Final Recall': f"{final_global['recall']:.4f}",
            'Final F1': f"{final_global['f1']:.4f}"
        })
    
    # Per-client final metrics
    for client_name, metrics in results['per_client_test'].items():
        if metrics:
            final = metrics[-1]
            short_name = client_name.split(': ')[1] if ': ' in client_name else client_name
            summary.append({
                'Dataset': f'{short_name} Test',
                'Final Accuracy': f"{final['accuracy']:.4f}",
                'Final Precision': f"{final['precision']:.4f}",
                'Final Recall': f"{final['recall']:.4f}",
                'Final F1': f"{final['f1']:.4f}"
            })
    
    return pd.DataFrame(summary)


def save_all_plots(
    results: Dict,
    output_dir: str,
    experiment_name: str,
    attack_config: Dict = None
) -> Dict[str, str]:
    """
    Generate and save all plots for an experiment.
    
    Args:
        results: Results dictionary from FLSimulator
        output_dir: Directory to save plots
        experiment_name: Name of the experiment
        attack_config: Attack configuration (if applicable)
    
    Returns:
        Dictionary of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_name}_{timestamp}"
    
    saved_files = {}
    
    # Global test metrics
    if results['global_test']:
        global_df = pd.DataFrame(results['global_test'])
        path = os.path.join(output_dir, f"{prefix}_global_test.png")
        plot_global_metrics(global_df, save_path=path)
        saved_files['global_test_plot'] = path
        plt.close()
    
    # Per-client test metrics
    test_dfs = []
    for client_name, metrics in results['per_client_test'].items():
        if metrics:
            df = pd.DataFrame(metrics)
            df['client'] = client_name
            test_dfs.append(df)
    
    if test_dfs:
        combined_df = pd.concat(test_dfs, ignore_index=True)
        path = os.path.join(output_dir, f"{prefix}_per_client_test.png")
        plot_per_client_metrics(combined_df, save_path=path)
        saved_files['per_client_test_plot'] = path
        plt.close()
    
    # Per-client train metrics
    train_dfs = []
    for client_name, metrics in results['per_client_train'].items():
        if metrics:
            df = pd.DataFrame(metrics)
            df['client'] = client_name
            train_dfs.append(df)
    
    if train_dfs:
        combined_df = pd.concat(train_dfs, ignore_index=True)
        path = os.path.join(output_dir, f"{prefix}_per_client_train.png")
        plot_training_metrics(combined_df, save_path=path)
        saved_files['per_client_train_plot'] = path
        plt.close()
    
    # Backdoor metrics
    if results.get('backdoor_metrics'):
        ba_df = pd.DataFrame(results['backdoor_metrics'])
        attack_start = attack_config.get('attack_start_round', 0) if attack_config else 0
        attack_end = attack_config.get('attack_end_round', None) if attack_config else None
        path = os.path.join(output_dir, f"{prefix}_backdoor_accuracy.png")
        plot_backdoor_accuracy(ba_df, attack_start, attack_end, save_path=path)
        saved_files['backdoor_accuracy_plot'] = path
        plt.close()
    
    return saved_files


if __name__ == "__main__":
    # Test visualization
    print("Visualization module loaded successfully")
