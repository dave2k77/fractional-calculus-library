"""
Plotting and visualization tools for fractional calculus computations.

This module provides tools for:
- Creating comparison plots between different methods
- Plotting convergence analysis
- Visualizing error analysis
- Saving and managing plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import warnings


class PlotManager:
    """Manager for creating and managing plots."""
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the plot manager.
        
        Args:
            style: Plotting style ('default', 'scientific', 'presentation')
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.setup_plotting_style(style)
    
    def setup_plotting_style(self, style: str = 'default') -> None:
        """
        Setup plotting style.
        
        Args:
            style: Style to use ('default', 'scientific', 'presentation')
        """
        if style == 'scientific':
            plt.style.use('seaborn-v0_8-whitegrid')
            rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'lines.linewidth': 2,
                'lines.markersize': 6,
            })
        elif style == 'presentation':
            plt.style.use('seaborn-v0_8-darkgrid')
            rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'lines.linewidth': 3,
                'lines.markersize': 8,
            })
        else:  # default
            plt.style.use('default')
            rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8,
                'figure.titlesize': 14,
                'lines.linewidth': 1.5,
                'lines.markersize': 4,
            })
    
    def create_comparison_plot(self, x: np.ndarray, data_dict: Dict[str, np.ndarray],
                              title: str = "Comparison Plot", xlabel: str = "x",
                              ylabel: str = "y", save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comparison plot of multiple datasets.
        
        Args:
            x: x-axis data
            data_dict: Dictionary of {label: y_data} pairs
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
        
        for i, (label, y_data) in enumerate(data_dict.items()):
            ax.plot(x, y_data, color=colors[i], label=label, linewidth=2)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig
    
    def plot_convergence(self, grid_sizes: List[int], errors: Dict[str, List[float]],
                        title: str = "Convergence Analysis", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot convergence analysis.
        
        Args:
            grid_sizes: List of grid sizes
            errors: Dictionary of {metric: error_list} pairs
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot errors vs grid size
        colors = plt.cm.tab10(np.linspace(0, 1, len(errors)))
        
        for i, (metric, error_list) in enumerate(errors.items()):
            ax1.loglog(grid_sizes, error_list, 'o-', color=colors[i], 
                      label=metric.upper(), linewidth=2, markersize=6)
        
        ax1.set_title("Error vs Grid Size", fontweight='bold')
        ax1.set_xlabel("Grid Size (N)")
        ax1.set_ylabel("Error")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot convergence rates
        convergence_rates = {}
        for metric, error_list in errors.items():
            try:
                # Compute convergence rate using linear regression
                log_n = np.log(np.array(grid_sizes))
                log_error = np.log(np.array(error_list))
                coeffs = np.polyfit(log_n, log_error, 1)
                convergence_rate = -coeffs[0]
                convergence_rates[metric] = convergence_rate
            except:
                convergence_rates[metric] = np.nan
        
        # Plot convergence rates
        metrics = list(convergence_rates.keys())
        rates = list(convergence_rates.values())
        
        bars = ax2.bar(metrics, rates, color=colors[:len(metrics)], alpha=0.7)
        ax2.set_title("Convergence Rates", fontweight='bold')
        ax2.set_ylabel("Convergence Rate (order)")
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            if not np.isnan(rate):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{rate:.2f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig
    
    def plot_error_analysis(self, x: np.ndarray, numerical: np.ndarray, analytical: np.ndarray,
                           title: str = "Error Analysis", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot error analysis between numerical and analytical solutions.
        
        Args:
            x: x-axis data
            numerical: Numerical solution
            analytical: Analytical solution
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot solutions
        ax1.plot(x, numerical, 'b-', label='Numerical', linewidth=2)
        ax1.plot(x, analytical, 'r--', label='Analytical', linewidth=2)
        ax1.set_title("Solutions Comparison", fontweight='bold')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot absolute error
        abs_error = np.abs(numerical - analytical)
        ax2.plot(x, abs_error, 'g-', linewidth=2)
        ax2.set_title("Absolute Error", fontweight='bold')
        ax2.set_xlabel("x")
        ax2.set_ylabel("|Error|")
        ax2.grid(True, alpha=0.3)
        
        # Plot relative error
        rel_error = np.abs(numerical - analytical) / (np.abs(analytical) + 1e-12)
        ax3.plot(x, rel_error, 'm-', linewidth=2)
        ax3.set_title("Relative Error", fontweight='bold')
        ax3.set_xlabel("x")
        ax3.set_ylabel("|Error|/|Analytical|")
        ax3.grid(True, alpha=0.3)
        
        # Error statistics
        error_stats = {
            'L1 Error': np.mean(abs_error),
            'L2 Error': np.sqrt(np.mean(abs_error**2)),
            'L∞ Error': np.max(abs_error),
            'Mean Rel. Error': np.mean(rel_error),
        }
        
        # Create text box with statistics
        stats_text = '\n'.join([f'{k}: {v:.2e}' for k, v in error_stats.items()])
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title("Error Statistics", fontweight='bold')
        ax4.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig
    
    def save_plot(self, fig: plt.Figure, path: str, dpi: int = 300, 
                  bbox_inches: str = 'tight') -> None:
        """
        Save a plot to file.
        
        Args:
            fig: Matplotlib figure object
            path: File path to save to
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Plot saved to: {path}")
        except Exception as e:
            warnings.warn(f"Failed to save plot to {path}: {e}")


# Convenience functions
def setup_plotting_style(style: str = 'default') -> None:
    """Setup plotting style."""
    manager = PlotManager(style=style)
    manager.setup_plotting_style(style)


def create_comparison_plot(x: np.ndarray, data_dict: Dict[str, np.ndarray],
                          title: str = "Comparison Plot", xlabel: str = "x",
                          ylabel: str = "y", save_path: Optional[str] = None) -> plt.Figure:
    """Create a comparison plot of multiple datasets."""
    manager = PlotManager()
    return manager.create_comparison_plot(x, data_dict, title, xlabel, ylabel, save_path)


def plot_convergence(grid_sizes: List[int], errors: Dict[str, List[float]],
                    title: str = "Convergence Analysis", save_path: Optional[str] = None) -> plt.Figure:
    """Plot convergence analysis."""
    manager = PlotManager()
    return manager.plot_convergence(grid_sizes, errors, title, save_path)


def plot_error_analysis(x: np.ndarray, numerical: np.ndarray, analytical: np.ndarray,
                       title: str = "Error Analysis", save_path: Optional[str] = None) -> plt.Figure:
    """Plot error analysis between numerical and analytical solutions."""
    manager = PlotManager()
    return manager.plot_error_analysis(x, numerical, analytical, title, save_path)


def save_plot(fig: plt.Figure, path: str, dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """Save a plot to file."""
    manager = PlotManager()
    manager.save_plot(fig, path, dpi, bbox_inches)
