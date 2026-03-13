"""Shared style constants and helpers for visualization scripts."""
import matplotlib.pyplot as plt

FIGSIZE_SINGLE = (7, 4.5)
FIGSIZE_WIDE   = (10, 4.5)
FIGSIZE_GRID   = (10, 7)

DPI = 300

MINERAL_COLORS = {
    'olivine':     '#4caf50',
    'lcp':         '#2196f3',
    'hcp':         '#ff9800',
    'plagioclase': '#9c27b0',
    'other':       '#9e9e9e',
}

LABEL_COLS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']


def apply_style():
    """Set global matplotlib rcParams for consistent figure style."""
    plt.rcParams.update({
        'font.size':       11,
        'axes.titlesize':  12,
        'axes.labelsize':  11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.grid':       True,
        'grid.alpha':      0.3,
    })


def despine(ax):
    """Remove top and right spines from axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
