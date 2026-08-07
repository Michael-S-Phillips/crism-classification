"""
Config loader that merges config.yaml with an optional machine-local override.

Priority (highest to lowest):
  1. CRISM_DATA_ROOT environment variable  → overrides data_root only
  2. config.local.yaml                     → overrides any key in config.yaml
  3. config.yaml                           → committed baseline

Usage in scripts:
    from config_loader import load_config
    cfg = load_config()              # finds config.yaml next to this file
    cfg = load_config('config.yaml') # explicit path

Each machine should have a config.local.yaml (gitignored) containing only
what differs, typically just:
    data_root: /mnt/whatever

All paths derived from data_root (output_dir, checkpoints_dir, gpkg_dir, etc.)
are recomputed automatically when data_root is overridden.
"""
import os
import yaml
from pathlib import Path


_PROJECT_ROOT = Path(__file__).parent


def load_config(config_path: str = None) -> dict:
    """
    Load and return the merged configuration dict.

    Parameters
    ----------
    config_path : str, optional
        Path to the base config.yaml. Defaults to config.yaml next to this file.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / 'config.yaml'
    config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Merge config.local.yaml if present (same directory as base config)
    local_path = config_path.parent / 'config.local.yaml'
    if local_path.exists():
        with open(local_path) as f:
            local = yaml.safe_load(f) or {}
        cfg.update(local)

    # CRISM_DATA_ROOT env var overrides data_root
    env_root = os.environ.get('CRISM_DATA_ROOT')
    if env_root:
        cfg['data_root'] = env_root

    # Recompute all paths derived from data_root whenever it was overridden
    _recompute_derived_paths(cfg)

    return cfg


def _recompute_derived_paths(cfg: dict) -> None:
    """
    Recompute paths that are conventionally derived from data_root.
    Only updates a key if it still points to the OLD data_root
    (i.e. don't clobber explicit overrides in config.local.yaml).
    """
    data_root = cfg.get('data_root', '')
    if not data_root:
        return

    project_root = os.path.join(data_root, 'crism_classification')

    defaults = {
        'project_root':    project_root,
        'output_dir':      os.path.join(project_root, 'data'),
        'checkpoints_dir': os.path.join(project_root, 'checkpoints'),
        'predictions_dir': os.path.join(project_root, 'predictions'),
        'reports_dir':     os.path.join(project_root, 'reports'),
        'patch_cache_dir': os.path.join(project_root, 'data', 'patch_cache'),
        'gpkg_dir':        os.path.join(data_root, 'categorized_mineral_units'),
    }

    for key, derived_value in defaults.items():
        current = cfg.get(key, '')
        # Update if not set, or if it still embeds an old data_root path
        # (allows explicit overrides in config.local.yaml to win)
        if not current or _path_under_any_old_root(current):
            cfg[key] = derived_value


_KNOWN_OLD_ROOTS = [
    '/mnt/gigas/CRISM/MRDR',
    '/mnt/crism/MRDR',
    '/mnt/mrdr',
]


def _path_under_any_old_root(path: str) -> bool:
    """True if this path starts with a known stale mount point."""
    return any(path.startswith(r) for r in _KNOWN_OLD_ROOTS)
