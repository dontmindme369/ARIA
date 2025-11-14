#!/usr/bin/env python3
"""
Configuration Loader for ARIA

Loads YAML configuration with portable path expansion.
All paths in config are expanded relative to project root or user home.

Usage:
    from utils.config_loader import load_config, get_config_value

    config = load_config()
    index_roots = config['paths']['index_roots']  # Already expanded Paths
    top_k = config['retrieval']['top_k']           # int

    # Or use helper:
    index_roots = get_config_value('paths.index_roots', default=[Path('./data')])
"""

from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import yaml
import os

from .paths import expand_path, get_default_config_path, get_project_root


def load_config(config_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    Load ARIA configuration from YAML file.

    All paths in the config are automatically expanded:
    - ~/... → user home
    - ./... → relative to project root
    - $VAR → environment variables

    Args:
        config_path: Path to config file (optional)
                    If None, uses default location (ARIA_CONFIG env var,
                    ./aria_config.yaml, or ~/.aria/config.yaml)

    Returns:
        Configuration dictionary with expanded paths

    Example:
        >>> config = load_config()
        >>> print(config['paths']['index_roots'])
        [PosixPath('/home/user/Documents/knowledge')]
    """
    if config_path is None:
        config_path = get_default_config_path()
    else:
        config_path = expand_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected at: {get_default_config_path()}\n"
            f"Run first-time setup or create aria_config.yaml"
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Expand all paths in the config
    config = _expand_paths_in_config(config)

    return config


def _expand_paths_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand all path strings in config to Path objects.

    Identifies path keys by name:
    - Any key containing 'path', 'dir', 'root', 'file'
    - Keys in 'paths' section

    Args:
        config: Raw config dictionary

    Returns:
        Config with paths expanded to Path objects
    """
    expanded = {}

    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively expand nested dicts
            expanded[key] = _expand_paths_in_config(value)
        elif isinstance(value, list):
            # Expand lists (like index_roots)
            if _is_path_key(key):
                expanded[key] = [expand_path(p) for p in value]
            else:
                expanded[key] = value
        elif isinstance(value, str):
            # Expand string paths
            if _is_path_key(key):
                expanded[key] = expand_path(value)
            else:
                expanded[key] = value
        else:
            # Keep other types as-is
            expanded[key] = value

    return expanded


def _is_path_key(key: str) -> bool:
    """
    Check if a config key likely contains a path.

    Args:
        key: Config key name

    Returns:
        True if key probably contains a path
    """
    path_indicators = ['path', 'dir', 'root', 'file', 'folder', 'location']
    key_lower = key.lower()
    return any(indicator in key_lower for indicator in path_indicators)


def get_config_value(
    key_path: str,
    config: Optional[Dict[str, Any]] = None,
    default: Any = None
) -> Any:
    """
    Get a config value by dot-separated key path.

    Args:
        key_path: Dot-separated path (e.g., 'paths.index_roots')
        config: Config dict (if None, loads from default location)
        default: Default value if key not found

    Returns:
        Config value at key path, or default if not found

    Examples:
        >>> get_config_value('retrieval.top_k', default=100)
        128

        >>> get_config_value('paths.index_roots')
        [PosixPath('/home/user/Documents/knowledge')]

        >>> get_config_value('nonexistent.key', default='fallback')
        'fallback'
    """
    if config is None:
        config = load_config()

    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def save_config(config: Dict[str, Any], config_path: Union[str, Path, None] = None) -> None:
    """
    Save configuration to YAML file.

    Converts Path objects back to strings for YAML serialization.

    Args:
        config: Configuration dictionary
        config_path: Where to save (if None, uses default location)

    Example:
        >>> config = load_config()
        >>> config['retrieval']['top_k'] = 256
        >>> save_config(config)
    """
    if config_path is None:
        config_path = get_default_config_path()
    else:
        config_path = expand_path(config_path)

    # Convert Path objects back to strings for YAML
    config_serializable = _paths_to_strings(config)

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            config_serializable,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )


def _paths_to_strings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Path objects in config to strings for YAML serialization.

    Args:
        config: Config dict with Path objects

    Returns:
        Config dict with paths as strings
    """
    serializable = {}

    for key, value in config.items():
        if isinstance(value, dict):
            serializable[key] = _paths_to_strings(value)
        elif isinstance(value, list):
            serializable[key] = [
                str(p) if isinstance(p, Path) else p
                for p in value
            ]
        elif isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value

    return serializable


def create_default_config(config_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    Create a default configuration file.

    Used for first-time setup.

    Args:
        config_path: Where to save config (if None, uses default location)

    Returns:
        Default configuration dictionary

    Example:
        >>> config = create_default_config()
        >>> print(config['paths']['index_roots'])
        [PosixPath('/home/user/Documents/knowledge')]
    """
    default_config = {
        'paths': {
            'index_roots': ['~/Documents/knowledge'],  # User home directory
            'output_dir': './aria_packs',             # Relative to project root
            'exemplars': './data/exemplars.txt',
            'bandit_state': '~/.aria/bandit_state.json',
            'domain_dictionaries': './data/domain_dictionaries',
        },
        'retrieval': {
            'top_k': 128,
            'per_file_limit': 10000,
            'max_per_file': 32,
            'max_files_to_scan': 10000,
            'use_semantic': True,
            'semantic_model': 'all-MiniLM-L6-v2',
            'semantic_weight': 0.7,
            'semantic_rotations': 3,
            'semantic_limit': 256,
        },
        'postfilter': {
            'enabled': True,
            'quality_filter': False,    # Disabled for small datasets
            'topic_filter': False,
            'diversity_filter': True,
            'max_per_source': 15,
            'min_keep': 20,
            'min_alpha_ratio': 0.2,
            'min_score': 0.001,
        },
        'bandit': {
            'enabled': True,
            'exploration_pulls': 20,
        },
        'context': {
            'max_chars': 500000,
            'max_item_chars': 5000,
            'min_item_chars': 1000,
            'separator': '\n\n---\n\n',
        },
        'terminal': {
            'colors': True,
            'verbose': False,
        },
    }

    # Expand paths and save
    expanded_config = _expand_paths_in_config(default_config)

    if config_path:
        save_config(expanded_config, config_path)

    return expanded_config
