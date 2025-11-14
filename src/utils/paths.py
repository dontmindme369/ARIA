#!/usr/bin/env python3
"""
Portable Path Resolution for ARIA

Provides utilities for resolving paths portably across different systems.
No hard-coded paths - everything is config-driven or relative to project root.

Usage:
    from utils.paths import get_project_root, get_config_path, expand_path

    root = get_project_root()
    index_dir = get_config_path('index_roots')[0]
    output_dir = get_config_path('output_dir')
"""

from pathlib import Path
from typing import List, Union
import os


def get_project_root() -> Path:
    """
    Get the ARIA project root directory.

    The project root is defined as the directory containing aria_main.py
    or the parent of the src/ directory.

    Returns:
        Path to project root (absolute)

    Example:
        >>> root = get_project_root()
        >>> print(root)
        /home/user/aria
    """
    # Start from this file's location
    current_file = Path(__file__).resolve()

    # Go up to src/, then to project root
    # paths.py is in src/utils/, so go up 2 levels
    src_dir = current_file.parent.parent
    project_root = src_dir.parent

    # Verify we're in the right place (check for aria_main.py)
    if (project_root / "aria_main.py").exists():
        return project_root
    elif (project_root / "src" / "aria_main.py").exists():
        # aria_main.py might be in src/
        return project_root
    else:
        # Fallback: return the parent directory
        return project_root


def expand_path(path: Union[str, Path]) -> Path:
    """
    Expand a path string to an absolute Path object.

    Handles:
    - Tilde expansion (~/ → user home)
    - Relative paths (./... → relative to project root)
    - Environment variables ($VAR or ${VAR})
    - Absolute paths (returned as-is)

    Args:
        path: Path string or Path object

    Returns:
        Expanded absolute Path

    Examples:
        >>> expand_path("~/Documents/knowledge")
        PosixPath('/home/user/Documents/knowledge')

        >>> expand_path("./aria_packs")
        PosixPath('/home/user/aria/aria_packs')

        >>> expand_path("$HOME/data")
        PosixPath('/home/user/data')
    """
    if isinstance(path, Path):
        path_str = str(path)
    else:
        path_str = path

    # Expand environment variables
    path_str = os.path.expandvars(path_str)

    # Convert to Path and expand user home (~)
    path_obj = Path(path_str).expanduser()

    # If relative, make it relative to project root
    if not path_obj.is_absolute():
        project_root = get_project_root()
        path_obj = (project_root / path_obj).resolve()

    return path_obj


def get_default_config_path() -> Path:
    """
    Get the default config file location.

    Checks in order:
    1. ARIA_CONFIG environment variable
    2. ./aria_config.yaml (project root)
    3. ~/.aria/config.yaml (user home)

    Returns:
        Path to config file (may not exist yet)
    """
    # Check environment variable first
    if "ARIA_CONFIG" in os.environ:
        return expand_path(os.environ["ARIA_CONFIG"])

    # Check project root
    project_root = get_project_root()
    project_config = project_root / "aria_config.yaml"
    if project_config.exists():
        return project_config

    # Check user home
    user_config = Path.home() / ".aria" / "config.yaml"
    if user_config.exists():
        return user_config

    # Default: project root
    return project_config


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Absolute Path to directory

    Example:
        >>> output_dir = ensure_dir("./aria_packs")
        >>> print(output_dir)
        /home/user/aria/aria_packs
    """
    dir_path = expand_path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_var_dir(subdir: str = "") -> Path:
    """
    Get the var/ directory for runtime data (state, cache, logs).

    Args:
        subdir: Optional subdirectory within var/

    Returns:
        Path to var directory (created if doesn't exist)

    Examples:
        >>> get_var_dir()
        /home/user/aria/var

        >>> get_var_dir("sessions")
        /home/user/aria/var/sessions
    """
    project_root = get_project_root()
    var_dir = project_root / "var"

    if subdir:
        var_dir = var_dir / subdir

    return ensure_dir(var_dir)


def get_cache_dir() -> Path:
    """Get the cache directory for temporary files."""
    return get_var_dir("cache")


def get_logs_dir() -> Path:
    """Get the logs directory for telemetry and logs."""
    return get_var_dir("logs")


def get_state_dir() -> Path:
    """Get the state directory for persistent state (bandit, watcher, etc.)."""
    return get_var_dir("state")


# Convenience exports
PROJECT_ROOT = get_project_root()
VAR_DIR = PROJECT_ROOT / "var"
CACHE_DIR = VAR_DIR / "cache"
LOGS_DIR = VAR_DIR / "logs"
STATE_DIR = VAR_DIR / "state"
