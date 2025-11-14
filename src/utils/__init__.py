"""ARIA Utilities - Portable paths, config loading, and helpers"""

from .paths import (
    get_project_root,
    expand_path,
    get_default_config_path,
    ensure_dir,
    get_var_dir,
    get_cache_dir,
    get_logs_dir,
    get_state_dir,
    PROJECT_ROOT,
    VAR_DIR,
    CACHE_DIR,
    LOGS_DIR,
    STATE_DIR,
)

from .config_loader import (
    load_config,
    get_config_value,
    save_config,
    create_default_config,
)

__all__ = [
    # Paths
    'get_project_root',
    'expand_path',
    'get_default_config_path',
    'ensure_dir',
    'get_var_dir',
    'get_cache_dir',
    'get_logs_dir',
    'get_state_dir',
    'PROJECT_ROOT',
    'VAR_DIR',
    'CACHE_DIR',
    'LOGS_DIR',
    'STATE_DIR',
    # Config
    'load_config',
    'get_config_value',
    'save_config',
    'create_default_config',
]
