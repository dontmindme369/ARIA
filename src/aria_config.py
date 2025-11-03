#!/usr/bin/env python3
"""
ARIA Configuration - YAML-based config with hot-reload support
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ARIAConfig:
    """ARIA configuration with hot-reload support"""
    
    # Paths - Use relative paths by default, override with YAML config
    index_roots: List[str] = field(default_factory=lambda: ["./data"])
    output_dir: str = "./output"
    exemplars: Optional[str] = "./data/exemplars.txt"
    bandit_state: str = "./var/bandit_state.json"
    
    # Retrieval
    retrieval_top_k: int = 32
    retrieval_per_file_limit: int = 3000
    retrieval_max_per_file: int = 4
    
    # Postfilter
    postfilter_enabled: bool = True
    postfilter_quality: bool = True
    postfilter_topic: bool = True
    postfilter_diversity: bool = True
    postfilter_max_per_source: int = 6
    postfilter_min_keep: int = 10
    postfilter_min_alpha_ratio: float = 0.15
    
    # Bandit
    bandit_enabled: bool = True
    bandit_exploration_pulls: int = 20
    
    # Context
    context_max_chars: int = 400000
    context_max_item_chars: int = 6000
    context_separator: str = "\n\n---\n\n"
    
    # Terminal
    terminal_enabled: bool = True
    terminal_show_chunks: bool = True
    terminal_show_metrics: bool = True
    terminal_show_resources: bool = True
    terminal_color_output: bool = True
    
    # Session
    session_enforce: bool = True
    session_ttl_hours: int = 24
    
    # Logging
    logging_save_telemetry: bool = True
    logging_debug_mode: bool = False
    logging_verbose: bool = False  # Control detailed [ARIA] messages
    
    # Curiosity
    curiosity_enabled: bool = True
    curiosity_gap_threshold: float = 0.3
    curiosity_personality: int = 7
    curiosity_enable_socratic: bool = True
    curiosity_mode: str = 'adaptive'
    
    # Phase 6: Resonance Detection
    resonance_enabled: bool = True
    resonance_history_size: int = 10
    resonance_stable_threshold: int = 5
    resonance_floor: float = 0.85
    resonance_auto_tune: bool = False
    resonance_save_interval: int = 5
    
    # Presets (stored as dict)
    presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Config file metadata
    _config_path: Optional[Path] = None
    _last_mtime: Optional[float] = None
    
    def needs_reload(self) -> bool:
        """Check if config file has been modified since last load"""
        if not self._config_path or not self._config_path.exists():
            return False
        
        try:
            current_mtime = self._config_path.stat().st_mtime
            if self._last_mtime is None or current_mtime > self._last_mtime:
                return True
        except Exception:
            pass
        
        return False
    
    def reload(self) -> bool:
        """Reload config from YAML file if it exists and has changed"""
        if not self.needs_reload():
            return False
        
        try:
            load_config(str(self._config_path), self)
            return True
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for easy access"""
        return {
            'paths': {
                'index_roots': self.index_roots,
                'output_dir': self.output_dir,
                'exemplars': self.exemplars,
                'bandit_state': self.bandit_state,
            },
            'retrieval': {
                'top_k': self.retrieval_top_k,
                'per_file_limit': self.retrieval_per_file_limit,
                'max_per_file': self.retrieval_max_per_file,
            },
            'postfilter': {
                'enabled': self.postfilter_enabled,
                'quality_filter': self.postfilter_quality,
                'topic_filter': self.postfilter_topic,
                'diversity_filter': self.postfilter_diversity,
                'max_per_source': self.postfilter_max_per_source,
                'min_keep': self.postfilter_min_keep,
                'min_alpha_ratio': self.postfilter_min_alpha_ratio,
            },
            'bandit': {
                'enabled': self.bandit_enabled,
                'exploration_pulls': self.bandit_exploration_pulls,
            },
            'context': {
                'max_chars': self.context_max_chars,
                'max_item_chars': self.context_max_item_chars,
                'separator': self.context_separator,
            },
            'terminal': {
                'enabled': self.terminal_enabled,
                'show_chunks': self.terminal_show_chunks,
                'show_metrics': self.terminal_show_metrics,
                'show_resources': self.terminal_show_resources,
                'color_output': self.terminal_color_output,
            },
            'session': {
                'enforce': self.session_enforce,
                'ttl_hours': self.session_ttl_hours,
            },
            'logging': {
                'save_telemetry': self.logging_save_telemetry,
                'debug_mode': self.logging_debug_mode,
                'verbose': self.logging_verbose,
            },
            'curiosity': {
                'enabled': self.curiosity_enabled,
                'gap_threshold': self.curiosity_gap_threshold,
                'personality': self.curiosity_personality,
                'enable_socratic': self.curiosity_enable_socratic,
                'mode': self.curiosity_mode,
            },
            'resonance': {
                'enabled': self.resonance_enabled,
                'history_size': self.resonance_history_size,
                'stable_threshold': self.resonance_stable_threshold,
                'resonance_floor': self.resonance_floor,
                'auto_tune': self.resonance_auto_tune,
                'save_interval': self.resonance_save_interval,
            },
            'presets': self.presets,
        }


def load_config(yaml_path: Optional[str] = None, existing_config: Optional[ARIAConfig] = None) -> ARIAConfig:
    """
    Load configuration from YAML file
    
    Args:
        yaml_path: Path to YAML config file (auto-detects if None)
        existing_config: Existing config to update (creates new if None)
    
    Returns:
        ARIAConfig instance
    """
    config = existing_config or ARIAConfig()
    
    # Auto-detect config file location
    if yaml_path is None:
        # Try common locations
        candidates = [
            Path.cwd() / "aria_config.yaml",
            Path.cwd() / "config.yaml",
            Path.home() / ".aria" / "config.yaml",
            Path.home() / ".config" / "aria" / "config.yaml",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                yaml_path = str(candidate)
                break
    
    # If no config file found, return defaults
    if not yaml_path or not Path(yaml_path).exists():
        return config
    
    try:
        config_path = Path(yaml_path)
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Update config from YAML
        if 'paths' in data:
            paths = data['paths']
            if 'index_roots' in paths:
                config.index_roots = [os.path.expanduser(p) for p in paths['index_roots']]
            if 'output_dir' in paths:
                config.output_dir = os.path.expanduser(paths['output_dir'])
            if 'exemplars' in paths:
                config.exemplars = os.path.expanduser(paths['exemplars']) if paths['exemplars'] else None
            if 'bandit_state' in paths:
                config.bandit_state = os.path.expanduser(paths['bandit_state'])
        
        if 'retrieval' in data:
            retr = data['retrieval']
            config.retrieval_top_k = retr.get('top_k', config.retrieval_top_k)
            config.retrieval_per_file_limit = retr.get('per_file_limit', config.retrieval_per_file_limit)
            config.retrieval_max_per_file = retr.get('max_per_file', config.retrieval_max_per_file)
        
        if 'postfilter' in data:
            pf = data['postfilter']
            config.postfilter_enabled = pf.get('enabled', config.postfilter_enabled)
            config.postfilter_quality = pf.get('quality_filter', config.postfilter_quality)
            config.postfilter_topic = pf.get('topic_filter', config.postfilter_topic)
            config.postfilter_diversity = pf.get('diversity_filter', config.postfilter_diversity)
            config.postfilter_max_per_source = pf.get('max_per_source', config.postfilter_max_per_source)
            config.postfilter_min_keep = pf.get('min_keep', config.postfilter_min_keep)
            config.postfilter_min_alpha_ratio = pf.get('min_alpha_ratio', config.postfilter_min_alpha_ratio)
        
        if 'bandit' in data:
            bd = data['bandit']
            config.bandit_enabled = bd.get('enabled', config.bandit_enabled)
            config.bandit_exploration_pulls = bd.get('exploration_pulls', config.bandit_exploration_pulls)
        
        if 'context' in data:
            ctx = data['context']
            config.context_max_chars = ctx.get('max_chars', config.context_max_chars)
            config.context_max_item_chars = ctx.get('max_item_chars', config.context_max_item_chars)
            config.context_separator = ctx.get('separator', config.context_separator)
        
        if 'terminal' in data:
            term = data['terminal']
            config.terminal_enabled = term.get('enabled', config.terminal_enabled)
            config.terminal_show_chunks = term.get('show_chunks', config.terminal_show_chunks)
            config.terminal_show_metrics = term.get('show_metrics', config.terminal_show_metrics)
            config.terminal_show_resources = term.get('show_resources', config.terminal_show_resources)
            config.terminal_color_output = term.get('color_output', config.terminal_color_output)
        
        if 'session' in data:
            sess = data['session']
            config.session_enforce = sess.get('enforce', config.session_enforce)
            config.session_ttl_hours = sess.get('ttl_hours', config.session_ttl_hours)
        
        if 'logging' in data:
            log = data['logging']
            config.logging_save_telemetry = log.get('save_telemetry', config.logging_save_telemetry)
            config.logging_debug_mode = log.get('debug_mode', config.logging_debug_mode)
            config.logging_verbose = log.get('verbose', config.logging_verbose)
        
        if 'curiosity' in data:
            cur = data['curiosity']
            config.curiosity_enabled = cur.get('enabled', config.curiosity_enabled)
            config.curiosity_gap_threshold = cur.get('gap_threshold', config.curiosity_gap_threshold)
            config.curiosity_personality = cur.get('personality', config.curiosity_personality)
            config.curiosity_enable_socratic = cur.get('enable_socratic', config.curiosity_enable_socratic)
            config.curiosity_mode = cur.get('mode', config.curiosity_mode)
        
        if 'resonance' in data:
            res = data['resonance']
            config.resonance_enabled = res.get('enabled', config.resonance_enabled)
            config.resonance_history_size = res.get('history_size', config.resonance_history_size)
            config.resonance_stable_threshold = res.get('stable_threshold', config.resonance_stable_threshold)
            config.resonance_floor = res.get('resonance_floor', config.resonance_floor)
            config.resonance_auto_tune = res.get('auto_tune', config.resonance_auto_tune)
            config.resonance_save_interval = res.get('save_interval', config.resonance_save_interval)
        
        if 'presets' in data:
            config.presets = data['presets']
        
        # Store metadata for hot-reload
        config._config_path = config_path
        config._last_mtime = config_path.stat().st_mtime
        
    except Exception as e:
        print(f"[ARIA Config] Warning: Failed to load config from {yaml_path}: {e}")
    
    return config


# Global config instance
_global_config: Optional[ARIAConfig] = None


def get_config(yaml_path: Optional[str] = None, force_reload: bool = False) -> ARIAConfig:
    """
    Get global ARIA configuration (singleton pattern with auto-reload)
    
    Args:
        yaml_path: Path to YAML config file
        force_reload: Force reload even if not modified
    
    Returns:
        ARIAConfig instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = load_config(yaml_path)
    elif force_reload or _global_config.needs_reload():
        _global_config.reload()
    
    return _global_config


if __name__ == '__main__':
    # Load and display config
    config = get_config()
    
    print("ARIA Configuration")
    print("=" * 70)
    print()
    
    cfg_dict = config.to_dict()
    
    for section, values in cfg_dict.items():
        print(f"{section.upper()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key:25}: {value}")
        print()
    
    # Verify paths
    print("PATH VERIFICATION:")
    print("-" * 70)
    
    for root in config.index_roots:
        exists = Path(root).exists()
        status = "ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ" if exists else "ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â"
        print(f"  {status} Index root: {root}")
    
    if config.exemplars:
        exists = Path(config.exemplars).exists()
        status = "ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ" if exists else "ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â"
        print(f"  {status} Exemplars: {config.exemplars}")
    
    output_path = Path(config.output_dir)
    print(f"  {'ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ' if output_path.parent.exists() else 'ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â'} Output dir: {config.output_dir}")
