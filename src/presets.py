#!/usr/bin/env python3
"""
presets.py - Preset configuration and utilities

Defines the canonical Preset class and converts preset objects to CLI arguments
"""

from typing import Any, Dict, List


class Preset:
    """Retrieval preset configuration"""
    
    def __init__(self, name: str, args: Dict[str, Any]):
        self.name = name
        self.args = args
    
    def __repr__(self) -> str:
        return f"Preset({self.name}, {self.args})"


def preset_to_cli_args(preset: Any) -> List[str]:
    """
    Convert preset object to command-line arguments
    
    Args:
        preset: Preset object with args dict
    
    Returns:
        List of command-line argument strings
    """
    flags = []
    
    # Handle preset as dict
    if isinstance(preset, dict):
        args = preset
    # Handle preset as object with args attribute
    elif hasattr(preset, 'args'):
        args = preset.args if isinstance(preset.args, dict) else {}
    else:
        return flags
    
    # Convert common parameters
    if 'top_k' in args:
        flags.extend(['--top-k', str(args['top_k'])])
    
    if 'sem_limit' in args:
        flags.extend(['--sem-limit', str(args['sem_limit'])])
    
    if 'rotations' in args and args['rotations'] > 1:
        flags.extend(['--rotations', str(args['rotations'])])
    
    if 'max_per_file' in args:
        flags.extend(['--max-per-file', str(args['max_per_file'])])
    
    if 'diversity_mm' in args and args['diversity_mm']:
        flags.append('--diversity-mm')
    
    if 'use_cache' in args and not args['use_cache']:
        flags.append('--no-cache')
    
    if 'boost_recent' in args and args['boost_recent']:
        flags.append('--boost-recent')
    
    return flags


__all__ = ['Preset', 'preset_to_cli_args']
