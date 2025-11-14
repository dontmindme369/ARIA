#!/usr/bin/env python3
"""
presets.py - Preset dataclass for ARIA retrieval strategies
"""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Preset:
    """Retrieval strategy preset"""
    name: str
    args: Dict[str, Any]

    def __repr__(self) -> str:
        return f"Preset(name='{self.name}', args={self.args})"
