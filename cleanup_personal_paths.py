#!/usr/bin/env python3
"""
cleanup_personal_paths.py

Removes all personal file paths from ARIA documentation and replaces them
with generic placeholders. Prepares documentation for public repository.

Usage:
    python cleanup_personal_paths.py /path/to/aria-github-clean
"""

import sys
import re
from pathlib import Path
from typing import Dict, List

# Personal paths to remove (add yours here)
PERSONAL_PATHS = [
    "/media/notapplicable/Internal-SSD/ai-quaternions-model",
    "/media/notapplicable/ARIA-knowledge",
    "/home/notapplicable/.lmstudio",
    "/media/notapplicable/Internal-SSD/venv",
    "/home/claude",
]

# Replacement patterns
PATH_REPLACEMENTS = {
    # Specific path patterns
    r"/media/notapplicable/Internal-SSD/ai-quaternions-model/src": "./src",
    r"/media/notapplicable/Internal-SSD/ai-quaternions-model": "./",
    r"/media/notapplicable/ARIA-knowledge/data": "./data",
    r"/media/notapplicable/ARIA-knowledge/aria-github-clean": "./",
    r"/media/notapplicable/ARIA-knowledge/rag_runs": "./output/rag_runs",
    r"/media/notapplicable/ARIA-knowledge": "./",
    r"/home/notapplicable/.lmstudio": "~/.lmstudio",
    r"/media/notapplicable/Internal-SSD/venv/bin/python3": "python",
    r"/home/claude": "./",
    
    # Generic personal path patterns
    r"/home/[^/\s]+": "~",
    r"/Users/[^/\s]+": "~",
    r"C:\\Users\\[^\\]+": "~",
}

# Files to clean
DOC_EXTENSIONS = [".md", ".txt", ".yaml", ".yml"]

def clean_file(file_path: Path) -> bool:
    """
    Clean personal paths from a single file.
    
    Returns:
        True if file was modified, False otherwise
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Apply all replacements
        for pattern, replacement in PATH_REPLACEMENTS.items():
            content = re.sub(pattern, replacement, content)
        
        # Check if content changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
        
        return False
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def clean_directory(root_dir: Path) -> Dict[str, int]:
    """
    Clean all documentation files in directory tree.
    
    Returns:
        Statistics about cleaned files
    """
    stats = {
        'total_files': 0,
        'modified_files': 0,
        'skipped_files': 0,
    }
    
    # Find all documentation files
    for ext in DOC_EXTENSIONS:
        for file_path in root_dir.rglob(f"*{ext}"):
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            # Skip cache, output, and build directories
            skip_dirs = {'cache', 'output', '__pycache__', 'node_modules', '.git'}
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                stats['skipped_files'] += 1
                continue
            
            stats['total_files'] += 1
            
            if clean_file(file_path):
                print(f"✅ Cleaned: {file_path.relative_to(root_dir)}")
                stats['modified_files'] += 1
            else:
                print(f"⏭️  No changes: {file_path.relative_to(root_dir)}")
    
    return stats

def verify_no_personal_paths(root_dir: Path) -> List[str]:
    """
    Verify that no personal paths remain in documentation.
    
    Returns:
        List of files still containing personal paths
    """
    violations = []
    
    for ext in DOC_EXTENSIONS:
        for file_path in root_dir.rglob(f"*{ext}"):
            # Skip hidden files and cache directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            skip_dirs = {'cache', 'output', '__pycache__', 'node_modules', '.git'}
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Check for personal path patterns
                for path_pattern in PERSONAL_PATHS:
                    if path_pattern in content:
                        violations.append(
                            f"{file_path.relative_to(root_dir)}: contains {path_pattern}"
                        )
            
            except Exception as e:
                print(f"⚠️  Could not verify {file_path}: {e}")
    
    return violations

def main():
    if len(sys.argv) < 2:
        print("Usage: python cleanup_personal_paths.py /path/to/aria-github-clean")
        sys.exit(1)
    
    root_dir = Path(sys.argv[1])
    
    if not root_dir.exists():
        print(f"❌ Directory not found: {root_dir}")
        sys.exit(1)
    
    print(f"🧹 Cleaning personal paths from: {root_dir}\n")
    
    # Clean all files
    stats = clean_directory(root_dir)
    
    print("\n" + "="*60)
    print("📊 Cleanup Statistics:")
    print(f"  Total files processed: {stats['total_files']}")
    print(f"  Files modified: {stats['modified_files']}")
    print(f"  Files skipped: {stats['skipped_files']}")
    print("="*60 + "\n")
    
    # Verify no personal paths remain
    print("🔍 Verifying no personal paths remain...\n")
    violations = verify_no_personal_paths(root_dir)
    
    if violations:
        print("⚠️  WARNING: Personal paths still found in:\n")
        for violation in violations:
            print(f"  - {violation}")
        print("\n❌ Please review and remove manually.")
        sys.exit(1)
    else:
        print("✅ All personal paths removed successfully!\n")
    
    print("🎉 Documentation ready for public repository!")

if __name__ == "__main__":
    main()
