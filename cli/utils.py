"""
Utility functions for VAD CLI.
"""

import functools
import sys
from pathlib import Path
from typing import Callable, Optional

from utils.common import format_duration


def ensure_project_root() -> Path:
    """
    Ensure the project root is in sys.path.
    
    Returns:
        Path to project root
    """
    # Get project root (parent of cli directory)
    cli_dir = Path(__file__).parent.resolve()
    project_root = cli_dir.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def print_error(message: str, suggestion: Optional[str] = None) -> None:
    """
    Print formatted error message.
    
    Args:
        message: Error description
        suggestion: Optional suggestion for fixing
    """
    print(f"\n❌ ERROR: {message}", file=sys.stderr)
    if suggestion:
        print(f"\n💡 Suggestion: {suggestion}", file=sys.stderr)
    print(file=sys.stderr)


def print_success(message: str) -> None:
    """Print success message."""
    print(f"✓ {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"⚠ {message}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"ℹ {message}")


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask user to confirm an action.
    
    Args:
        message: Confirmation prompt
        default: Default answer if user just presses enter
        
    Returns:
        True if confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    response = input(f"{message} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    return response in ('y', 'yes')


def format_size(size_bytes: float) -> str:
    """
    Format size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def with_project_root(func: Callable) -> Callable:
    """
    Decorator to ensure project root is in sys.path before executing function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ensure_project_root()
        return func(*args, **kwargs)
    return wrapper


def get_device_preference(preferred: Optional[str] = None) -> str:
    """
    Get device preference, with auto-detection.
    
    Args:
        preferred: Preferred device (cpu/cuda/mps/auto)
        
    Returns:
        Device string
    """
    if preferred and preferred != 'auto':
        return preferred
    
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    
    return 'cpu'


class ProgressTracker:
    """Simple progress tracker for batch operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
    
    def update(self, n: int = 1) -> None:
        """Update progress."""
        self.current += n
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print progress bar."""
        if self.total == 0:
            return
        
        percent = 100 * self.current / self.total
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f'\r{self.description}: |{bar}| {percent:.1f}% ({self.current}/{self.total})', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def close(self) -> None:
        """Close tracker."""
        if self.current < self.total:
            self.current = self.total
            self._print_progress()
