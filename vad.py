#!/usr/bin/env python3
"""
Unified CLI for VAD Distillation Project.

Single entry point for all project operations:
    python vad.py [command] [options]

Commands:
    setup      - Environment setup and validation
    validate   - Data and configuration validation
    train      - Training operations
    baseline   - Run baseline methods
    sweep      - Hyperparameter sweeps
    analyze    - Analysis and visualization
    status     - Show training status
    clean      - Cleanup operations
    export     - Model export

Examples:
    # Setup and validate
    python vad.py setup
    python vad.py validate

    # Training
    python vad.py train --fold F01
    python vad.py train --all --parallel 2
    python vad.py train --quick

    # Baselines
    python vad.py baseline silero
    python vad.py baseline all

    # Analysis
    python vad.py analyze
    python vad.py analyze report

    # Utilities
    python vad.py status
    python vad.py export --format onnx

For detailed help on any command:
    python vad.py [command] --help
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional

# Project metadata
__version__ = "1.0.0"
__author__ = "VAD Distillation Team"

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class VADCli:
    """Main CLI dispatcher for VAD distillation project."""
    
    # Command registry: name -> (module_path, help_text)
    COMMANDS = {
        'setup': ('cli.commands.setup', 'Environment setup and dependency installation'),
        'validate': ('cli.commands.validate', 'Validate data, configs, and environment'),
        'train': ('cli.commands.train', 'Training operations'),
        'baseline': ('cli.commands.baseline', 'Run baseline methods'),
        'sweep': ('cli.commands.sweep', 'Hyperparameter sweeps'),
        'analyze': ('cli.commands.analyze', 'Analysis and visualization'),
        'status': ('cli.commands.status', 'Show training status overview'),
        'clean': ('cli.commands.clean', 'Cleanup operations'),
        'export': ('cli.commands.export', 'Model export to various formats'),
    }
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            prog='vad.py',
            description='Unified CLI for VAD Distillation Project',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog()
        )
        
        # Global options
        parser.add_argument(
            '--version', '-v',
            action='version',
            version=f'%(prog)s {__version__}'
        )
        parser.add_argument(
            '--verbose', '-V',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-error output'
        )
        parser.add_argument(
            '--config', '-c',
            type=str,
            default=None,
            help='Global config file (overrides default)'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            metavar='COMMAND',
            help='Available commands'
        )
        
        # Add subcommand parsers
        for name, (module_path, help_text) in self.COMMANDS.items():
            self._add_subcommand(subparsers, name, module_path, help_text)
        
        return parser
    
    def _add_subcommand(
        self,
        subparsers: argparse._SubParsersAction,
        name: str,
        module_path: str,
        help_text: str
    ) -> None:
        """Add a subcommand parser with arguments from command module."""
        # Create subparser
        subparser = subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Try to import the command module and add its arguments
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, 'add_arguments'):
                module.add_arguments(subparser)
        except ImportError:
            # Module not yet implemented, that's okay
            pass
        
        # Store module path for dispatch
        subparser.set_defaults(_module_path=module_path)
    
    def _get_epilog(self) -> str:
        """Generate help epilog with examples."""
        return """
Examples:
  # Setup and validation
  python vad.py setup
  python vad.py validate

  # Training
  python vad.py train --fold F01
  python vad.py train --all --parallel 2
  python vad.py train --quick

  # Baselines and sweeps
  python vad.py baseline silero
  python vad.py sweep --param alpha --values 0.3 0.5 --folds F01

  # Analysis
  python vad.py analyze
  python vad.py status

For more help: https://github.com/yourorg/vad-distillation
        """.strip()
    
    def dispatch(self, args: Optional[List[str]] = None) -> int:
        """
        Parse arguments and dispatch to appropriate command handler.
        
        Args:
            args: Command line arguments (defaults to sys.argv[1:])
            
        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        parsed_args = self.parser.parse_args(args)
        
        # Check if a command was specified
        if not parsed_args.command:
            self.parser.print_help()
            return 0
        
        # Get command module path
        module_path = getattr(parsed_args, '_module_path', None)
        if not module_path:
            print(f"Error: Unknown command '{parsed_args.command}'", file=sys.stderr)
            return 1
        
        # Import and execute command module
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, 'main'):
                print(
                    f"Error: Command module '{module_path}' missing 'main' function",
                    file=sys.stderr
                )
                return 1
            
            # Execute command
            return module.main(parsed_args) or 0
            
        except ImportError as e:
            print(
                f"Error: Failed to load command '{parsed_args.command}': {e}",
                file=sys.stderr
            )
            print(
                f"Note: Command module '{module_path}' not found. "
                "CLI may not be fully implemented yet.",
                file=sys.stderr
            )
            return 1
        except Exception as e:
            print(f"Error executing command: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    cli = VADCli()
    return cli.dispatch(args)


if __name__ == '__main__':
    sys.exit(main())
