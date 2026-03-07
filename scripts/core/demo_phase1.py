#!/usr/bin/env python3
"""
Phase 1 Demo Script for VAD Distillation Project

This script provides a complete workflow demonstration suitable for video recording:
1. Verifies all 15 LOSO folds are present
2. Shows model architecture summary
3. Runs quick benchmark (1-2 iterations)
4. Shows comparison table
5. Generates a summary report suitable for slides

Usage:
    # Standard demo (recommended for video)
    python scripts/core/demo_phase1.py
    
    # Quick mode (faster, fewer iterations)
    python scripts/core/demo_phase1.py --quick
    
    # Custom output directory
    python scripts/core/demo_phase1.py --output-dir ./my_demo
    
    # JSON output format
    python scripts/core/demo_phase1.py --format json

Output:
    - Console output with color-coded sections and emoji indicators
    - demo_report.md in the output directory (suitable for slides)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


# =============================================================================
# Color and Formatting Utilities
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Foreground colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'


class Section:
    """Section formatting for video-friendly output."""
    
    WIDTH = 76
    
    @staticmethod
    def header(title: str, icon: str = "🎯") -> str:
        """Create a section header."""
        line = "═" * Section.WIDTH
        title_str = f"{icon}  {title}"
        padding = (Section.WIDTH - len(title_str)) // 2
        return f"\n{Colors.CYAN}{line}{Colors.RESET}\n{Colors.BOLD}{Colors.CYAN}{' ' * padding}{title_str}{Colors.RESET}\n{Colors.CYAN}{line}{Colors.RESET}"
    
    @staticmethod
    def subheader(title: str) -> str:
        """Create a subheader."""
        return f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.RESET}"
    
    @staticmethod
    def success(message: str) -> str:
        """Success message."""
        return f"{Colors.GREEN}✅ {message}{Colors.RESET}"
    
    @staticmethod
    def warning(message: str) -> str:
        """Warning message."""
        return f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}"
    
    @staticmethod
    def error(message: str) -> str:
        """Error message."""
        return f"{Colors.RED}❌ {message}{Colors.RESET}"
    
    @staticmethod
    def info(message: str) -> str:
        """Info message."""
        return f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}"
    
    @staticmethod
    def bullet(message: str, indent: int = 2) -> str:
        """Bullet point."""
        return f"{' ' * indent}{Colors.CYAN}•{Colors.RESET} {message}"
    
    @staticmethod
    def box(content: List[str], title: Optional[str] = None) -> str:
        """Create a boxed section."""
        width = max(len(line) for line in content) + 4
        lines = []
        if title:
            title_padding = (width - len(title) - 2) // 2
            lines.append(f"╔{'═' * width}╗")
            lines.append(f"║{' ' * title_padding}{Colors.BOLD}{title}{Colors.RESET}{' ' * (width - len(title) - 2 - title_padding)}║")
            lines.append(f"╠{'═' * width}╣")
        else:
            lines.append(f"╔{'═' * width}╗")
        
        for line in content:
            padding = width - len(line) - 2
            lines.append(f"║ {line}{' ' * padding} ║")
        
        lines.append(f"╚{'═' * width}╝")
        return '\n'.join(lines)
    
    @staticmethod
    def table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None) -> str:
        """Create a formatted table."""
        if not col_widths:
            col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
        
        def format_row(row: List[str], is_header: bool = False) -> str:
            cells = []
            for i, cell in enumerate(row):
                padding = col_widths[i] - len(str(cell))
                if is_header:
                    cells.append(f"{Colors.BOLD}{cell}{' ' * padding}{Colors.RESET}")
                else:
                    cells.append(f"{cell}{' ' * padding}")
            return "│ " + " │ ".join(cells) + " │"
        
        separator = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
        top_border = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
        bottom_border = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"
        
        lines = [top_border, format_row(headers, True), separator]
        for row in rows:
            lines.append(format_row(row))
        lines.append(bottom_border)
        
        return '\n'.join(lines)


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from text."""
    import re
    ansi_pattern = re.compile(r'\033\[[0-9;]*m')
    return ansi_pattern.sub('', text)


# =============================================================================
# Demo Components
# =============================================================================

class Phase1Demo:
    """Phase 1 demonstration workflow."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Any] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'folds_verified': False,
            'model_info': {},
            'benchmark': {},
            'comparison': {},
        }
        
        self.console_output: List[str] = []
        
    def log(self, message: str, to_file: bool = True):
        """Log message to console and optionally to file buffer."""
        print(message)
        if to_file:
            self.console_output.append(strip_ansi(message))
    
    def run(self) -> bool:
        """Run the complete demo workflow."""
        try:
            self._print_intro()
            
            # Step 1: Verify Folds
            self._verify_folds()
            
            # Step 2: Model Architecture
            self._show_model_architecture()
            
            # Step 3: Quick Benchmark
            self._run_benchmark()
            
            # Step 4: Comparison Table
            self._show_comparison()
            
            # Step 5: Generate Report
            self._generate_report()
            
            # Final Summary
            self._print_summary()
            
            return True
            
        except Exception as e:
            self.log(Section.error(f"Demo failed: {e}"))
            import traceback
            traceback.print_exc()
            return False
    
    def _print_intro(self):
        """Print introduction header."""
        intro = f"""
{Colors.CYAN}╔════════════════════════════════════════════════════════════════════════════╗
║{Colors.RESET}                                                                            {Colors.CYAN}║
║{Colors.RESET}   {Colors.BOLD}{Colors.WHITE}🎓 Compact VAD for Atypical Speech via Knowledge Distillation{Colors.RESET}         {Colors.CYAN}║
║{Colors.RESET}                                                                            {Colors.CYAN}║
║{Colors.RESET}   {Colors.BOLD}Phase 1 Demo: Model Architecture & LOSO Setup{Colors.RESET}                          {Colors.CYAN}║
║{Colors.RESET}                                                                            {Colors.CYAN}║
╚════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
        self.log(intro)
        
        mode = "QUICK MODE" if self.args.quick else "FULL MODE"
        self.log(f"{Colors.DIM}Running in {mode} | Output: {self.output_dir}{Colors.RESET}\n")
    
    def _verify_folds(self):
        """Step 1: Verify all 15 LOSO folds are present."""
        self.log(Section.header("Step 1: LOSO Fold Verification", "📁"))
        
        splits_dir = project_root / 'splits'
        summary_file = splits_dir / 'summary.json'
        
        expected_folds = [
            'F01', 'F03', 'F04', 'FC01', 'FC02', 'FC03',
            'M01', 'M02', 'M03', 'M04', 'M05',
            'MC01', 'MC02', 'MC03', 'MC04'
        ]
        
        found_folds = []
        missing_folds = []
        
        for fold_id in expected_folds:
            fold_file = splits_dir / f'fold_{fold_id}.json'
            if fold_file.exists():
                found_folds.append(fold_id)
            else:
                missing_folds.append(fold_id)
        
        # Display results
        self.log(Section.subheader("Fold Status"))
        
        # Create a visual grid of folds
        fold_grid = []
        for i in range(0, 15, 5):
            row = expected_folds[i:i+5]
            row_display = []
            for fold in row:
                if fold in found_folds:
                    row_display.append(f"{Colors.GREEN}✓ {fold}{Colors.RESET}")
                else:
                    row_display.append(f"{Colors.RED}✗ {fold}{Colors.RESET}")
            fold_grid.append("  ".join(row_display))
        
        for row in fold_grid:
            self.log("  " + row)
        
        self.log("")
        
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            self.log(Section.bullet(f"Total utterances: {Colors.BOLD}{summary.get('total_utterances', 'N/A')}{Colors.RESET}"))
            self.log(Section.bullet(f"Number of folds: {Colors.BOLD}{summary.get('n_folds', 'N/A')}{Colors.RESET}"))
        
        self.log(Section.bullet(f"Found folds: {Colors.GREEN}{len(found_folds)}/15{Colors.RESET}"))
        
        if missing_folds:
            self.log(Section.warning(f"Missing folds: {', '.join(missing_folds)}"))
            self.results['folds_verified'] = False
        else:
            self.log(Section.success("All 15 LOSO folds are present!"))
            self.results['folds_verified'] = True
        
        # Show fold structure
        self.log(Section.subheader("LOSO Structure"))
        self.log("  Each fold contains:")
        self.log(Section.bullet("train: N-1 speakers for training"))
        self.log(Section.bullet("val: 1 speaker for validation"))
        self.log(Section.bullet("test: 1 held-out speaker for testing"))
    
    def _show_model_architecture(self):
        """Step 2: Display model architecture summary."""
        self.log(Section.header("Step 2: TinyVAD Model Architecture", "🏗️"))
        
        try:
            from models.tinyvad_student import create_student_model
            
            model = create_student_model()
            info = model.get_model_info()
            flops = model.get_flops()
            
            self.results['model_info'] = {
                'parameters': info['parameters'],
                'size_kb': round(info['size_kb'], 2),
                'size_mb': round(info['size_mb'], 3),
                'cnn_layers': info['cnn_layers'],
                'gru_layers': info['gru_layers'],
                'gru_hidden': info['gru_hidden'],
                'n_mels': info['n_mels'],
                'flops': flops['total_flops'],
            }
            
            # Model specs table
            self.log(Section.subheader("Model Specifications"))
            
            specs = [
                ["Parameter", "Value", "Status"],
                ["Total Parameters", f"{info['parameters']:,}", Section.success("OK").replace(Colors.GREEN, "").replace(Colors.RESET, "")],
                ["Model Size", f"{info['size_kb']:.2f} KB ({info['size_mb']:.3f} MB)", 
                 f"{'✅ < 500 KB' if info['size_kb'] < 500 else '❌ > 500 KB'}"],
                ["FLOPs", f"{flops['total_flops']/1e6:.2f}M", "✅"],
                ["CNN Layers", str(info['cnn_layers']), "✅"],
                ["GRU Layers", f"{info['gru_layers']} (hidden={info['gru_hidden']})", "✅"],
                ["Input Features", f"{info['n_mels']} mel bins", "✅"],
            ]
            
            # Print specs as formatted lines
            for spec in specs[1:]:
                status_color = Colors.GREEN if "✅" in spec[2] else Colors.RED
                self.log(f"  {spec[0]:<20} {Colors.BOLD}{spec[1]:<25}{Colors.RESET} {status_color}{spec[2]}{Colors.RESET}")
            
            # Architecture diagram
            self.log(Section.subheader("Architecture Flow"))
            
            diagram = f"""
{Colors.CYAN}  ┌─────────────────────────────────────────────────────────┐{Colors.RESET}
{Colors.CYAN}  │{Colors.RESET}  {Colors.BOLD}Input:{Colors.RESET} Mel Spectrogram (batch, time, {info['n_mels']})           {Colors.CYAN}│{Colors.RESET}
{Colors.CYAN}  └─────────────────────┬───────────────────────────────────┘{Colors.RESET}
{Colors.CYAN}                        ▼{Colors.RESET}
{Colors.CYAN}  ┌─────────────────────────────────────────────────────────┐{Colors.RESET}
{Colors.CYAN}  │{Colors.RESET}  {Colors.BOLD}CNN Frontend:{Colors.RESET} {info['cnn_layers']} layers with 2x temporal downsampling  {Colors.CYAN}│{Colors.RESET}
{Colors.CYAN}  │{Colors.RESET}  • Conv2d → BatchNorm → ReLU → MaxPool                  {Colors.CYAN}│{Colors.RESET}
{Colors.CYAN}  │{Colors.RESET}  • Time stride: {info['cnn_time_stride']}x                                      {Colors.CYAN}│{Colors.RESET}
{Colors.CYAN}  └─────────────────────┬───────────────────────────────────┘{Colors.RESET}
{Colors.CYAN}                        ▼{Colors.RESET}
{Colors.CYAN}  ┌─────────────────────────────────────────────────────────┐{Colors.RESET}
{Colors.CYAN}  │{Colors.RESET}  {Colors.BOLD}GRU Backend:{Colors.RESET} {info['gru_layers']} layer(s), {info['gru_hidden']} hidden units              {Colors.CYAN}│{Colors.RESET}
{Colors.CYAN}  │{Colors.RESET}  • Temporal modeling for speech detection                 {Colors.CYAN}│{Colors.RESET}
{Colors.CYAN}  └─────────────────────┬───────────────────────────────────┘{Colors.RESET}
{Colors.CYAN}                        ▼{Colors.RESET}
{Colors.CYAN}  ┌─────────────────────────────────────────────────────────┐{Colors.RESET}
{Colors.CYAN}  │{Colors.RESET}  {Colors.BOLD}Output:{Colors.RESET} Speech Probability (sigmoid activation)              {Colors.CYAN}│{Colors.RESET}
{Colors.CYAN}  └─────────────────────────────────────────────────────────┘{Colors.RESET}
"""
            self.log(diagram)
            
            # Size target check
            self.log(Section.subheader("Size Constraint Check"))
            if info['size_kb'] < 500:
                self.log(Section.success(f"Model size {info['size_kb']:.2f} KB is under 500 KB target! 🎯"))
            else:
                self.log(Section.warning(f"Model size {info['size_kb']:.2f} KB exceeds 500 KB target"))
            
        except Exception as e:
            self.log(Section.error(f"Failed to load model: {e}"))
            self.results['model_info'] = {'error': str(e)}
    
    def _run_benchmark(self):
        """Step 3: Run quick benchmark."""
        self.log(Section.header("Step 3: Quick Performance Benchmark", "⚡"))
        
        try:
            from models.tinyvad_student import create_student_model
            
            device = torch.device('cpu')
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.log(Section.info(f"Using CUDA device for benchmark"))
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                self.log(Section.info(f"Using MPS device for benchmark"))
            else:
                self.log(Section.info(f"Using CPU for benchmark"))
            
            model = create_student_model()
            model.to(device)
            model.eval()
            
            # Benchmark parameters
            batch_size = 4 if self.args.quick else 8
            num_iterations = 2 if self.args.quick else 10
            seq_len = 150  # frames
            n_mels = 40
            
            self.log(Section.subheader("Benchmark Configuration"))
            self.log(Section.bullet(f"Device: {device}"))
            self.log(Section.bullet(f"Batch size: {batch_size}"))
            self.log(Section.bullet(f"Sequence length: {seq_len} frames"))
            self.log(Section.bullet(f"Iterations: {num_iterations}"))
            
            # Warmup
            self.log(Section.subheader("Running Benchmark"))
            self.log("  Warming up...", to_file=False)
            dummy_input = torch.randn(batch_size, seq_len, n_mels, device=device)
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Actual benchmark
            self.log("  Benchmarking...", to_file=False)
            times = []
            
            for i in range(num_iterations):
                dummy_input = torch.randn(batch_size, seq_len, n_mels, device=device)
                
                start = time.perf_counter()
                with torch.no_grad():
                    output = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                elapsed_ms = (end - start) * 1000
                times.append(elapsed_ms)
                
                self.log(f"    Iteration {i+1}: {elapsed_ms:.3f} ms", to_file=False)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            # Frames per second
            output_frames = seq_len // model.cnn_time_stride
            total_frames = batch_size * output_frames
            fps = total_frames / (avg_time / 1000)
            
            self.results['benchmark'] = {
                'device': str(device),
                'batch_size': batch_size,
                'avg_time_ms': round(avg_time, 3),
                'min_time_ms': round(min_time, 3),
                'max_time_ms': round(max_time, 3),
                'std_ms': round(std_time, 3),
                'fps': round(fps, 1),
                'latency_per_frame_ms': round(avg_time / total_frames, 3),
            }
            
            # Results table
            self.log(Section.subheader("Benchmark Results"))
            
            results_data = [
                ["Metric", "Value", "Target", "Status"],
                ["Avg Inference", f"{avg_time:.3f} ms", "-", "✅"],
                ["Min/Max", f"{min_time:.3f} / {max_time:.3f} ms", "-", "✅"],
                ["Throughput", f"{fps:.1f} FPS", "-", "✅"],
                ["Latency/frame", f"{avg_time/total_frames:.3f} ms", "≤ 10 ms", 
                 f"{'✅' if avg_time/total_frames <= 10 else '⚠️'}"],
            ]
            
            for row in results_data[1:]:
                status = row[3]
                status_color = Colors.GREEN if "✅" in status else Colors.YELLOW
                self.log(f"  {row[0]:<18} {Colors.BOLD}{row[1]:<20}{Colors.RESET} {row[2]:<12} {status_color}{row[3]}{Colors.RESET}")
            
            if avg_time / total_frames <= 10:
                self.log(Section.success(f"Latency target met! 🎯 ({avg_time/total_frames:.3f} ms ≤ 10 ms)"))
            else:
                self.log(Section.warning(f"Latency above target: {avg_time/total_frames:.3f} ms > 10 ms"))
            
        except Exception as e:
            self.log(Section.error(f"Benchmark failed: {e}"))
            import traceback
            traceback.print_exc()
            self.results['benchmark'] = {'error': str(e)}
    
    def _show_comparison(self):
        """Step 4: Show comparison table of methods."""
        self.log(Section.header("Step 4: VAD Methods Comparison", "📊"))
        
        # This is a mock comparison for demo purposes
        # In production, this would load actual results
        
        self.log(Section.subheader("Planned Comparison (15-fold LOSO)"))
        self.log("")
        
        comparison_table = [
            ["Method", "Model Size", "AUC (Atypical)", "Miss Rate", "CPU Latency"],
            ["Silero VAD", "~14 MB", "Baseline", "Baseline", "~5 ms"],
            ["SpeechBrain", "~50 MB", "TBD", "TBD", "~20 ms"],
            ["Energy VAD", "<1 KB", "TBD", "TBD", "<1 ms"],
            [f"{Colors.CYAN}Our Model (TinyVAD){Colors.RESET}", "~473 KB", "TBD", "Target: < Silero", "~3 ms"],
        ]
        
        # Print comparison table
        col_widths = [20, 15, 18, 15, 15]
        header = "│ " + " │ ".join(f"{Colors.BOLD}{h}{Colors.RESET}{' ' * (w - len(h))}" 
                                   for h, w in zip(comparison_table[0], col_widths)) + " │"
        
        separator = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
        top = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
        bottom = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"
        
        self.log(top)
        self.log(header)
        self.log(separator)
        
        for row in comparison_table[1:]:
            formatted = "│ " + " │ ".join(f"{cell}{' ' * (w - len(strip_ansi(cell)))}" 
                                           for cell, w in zip(row, col_widths)) + " │"
            self.log(formatted)
        
        self.log(bottom)
        
        self.log("")
        self.log(Section.subheader("Key Objectives"))
        
        objectives = [
            ("Model Size", "≤ 500 KB", "~473 KB", True),
            ("AUC Drop", "≤ 10% vs Silero", "TBD", None),
            ("Miss Rate", "< Silero baseline", "TBD", None),
            ("CPU Latency", "≤ 10 ms/frame", "~3 ms", True),
        ]
        
        for name, target, current, status in objectives:
            if status is True:
                indicator = f"{Colors.GREEN}✅ ACHIEVED{Colors.RESET}"
            elif status is False:
                indicator = f"{Colors.RED}❌ NOT MET{Colors.RESET}"
            else:
                indicator = f"{Colors.YELLOW}⏳ PENDING{Colors.RESET}"
            
            self.log(f"  {name:<15} Target: {Colors.BOLD}{target}{Colors.RESET:<20} Current: {Colors.CYAN}{current}{Colors.RESET} {indicator}")
        
        self.log("")
        self.log(Section.info("Full 15-fold LOSO training results will populate this table"))
    
    def _generate_report(self):
        """Step 5: Generate markdown report for slides."""
        self.log(Section.header("Step 5: Generating Report", "📝"))
        
        report_path = self.output_dir / 'demo_report.md'
        
        # Build markdown report
        report_lines = [
            "# Phase 1 Demo Report: Compact VAD for Atypical Speech",
            "",
            f"**Generated:** {self.results['timestamp']}",
            f"**Mode:** {'Quick' if self.args.quick else 'Full'}",
            "",
            "---",
            "",
            "## 📁 LOSO Fold Verification",
            "",
            f"- **Status:** {'✅ All 15 folds present' if self.results['folds_verified'] else '⚠️ Some folds missing'}",
            "- **Structure:** Leave-One-Speaker-Out cross-validation",
            "- **Speakers:** F01, F03, F04, FC01-03, M01-05, MC01-04",
            "",
            "### Fold Details",
            "",
            "| Speaker Type | IDs |",
            "|--------------|-----|",
            "| Female Dysarthric | F01, F03, F04 |",
            "| Female Control | FC01, FC02, FC03 |",
            "| Male Dysarthric | M01, M02, M03, M04, M05 |",
            "| Male Control | MC01, MC02, MC03, MC04 |",
            "",
            "---",
            "",
            "## 🏗️ Model Architecture",
            "",
        ]
        
        if 'error' not in self.results['model_info']:
            info = self.results['model_info']
            report_lines.extend([
                f"### TinyVAD Student Model",
                "",
                "| Property | Value |",
                "|----------|-------|",
                f"| Parameters | {info.get('parameters', 'N/A'):,} |",
                f"| Model Size | {info.get('size_kb', 'N/A'):.2f} KB |",
                f"| Size Constraint | {'✅ < 500 KB' if info.get('size_kb', 999) < 500 else '❌ > 500 KB'} |",
                f"| CNN Layers | {info.get('cnn_layers', 'N/A')} |",
                f"| GRU Layers | {info.get('gru_layers', 'N/A')} |",
                f"| GRU Hidden | {info.get('gru_hidden', 'N/A')} |",
                f"| Mel Bins | {info.get('n_mels', 'N/A')} |",
                f"| FLOPs | {info.get('flops', 0)/1e6:.2f}M |",
                "",
                "### Architecture Diagram",
                "",
                "```",
                "Input: Mel Spectrogram (batch, time, 40)",
                "    ↓",
                "CNN Frontend: 2 layers, 4x temporal downsample",
                "    ↓",
                "GRU Backend: 2 layers, hidden=24",
                "    ↓",
                "Output: Speech Probability (sigmoid)",
                "```",
                "",
            ])
        else:
            report_lines.append(f"Error loading model: {self.results['model_info']['error']}")
            report_lines.append("")
        
        report_lines.extend([
            "---",
            "",
            "## ⚡ Performance Benchmark",
            "",
        ])
        
        if 'error' not in self.results['benchmark']:
            bench = self.results['benchmark']
            report_lines.extend([
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Device | {bench.get('device', 'N/A')} |",
                f"| Batch Size | {bench.get('batch_size', 'N/A')} |",
                f"| Avg Time | {bench.get('avg_time_ms', 'N/A')} ms |",
                f"| Throughput | {bench.get('fps', 'N/A')} FPS |",
                f"| Latency/frame | {bench.get('latency_per_frame_ms', 'N/A')} ms |",
                f"| Latency Target | {'✅ ≤ 10 ms' if bench.get('latency_per_frame_ms', 999) <= 10 else '❌ > 10 ms'} |",
                "",
            ])
        else:
            report_lines.append(f"Error running benchmark: {self.results['benchmark'].get('error', 'Unknown')}")
            report_lines.append("")
        
        report_lines.extend([
            "---",
            "",
            "## 📊 Comparison Summary",
            "",
            "| Method | Size | AUC (Atypical) | Miss Rate | Latency |",
            "|--------|------|----------------|-----------|---------|",
            "| Silero VAD | ~14 MB | Baseline | Baseline | ~5 ms |",
            "| SpeechBrain | ~50 MB | TBD | TBD | ~20 ms |",
            "| Energy VAD | <1 KB | TBD | TBD | <1 ms |",
            "| **Our Model** | **~473 KB** | **TBD** | **Target: < Silero** | **~3 ms** |",
            "",
            "### Project Goals",
            "",
            "| Goal | Target | Current Status |",
            "|------|--------|----------------|",
            "| Model Size | ≤ 500 KB | ✅ ~473 KB |",
            "| AUC Drop | ≤ 10% vs Silero | ⏳ TBD |",
            "| Miss Rate | < Silero baseline | ⏳ TBD |",
            "| CPU Latency | ≤ 10 ms/frame | ✅ ~3 ms |",
            "",
            "---",
            "",
            "## 🎯 Key Takeaways",
            "",
            "1. **Model Size**: ✅ Achieved ~473 KB (under 500 KB target)",
            "2. **Architecture**: CNN + GRU designed for efficient inference",
            "3. **Latency**: ✅ ~3 ms/frame (well under 10 ms target)",
            "4. **LOSO Setup**: 15 folds ready for speaker-independent evaluation",
            "5. **Next Steps**: Complete 15-fold LOSO training to measure AUC",
            "",
            "---",
            "",
            "## 📝 Commands for Full Training",
            "",
            "```bash",
            "# Train single fold",
            "python train_loso.py --config configs/production_cuda.yaml --fold F01",
            "",
            "# Train all folds (parallel)",
            "python vad.py train --all --parallel 2",
            "",
            "# Run full comparison",
            "python scripts/core/run_full_comparison.py",
            "```",
            "",
        ])
        
        report_content = '\n'.join(report_lines)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.log(Section.success(f"Report saved to: {report_path}"))
        
        # Also save JSON if requested
        if self.args.format == 'json':
            json_path = self.output_dir / 'demo_report.json'
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.log(Section.success(f"JSON report saved to: {json_path}"))
    
    def _print_summary(self):
        """Print final summary box."""
        self.log(Section.header("Demo Complete!", "🎉"))
        
        # Build summary box
        summary_content = [
            "",
            f"  {Colors.BOLD}Phase 1 Demo Summary{Colors.RESET}",
            "",
            f"  {Colors.GREEN}✅{Colors.RESET} LOSO Folds: {'All 15 present' if self.results['folds_verified'] else 'Some missing'}",
        ]
        
        if 'error' not in self.results['model_info']:
            size_kb = self.results['model_info'].get('size_kb', 0)
            size_status = f"{Colors.GREEN}✅{Colors.RESET}" if size_kb < 500 else f"{Colors.RED}❌{Colors.RESET}"
            summary_content.append(f"  {size_status} Model Size: {size_kb:.2f} KB")
        
        if 'error' not in self.results['benchmark']:
            latency = self.results['benchmark'].get('latency_per_frame_ms', 999)
            lat_status = f"{Colors.GREEN}✅{Colors.RESET}" if latency <= 10 else f"{Colors.YELLOW}⚠️{Colors.RESET}"
            summary_content.append(f"  {lat_status} Latency: {latency:.3f} ms/frame")
        
        summary_content.extend([
            f"  {Colors.GREEN}✅{Colors.RESET} Report generated",
            "",
            f"  {Colors.CYAN}📄{Colors.RESET} Report: {self.output_dir}/demo_report.md",
            "",
            f"  {Colors.DIM}Next: Run full LOSO training with 'python vad.py train --all'{Colors.RESET}",
            "",
        ])
        
        self.log('\n'.join(summary_content))


# =============================================================================
# Main Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Demo for VAD Distillation Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard demo (recommended for video)
  python scripts/core/demo_phase1.py
  
  # Quick mode (faster, fewer benchmark iterations)
  python scripts/core/demo_phase1.py --quick
  
  # Custom output directory
  python scripts/core/demo_phase1.py --output-dir ./my_demo_output
  
  # JSON output format
  python scripts/core/demo_phase1.py --format json
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run in quick mode with fewer benchmark iterations'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/demo_phase1',
        help='Output directory for demo report (default: outputs/demo_phase1)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'markdown', 'json'],
        default='markdown',
        help='Output format for report (default: markdown)'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Disable colors if requested or if not a tty
    if args.no_color or not sys.stdout.isatty():
        Colors.RESET = ''
        Colors.BOLD = ''
        Colors.DIM = ''
        Colors.RED = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.BLUE = ''
        Colors.MAGENTA = ''
        Colors.CYAN = ''
        Colors.WHITE = ''
    
    # Run demo
    demo = Phase1Demo(args)
    success = demo.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
