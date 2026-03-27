#!/usr/bin/env python3
"""
Latency Benchmark Script for VAD Student Model

Benchmarks the TinyVAD student model on synthetic audio of varying durations
to measure inference latency, throughput, and memory usage on CPU (edge deployment target).

Engineering Targets:
    - Model Size: ≤ 500 KB (achieved: ~473 KB)
    - CPU Latency: ≤ 10 ms/frame
    - Real-time Factor: < 1.0 (faster than real-time)

Usage:
    # Run full benchmark
    python scripts/analysis/benchmark_latency.py
    
    # Quick benchmark (fewer iterations)
    python scripts/analysis/benchmark_latency.py --quick
    
    # Custom checkpoint path
    python scripts/analysis/benchmark_latency.py --checkpoint path/to/model.pt
    
    # Compare with baselines
    python scripts/analysis/benchmark_latency.py --compare-baselines
    
    # Specific audio durations
    python scripts/analysis/benchmark_latency.py --durations 1 5 10

Outputs:
    - Console: Formatted benchmark results table
    - JSON: analysis/benchmark_results.json (detailed results)
    - Plot: analysis/latency_comparison.png (if matplotlib available)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.tinyvad_student import TinyVAD, create_student_model


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "production_cuda" / "checkpoints"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_DURATIONS = [1, 5, 10, 30]  # seconds
DEFAULT_ITERATIONS = 10
WARMUP_ITERATIONS = 3
SAMPLE_RATE = 16000
N_MELS = 40

# Engineering targets
TARGET_LATENCY_MS_PER_FRAME = 10.0
TARGET_MODEL_SIZE_KB = 500.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""
    duration_sec: float
    audio_samples: int
    num_frames: int
    total_time_ms: float
    ms_per_frame: float
    ms_per_second_audio: float
    real_time_factor: float
    throughput_fps: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBenchmark:
    """Complete benchmark results for a model."""
    model_name: str
    model_size_kb: float
    num_parameters: int
    device: str
    
    # Benchmark results per duration
    results: List[BenchmarkResult] = field(default_factory=list)
    
    # Memory usage (if available)
    peak_memory_mb: Optional[float] = None
    avg_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_size_kb": self.model_size_kb,
            "num_parameters": self.num_parameters,
            "device": self.device,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
            "results": [r.to_dict() for r in self.results]
        }


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark TinyVAD student model latency on CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark with best checkpoint
  python scripts/analysis/benchmark_latency.py
  
  # Quick benchmark (fewer iterations)
  python scripts/analysis/benchmark_latency.py --quick
  
  # Custom checkpoint
  python scripts/analysis/benchmark_latency.py --checkpoint outputs/my_model/best.pt
  
  # Compare with energy baseline
  python scripts/analysis/benchmark_latency.py --compare-baselines
  
  # Custom output directory
  python scripts/analysis/benchmark_latency.py --output-dir results/
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-detect from outputs/production_cuda/checkpoints/)"
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="*_latest_best.pt",
        help="Pattern for auto-detecting checkpoint (default: *_latest_best.pt)"
    )
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=DEFAULT_DURATIONS,
        help=f"Audio durations to test in seconds (default: {DEFAULT_DURATIONS})"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of inference iterations per duration (default: {DEFAULT_ITERATIONS})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_ITERATIONS,
        help=f"Number of warmup iterations (default: {WARMUP_ITERATIONS})"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 iterations, 1 warmup"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu for edge deployment target)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=str(DEFAULT_EVALUATION_DIR),
        help=f"Directory to copy final evaluation artifacts (default: {DEFAULT_EVALUATION_DIR})"
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Also benchmark energy-based baseline for comparison"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6140,
        help="Random seed for reproducibility (default: 6140)"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only save JSON output, skip plots"
    )
    
    return parser.parse_args()


# =============================================================================
# Model Loading
# =============================================================================

def find_checkpoint(checkpoint_path: Optional[str], pattern: str = "*_latest_best.pt") -> Optional[Path]:
    """Find checkpoint file."""
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            return path
        print(f"Warning: Specified checkpoint not found: {checkpoint_path}")
        return None
    
    # Auto-detect from default directory
    if DEFAULT_CHECKPOINT_DIR.exists():
        checkpoints = list(DEFAULT_CHECKPOINT_DIR.glob(pattern))
        if checkpoints:
            # Sort by name and return first (consistent ordering)
            checkpoints.sort()
            return checkpoints[0]
    
    return None


def load_student_model(checkpoint_path: Path, device: torch.device) -> Tuple[TinyVAD, Dict]:
    """Load student model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config
    if "config" in checkpoint:
        raw_config = checkpoint["config"]
    elif "model_config" in checkpoint:
        raw_config = checkpoint["model_config"]
    else:
        raw_config = {}
    
    # Filter only valid TinyVAD parameters
    valid_keys = {"n_mels", "cnn_channels", "gru_hidden", "gru_layers", 
                  "dropout", "sample_rate", "hop_length"}
    config = {k: v for k, v in raw_config.items() if k in valid_keys}
    
    # Apply defaults for missing keys
    default_config = {
        "n_mels": N_MELS,
        "cnn_channels": [14, 28],
        "gru_hidden": 32,
        "gru_layers": 2,
        "dropout": 0.1,
    }
    default_config.update(config)
    config = default_config
    
    # Create model
    model = TinyVAD(**config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config


def get_model_size_from_checkpoint(checkpoint_path: Path) -> float:
    """Get model size in KB from checkpoint file."""
    return checkpoint_path.stat().st_size / 1024


# =============================================================================
# Synthetic Audio Generation
# =============================================================================

def generate_synthetic_audio(duration_sec: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate synthetic audio that mimics speech-like characteristics.
    
    Creates audio with alternating voiced and unvoiced segments to simulate
    realistic VAD input patterns.
    """
    np.random.seed(6140)  # Consistent seed for reproducibility
    
    samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, samples)
    
    # Create base signal with speech-like characteristics
    audio = np.zeros(samples)
    
    # Add some "speech-like" segments (modulated tones)
    segment_duration = 0.5  # 500ms segments
    num_segments = int(duration_sec / segment_duration)
    
    for i in range(num_segments):
        start_sec = i * segment_duration
        end_sec = min((i + 1) * segment_duration, duration_sec)
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        
        # Alternate between speech-like and silence
        if i % 2 == 0:
            # Speech-like: modulated sine wave
            freq = 150 + (i * 20) % 150  # 150-300 Hz fundamental
            modulation = 1 + 0.5 * np.sin(2 * np.pi * 4 * (t[start_idx:end_idx] - start_sec))
            audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * freq * t[start_idx:end_idx]) * modulation
        else:
            # Silence: low-level noise
            audio[start_idx:end_idx] = 0.01 * np.random.randn(end_idx - start_idx)
    
    # Add slight background noise
    audio += 0.005 * np.random.randn(samples)
    
    return audio.astype(np.float32)


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_model_inference(
    model: TinyVAD,
    audio: np.ndarray,
    device: torch.device,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS
) -> Tuple[float, int]:
    """
    Benchmark model inference.
    
    Returns:
        total_time_ms: Total inference time in milliseconds
        num_frames: Number of output frames
    """
    # Warmup iterations
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.predict(audio, device=device, return_numpy=True)
    
    # Synchronize if using CUDA (for accurate timing)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Actual benchmark
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        with torch.no_grad():
            probs = model.predict(audio, device=device, return_numpy=True)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000  # Convert to ms
    num_frames = len(probs) if probs.ndim == 1 else probs.shape[1]
    
    return total_time_ms, num_frames


def get_memory_usage() -> Optional[Tuple[float, float]]:
    """Get current and peak memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return None
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    current_mb = memory_info.rss / (1024 * 1024)
    
    # Peak memory (since process start)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if sys.platform == "darwin":  # macOS reports in bytes
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    
    return current_mb, peak_mb


def benchmark_duration(
    model: TinyVAD,
    duration_sec: float,
    device: torch.device,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS
) -> BenchmarkResult:
    """Benchmark model on audio of specific duration."""
    # Generate synthetic audio
    audio = generate_synthetic_audio(duration_sec)
    
    # Measure memory before (if available)
    mem_before = get_memory_usage()
    
    # Run benchmark
    total_time_ms, num_frames = benchmark_model_inference(
        model, audio, device, iterations, warmup
    )
    
    # Measure memory after
    mem_after = get_memory_usage()
    
    # Calculate metrics
    avg_time_ms = total_time_ms / iterations
    ms_per_frame = avg_time_ms / num_frames
    ms_per_second_audio = avg_time_ms / duration_sec
    real_time_factor = avg_time_ms / (duration_sec * 1000)  # < 1.0 means faster than real-time
    throughput_fps = num_frames / (avg_time_ms / 1000)
    
    return BenchmarkResult(
        duration_sec=duration_sec,
        audio_samples=len(audio),
        num_frames=num_frames,
        total_time_ms=avg_time_ms,
        ms_per_frame=ms_per_frame,
        ms_per_second_audio=ms_per_second_audio,
        real_time_factor=real_time_factor,
        throughput_fps=throughput_fps
    )


def run_student_benchmark(args: argparse.Namespace) -> Optional[ModelBenchmark]:
    """Run complete benchmark on student model."""
    print("=" * 70)
    print("TinyVAD Student Model - Latency Benchmark")
    print("=" * 70)
    
    # Set device
    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Find and load checkpoint
    checkpoint_path = find_checkpoint(args.checkpoint, args.checkpoint_pattern)
    if checkpoint_path is None:
        print("\n❌ Error: No checkpoint found!")
        print(f"   Searched in: {DEFAULT_CHECKPOINT_DIR}")
        print("   Run training first or specify --checkpoint")
        return None
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    try:
        model, config = load_student_model(checkpoint_path, device)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        traceback.print_exc()
        return None
    
    # Get model info
    model_size_kb = model.get_model_size_kb()
    num_params = model.count_parameters()
    
    print(f"  Model parameters: {num_params:,}")
    print(f"  Model size: {model_size_kb:.2f} KB ({model_size_kb/1024:.3f} MB)")
    print(f"  Checkpoint size: {get_model_size_from_checkpoint(checkpoint_path):.2f} KB")
    
    # Check against target
    if model_size_kb <= TARGET_MODEL_SIZE_KB:
        print(f"  ✅ Size target met (≤ {TARGET_MODEL_SIZE_KB} KB)")
    else:
        print(f"  ⚠️  Size exceeds target ({model_size_kb:.2f} > {TARGET_MODEL_SIZE_KB} KB)")
    
    # Create benchmark object
    benchmark = ModelBenchmark(
        model_name="TinyVAD Student",
        model_size_kb=model_size_kb,
        num_parameters=num_params,
        device=str(device)
    )
    
    # Run benchmarks for each duration
    iterations = 3 if args.quick else args.iterations
    warmup = 1 if args.quick else args.warmup
    
    print(f"\nRunning benchmarks ({iterations} iterations, {warmup} warmup)...")
    print("-" * 70)
    
    for duration in args.durations:
        print(f"  Duration: {duration}s...", end=" ", flush=True)
        
        try:
            result = benchmark_duration(model, duration, device, iterations, warmup)
            benchmark.results.append(result)
            
            # Print quick summary
            print(f"✅ {result.ms_per_frame:.3f} ms/frame, RTF: {result.real_time_factor:.4f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    return benchmark


def run_energy_baseline_benchmark(args: argparse.Namespace) -> Optional[ModelBenchmark]:
    """Run benchmark on energy-based baseline for comparison."""
    try:
        from baselines.energy_vad import EnergyVAD
    except ImportError:
        print("\n⚠️  Energy VAD baseline not available, skipping comparison")
        return None
    
    print("\n" + "=" * 70)
    print("Energy VAD Baseline - Latency Benchmark")
    print("=" * 70)
    
    vad = EnergyVAD()
    
    # Energy VAD has no learnable parameters
    benchmark = ModelBenchmark(
        model_name="Energy VAD (Baseline)",
        model_size_kb=0,  # No model file
        num_parameters=0,  # No parameters
        device="cpu"
    )
    
    iterations = 3 if args.quick else args.iterations
    
    print(f"\nRunning benchmarks ({iterations} iterations)...")
    print("-" * 70)
    
    for duration in args.durations:
        print(f"  Duration: {duration}s...", end=" ", flush=True)
        
        try:
            audio = generate_synthetic_audio(duration)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = vad.get_frame_probs(audio, sr=SAMPLE_RATE)
            end_time = time.perf_counter()
            
            total_time_ms = (end_time - start_time) * 1000
            avg_time_ms = total_time_ms / iterations
            
            probs, _ = vad.get_frame_probs(audio, sr=SAMPLE_RATE)
            num_frames = len(probs)
            
            ms_per_frame = avg_time_ms / num_frames
            ms_per_second_audio = avg_time_ms / duration
            real_time_factor = avg_time_ms / (duration * 1000)
            throughput_fps = num_frames / (avg_time_ms / 1000)
            
            result = BenchmarkResult(
                duration_sec=duration,
                audio_samples=len(audio),
                num_frames=num_frames,
                total_time_ms=avg_time_ms,
                ms_per_frame=ms_per_frame,
                ms_per_second_audio=ms_per_second_audio,
                real_time_factor=real_time_factor,
                throughput_fps=throughput_fps
            )
            
            benchmark.results.append(result)
            print(f"✅ {result.ms_per_frame:.3f} ms/frame, RTF: {result.real_time_factor:.4f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    return benchmark


# =============================================================================
# Output Functions
# =============================================================================

def print_results_table(benchmarks: List[ModelBenchmark]):
    """Print formatted results table to console."""
    print("\n" + "=" * 70)
    print("Benchmark Results Summary")
    print("=" * 70)
    
    for benchmark in benchmarks:
        if not benchmark.results:
            continue
        
        print(f"\n{benchmark.model_name}")
        print(f"  Model Size: {benchmark.model_size_kb:.2f} KB")
        print(f"  Parameters: {benchmark.num_parameters:,}")
        print(f"  Device: {benchmark.device}")
        
        # Table header
        print("\n  " + "-" * 100)
        print(f"  {'Duration':>10} | {'Frames':>8} | {'Total (ms)':>12} | "
              f"{'ms/frame':>10} | {'ms/sec audio':>12} | {'RTF':>8} | {'Status':>8}")
        print("  " + "-" * 100)
        
        # Table rows
        for result in benchmark.results:
            # Determine status
            if result.ms_per_frame <= TARGET_LATENCY_MS_PER_FRAME:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            print(f"  {result.duration_sec:>9.1f}s | {result.num_frames:>8} | "
                  f"{result.total_time_ms:>12.2f} | {result.ms_per_frame:>10.3f} | "
                  f"{result.ms_per_second_audio:>12.3f} | {result.real_time_factor:>8.4f} | "
                  f"{status:>8}")
        
        print("  " + "-" * 100)
        
        # Engineering target summary
        all_pass = all(r.ms_per_frame <= TARGET_LATENCY_MS_PER_FRAME for r in benchmark.results)
        avg_latency = np.mean([r.ms_per_frame for r in benchmark.results])
        
        print(f"\n  Engineering Targets:")
        print(f"    Target latency: ≤ {TARGET_LATENCY_MS_PER_FRAME} ms/frame")
        print(f"    Average latency: {avg_latency:.3f} ms/frame")
        
        if all_pass:
            print(f"    ✅ Latency target met for all durations")
        else:
            print(f"    ❌ Latency target NOT met for some durations")
        
        if benchmark.model_size_kb <= TARGET_MODEL_SIZE_KB:
            print(f"    ✅ Size target met (≤ {TARGET_MODEL_SIZE_KB} KB)")
        else:
            print(f"    ❌ Size target NOT met")


def save_json_report(benchmarks: List[ModelBenchmark], output_path: Path):
    """Save benchmark results to JSON."""
    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "targets": {
            "max_latency_ms_per_frame": TARGET_LATENCY_MS_PER_FRAME,
            "max_model_size_kb": TARGET_MODEL_SIZE_KB
        },
        "models": [b.to_dict() for b in benchmarks if b.results]
    }
    
    # Add summary statistics
    for model_data in report["models"]:
        latencies = [r["ms_per_frame"] for r in model_data["results"]]
        rtfs = [r["real_time_factor"] for r in model_data["results"]]
        
        model_data["summary"] = {
            "avg_latency_ms": float(np.mean(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "avg_rtf": float(np.mean(rtfs)),
            "latency_target_met": all(l <= TARGET_LATENCY_MS_PER_FRAME for l in latencies),
            "size_target_met": model_data["model_size_kb"] <= TARGET_MODEL_SIZE_KB
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 JSON report saved: {output_path}")


def plot_latency_comparison(benchmarks: List[ModelBenchmark], output_path: Path):
    """Create latency comparison plot."""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  matplotlib not available, skipping plot")
        return
    
    valid_benchmarks = [b for b in benchmarks if b.results]
    if not valid_benchmarks:
        print("\n⚠️  No valid benchmark results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TinyVAD Latency Benchmark Results", fontsize=14, fontweight='bold')
    
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
    
    # Plot 1: ms/frame vs Duration
    ax = axes[0, 0]
    for i, benchmark in enumerate(valid_benchmarks):
        durations = [r.duration_sec for r in benchmark.results]
        latencies = [r.ms_per_frame for r in benchmark.results]
        ax.plot(durations, latencies, marker='o', linewidth=2, 
                label=benchmark.model_name, color=colors[i % len(colors)])
    
    ax.axhline(y=TARGET_LATENCY_MS_PER_FRAME, color='r', linestyle='--', 
               linewidth=2, label=f'Target ({TARGET_LATENCY_MS_PER_FRAME} ms)')
    ax.set_xlabel('Audio Duration (s)')
    ax.set_ylabel('Latency (ms/frame)')
    ax.set_title('Latency vs Audio Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Real-time Factor vs Duration
    ax = axes[0, 1]
    for i, benchmark in enumerate(valid_benchmarks):
        durations = [r.duration_sec for r in benchmark.results]
        rtfs = [r.real_time_factor for r in benchmark.results]
        ax.plot(durations, rtfs, marker='s', linewidth=2,
                label=benchmark.model_name, color=colors[i % len(colors)])
    
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2,
               label='Real-time threshold (1.0)')
    ax.set_xlabel('Audio Duration (s)')
    ax.set_ylabel('Real-time Factor')
    ax.set_title('Real-time Factor vs Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Throughput vs Duration
    ax = axes[1, 0]
    for i, benchmark in enumerate(valid_benchmarks):
        durations = [r.duration_sec for r in benchmark.results]
        throughputs = [r.throughput_fps for r in benchmark.results]
        ax.plot(durations, throughputs, marker='^', linewidth=2,
                label=benchmark.model_name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Audio Duration (s)')
    ax.set_ylabel('Throughput (frames/sec)')
    ax.set_title('Throughput vs Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Model Size Comparison (bar chart)
    ax = axes[1, 1]
    names = [b.model_name for b in valid_benchmarks]
    sizes = [b.model_size_kb for b in valid_benchmarks]
    
    bars = ax.barh(names, sizes, color=colors[:len(names)])
    ax.axvline(x=TARGET_MODEL_SIZE_KB, color='r', linestyle='--', linewidth=2,
               label=f'Target ({TARGET_MODEL_SIZE_KB} KB)')
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {size:.1f} KB', va='center', fontsize=10)
    
    ax.set_xlabel('Model Size (KB)')
    ax.set_title('Model Size Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {output_path}")


def generate_markdown_report(benchmarks: List[ModelBenchmark], output_path: Path):
    """Generate markdown report."""
    lines = [
        "# VAD Latency Benchmark Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Engineering Targets",
        "",
        f"| Target | Value |",
        f"|--------|-------|",
        f"| Max Latency | {TARGET_LATENCY_MS_PER_FRAME} ms/frame |",
        f"| Max Model Size | {TARGET_MODEL_SIZE_KB} KB |",
        f"| Real-time Factor | < 1.0 |",
        "",
        "## Results",
        ""
    ]
    
    for benchmark in benchmarks:
        if not benchmark.results:
            continue
        
        lines.extend([
            f"### {benchmark.model_name}",
            "",
            f"- **Model Size:** {benchmark.model_size_kb:.2f} KB",
            f"- **Parameters:** {benchmark.num_parameters:,}",
            f"- **Device:** {benchmark.device}",
            "",
            "| Duration | Frames | Total (ms) | ms/frame | ms/sec | RTF | Status |",
            "|----------|--------|------------|----------|--------|-----|--------|"
        ])
        
        for result in benchmark.results:
            status = "✅" if result.ms_per_frame <= TARGET_LATENCY_MS_PER_FRAME else "❌"
            lines.append(
                f"| {result.duration_sec:.1f}s | {result.num_frames} | "
                f"{result.total_time_ms:.2f} | {result.ms_per_frame:.3f} | "
                f"{result.ms_per_second_audio:.3f} | {result.real_time_factor:.4f} | {status} |"
            )
        
        # Summary
        latencies = [r.ms_per_frame for r in benchmark.results]
        avg_latency = np.mean(latencies)
        all_pass = all(l <= TARGET_LATENCY_MS_PER_FRAME for l in latencies)
        
        lines.extend([
            "",
            "**Summary:**",
            "",
            f"- Average latency: {avg_latency:.3f} ms/frame",
            f"- Latency target: {'✅ Met' if all_pass else '❌ Not met'}",
            f"- Size target: {'✅ Met' if benchmark.model_size_kb <= TARGET_MODEL_SIZE_KB else '❌ Not met'}",
            ""
        ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"📝 Markdown report saved: {output_path}")


def save_summary_metadata(benchmarks: List[ModelBenchmark], output_dir: Path):
    """
    Save lightweight timing/model-size metadata used by comparison scripts.
    """
    name_map = {
        "TinyVAD Student": "Our Model",
        "Energy VAD (Baseline)": "Energy",
    }

    timing = {}
    model_sizes = {}

    for bench in benchmarks:
        if not bench.results:
            continue
        method_name = name_map.get(bench.model_name, bench.model_name)
        avg_latency = float(np.mean([r.ms_per_frame for r in bench.results]))
        timing[method_name] = {
            "latency_ms": avg_latency,
            "source": "benchmark_latency.py",
        }
        model_sizes[method_name] = {
            "size_kb": float(bench.model_size_kb),
            "source": "benchmark_latency.py",
        }

    timing_path = output_dir / "timing.json"
    size_path = output_dir / "model_sizes.json"
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)
    with open(size_path, 'w') as f:
        json.dump(model_sizes, f, indent=2)

    print(f"📄 Timing metadata saved: {timing_path}")
    print(f"📄 Model-size metadata saved: {size_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    # Check for resource module (Unix only)
    global resource
    try:
        import resource
    except ImportError:
        resource = None
    
    benchmarks = []
    
    # Run student model benchmark
    student_benchmark = run_student_benchmark(args)
    if student_benchmark:
        benchmarks.append(student_benchmark)
    
    # Run baseline comparison if requested
    if args.compare_baselines:
        baseline_benchmark = run_energy_baseline_benchmark(args)
        if baseline_benchmark:
            benchmarks.append(baseline_benchmark)
    
    if not benchmarks:
        print("\n❌ No benchmarks completed successfully")
        sys.exit(1)
    
    # Print results
    print_results_table(benchmarks)
    
    # Save outputs
    output_dir = Path(args.output_dir)
    evaluation_dir = Path(args.evaluation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    json_path = output_dir / "benchmark_results.json"
    save_json_report(benchmarks, json_path)
    
    # Markdown report
    md_path = output_dir / "benchmark_report.md"
    generate_markdown_report(benchmarks, md_path)
    save_summary_metadata(benchmarks, output_dir)
    
    # Plot (if not disabled)
    if not args.json_only and MATPLOTLIB_AVAILABLE:
        plot_path = output_dir / "latency_comparison.png"
        plot_latency_comparison(benchmarks, plot_path)

    # Copy final benchmark artifacts for report packaging.
    for src in [json_path, md_path, output_dir / "latency_comparison.png"]:
        if src.exists():
            dst = evaluation_dir / src.name
            dst.write_bytes(src.read_bytes())
            print(f"📦 Copied to evaluation artifacts: {dst}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {json_path.name}")
    print(f"  - {md_path.name}")
    if not args.json_only and MATPLOTLIB_AVAILABLE:
        print(f"  - {plot_path.name}")
    
    # Engineering target status
    student_bench = next((b for b in benchmarks if "Student" in b.model_name), None)
    if student_bench:
        latencies = [r.ms_per_frame for r in student_bench.results]
        avg_latency = np.mean(latencies)
        latency_pass = all(l <= TARGET_LATENCY_MS_PER_FRAME for l in latencies)
        size_pass = student_bench.model_size_kb <= TARGET_MODEL_SIZE_KB
        
        print("\n" + "=" * 70)
        print("Engineering Target Status")
        print("=" * 70)
        print(f"\n  Model Size: {student_bench.model_size_kb:.2f} KB "
              f"({'✅ PASS' if size_pass else '❌ FAIL'})")
        print(f"  Latency: {avg_latency:.3f} ms/frame "
              f"({'✅ PASS' if latency_pass else '❌ FAIL'})")
        
        if size_pass and latency_pass:
            print("\n  ✅ ALL ENGINEERING TARGETS MET")
        else:
            print("\n  ❌ SOME ENGINEERING TARGETS NOT MET")
        
        print("=" * 70)
    
    print()


if __name__ == "__main__":
    main()
