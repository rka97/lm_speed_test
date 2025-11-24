"""
Benchmark Results Comparison Script

Reads JAX and PyTorch benchmark CSV files and prints a comparison table.
"""

import csv
import argparse
import os
from typing import Dict, Any, Optional


def read_csv_results(filepath: str) -> Optional[Dict[str, Any]]:
    """Read benchmark results from CSV file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        results = next(reader, None)
    
    if results is None:
        print(f"Warning: No data in file: {filepath}")
        return None
    
    # Convert numeric values
    numeric_keys = [
        'model_dim', 'num_heads', 'seq_len', 'num_layers', 'vocab_size',
        'batch_size', 'num_params', 'num_devices', 'num_iterations',
        'avg_forward_time_ms', 'min_forward_time_ms', 'max_forward_time_ms',
        'avg_fwd_bwd_time_ms', 'min_fwd_bwd_time_ms', 'max_fwd_bwd_time_ms',
        'forward_throughput_tokens_sec', 'fwd_bwd_throughput_tokens_sec'
    ]
    
    for key in numeric_keys:
        if key in results:
            try:
                results[key] = float(results[key])
            except (ValueError, TypeError):
                pass
    
    return results


def format_number(n: float, precision: int = 2) -> str:
    """Format number with commas and precision."""
    if abs(n) >= 1000000:
        return f"{n/1000000:,.{precision}f}M"
    elif abs(n) >= 1000:
        return f"{n/1000:,.{precision}f}K"
    else:
        return f"{n:,.{precision}f}"


def print_comparison_table(jax_results: Optional[Dict], pytorch_results: Optional[Dict]):
    """Print a nicely formatted comparison table."""
    
    print("\n" + "=" * 80)
    print("  TRANSFORMER BENCHMARK COMPARISON: JAX vs PyTorch DDP")
    print("=" * 80)
    
    # Check if we have results to compare
    if jax_results is None and pytorch_results is None:
        print("\nNo benchmark results available!")
        return
    
    # Use whichever results are available for config info
    config_source = jax_results or pytorch_results
    
    # Print configuration
    print("\n┌" + "─" * 78 + "┐")
    print("│" + " MODEL CONFIGURATION".center(78) + "│")
    print("├" + "─" * 78 + "┤")
    
    config_items = [
        ("Model Dimension", config_source.get('model_dim', 'N/A')),
        ("Number of Heads", config_source.get('num_heads', 'N/A')),
        ("Sequence Length", config_source.get('seq_len', 'N/A')),
        ("Number of Layers", config_source.get('num_layers', 'N/A')),
        ("Vocabulary Size", format_number(config_source.get('vocab_size', 0), 0)),
        ("Parameters", format_number(config_source.get('num_params', 0), 2)),
    ]
    
    for name, value in config_items:
        print(f"│  {name:<24} {str(value):>50} │")
    
    print("└" + "─" * 78 + "┘")
    
    # Print benchmark settings
    print("\n┌" + "─" * 78 + "┐")
    print("│" + " BENCHMARK SETTINGS".center(78) + "│")
    print("├" + "─" * 39 + "┬" + "─" * 38 + "┤")
    print("│" + " JAX".center(39) + "│" + " PyTorch DDP".center(38) + "│")
    print("├" + "─" * 39 + "┼" + "─" * 38 + "┤")
    
    settings = [
        ("Batch Size", 'batch_size'),
        ("Num Devices", 'num_devices'),
        ("Device Type", 'device_type'),
        ("Iterations", 'num_iterations'),
    ]
    
    for name, key in settings:
        jax_val = str(jax_results.get(key, 'N/A')) if jax_results else 'N/A'
        pytorch_val = str(pytorch_results.get(key, 'N/A')) if pytorch_results else 'N/A'
        # Truncate device type if too long
        if key == 'device_type':
            jax_val = jax_val[:35] if len(jax_val) > 35 else jax_val
            pytorch_val = pytorch_val[:32] if len(pytorch_val) > 32 else pytorch_val
        print(f"│  {name:<15} {jax_val:>20} │ {pytorch_val:>35} │")
    
    print("└" + "─" * 39 + "┴" + "─" * 38 + "┘")
    
    # Print timing results
    print("\n┌" + "─" * 78 + "┐")
    print("│" + " TIMING RESULTS (milliseconds)".center(78) + "│")
    print("├" + "─" * 26 + "┬" + "─" * 16 + "┬" + "─" * 16 + "┬" + "─" * 17 + "┤")
    print("│" + " Metric".center(26) + "│" + " JAX".center(16) + "│" + " PyTorch".center(16) + "│" + " Speedup".center(17) + "│")
    print("├" + "─" * 26 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┼" + "─" * 17 + "┤")
    
    timing_metrics = [
        ("Forward (avg)", 'avg_forward_time_ms'),
        ("Forward (min)", 'min_forward_time_ms'),
        ("Forward (max)", 'max_forward_time_ms'),
        ("Fwd+Bwd (avg)", 'avg_fwd_bwd_time_ms'),
        ("Fwd+Bwd (min)", 'min_fwd_bwd_time_ms'),
        ("Fwd+Bwd (max)", 'max_fwd_bwd_time_ms'),
    ]
    
    for name, key in timing_metrics:
        jax_val = jax_results.get(key, None) if jax_results else None
        pytorch_val = pytorch_results.get(key, None) if pytorch_results else None
        
        jax_str = f"{jax_val:.2f}" if jax_val is not None else "N/A"
        pytorch_str = f"{pytorch_val:.2f}" if pytorch_val is not None else "N/A"
        
        # Calculate speedup (JAX relative to PyTorch)
        if jax_val is not None and pytorch_val is not None and jax_val > 0:
            speedup = pytorch_val / jax_val
            if speedup >= 1:
                speedup_str = f"JAX {speedup:.2f}x"
            else:
                speedup_str = f"PT {1/speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"│  {name:<24}│{jax_str:>15} │{pytorch_str:>15} │{speedup_str:>16} │")
    
    print("└" + "─" * 26 + "┴" + "─" * 16 + "┴" + "─" * 16 + "┴" + "─" * 17 + "┘")
    
    # Print throughput results
    print("\n┌" + "─" * 78 + "┐")
    print("│" + " THROUGHPUT (tokens/second)".center(78) + "│")
    print("├" + "─" * 26 + "┬" + "─" * 16 + "┬" + "─" * 16 + "┬" + "─" * 17 + "┤")
    print("│" + " Metric".center(26) + "│" + " JAX".center(16) + "│" + " PyTorch".center(16) + "│" + " Speedup".center(17) + "│")
    print("├" + "─" * 26 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┼" + "─" * 17 + "┤")
    
    throughput_metrics = [
        ("Forward", 'forward_throughput_tokens_sec'),
        ("Forward+Backward", 'fwd_bwd_throughput_tokens_sec'),
    ]
    
    for name, key in throughput_metrics:
        jax_val = jax_results.get(key, None) if jax_results else None
        pytorch_val = pytorch_results.get(key, None) if pytorch_results else None
        
        jax_str = format_number(jax_val, 0) if jax_val is not None else "N/A"
        pytorch_str = format_number(pytorch_val, 0) if pytorch_val is not None else "N/A"
        
        # Calculate speedup (higher throughput is better)
        if jax_val is not None and pytorch_val is not None and pytorch_val > 0:
            speedup = jax_val / pytorch_val
            if speedup >= 1:
                speedup_str = f"JAX {speedup:.2f}x"
            else:
                speedup_str = f"PT {1/speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"│  {name:<24}│{jax_str:>15} │{pytorch_str:>15} │{speedup_str:>16} │")
    
    print("└" + "─" * 26 + "┴" + "─" * 16 + "┴" + "─" * 16 + "┴" + "─" * 17 + "┘")
    
    # Print summary
    print("\n┌" + "─" * 78 + "┐")
    print("│" + " SUMMARY".center(78) + "│")
    print("├" + "─" * 78 + "┤")
    
    if jax_results and pytorch_results:
        jax_fwd = jax_results.get('avg_forward_time_ms', float('inf'))
        pytorch_fwd = pytorch_results.get('avg_forward_time_ms', float('inf'))
        jax_fwdbwd = jax_results.get('avg_fwd_bwd_time_ms', float('inf'))
        pytorch_fwdbwd = pytorch_results.get('avg_fwd_bwd_time_ms', float('inf'))
        
        if jax_fwd < pytorch_fwd:
            fwd_winner = f"JAX is {pytorch_fwd/jax_fwd:.2f}x faster for forward pass"
        else:
            fwd_winner = f"PyTorch is {jax_fwd/pytorch_fwd:.2f}x faster for forward pass"
        
        if jax_fwdbwd < pytorch_fwdbwd:
            fwdbwd_winner = f"JAX is {pytorch_fwdbwd/jax_fwdbwd:.2f}x faster for forward+backward"
        else:
            fwdbwd_winner = f"PyTorch is {jax_fwdbwd/pytorch_fwdbwd:.2f}x faster for forward+backward"
        
        print(f"│  • {fwd_winner:<73} │")
        print(f"│  • {fwdbwd_winner:<73} │")
    else:
        print("│  • Cannot compare - missing benchmark results.".ljust(77) + " │")
    
    print("└" + "─" * 78 + "┘")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare JAX and PyTorch benchmark results')
    parser.add_argument('--jax', type=str, default='jax_benchmark_results.csv',
                        help='Path to JAX benchmark results CSV')
    parser.add_argument('--pytorch', type=str, default='pytorch_benchmark_results.csv',
                        help='Path to PyTorch benchmark results CSV')
    
    args = parser.parse_args()
    
    # Read results
    jax_results = read_csv_results(args.jax)
    pytorch_results = read_csv_results(args.pytorch)
    
    # Print comparison
    print_comparison_table(jax_results, pytorch_results)


if __name__ == '__main__':
    main()