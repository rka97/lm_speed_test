"""
Compare JAX and PyTorch DDP benchmark results.
"""

import csv
import argparse
import sys


def load_results(filepath):
    """Load benchmark results from CSV file."""
    if not filepath:
        return None
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            results = next(reader)
            # Convert numeric fields
            for key in results:
                try:
                    if '.' in results[key]:
                        results[key] = float(results[key])
                    else:
                        results[key] = int(results[key])
                except (ValueError, TypeError):
                    pass
            return results
    except FileNotFoundError:
        print(f"Warning: Could not find file {filepath}")
        return None
    except StopIteration:
        print(f"Warning: File {filepath} is empty")
        return None


def format_speedup(jax_val, pytorch_val):
    """Format speedup ratio with direction indicator."""
    if jax_val is None or pytorch_val is None:
        return "N/A"
    if pytorch_val == 0:
        return "N/A"
    ratio = jax_val / pytorch_val
    if ratio > 1:
        return f"{ratio:.2f}x (PyTorch faster)"
    else:
        return f"{1/ratio:.2f}x (JAX faster)"


def format_throughput_comparison(jax_val, pytorch_val):
    """Format throughput comparison with direction indicator."""
    if jax_val is None or pytorch_val is None:
        return "N/A"
    if jax_val == 0:
        return "N/A"
    ratio = pytorch_val / jax_val
    if ratio > 1:
        return f"{ratio:.2f}x (PyTorch higher)"
    else:
        return f"{1/ratio:.2f}x (JAX higher)"


def print_section(title, width=70):
    """Print a section header."""
    print(f"\n{'='*width}")
    print(f" {title}")
    print(f"{'='*width}")


def print_comparison_row(label, jax_val, pytorch_val, unit="", comparison_fn=None):
    """Print a comparison row."""
    jax_str = f"{jax_val:,.2f}" if isinstance(jax_val, float) else (f"{jax_val:,}" if jax_val is not None else "N/A")
    pytorch_str = f"{pytorch_val:,.2f}" if isinstance(pytorch_val, float) else (f"{pytorch_val:,}" if pytorch_val is not None else "N/A")
    
    if unit:
        jax_str += f" {unit}"
        pytorch_str += f" {unit}"
    
    comparison = ""
    if comparison_fn and jax_val is not None and pytorch_val is not None:
        comparison = comparison_fn(jax_val, pytorch_val)
    
    print(f"  {label:40s} | {jax_str:>18s} | {pytorch_str:>18s} | {comparison}")


def compare_results(jax_file, pytorch_file):
    """Compare JAX and PyTorch benchmark results."""
    jax_results = load_results(jax_file)
    pytorch_results = load_results(pytorch_file)
    
    if jax_results is None and pytorch_results is None:
        print("Error: No results to compare!")
        sys.exit(1)
    
    print("\n" + "="*90)
    print(" "*30 + "BENCHMARK COMPARISON")
    print("="*90)
    
    # Configuration comparison
    print_section("Configuration")
    print(f"  {'Parameter':<40s} | {'JAX':>18s} | {'PyTorch':>18s}")
    print(f"  {'-'*40} | {'-'*18} | {'-'*18}")
    
    config_keys = ['model_dim', 'num_heads', 'seq_len', 'num_layers', 'vocab_size', 
                   'batch_size', 'num_params', 'num_devices']
    
    for key in config_keys:
        jax_val = jax_results.get(key) if jax_results else None
        pytorch_val = pytorch_results.get(key) if pytorch_results else None
        print_comparison_row(key, jax_val, pytorch_val)
    
    # Device info
    jax_device = jax_results.get('device_type', 'N/A') if jax_results else 'N/A'
    pytorch_device = pytorch_results.get('device_type', 'N/A') if pytorch_results else 'N/A'
    print(f"  {'device_type':<40s} | {str(jax_device):>18s} | {str(pytorch_device):>18s}")
    
    # Full Model Performance comparison
    print_section("Full Model Performance")
    print(f"  {'Metric':<40s} | {'JAX':>18s} | {'PyTorch':>18s} | Comparison")
    print(f"  {'-'*40} | {'-'*18} | {'-'*18} | {'-'*25}")
    
    # Forward pass
    jax_fwd = jax_results.get('avg_forward_time_ms') if jax_results else None
    pytorch_fwd = pytorch_results.get('avg_forward_time_ms') if pytorch_results else None
    print_comparison_row("Avg Forward Time", jax_fwd, pytorch_fwd, "ms", format_speedup)
    
    # Forward+Backward pass
    jax_fwd_bwd = jax_results.get('avg_fwd_bwd_time_ms') if jax_results else None
    pytorch_fwd_bwd = pytorch_results.get('avg_fwd_bwd_time_ms') if pytorch_results else None
    print_comparison_row("Avg Forward+Backward Time", jax_fwd_bwd, pytorch_fwd_bwd, "ms", format_speedup)
    
    # Throughput
    jax_fwd_tput = jax_results.get('forward_throughput_tokens_sec') if jax_results else None
    pytorch_fwd_tput = pytorch_results.get('forward_throughput_tokens_sec') if pytorch_results else None
    print_comparison_row("Forward Throughput", jax_fwd_tput, pytorch_fwd_tput, "tok/s", format_throughput_comparison)
    
    jax_fwd_bwd_tput = jax_results.get('fwd_bwd_throughput_tokens_sec') if jax_results else None
    pytorch_fwd_bwd_tput = pytorch_results.get('fwd_bwd_throughput_tokens_sec') if pytorch_results else None
    print_comparison_row("Forward+Backward Throughput", jax_fwd_bwd_tput, pytorch_fwd_bwd_tput, "tok/s", format_throughput_comparison)
    
    # Component-level benchmarks
    # Map JAX component names to PyTorch component names
    component_mapping = [
        ('Mlp', 'MLP', 'MLP / Mlp'),
        ('CausalAttn', 'Attention', 'Attention / CausalAttn'),
        ('TBlock', 'Block', 'Block / TBlock'),
    ]
    
    for jax_name, pytorch_name, display_name in component_mapping:
        # Check if component results exist
        jax_key = f'{jax_name}_avg_forward_time_ms'
        pytorch_key = f'{pytorch_name}_avg_forward_time_ms'
        
        jax_has_component = jax_results and jax_key in jax_results
        pytorch_has_component = pytorch_results and pytorch_key in pytorch_results
        
        if jax_has_component or pytorch_has_component:
            print_section(f"Component: {display_name}")
            print(f"  {'Metric':<40s} | {'JAX':>18s} | {'PyTorch':>18s} | Comparison")
            print(f"  {'-'*40} | {'-'*18} | {'-'*18} | {'-'*25}")
            
            # Parameters
            jax_params = jax_results.get(f'{jax_name}_num_params') if jax_results else None
            pytorch_params = pytorch_results.get(f'{pytorch_name}_num_params') if pytorch_results else None
            print_comparison_row("Parameters", jax_params, pytorch_params)
            
            # Forward time
            jax_fwd = jax_results.get(f'{jax_name}_avg_forward_time_ms') if jax_results else None
            pytorch_fwd = pytorch_results.get(f'{pytorch_name}_avg_forward_time_ms') if pytorch_results else None
            print_comparison_row("Avg Forward Time", jax_fwd, pytorch_fwd, "ms", format_speedup)
            
            # Forward+Backward time
            jax_fwd_bwd = jax_results.get(f'{jax_name}_avg_fwd_bwd_time_ms') if jax_results else None
            pytorch_fwd_bwd = pytorch_results.get(f'{pytorch_name}_avg_fwd_bwd_time_ms') if pytorch_results else None
            print_comparison_row("Avg Forward+Backward Time", jax_fwd_bwd, pytorch_fwd_bwd, "ms", format_speedup)
            
            # Forward throughput
            jax_fwd_tput = jax_results.get(f'{jax_name}_forward_throughput_tokens_sec') if jax_results else None
            pytorch_fwd_tput = pytorch_results.get(f'{pytorch_name}_forward_throughput_tokens_sec') if pytorch_results else None
            print_comparison_row("Forward Throughput", jax_fwd_tput, pytorch_fwd_tput, "tok/s", format_throughput_comparison)
            
            # Forward+Backward throughput
            jax_fwd_bwd_tput = jax_results.get(f'{jax_name}_fwd_bwd_throughput_tokens_sec') if jax_results else None
            pytorch_fwd_bwd_tput = pytorch_results.get(f'{pytorch_name}_fwd_bwd_throughput_tokens_sec') if pytorch_results else None
            print_comparison_row("Forward+Backward Throughput", jax_fwd_bwd_tput, pytorch_fwd_bwd_tput, "tok/s", format_throughput_comparison)
    
    # Summary
    print_section("Summary")
    if jax_results and pytorch_results:
        full_fwd_jax = jax_results.get('avg_forward_time_ms', 0)
        full_fwd_pytorch = pytorch_results.get('avg_forward_time_ms', 0)
        full_fwd_bwd_jax = jax_results.get('avg_fwd_bwd_time_ms', 0)
        full_fwd_bwd_pytorch = pytorch_results.get('avg_fwd_bwd_time_ms', 0)
        
        if full_fwd_jax and full_fwd_pytorch:
            if full_fwd_jax < full_fwd_pytorch:
                print(f"  Full Model Forward: JAX is {full_fwd_pytorch/full_fwd_jax:.2f}x faster")
            else:
                print(f"  Full Model Forward: PyTorch is {full_fwd_jax/full_fwd_pytorch:.2f}x faster")
        
        if full_fwd_bwd_jax and full_fwd_bwd_pytorch:
            if full_fwd_bwd_jax < full_fwd_bwd_pytorch:
                print(f"  Full Model Forward+Backward: JAX is {full_fwd_bwd_pytorch/full_fwd_bwd_jax:.2f}x faster")
            else:
                print(f"  Full Model Forward+Backward: PyTorch is {full_fwd_bwd_jax/full_fwd_bwd_pytorch:.2f}x faster")
        
        # Component summaries
        print("\n  Component-level summary:")
        for jax_name, pytorch_name, display_name in component_mapping:
            jax_fwd = jax_results.get(f'{jax_name}_avg_forward_time_ms')
            pytorch_fwd = pytorch_results.get(f'{pytorch_name}_avg_forward_time_ms')
            
            if jax_fwd and pytorch_fwd:
                if jax_fwd < pytorch_fwd:
                    print(f"    {display_name:30s} Forward: JAX is {pytorch_fwd/jax_fwd:.2f}x faster")
                else:
                    print(f"    {display_name:30s} Forward: PyTorch is {jax_fwd/pytorch_fwd:.2f}x faster")
    
    print("\n" + "="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare JAX and PyTorch benchmark results')
    parser.add_argument('--jax', type=str, default='jax_benchmark_results.csv',
                       help='JAX benchmark results CSV file')
    parser.add_argument('--pytorch', type=str, default='pytorch_benchmark_results.csv',
                       help='PyTorch benchmark results CSV file')
    
    args = parser.parse_args()
    compare_results(args.jax, args.pytorch)


if __name__ == '__main__':
    main()