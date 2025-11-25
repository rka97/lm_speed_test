"""
PyTorch DDP Transformer Benchmark Script
Times forward and backward passes and writes results to CSV.

Run with:
    torchrun --nproc_per_node=NUM_GPUS benchmark_torch_ddp.py [args]
    
Or for single GPU/CPU:
    python benchmark_torch_ddp.py [args]
"""

import math
import time
import csv
import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class ModelConfig:
    model_dim: int
    num_heads: int
    seq_len: int
    num_layers: int
    vocab_size: int
    expanded_model_dim: int
    multiple_of: int = 256
    rmsnorm_epsilon: float = 1e-6
    qknorm_epsilon: float = 1e-6
    use_residual_scaling: bool = True
    tie_embeddings: bool = True
    dtype: torch.dtype = torch.bfloat16


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, dtype: torch.dtype = torch.float32):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False, dtype=dtype)
        self.glu = nn.GLU(dim=-1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x):
        return self.fc2(self.glu(self.fc1(x)))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
    inv_freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.device('cpu')) / dim)
    )
    t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device) / condense_ratio
    freqs = torch.outer(t, inv_freqs).float()
    return torch.stack(
        [torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]],
        dim=4,
    )


def apply_rotary_emb_complex_like(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()
    rotated_qk_r2 = torch.stack(
        [
            qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
            qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    ).flatten(3)
    rotated_qk = rotated_qk_r2
    return torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.model_dim % cfg.num_heads == 0
        self.dim = cfg.model_dim
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.model_dim // cfg.num_heads
        self.seq_len = cfg.seq_len

        self.w_qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim, bias=False, dtype=cfg.dtype)
        self.w_out = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False, dtype=cfg.dtype)
        wq, wk, wv = torch.chunk(self.w_qkv.weight, 3, dim=0)
        for w in [wq, wk, wv]:
            nn.init.normal_(w, std=0.02)
        nn.init.normal_(self.w_out.weight, std=0.02)

        self.eps = cfg.qknorm_epsilon
        seq_len = cfg.seq_len
        attn_scale0 = math.log2(seq_len**2 - seq_len)
        self.attn_scale = nn.Parameter(torch.tensor(attn_scale0, dtype=cfg.dtype))
        
        # Pre-compute freqs_cis for standalone usage
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(self.head_dim, cfg.seq_len, 500000)[0:cfg.seq_len],
            persistent=False,
        )

    def forward(self, x, freqs_cis=None):
        bsz, seqlen, d = x.shape
        
        # Use provided freqs_cis or fall back to self.freqs_cis
        if freqs_cis is None:
            freqs_cis = self.freqs_cis[:, :seqlen, :].to(x.device)

        q, k, v = self.w_qkv(x).split(d, dim=2)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q / (torch.norm(q, dim=-1, keepdim=True) + self.eps)
        k = k / (torch.norm(k, dim=-1, keepdim=True) + self.eps)
        q = q * self.attn_scale

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)

        return self.w_out(out)


class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.attn_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rmsnorm_epsilon, dtype=cfg.dtype)
        self.mlp = MLP(
            dim=cfg.model_dim,
            hidden_dim=cfg.expanded_model_dim,
            multiple_of=cfg.multiple_of,
            dtype=cfg.dtype,
        )
        self.mlp_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rmsnorm_epsilon, dtype=cfg.dtype)
        self.layer_id = layer_id

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_layers = cfg.num_layers
        self.cfg = cfg
        head_dim = cfg.model_dim // cfg.num_heads
        assert cfg.model_dim % cfg.num_heads == 0

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList([Block(idx, cfg) for idx in range(cfg.num_layers)])
        self.out_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rmsnorm_epsilon, dtype=cfg.dtype)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0:cfg.seq_len],
            persistent=False,
        )

        self.apply(self._init_weights)
        self._scale_residual_branches()

        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x, targets=None):
        x = self.embed_tokens(x)
        L = x.shape[1]

        if L > self.freqs_cis.shape[1]:
            head_dim = self.cfg.model_dim // self.cfg.num_heads
            new_freqs = precompute_freqs_cis(head_dim, max(L, self.cfg.seq_len), 500000)
            self.register_buffer(
                'freqs_cis', new_freqs[0:max(L, self.cfg.seq_len)], persistent=False
            )

        freqs_cis = self.freqs_cis[:, :L, :].to(x.device)

        for layer in self.layers:
            x = layer(x, freqs_cis)
        out = self.lm_head(self.out_norm(x))
        
        if targets is not None:
            loss = F.cross_entropy(out.view(-1, out.size(-1)), targets.view(-1), ignore_index=-100)
            return out, loss
        return out

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def _scale_residual_branches(self):
        for n, p in self.named_parameters():
            if n.endswith('fc2.weight') or n.endswith('w_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if self.lm_head.weight is not self.embed_tokens.weight:
                n_params -= self.lm_head.weight.numel()
        return n_params


# ============== Benchmark Functions ==============

def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def benchmark_component(component_name: str, model: nn.Module, cfg: ModelConfig,
                       batch_size: int, num_warmup: int, num_iterations: int,
                       device: torch.device, world_size: int, local_rank: int,
                       is_main: bool) -> Dict[str, Any]:
    """Benchmark a single component (MLP, Attention, or Block)."""
    if is_main:
        print(f"\n--- {component_name} Benchmark ---")
    
    model = model.to(device)
    if cfg.dtype == torch.bfloat16:
        model = model.bfloat16()
    #model = torch.compile(model, fullgraph=True, mode='max-autotune', dynamic=False)
    model = torch.compile(model, fullgraph=True)
    
    # Count parameters before wrapping with DDP
    param_count = sum(p.numel() for p in model.parameters())
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    if is_main:
        print(f"  Parameters: {param_count:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Generate synthetic data (float input for components)
    torch.manual_seed(456 + local_rank)
    batch = torch.randn(batch_size, cfg.seq_len, cfg.model_dim, device=device, dtype=cfg.dtype)
    
    # ============== Forward-only benchmark ==============
    model.eval()
    
    if is_main:
        print(f"  Running {num_warmup} warmup iterations (forward)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            output = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    if is_main:
        print(f"  Running {num_iterations} benchmark iterations (forward)...")
    forward_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            forward_times.append(end - start)
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    min_forward_time = min(forward_times)
    max_forward_time = max(forward_times)
    
    if is_main:
        print(f"  Avg forward time: {avg_forward_time*1000:.2f} ms")
    
    # ============== Forward + Backward benchmark ==============
    model.train()
    
    @torch.compile
    def _component_step(batch):
        optimizer.zero_grad()
        output = model(batch)
        # Simple MSE loss to enable gradient computation
        loss = (output ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss
    
    if is_main:
        print(f"  Running {num_warmup} warmup iterations (forward+backward)...")
    for _ in range(num_warmup):
        _component_step(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    if is_main:
        print(f"  Running {num_iterations} benchmark iterations (forward+backward)...")
    fwd_bwd_times = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _component_step(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        fwd_bwd_times.append(end - start)
    
    avg_fwd_bwd_time = sum(fwd_bwd_times) / len(fwd_bwd_times)
    min_fwd_bwd_time = min(fwd_bwd_times)
    max_fwd_bwd_time = max(fwd_bwd_times)
    
    if is_main:
        print(f"  Avg forward+backward time: {avg_fwd_bwd_time*1000:.2f} ms")
    
    # Calculate throughput
    tokens_per_batch = batch_size * cfg.seq_len * world_size
    forward_throughput = tokens_per_batch / avg_forward_time
    fwd_bwd_throughput = tokens_per_batch / avg_fwd_bwd_time
    
    if is_main:
        print(f"  Forward throughput: {forward_throughput:,.0f} tokens/sec")
        print(f"  Forward+Backward throughput: {fwd_bwd_throughput:,.0f} tokens/sec")
    
    return {
        f'{component_name}_num_params': param_count,
        f'{component_name}_avg_forward_time_ms': avg_forward_time * 1000,
        f'{component_name}_min_forward_time_ms': min_forward_time * 1000,
        f'{component_name}_max_forward_time_ms': max_forward_time * 1000,
        f'{component_name}_avg_fwd_bwd_time_ms': avg_fwd_bwd_time * 1000,
        f'{component_name}_min_fwd_bwd_time_ms': min_fwd_bwd_time * 1000,
        f'{component_name}_max_fwd_bwd_time_ms': max_fwd_bwd_time * 1000,
        f'{component_name}_forward_throughput_tokens_sec': forward_throughput,
        f'{component_name}_fwd_bwd_throughput_tokens_sec': fwd_bwd_throughput,
    }


def run_benchmark(cfg, batch_size, num_warmup, num_iterations, output_file):
    """Run the PyTorch benchmark and save results."""
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    if is_main:
        print(f"\n{'='*60}")
        print("PyTorch DDP Transformer Benchmark")
        print(f"{'='*60}")
        
        print(f"\nConfiguration:")
        print(f"  Model dim: {cfg.model_dim}")
        print(f"  Num heads: {cfg.num_heads}")
        print(f"  Seq len: {cfg.seq_len}")
        print(f"  Num layers: {cfg.num_layers}")
        print(f"  Vocab size: {cfg.vocab_size}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  World size (num processes): {world_size}")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  Warmup iterations: {num_warmup}")
        print(f"  Benchmark iterations: {num_iterations}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        device_type = f'cuda:{torch.cuda.get_device_name(local_rank)}'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
    
    if is_main:
        print(f"\nDevice: {device_type}")
    
    # Initialize model
    model = Transformer(cfg)
    model = model.to(device)
    if cfg.dtype == torch.bfloat16:
        model = model.bfloat16()
    # TODO(rka97): make this an option
    #model = torch.compile(model, fullgraph=True, mode='max-autotune', dynamic=False)
    model = torch.compile(model, fullgraph=True)
    
    # Count parameters before wrapping with DDP
    param_count = sum(p.numel() for p in model.parameters())
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    if is_main:
        print(f"  Total parameters: {param_count:,}")
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Generate synthetic data (same seed for reproducibility)
    torch.manual_seed(123 + rank)  # Different data per rank for DDP
    batch = torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len), dtype=torch.long, device=device)
    targets = batch.clone()  # For language modeling
    
    # ============== Forward-only benchmark ==============
    if is_main:
        print(f"\n--- Forward-only Benchmark ---")
    
    model.eval()
    
    # Warmup
    if is_main:
        print(f"Running {num_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            logits = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Benchmark
    if is_main:
        print(f"Running {num_iterations} benchmark iterations...")
    forward_times = []
    with torch.no_grad():
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            logits = model(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            forward_times.append(end - start)
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    min_forward_time = min(forward_times)
    max_forward_time = max(forward_times)
    
    if is_main:
        print(f"  Avg forward time: {avg_forward_time*1000:.2f} ms")
        print(f"  Min forward time: {min_forward_time*1000:.2f} ms")
        print(f"  Max forward time: {max_forward_time*1000:.2f} ms")
    
    # ============== Forward + Backward benchmark ==============
    if is_main:
        print(f"\n--- Forward + Backward Benchmark ---")
    
    model.train()

    @torch.compile
    def _full_step(batch, targets):
        optimizer.zero_grad()
        logits, loss = model(batch, targets=targets)
        loss.backward()
        optimizer.step()
    
    # Warmup
    if is_main:
        print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _full_step(batch, targets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    if is_main:
        print(f"Running {num_iterations} benchmark iterations...")
    fwd_bwd_times = []
    for i in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _full_step(batch, targets) 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        fwd_bwd_times.append(end - start)
    
    avg_fwd_bwd_time = sum(fwd_bwd_times) / len(fwd_bwd_times)
    min_fwd_bwd_time = min(fwd_bwd_times)
    max_fwd_bwd_time = max(fwd_bwd_times)
    
    if is_main:
        print(f"  Avg forward+backward time: {avg_fwd_bwd_time*1000:.2f} ms")
        print(f"  Min forward+backward time: {min_fwd_bwd_time*1000:.2f} ms")
        print(f"  Max forward+backward time: {max_fwd_bwd_time*1000:.2f} ms")
    
    # Calculate throughput (considering all GPUs)
    tokens_per_batch = batch_size * cfg.seq_len * world_size  # Total tokens across all GPUs
    forward_throughput = tokens_per_batch / avg_forward_time
    fwd_bwd_throughput = tokens_per_batch / avg_fwd_bwd_time
    
    if is_main:
        print(f"\n  Forward throughput: {forward_throughput:,.0f} tokens/sec")
        print(f"  Forward+Backward throughput: {fwd_bwd_throughput:,.0f} tokens/sec")
    
    # ============== Component Benchmarks ==============
    if is_main:
        print(f"\n{'='*60}")
        print("Component-Level Benchmarks")
        print(f"{'='*60}")
    
    component_results = {}
    
    # Benchmark MLP
    mlp_model = MLP(
        dim=cfg.model_dim,
        hidden_dim=cfg.expanded_model_dim,
        multiple_of=cfg.multiple_of,
        dtype=cfg.dtype,
    )
    mlp_results = benchmark_component('MLP', mlp_model, cfg, batch_size, num_warmup,
                                      num_iterations, device, world_size, local_rank, is_main)
    component_results.update(mlp_results)
    
    # Benchmark Attention
    attn_model = Attention(cfg)
    attn_results = benchmark_component('Attention', attn_model, cfg, batch_size, num_warmup,
                                       num_iterations, device, world_size, local_rank, is_main)
    component_results.update(attn_results)
    
    # Benchmark Block
    block_model = Block(layer_id=0, cfg=cfg)
    block_results = benchmark_component('Block', block_model, cfg, batch_size, num_warmup,
                                        num_iterations, device, world_size, local_rank, is_main)
    component_results.update(block_results)
    
    # Save results to CSV (only main process)
    if is_main:
        results = {
            'framework': 'PyTorch_DDP',
            'model_dim': cfg.model_dim,
            'num_heads': cfg.num_heads,
            'seq_len': cfg.seq_len,
            'num_layers': cfg.num_layers,
            'vocab_size': cfg.vocab_size,
            'batch_size': batch_size * world_size,  # Effective batch size
            'batch_size_per_gpu': batch_size,
            'num_params': param_count,
            'num_devices': world_size,
            'device_type': device_type,
            'num_iterations': num_iterations,
            'avg_forward_time_ms': avg_forward_time * 1000,
            'min_forward_time_ms': min_forward_time * 1000,
            'max_forward_time_ms': max_forward_time * 1000,
            'avg_fwd_bwd_time_ms': avg_fwd_bwd_time * 1000,
            'min_fwd_bwd_time_ms': min_fwd_bwd_time * 1000,
            'max_fwd_bwd_time_ms': max_fwd_bwd_time * 1000,
            'forward_throughput_tokens_sec': forward_throughput,
            'fwd_bwd_throughput_tokens_sec': fwd_bwd_throughput,
        }
        
        # Add component results
        results.update(component_results)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
        
        print(f"\nResults saved to: {output_file}")
    
    cleanup_distributed()
    
    if is_main:
        return results
    return None


def main():
    parser = argparse.ArgumentParser(description='PyTorch DDP Transformer Benchmark')
    parser.add_argument('--model_dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--num_iterations', type=int, default=20, help='Number of benchmark iterations')
    parser.add_argument('--output', type=str, default='pytorch_benchmark_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create model config
    cfg = ModelConfig(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        expanded_model_dim=4 * args.model_dim,
    )
    
    run_benchmark(
        cfg=cfg,
        batch_size=args.batch_size,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()