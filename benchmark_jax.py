"""
JAX Transformer Benchmark Script
Times forward and backward passes and writes results to CSV.

Uses JAX's sharding API for data parallelism across multiple devices,
comparable to PyTorch's DistributedDataParallel (DDP).
"""

import dataclasses
import time
import csv
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from flax import linen as nn
from flax.training import train_state
import optax


@dataclasses.dataclass
class ModelConfig:
    """Hyper-parameters for Transformer decoder-only."""
    model_dim: int
    num_heads: int
    seq_len: int
    num_layers: int
    vocab_size: int
    expanded_model_dim: int
    multiple_of: int = 256
    rmsnorm_epsilon: float = 1e-6
    use_residual_scaling: bool = True
    tie_embeddings: bool = True
    qknorm_epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    attention_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)
    linear_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)
    embed_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)

    def __post_init__(self):
        self.residual_init = nn.initializers.normal(
            stddev=0.02 / jnp.sqrt(2 * self.num_layers)
        )


class Mlp(nn.Module):
    """Multilayer perceptron with GLU activation."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x_BxLxD: jax.Array):
        cfg = self.cfg
        linear = partial(
            nn.Dense, kernel_init=cfg.linear_init, use_bias=False, dtype=cfg.dtype
        )
        hidden_dim = cfg.expanded_model_dim * 2 / 3
        hidden_dim = cfg.multiple_of * (
            (cfg.expanded_model_dim + cfg.multiple_of - 1) // cfg.multiple_of
        )
        x_BxLx2F = linear(int(2 * hidden_dim))(x_BxLxD)
        x_BxLxF = nn.glu(x_BxLx2F, axis=-1)
        x_BxLxD = nn.Dense(
            cfg.model_dim,
            use_bias=False,
            dtype=cfg.dtype,
            kernel_init=cfg.residual_init if cfg.use_residual_scaling else cfg.linear_init,
        )(x_BxLxF)
        return x_BxLxD


@partial(jax.jit, static_argnums=(0, 1, 2))
def init_rope(dim=256, seq_len=128, n_heads=4):
    """Initialize rotary embeddings."""
    def precompute_freqs_cis_jax(dim, end, theta=10000.0):
        inv_freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
        t = jnp.arange(end) / 1.0
        freqs = jnp.outer(t, inv_freqs).astype(jnp.float32)
        return jnp.stack(
            [jnp.cos(freqs)[None, :, None, :], jnp.sin(freqs)[None, :, None, :]],
            axis=3,
        )
    freqs_cis = precompute_freqs_cis_jax(dim // n_heads, seq_len, theta=500000)
    return freqs_cis.transpose(0, 1, 2, 4, 3)


@jax.jit
def apply_rope(q, k, freqs_cis):
    """Apply rotary embeddings to Q and K."""
    def rotate_tensor(x):
        x_r2 = x.reshape(*x.shape[:-1], -1, 2)
        L = x.shape[1]
        freqs = freqs_cis[:, :L, :, :, :]
        rotated_x_r2 = jnp.stack(
            [
                x_r2[..., 0] * freqs[..., 0] - x_r2[..., 1] * freqs[..., 1],
                x_r2[..., 1] * freqs[..., 0] + x_r2[..., 0] * freqs[..., 1],
            ],
            axis=-1,
        )
        return rotated_x_r2.reshape(*x.shape)
    rotated_q = rotate_tensor(q)
    rotated_k = rotate_tensor(k)
    return rotated_q, rotated_k


class CausalAttn(nn.Module):
    """Causal attention layer with rotary embeddings."""
    cfg: ModelConfig

    def setup(self):
        cfg = self.cfg
        assert cfg.model_dim % cfg.num_heads == 0
        self.Dh = cfg.model_dim // cfg.num_heads
        self.eps = cfg.qknorm_epsilon
        self.freqs_cis = init_rope(cfg.model_dim, cfg.seq_len, cfg.num_heads)
        self.multilinear = partial(
            nn.DenseGeneral,
            axis=-1,
            features=(cfg.num_heads, self.Dh),
            kernel_init=cfg.attention_init,
            use_bias=False,
            dtype=cfg.dtype,
        )
        self.multilinear_query = self.multilinear(name='query')
        self.multilinear_key = self.multilinear(name='key')
        self.multilinear_value = self.multilinear(name='value')
        seq_len = cfg.seq_len
        attn_scale0 = jnp.log2(seq_len**2 - seq_len)
        self.attn_scale = self.param('attn_scale', nn.initializers.constant(attn_scale0), ())
        self.output_projection = nn.DenseGeneral(
            features=cfg.model_dim,
            name='attn_out_proj',
            kernel_init=cfg.residual_init if cfg.use_residual_scaling else cfg.linear_init,
            use_bias=False,
            dtype=cfg.dtype,
        )

    def __call__(self, x_BxLxD: jax.Array):
        cfg = self.cfg
        q_BxLxHxDh = self.multilinear_query(x_BxLxD)
        k_BxLxHxDh = self.multilinear_key(x_BxLxD)
        v_BxLxHxDh = self.multilinear_value(x_BxLxD)
        q_BxLxHxDh, k_BxLxHxDh = apply_rope(q_BxLxHxDh, k_BxLxHxDh, self.freqs_cis)
        q_BxLxHxDh /= jnp.linalg.norm(q_BxLxHxDh, axis=-1, keepdims=True) + self.eps
        k_BxLxHxDh /= jnp.linalg.norm(k_BxLxHxDh, axis=-1, keepdims=True) + self.eps
        att_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxLxHxDh)
        L = x_BxLxD.shape[1]
        mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
        _NEG_INF = jnp.finfo(cfg.dtype).min
        att_BxHxLxL = jnp.where(mask_1x1xLxL, att_BxHxLxL, _NEG_INF)
        att_BxHxLxL = self.attn_scale * att_BxHxLxL
        att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
        att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)
        out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxL, v_BxLxHxDh)
        out_BxLxD = out_BxLxHxDh.reshape(*x_BxLxD.shape)
        out_BxLxD = self.output_projection(out_BxLxD)
        return out_BxLxD


class TBlock(nn.Module):
    """Transformer Block."""
    docfg: ModelConfig

    @nn.compact
    def __call__(self, in_BxLxD: jax.Array):
        cfg = self.docfg
        x_BxLxD = nn.RMSNorm(param_dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)(in_BxLxD)
        x_BxLxD = CausalAttn(cfg)(x_BxLxD)
        x_BxLxD += in_BxLxD
        z_BxLxD = nn.RMSNorm(param_dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)(x_BxLxD)
        z_BxLxD = Mlp(cfg)(z_BxLxD)
        return x_BxLxD + z_BxLxD


class TransformerDo(nn.Module):
    """Transformer decoder-only."""
    docfg: ModelConfig

    def setup(self):
        cfg = self.docfg
        self.embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.model_dim,
            embedding_init=cfg.embed_init,
        )
        self.blocks = [TBlock(cfg) for _ in range(cfg.num_layers)]
        self.out_ln = nn.RMSNorm(param_dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)
        if cfg.tie_embeddings:
            self.output_proj = lambda x: self.embed.attend(x.astype(jnp.float32))
        else:
            self.output_proj = nn.Dense(
                cfg.vocab_size,
                kernel_init=cfg.embed_init,
                dtype=cfg.dtype,
                name='output_proj',
            )

    def __call__(self, y_BxL: jax.Array):
        y_BxLxD = self.embed(y_BxL)
        for block in self.blocks:
            y_BxLxD = block(y_BxLxD)
        y_BxLxD = self.out_ln(y_BxLxD)
        logits_BxLxV = self.output_proj(y_BxLxD)
        return logits_BxLxV


# ============== Benchmark Functions ==============

def create_mesh(num_devices=None):
    """Create a device mesh for data parallelism."""
    devices = jax.devices()
    if num_devices is not None:
        devices = devices[:num_devices]
    
    # Create 1D mesh for data parallelism (batch dimension)
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),), devices), ('data',))
    return mesh


def create_train_state(rng, model, learning_rate, input_shape, mesh):
    """Creates initial TrainState with sharded parameters."""
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    tx = optax.adamw(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    
    # Replicate params and optimizer state across all devices
    # (data parallelism = replicated params, sharded data)
    replicate_sharding = NamedSharding(mesh, P())  # No partitioning = replicate
    state = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, replicate_sharding),
        state
    )
    return state


def compute_loss(params, apply_fn, batch, targets):
    """Compute cross-entropy loss."""
    logits = apply_fn(params, batch)
    # Shift logits and targets for next-token prediction
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
    return jnp.mean(loss)


def create_sharded_train_step(mesh, apply_fn):
    """Create a sharded training step function."""
    # Data is sharded along batch dimension, params are replicated
    data_sharding = NamedSharding(mesh, P('data'))  # Shard batch dim
    replicate_sharding = NamedSharding(mesh, P())   # Replicate
    
    # Alternative without pmean - using automatic gradient aggregation
    @partial(jax.jit, in_shardings=(replicate_sharding, data_sharding),
             out_shardings=(replicate_sharding, replicate_sharding))
    def train_step(state, batch):
        """Training step - gradients auto-averaged by sharding."""
        targets = batch
        
        def loss_fn(params):
            return compute_loss(params, apply_fn, batch, targets)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    return train_step


def create_sharded_forward(mesh, apply_fn):
    """Create a sharded forward pass function."""
    data_sharding = NamedSharding(mesh, P('data'))
    replicate_sharding = NamedSharding(mesh, P())
    output_sharding = NamedSharding(mesh, P('data'))  # Output also sharded
    
    @partial(jax.jit, in_shardings=(replicate_sharding, data_sharding),
             out_shardings=output_sharding)
    def forward_only(params, batch):
        """Forward pass only."""
        return apply_fn(params, batch)
    
    return forward_only


def shard_batch(batch, mesh):
    """Shard a batch across the data dimension."""
    data_sharding = NamedSharding(mesh, P('data'))
    return jax.device_put(batch, data_sharding)


def run_benchmark(cfg, batch_size, num_warmup, num_iterations, output_file, num_devices=None):
    """Run the JAX benchmark and save results."""
    print(f"\n{'='*60}")
    print("JAX Transformer Benchmark (Data Parallel)")
    print(f"{'='*60}")
    
    # Create mesh for data parallelism
    all_devices = jax.devices()
    if num_devices is not None:
        devices_to_use = all_devices[:num_devices]
    else:
        devices_to_use = all_devices
    
    mesh = create_mesh(len(devices_to_use))
    num_devices_used = len(devices_to_use)
    
    # Total batch size = per-device batch size * num devices
    total_batch_size = batch_size * num_devices_used
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model dim: {cfg.model_dim}")
    print(f"  Num heads: {cfg.num_heads}")
    print(f"  Seq len: {cfg.seq_len}")
    print(f"  Num layers: {cfg.num_layers}")
    print(f"  Vocab size: {cfg.vocab_size}")
    print(f"  Batch size per device: {batch_size}")
    print(f"  Total batch size: {total_batch_size}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark iterations: {num_iterations}")
    
    print(f"\nDevices:")
    print(f"  Available devices: {len(all_devices)}")
    print(f"  Using devices: {num_devices_used}")
    print(f"  Device type: {devices_to_use[0]}")
    print(f"  Mesh shape: {mesh.shape}")
    
    # Initialize model
    model = TransformerDo(cfg)
    rng = jax.random.PRNGKey(42)
    
    # Create train state with sharding
    input_shape = (total_batch_size, cfg.seq_len)
    with mesh:
        state = create_train_state(rng, model, learning_rate=1e-4, input_shape=input_shape, mesh=mesh)
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Total parameters: {param_count:,}")
    
    # Generate synthetic data and shard it
    data_rng = jax.random.PRNGKey(123)
    batch = jax.random.randint(data_rng, shape=input_shape, minval=0, maxval=cfg.vocab_size, dtype=jnp.int32)
    batch = shard_batch(batch, mesh)
    
    print(f"\nData sharding: {batch.sharding}")
    
    # Create sharded functions
    with mesh:
        forward_fn = create_sharded_forward(mesh, model.apply)
        train_step_fn = create_sharded_train_step(mesh, model.apply)
    
    # ============== Forward-only benchmark ==============
    print(f"\n--- Forward-only Benchmark ---")
    
    # Warmup
    print(f"Running {num_warmup} warmup iterations...")
    with mesh:
        for _ in range(num_warmup):
            logits = forward_fn(state.params, batch)
            logits.block_until_ready()
    
    # Benchmark
    print(f"Running {num_iterations} benchmark iterations...")
    forward_times = []
    with mesh:
        for i in range(num_iterations):
            start = time.perf_counter()
            logits = forward_fn(state.params, batch)
            logits.block_until_ready()
            end = time.perf_counter()
            forward_times.append(end - start)
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    min_forward_time = min(forward_times)
    max_forward_time = max(forward_times)
    print(f"  Avg forward time: {avg_forward_time*1000:.2f} ms")
    print(f"  Min forward time: {min_forward_time*1000:.2f} ms")
    print(f"  Max forward time: {max_forward_time*1000:.2f} ms")
    
    # ============== Forward + Backward benchmark ==============
    print(f"\n--- Forward + Backward Benchmark ---")
    
    # Warmup
    print(f"Running {num_warmup} warmup iterations...")
    with mesh:
        for _ in range(num_warmup):
            state, loss = train_step_fn(state, batch)
            loss.block_until_ready()
    
    # Benchmark
    print(f"Running {num_iterations} benchmark iterations...")
    fwd_bwd_times = []
    with mesh:
        for i in range(num_iterations):
            start = time.perf_counter()
            state, loss = train_step_fn(state, batch)
            loss.block_until_ready()
            end = time.perf_counter()
            fwd_bwd_times.append(end - start)
    
    avg_fwd_bwd_time = sum(fwd_bwd_times) / len(fwd_bwd_times)
    min_fwd_bwd_time = min(fwd_bwd_times)
    max_fwd_bwd_time = max(fwd_bwd_times)
    print(f"  Avg forward+backward time: {avg_fwd_bwd_time*1000:.2f} ms")
    print(f"  Min forward+backward time: {min_fwd_bwd_time*1000:.2f} ms")
    print(f"  Max forward+backward time: {max_fwd_bwd_time*1000:.2f} ms")
    
    # Calculate throughput (total tokens across all devices)
    tokens_per_batch = total_batch_size * cfg.seq_len
    forward_throughput = tokens_per_batch / avg_forward_time
    fwd_bwd_throughput = tokens_per_batch / avg_fwd_bwd_time
    print(f"\n  Forward throughput: {forward_throughput:,.0f} tokens/sec")
    print(f"  Forward+Backward throughput: {fwd_bwd_throughput:,.0f} tokens/sec")
    
    # Save results to CSV
    results = {
        'framework': 'JAX_DataParallel',
        'model_dim': cfg.model_dim,
        'num_heads': cfg.num_heads,
        'seq_len': cfg.seq_len,
        'num_layers': cfg.num_layers,
        'vocab_size': cfg.vocab_size,
        'batch_size': total_batch_size,
        'batch_size_per_device': batch_size,
        'num_params': param_count,
        'num_devices': num_devices_used,
        'device_type': str(devices_to_use[0]),
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
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    print(f"\nResults saved to: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='JAX Transformer Benchmark (Data Parallel)')
    parser.add_argument('--model_dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--num_iterations', type=int, default=20, help='Number of benchmark iterations')
    parser.add_argument('--num_devices', type=int, default=None, help='Number of devices to use (default: all available)')
    parser.add_argument('--output', type=str, default='jax_benchmark_results.csv', help='Output CSV file')
    
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
        num_devices=args.num_devices,
    )


if __name__ == '__main__':
    main()