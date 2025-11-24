#!/bin/bash
#
# Transformer Benchmark Runner
# Runs JAX and PyTorch DDP benchmarks and compares results
#

set -e

# Default configuration
MODEL_DIM=${MODEL_DIM:-1024}
NUM_HEADS=${NUM_HEADS:-8}
SEQ_LEN=${SEQ_LEN:-1024}
NUM_LAYERS=${NUM_LAYERS:-12}
VOCAB_SIZE=${VOCAB_SIZE:-50257}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WARMUP=${NUM_WARMUP:-5}
NUM_ITERATIONS=${NUM_ITERATIONS:-20}
NUM_GPUS=${NUM_GPUS:-4}
RUN_JAX=${RUN_JAX:-1}  # Set to 0 to skip JAX benchmark

# Output files
JAX_OUTPUT="jax_benchmark_results.csv"
PYTORCH_OUTPUT="pytorch_benchmark_results.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "          TRANSFORMER BENCHMARK SUITE"
echo "============================================================"
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model Dimension:    $MODEL_DIM"
echo "  Number of Heads:    $NUM_HEADS"
echo "  Sequence Length:    $SEQ_LEN"
echo "  Number of Layers:   $NUM_LAYERS"
echo "  Vocabulary Size:    $VOCAB_SIZE"
echo "  Batch Size:         $BATCH_SIZE"
echo "  Warmup Iterations:  $NUM_WARMUP"
echo "  Benchmark Iters:    $NUM_ITERATIONS"
echo "  Number of GPUs:     $NUM_GPUS"
echo ""

echo -e "${YELLOW}Benchmarks to run:${NC}"
echo "  - Full Transformer Model"
echo "  - MLP / Mlp component"
echo "  - Attention / CausalAttn component"
echo "  - Block / TBlock component"
echo ""

# Common arguments
COMMON_ARGS="--model_dim $MODEL_DIM --num_heads $NUM_HEADS --seq_len $SEQ_LEN \
    --num_layers $NUM_LAYERS --vocab_size $VOCAB_SIZE --batch_size $BATCH_SIZE \
    --num_warmup $NUM_WARMUP --num_iterations $NUM_ITERATIONS"
eval "$(conda shell.bash hook)"
# Run JAX benchmark
if [ $RUN_JAX -gt 0 ]; then
  echo -e "${GREEN}[1/3] Running JAX Benchmark...${NC}"
  echo "─────────────────────────────────────────────────────────────"
  conda activate ap11_jax
  python benchmark_jax.py $COMMON_ARGS --output $JAX_OUTPUT

  if [ $? -ne 0 ]; then
      echo -e "${RED}JAX benchmark failed!${NC}"
      exit 1
  fi
else
  echo -e "${YELLOW}Skipping JAX benchmark as per configuration.${NC}"
  JAX_OUTPUT=""
fi

echo ""

# Run PyTorch DDP benchmark
echo -e "${GREEN}[2/3] Running PyTorch DDP Benchmark...${NC}"
echo "─────────────────────────────────────────────────────────────"
conda activate ap11_torch_latest
if [ $NUM_GPUS -gt 1 ]; then
    # Use torchrun for multi-GPU
    torchrun --nproc_per_node=$NUM_GPUS benchmark_torch_ddp.py $COMMON_ARGS --output $PYTORCH_OUTPUT
else
    # Single GPU/CPU mode
    python benchmark_torch_ddp.py $COMMON_ARGS --output $PYTORCH_OUTPUT
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}PyTorch benchmark failed!${NC}"
    exit 1
fi

echo ""

# Compare results
echo -e "${GREEN}[3/3] Comparing Results...${NC}"
echo "─────────────────────────────────────────────────────────────"
python compare_results.py --jax $JAX_OUTPUT --pytorch $PYTORCH_OUTPUT

echo -e "${BLUE}"
echo "============================================================"
echo "                    BENCHMARK COMPLETE"
echo "============================================================"
echo -e "${NC}"

echo "Results saved to:"
echo "  - JAX:     $JAX_OUTPUT"
echo "  - PyTorch: $PYTORCH_OUTPUT"