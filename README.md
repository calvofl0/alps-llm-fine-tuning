# LLM Fine-tuning with DeepSpeed on CSCS Alps

This repository contains the setup for fine-tuning Large Language Models (LLMs) using DeepSpeed on the CSCS Alps cluster. The setup leverages containerization with Podman/Enroot and the SLURM workload manager for distributed training.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
  - [1. Container Build](#1-container-build)
  - [2. Environment Configuration](#2-environment-configuration)
  - [3. Job Configuration](#3-job-configuration)
  - [4. DeepSpeed Configuration](#4-deepspeed-configuration)
- [Running the Job](#running-the-job)
- [Understanding the Configuration](#understanding-the-configuration)
- [Troubleshooting](#troubleshooting)

## Overview

This setup uses:
- **Base Image**: NVIDIA PyTorch 24.06 container (with DeepSpeed additions)
- **Container Engine**: CSCS Container Engine (Enroot) with EDF (Environment Description Files)
- **Scheduler**: SLURM
- **Training Framework**: DeepSpeed with ZeRO Stage 3 optimization

DeepSpeed ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients, and model parameters across multiple GPUs to enable training of very large models that wouldn't fit in single-GPU memory.

## Prerequisites

- Access to CSCS Alps cluster
- Working directory with sufficient storage (preferably on `/capstor` or `/iopsstor`)
- Basic familiarity with SLURM and containerization

## Quick Start

For those who just want to run:

```bash
# 1. Build the container (one-time setup)
cd ngc-pytorch-deepspeed-24.06
podman build -t ngc-pytorch-deepspeed:24.06 .
enroot import -x mount -o ngc-pytorch-deepspeed-24.06.sqsh podman://ngc-pytorch-deepspeed:24.06
cd ..

# 2. Edit the environment file
# Update ${HOME}/workdir paths in ngc-pytorch-deepspeed-24.06.toml

# 3. Configure your job
# Edit deepspeed.sbatch:
#   - Set your email address
#   - Set MODEL_ID (the model to fine-tune)
#   - Set OUTPUT_DIR (where to save results)

# 4. Submit the job
sbatch deepspeed.sbatch
```

## Detailed Setup

### 1. Container Build

The base NVIDIA PyTorch container needs to be customized with DeepSpeed and converted to a format compatible with the CSCS Container Engine.

#### Build the Podman container:

```bash
cd ngc-pytorch-deepspeed-24.06
podman build -t ngc-pytorch-deepspeed:24.06 .
```

This reads the `Containerfile` in the directory and builds a container tagged as `ngc-pytorch-deepspeed:24.06`.

#### Convert to SquashFS format:

```bash
enroot import -x mount -o ngc-pytorch-deepspeed-24.06.sqsh podman://ngc-pytorch-deepspeed:24.06
```

This command:
- Converts the Podman container to SquashFS format (`.sqsh`)
- SquashFS is a compressed read-only filesystem ideal for container images
- The resulting file can be mounted directly without extraction, saving disk space and improving startup time

**Note**: The `-x mount` flag keeps the SquashFS file mounted during import, and `-o` specifies the output filename.

### 2. Environment Configuration

The Environment Description File (EDF) `ngc-pytorch-deepspeed-24.06.toml` tells the CSCS Container Engine how to run your container. You need to update the working directory paths.

#### Edit `ngc-pytorch-deepspeed-24.06.toml`:

Find these lines and replace `${HOME}/workdir` with your actual working directory path:

```toml
mounts = [
    "/capstor",
    "/iopsstor",
    "/dev/shm/${USER}",
    "${HOME}/.ssh",
    "${HOME}/workdir"  # <-- Update this
] 
workdir = "${HOME}/workdir"  # <-- Update this
```

**Example**: If your working directory is `/capstor/projects/csstaff/aluco/deepspeed-llm`, change both occurrences to that path.

#### Understanding the EDF Configuration:

The EDF file configures several important aspects:

- **image**: Points to the SquashFS container image
- **mounts**: Directories from the host system that will be accessible inside the container
  - `/capstor` and `/iopsstor`: High-performance storage filesystems
  - `/dev/shm/${USER}`: Shared memory for the user
  - `${HOME}/.ssh`: SSH keys for potential remote operations
  - Your working directory: Where your code and data reside
  
- **annotations**: Enable AWS OFI NCCL plugin for optimized GPU communication
  - `com.hooks.aws_ofi_nccl.enabled = "true"`: Activates the plugin
  - `com.hooks.aws_ofi_nccl.variant = "cuda12"`: Specifies CUDA 12 compatibility

- **Environment variables**:
  - `NCCL_DEBUG = "INFO"`: Enables detailed NCCL logging for debugging
  - `CUDA_CACHE_DISABLE = "1"`: Disables CUDA kernel cache to avoid issues
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"`: Enables async error handling for better error reporting
  - `MPICH_GPU_SUPPORT_ENABLED = "0"`: Disables MPI GPU support (using NCCL instead)

### 3. Job Configuration

The SLURM batch script `deepspeed.sbatch` defines your training job parameters.

#### Required edits:

```bash
#SBATCH --account <YOUR_ACCOUNT>  # Replace with your account
#SBATCH --mail-user <YOUR_EMAIL>  # Replace with your email address

# In the script body:
MODEL_ID="meta-llama/Llama-2-7b-hf"  # Model to fine-tune
OUTPUT_DIR="./output"                 # Where to save results
```

#### Understanding the SLURM directives:

The script includes several SLURM directives that control job execution:

```bash
#SBATCH --job-name=deepspeed-training    # Job name in queue
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=4               # GPUs per node (Alps has 4 GPUs/node)
#SBATCH --cpus-per-task=16                # CPU cores per GPU task
#SBATCH --time=04:00:00                   # Maximum runtime (4 hours)
#SBATCH --partition=gpu                   # GPU partition
#SBATCH --account=<your_account>          # Your allocation account
#SBATCH --mail-type=END,FAIL              # Email on job end or failure
```

**Important**: The total number of GPUs used will be `nodes × ntasks-per-node` (in the example: 2 × 4 = 8 GPUs).

### 4. DeepSpeed Configuration

The `deepspeed_config.json` file controls DeepSpeed's optimization strategy. The current configuration uses ZeRO Stage 3 with activation checkpointing.

#### Key Configuration Options:

##### Basic Settings:

- **`train_micro_batch_size_per_gpu`**: Number of samples processed per GPU per step
  - Set to `1` for very large models
  - Increase if you have memory headroom for better throughput
  - Must be balanced with gradient accumulation

- **`bf16.enabled`**: Uses BFloat16 precision for training
  - Provides similar numerical stability to FP32
  - Reduces memory usage and increases speed
  - Recommended for modern GPUs (A100, H100)

##### ZeRO Stage 3 Optimization:

ZeRO Stage 3 partitions the model parameters, gradients, and optimizer states across all GPUs, dramatically reducing per-GPU memory requirements.

- **`stage: 3`**: Enables full ZeRO optimization
  - Stage 1: Partitions optimizer states only
  - Stage 2: Partitions optimizer states + gradients
  - Stage 3: Partitions optimizer states + gradients + parameters

- **`allgather_bucket_size`**, **`reduce_bucket_size`**, **`prefetch_bucket_size`**: All set to `104857600` (100 MB)
  - Controls the size of communication buffers
  - Larger values: Better bandwidth utilization, higher memory usage
  - Smaller values: More frequent communication, lower memory usage
  - 100 MB is a balanced default for modern interconnects

- **`overlap_comm: false`**: Communication is not overlapped with computation
  - When `true`: Can hide communication latency but requires more memory
  - When `false`: Safer for memory-constrained scenarios
  - Consider enabling if training is bandwidth-bound

- **`contiguous_gradients: false`**: Gradients are not copied to contiguous buffers
  - When `true`: Reduces memory fragmentation, increases memory overhead
  - When `false`: Lower memory usage, potential fragmentation

- **`stage3_param_persistence_threshold: 0`**: All parameters are partitioned
  - Larger values keep small parameters unpartitioned (reduces communication)
  - `0` maximizes memory savings by partitioning everything

- **`stage3_gather_16bit_weights_on_model_save: false`**: Model is saved in partitioned format
  - When `true`: Gathers full model in FP16 before saving (easier to load later)
  - When `false`: Saves in partitioned format (saves memory during checkpointing)

##### Activation Checkpointing:

Activation checkpointing (also called gradient checkpointing) trades computation for memory by recomputing activations during the backward pass instead of storing them.

- **`partition_activations: true`**: Activations are partitioned across GPUs
  - Distributes activation memory across all GPUs
  - Essential for very large models

- **`contiguous_memory_optimization: true`**: Optimizes activation memory layout
  - Reduces memory fragmentation
  - Can improve performance

##### Checkpoint Configuration:

- **`load_universal: true`**: Enables loading checkpoints created with different world sizes
  - Allows flexibility in resuming training with different GPU counts
  - Useful for dynamic resource allocation

- **`use_node_local_storage: true`**: Stores checkpoints on local node storage
  - Faster checkpoint writes/reads
  - Doesn't require shared filesystem access during checkpointing
  - Final checkpoints should still be copied to persistent storage

#### When to Modify the Configuration:

You might want to modify `deepspeed_config.json` if:

- **Running smaller models**: Increase `train_micro_batch_size_per_gpu` for better throughput
- **Out of memory errors**: 
  - Decrease batch size
  - Enable `overlap_comm` if you have memory headroom
  - Ensure `stage3_param_persistence_threshold: 0`
- **Communication bottlenecks**: 
  - Increase bucket sizes
  - Enable `overlap_comm: true`
- **Different precision needs**: Switch between `fp16` and `bf16` based on your hardware
- **Checkpoint compatibility**: Set `stage3_gather_16bit_weights_on_model_save: true` for easier model distribution

For a complete reference of all DeepSpeed configuration options, see: https://www.deepspeed.ai/docs/config-json/

## Running the Job

Once everything is configured:

```bash
sbatch deepspeed.sbatch
```

### Monitoring your job:

```bash
# Check job status
squeue -u $USER

# View job output (while running or after completion)
tail -f slurm-<jobid>.out

# Cancel a job if needed
scancel <jobid>
```

### Understanding the Output:

DeepSpeed will print training progress including:
- Loss values
- Training throughput (samples/second)
- Memory usage per GPU
- Communication timing (if `NCCL_DEBUG=INFO`)

## Understanding the Configuration

### Container Engine on Alps

The CSCS Container Engine uses a customized version of NVIDIA Pyxis to integrate containers with SLURM. Key features:

- **Automatic image caching**: Remote images are cached in `${SCRATCH}/.edf_imagestore`
- **Shared containers**: All tasks on a node share the same container instance
- **Registry authentication**: Configure in `~/.config/enroot/.credentials` for private registries
- **EDF search path**: EDFs can be placed in `~/.edf/` and referenced by name (without `.toml` extension)

To use the EDF by name instead of path:

```bash
mkdir -p ~/.edf
cp ngc-pytorch-deepspeed-24.06.toml ~/.edf/
# Then in your sbatch script, use: --environment=ngc-pytorch-deepspeed-24.06
```

### DeepSpeed + SLURM Integration

The training script automatically detects SLURM environment variables:
- `SLURM_PROCID`: Global rank of the process
- `SLURM_LOCALID`: Local rank on the node
- `SLURM_NTASKS`: Total number of processes
- `SLURM_STEP_NODELIST`: List of nodes in the job

DeepSpeed uses these to configure distributed training without additional setup.

### Memory Optimization Strategy

The configuration uses multiple complementary techniques to reduce memory:

1. **ZeRO Stage 3**: Partitions all model state across GPUs
2. **BFloat16**: Reduces precision from 32-bit to 16-bit
3. **Activation Checkpointing**: Recomputes instead of storing activations
4. **Gradient Accumulation** (if configured): Splits effective batch into smaller micro-batches

Together, these enable training models that are 10-100x larger than what would fit in GPU memory otherwise.

## Troubleshooting

### Container build fails

```bash
# Check if you have podman installed and configured
podman --version

# Ensure you have sufficient disk space
df -h .
```

### Enroot import fails

```bash
# Verify the container exists in podman
podman images

# Check for sufficient disk space
df -h .
```

### Job fails immediately

```bash
# Check the SLURM output file for errors
cat slurm-<jobid>.out

# Common issues:
# - Wrong account name
# - Insufficient allocation hours
# - Invalid partition name
```

### Out of Memory (OOM) errors

If you encounter OOM errors:

1. **Reduce batch size**: Decrease `train_micro_batch_size_per_gpu` in `deepspeed_config.json`
2. **Enable communication overlap**: Set `overlap_comm: true` (if you have some memory headroom)
3. **Verify ZeRO Stage 3**: Ensure `stage: 3` is set
4. **Check activation checkpointing**: Ensure `partition_activations: true`
5. **Reduce model size**: Consider using a smaller model for initial testing

### NCCL/Communication errors

If you see NCCL timeout or communication errors:

1. **Check interconnect**: Ensure AWS OFI NCCL plugin is enabled in the EDF
2. **Increase timeout**: Add `NCCL_TIMEOUT=1800` to the environment section of the EDF
3. **Verify GPU connectivity**: Run `nvidia-smi topo -m` on a compute node to check GPU topology
4. **Check for hardware issues**: Some nodes may have faulty GPUs or network adapters

### Container runtime errors

```bash
# Verify the EDF file syntax
cat ngc-pytorch-deepspeed-24.06.toml

# Check if mounts are accessible
ls -la /capstor /iopsstor

# Ensure the container image path is correct
ls -la ngc-pytorch-deepspeed-24.06/ngc-pytorch-deepspeed-24.06.sqsh
```

### Permission errors

```bash
# Verify you have write access to output directory
mkdir -p $OUTPUT_DIR && touch $OUTPUT_DIR/test && rm $OUTPUT_DIR/test

# Check ownership of working directory
ls -la $(dirname $PWD)
```

## Additional Resources

- [CSCS Container Engine Documentation](https://docs.cscs.ch/software/container-engine/run/)
- [DeepSpeed Configuration Reference](https://www.deepspeed.ai/docs/config-json/)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [NVIDIA PyTorch Container Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
- [Alps System Documentation](https://docs.cscs.ch/)

## Support

For issues specific to:
- **CSCS infrastructure**: Contact CSCS support
- **DeepSpeed configuration**: Consult DeepSpeed documentation or GitHub issues
- **This setup**: Review the troubleshooting section above

---

**Note**: This README assumes you have your training script properly configured to use DeepSpeed. Ensure your training code imports DeepSpeed and initializes it correctly with the configuration file.
