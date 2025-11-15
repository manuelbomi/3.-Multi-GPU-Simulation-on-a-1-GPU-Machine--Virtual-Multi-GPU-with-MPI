# Multi-GPU Simulation on a 1‑GPU Machine (Virtual Multi‑GPU with MPI)

This project simulates multi-GPU distributed training using MPI on a single physical GPU (NVIDIA RTX 4070). 

The example uses a minimal CUDA kernel to represent model computation, hosts per-rank synthetic data, and implements an AllReduce (gradient averaging) manually using MPI point-to-point operations.

Goal: implement rank-to-GPU mapping, gradient averaging, DDP internals, and how to scale concepts even on a single GPU.

In total, this project shows:

#### <ins>Rank-to-GPU mapping</ins>

- How frameworks like PyTorch assign each training process to a GPU.

#### <ins>Simulated multi-GPU compute</ins>

Each MPI rank has:

- its own “model shard”

- its own synthetic data

- its own CUDA kernel execution

- its own gradient buffer

#### <ins> Manual gradient AllReduce</ins>

#### The project implement the same technique NCCL uses:

- Each rank computes a local gradient

- All ranks exchange gradients

- Gradients are averaged

- Each rank updates its portion of the model


#### <ins> How real clusters work</ins>

#### This project applies directly to training on:

- DGX stations

- multi-GPU servers

- 8/32/256-GPU supercomputers

#### Even though you only have one GPU (Nvidia RTX 4070).



```python
project3_mpi_cuda/
├─ README.md # this document (short) + run notes
├─ Makefile # build commands
├─ run.sh # quick run script (mpirun wrapper)
├─ src/
│ ├─ mpi_cuda_train.cu # primary program (CUDA + MPI)
│ └─ utils.h # small helpers
└─ env_notes.md # some notes to assist @emmanuel oyekanlu 
```
