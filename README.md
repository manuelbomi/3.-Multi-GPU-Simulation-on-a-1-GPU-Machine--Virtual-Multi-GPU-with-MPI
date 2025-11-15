# Multi-GPU Simulation on a 1‑GPU Machine (Virtual Multi‑GPU with MPI)

#### This project simulates multi-GPU distributed training using MPI on a single physical GPU (NVIDIA RTX 4070). 

#### The project uses a minimal CUDA kernel to represent model computation, hosts per-rank synthetic data, and implements an AllReduce (gradient averaging) manually using MPI point-to-point operations.

<ins>Goal</ins>: implement rank-to-GPU mapping, gradient averaging, DDP internals, and how to scale concepts even on a single GPU.

#### In total, this project shows:

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

#### Even though we have only have one GPU (Nvidia RTX 4070).

---


#### Interested readers may benefit from reading the preceding projects in this series. They are available here:

*https://github.com/manuelbomi/1.-HPC-Deep-Learning--Training-Workflow-Simulation-with-CUDA-MPI-Nvidia-RTX4070-GPU-and-Rocky-Linux*

and here: 

*https://github.com/manuelbomi/2.-End-to-End-HPC-AI-Training-Simulation-with-CUDA-MPI-and-Slurm-on-Rocky-Linux-----RHEL-Compatible*

---

## Project Structure

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

## Environment Requirements

- Rocky Linux (RHEL compatible)

- Nvidia RTX 4070 GPU 

- CUDA Toolkit installed (nvcc works)

- OpenMPI installed (openmpi + openmpi-devel)

---

## Design notes 
    • In our design, each MPI rank acts like one GPU. Each rank has its own model shard (a contiguous block of the weight vector), synthetic data (random but seeded by rank), and computes its own local gradient.
    • After computing the local gradient, ranks participate in a manual AllReduce implemented using MPI point-to-point operations to compute the elementwise average of gradients across ranks.
    • Aggregated gradients are then applied to each rank's shard locally.
    • The example is intentionally small and fully commented so you can follow the exact data movement (GPU ↔ host ↔ MPI) that mirrors real distributed training.

#### All the codes are provided below:

<ins>Makefile</ins>

```python
NVCC = nvcc
CXXFLAGS = -std=c++14 -Xcompiler -fPIC
MPI_INC = /usr/include/openmpi-x86_64
MPI_LIB = /usr/lib64/openmpi/lib

all: mpi_cuda_train

mpi_cuda_train: src/mpi_cuda_train.cu
	$(NVCC) $(CXXFLAGS) -I$(MPI_INC) -L$(MPI_LIB) \
	-o mpi_cuda_train src/mpi_cuda_train.cu -lmpi

clean:
	rm -f mpi_cuda_train

```

> [!NOTE]
> System MPI/CUDA setups vary. If nvcc linking fails on your machine, try replacing the compile line with an mpicxx command and add -L/-I for CUDA.
> Example alternative:
> # Alternative compile (if required):
> # mpicxx -std=c++14 -o $(TARGET) $(SRC) -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

---

<ins>run.sh</ins>
```python
#!/usr/bin/env bash

if [ -z "$1" ]; then
echo "Usage: $0 <num_ranks>"
exit 1
fi

NUM_RANKS=$1

# Map all ranks to the same physical GPU (0) but leave CUDA_VISIBLE_DEVICES per-process
# This trick helps reproduce multi-process single-GPU behavior: each process believes it owns a GPU.
export CUDA_VISIBLE_DEVICES=0
# Recommended mpirun flags: bind-to none so processes can create CUDA contexts

mpirun -np ${NUM_RANKS} --bind-to none --map-by slot ./mpi_cuda_train

Make the script executable: chmod +x run.sh.

```

---

