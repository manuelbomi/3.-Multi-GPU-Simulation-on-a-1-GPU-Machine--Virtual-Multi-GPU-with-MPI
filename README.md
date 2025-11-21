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
	• Create a virtual environemnt in your Rocky Linux
    • Create new project file; e.g.  *mkdir -p ~/projects/project3_mpi_cuda && cd ~/projects/project3_mpi_cuda*
	• Copy the codes from below or clone this repository
	• Paste each code into its file if you choose to copy. You can use nano to create each file. Give each file the necessary execution permissions (chmod +x filename)

#### For example, to use nano to create and paste code into run.sh, follow the sequence below:

*nano run.sh*           (create nano file run.sh)

*ctrl+shift+v*          (to paste the code into the nano file)

*ctrl+o*                (to save code)

*press Enter*

*ctrl+x*                (to move back from nano to terminal)



*chmod +x filename*     (to give each file the necessary execution permission)

---
	
	

#### Do the same process for all the codes. All the codes are provided below:

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

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/29429fb9-aa61-49c1-8e8c-6ae913fa2869" />

> [!NOTE]
> System MPI/CUDA setups vary. If nvcc linking fails on your machine, try replacing the compile line with an mpicxx command and add -L/-I for CUDA.
> 
> Alternative compile (if required):
> 
> mpicxx -std=c++14 -o $(TARGET) $(SRC) -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

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

<ins>src/utils.h</ins>
```python
#pragma once
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

inline void checkCuda(cudaError_t e, const char* file, int line) {
if (e != cudaSuccess) {
fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), file, line);
MPI_Abort(MPI_COMM_WORLD, -1);
}
}
#define CHECK_CUDA(x) checkCuda((x), __FILE__, __LINE__)

inline void checkMPI(int rc, const char* file, int line) {
if (rc != MPI_SUCCESS) {
fprintf(stderr, "MPI error %d at %s:%d\n", rc, file, line);
MPI_Abort(MPI_COMM_WORLD, rc);
}
}
#define CHECK_MPI(x) checkMPI((x), __FILE__, __LINE__)

```

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/23d23e4a-01ab-4342-9860-a652190b638b" />

---

<ins>src/mpi_cuda_train.cu</ins>
```python
// Allocate device arrays for weights and gradient (per-shard)
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

// -----------------------------------------------
// Simple GPU kernel: multiply each element by rank
// -----------------------------------------------
__global__ void scale_kernel(float* data, int N, int rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= rank;
}

// -----------------------------------------------
// Main Program
// -----------------------------------------------
int main(int argc, char** argv) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N = 1024;

    // Host buffer
    float* host_data = (float*) malloc(N * sizeof(float));

    // Initialize host data (no CURAND — deterministic)
    for (int i = 0; i < N; i++) {
        host_data[i] = 1.0f;  // constant data for demonstration
    }

    // Allocate GPU memory
    float* dev_data;
    cudaMalloc(&dev_data, N * sizeof(float));
    cudaMemcpy(dev_data, host_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(dev_data, N, world_rank);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(host_data, dev_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few values to confirm each rank did GPU work
    printf("Rank %d sample values: %f %f %f\n",
           world_rank, host_data[0], host_data[1], host_data[2]);

    // Cleanup
    cudaFree(dev_data);
    free(host_data);

    MPI_Finalize();
    return 0;
}
```

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/a092ddc1-6779-4bcb-bc72-489d63528034" />

---

## How to Run the Project

#### 1. Go to the project root

*cd ~/projects/project3_mpi_cuda*

#### 2. Build the MPI + CUDA program

*make*

#### This compiles src/mpi_cuda_train.cu into:

*./mpi_cuda_train*

#### 3. Run with 4 simulated GPUs (MPI ranks)

*./run.sh 4*

#### run.sh will then execute:

*mpirun -np 4 --bind-to none --map-by slot ./mpi_cuda_train* ; (i.e. run with 4 MPI ranks)

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/43d7581b-e44a-49eb-bbe0-200a2819bbfe" />

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/d119d28b-9855-4bb1-bd4f-cd74a53a8559" />

#### Alternatively, just run the program with:  

*mpirun -np 2 ./mpi_cuda_train* ; (i.e. run with 2 MPI ranks)

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/6e8dfcf3-2403-4eff-b677-0d5b1fc49219" />


#### All 4 MPI ranks will run simulated GPUs on your single RTX 4070, each with its own:
    • model shard
    • synthetic data
    • gradient
    • manual AllReduce

#### You should see output like:

```python
rank 0 step 0 shard_len 256 averaged_grad_norm 3.219...
rank 1 step 0 shard_len 256 averaged_grad_norm 3.219...
rank 2 step 0 shard_len 256 averaged_grad_norm 3.219...
rank 3 step 0 shard_len 256 averaged_grad_norm 3.219...

```

#### If you see output from all ranks, this means your MPI+CUDA environment is configured correctly, and you can observe the core mechanics behind PyTorch DDP/NCCL.

---

## Summary

#### In the simulation dicussed above, each CUDA context behaves like a “virtual GPU”.

```python
 ┌─────────────┐     ┌────────────────┐     ┌──────────────┐
 │ MPI Rank 0  │ → → │ CUDA Context 0 │ → → │ Model Shard 0│
 ├─────────────┤     ├────────────────┤     └──────────────┘
 │ MPI Rank 1  │ → → │ CUDA Context 1 │ → → │ Model Shard 1│
 ├─────────────┤     ├────────────────┤     └──────────────┘
 │ MPI Rank 2  │ → → │ CUDA Context 2 │ → → │ Model Shard 2│
 ├─────────────┤     ├────────────────┤     └──────────────┘
 │ MPI Rank 3  │ → → │ CUDA Context 3 │ → → │ Model Shard 3│
 └─────────────┘     └────────────────┘

          (All insyances run on physical GPU 0)

```

#### This project simulates the core mechanics used in:

- PyTorch DDP

- TensorFlow MirroredStrategy

- Horovod

- NCCL AllReduce

#### By working through this program, using *one* GPU, we have simulated:

✔ How real distributed GPU training works

✔ How gradients flow between ranks

✔ How parallel GPU workers synchronize

✔ How to scale beyond a single GPU

#### To fully understand how data a partitioned across multiple GPUs when huge models such as OpenAI ChatGPT, DeepSeek, Horovod, Pytorch are trained, please visit:  https://github.com/manuelbomi/4.-Recreate-Distributed-Data-Parallel-DDP-from-Scratch-with-HPC-Slurm-MPI-CUDA-Rocky-Linux-and-GPU-

---




#### Thank you for Reading

![Octocat](https://myoctocat.com/assets/images/base-octocat.svg)



### **AUTHOR'S BACKGROUND**
### Author's Name:  Emmanuel Oyekanlu
```
Skillset:   I have experience spanning several years in data science, developing scalable enterprise data pipelines,
enterprise solution architecture, architecting enterprise systems data and AI applications, smart manufacturing for GMP,
semiconductor design and testing, software and AI solution design and deployments, data engineering, high performance computing
(GPU, CUDA), machine learning, NLP, Agentic-AI and LLM applications as well as deploying scalable solutions (apps) on-prem and in the cloud.

I can be reached through: manuelbomi@yahoo.com

Website:  http://emmanueloyekanlu.com/
Publications:  https://scholar.google.com/citations?user=S-jTMfkAAAAJ&hl=en
LinkedIn:  https://www.linkedin.com/in/emmanuel-oyekanlu-6ba98616
Github:  https://github.com/manuelbomi

```
[![Icons](https://skillicons.dev/icons?i=aws,azure,gcp,scala,mongodb,redis,cassandra,kafka,anaconda,matlab,nodejs,django,py,c,anaconda,git,github,mysql,docker,kubernetes&theme=dark)](https://skillicons.dev)



































