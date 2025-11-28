# Homework 12 - CUDA GPU Programming Analysis

**Author:** Mingxu Zhang  
**Date:** November 28, 2025

---

## Environment Notes

**System Configuration:**
- GPU: NVIDIA GeForce RTX 3090 (24GB) × 3
- Driver Version: 515.65.01
- System CUDA Version: 11.7
- Julia Version: 1.12.0

**Compatibility Issue:**

Due to version incompatibility between Julia 1.12.0 and the system's CUDA 11.7, CUDA code cannot be executed on this system.

**Issue Details:**
1. The latest CUDA.jl (v5.9.5) requires CUDA 12.x or higher
2. When attempting to install older CUDA.jl (v5.2.0), dependency conflicts with GPUCompiler.jl prevent installation on Julia 1.12

```
# Error 1: CUDA.jl v5.9.5 requires CUDA 12.x
┌ Error: This version of CUDA.jl requires an NVIDIA driver for CUDA 12.x or higher 
│        (yours only supports up to CUDA 11.7.0)
└ @ CUDA ~/.julia/packages/CUDA/x8d2s/src/initialization.jl:70

# Error 2: Dependency conflict when trying to install older version
ERROR: Unsatisfiable requirements detected for package GPUCompiler [61eb1bfa]:
 GPUCompiler [61eb1bfa] log:
 ├─possible versions are: 0.1.0 - 1.7.5 or uninstalled
 ├─restricted by julia compatibility requirements to versions: 1.6.0 - 1.7.5 or uninstalled
 └─restricted by compatibility requirements with CUDA [052768ef] to versions: 0.24.0 - 0.25.0 — no versions left
```

**Possible Solutions (not implemented):**
- Upgrade system CUDA to version 12.x (requires administrator privileges)
- Or use Julia 1.10.x (compatible with CUDA.jl 5.2.0)

**Note on This Assignment:**
Since this is a **code analysis** assignment, the focus is on understanding GPU programming concepts, including:
- CPU-GPU data transfer overhead
- Warp Divergence
- Memory Coalescing
- Kernel Fusion
- GPU library usage

The analysis below is based on theoretical principles of GPU architecture and CUDA programming, which are hardware-independent and applicable to all NVIDIA GPUs.

---

## Problem 1: Performance Anti-patterns

### Code Analysis

**Snippet A (Inefficient):**
```julia
x_cpu = randn(10000)
for i in 1:100
    x_gpu = CuArray(x_cpu)    # Upload
    x_gpu .+= 1
    x_cpu = Array(x_gpu)       # Download
end
```

**Snippet B (Efficient):**
```julia
x_gpu = CuArray(randn(10000))  # Upload once
for i in 1:100
    x_gpu .+= 1                # All on GPU
end
result = Array(x_gpu)          # Download once
```

### Answers

#### Q1: Which code will run faster and by approximately how much?

**Snippet B will be significantly faster, approximately 50-200x faster than Snippet A.**

**Reasoning:**
- Snippet A performs 200 PCIe transfers (100 uploads + 100 downloads)
- Snippet B performs only 2 PCIe transfers (1 upload + 1 download)

**Time estimation:**
- Array size: 10,000 × 8 bytes = 80 KB
- PCIe 3.0 bandwidth: ~12 GB/s (practical), latency ~1-10 μs
- Each transfer: ~80KB / 12GB/s + latency ≈ 7-20 μs
- Snippet A transfers: 200 × 20μs = 4,000 μs = 4 ms
- Snippet B transfers: 2 × 20μs = 40 μs
- GPU computation (100 additions): ~10-50 μs (negligible)

**Speedup: ~100x** (transfer-dominated workload)

#### Q2: What is the performance problem in Snippet A?

**The problem is repeated CPU-GPU data transfers inside a loop.**

This occurs because:
1. **PCIe bus is the bottleneck**: Data must traverse the PCIe bus between CPU and GPU memory, which is orders of magnitude slower than GPU memory bandwidth
2. **Transfer overhead**: Each `CuArray()` and `Array()` call incurs:
   - Memory allocation overhead
   - DMA (Direct Memory Access) setup
   - Synchronization overhead
   - PCIe latency (~1-10 μs per transfer)
3. **The actual computation (`x .+= 1`) takes nanoseconds**, but transfers take microseconds to milliseconds

#### Q3: What is the "golden rule" that Snippet B follows but Snippet A violates?

**The Golden Rule: "Minimize data transfers between CPU and GPU"**

More specifically:
> **"Move data to the GPU once, perform all computations there, and only transfer results back when finished."**

This principle is also known as:
- **"Keep data on the device"**
- **"Maximize arithmetic intensity"** (compute operations per byte transferred)
- **"Amortize transfer costs over many operations"**

**Snippet B follows this rule by:**
- Uploading once at the beginning
- Performing all 100 iterations on GPU memory
- Downloading once at the end

**Snippet A violates this rule by:**
- Transferring data 200 times for a simple operation that could be done entirely on GPU

#### Q4: Estimate the bandwidth difference between PCIe transfer and GPU memory operations

| Memory Type | Bandwidth | Relative Speed |
|-------------|-----------|----------------|
| **GPU Global Memory (HBM2)** | 900-2000 GB/s | 1x (baseline) |
| **GPU Shared Memory** | ~12,000 GB/s | 6-12x faster |
| **PCIe 3.0 x16** | ~12-16 GB/s | 60-150x slower |
| **PCIe 4.0 x16** | ~25-32 GB/s | 30-80x slower |
| **PCIe 5.0 x16** | ~50-64 GB/s | 15-40x slower |
| **NVLink** | 300-900 GB/s | Similar to HBM |

**Key insight:** GPU memory bandwidth is **50-150x faster** than PCIe bandwidth. This is why minimizing CPU-GPU transfers is critical for performance.

---

## Problem 2: Kernel Divergence and Memory Access

### Code Analysis

**Kernel A (Warp Divergence):**
```julia
function divergent_kernel(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i % 2 == 0
        A[i] = sin(A[i])    # Half of warp
    else
        A[i] = cos(A[i])    # Other half
    end
    return nothing
end
```

**Kernel B (Non-coalesced Access):**
```julia
function bad_memory_kernel(A, B, stride)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(A)
        B[i] = A[i * stride]  # Non-sequential access!
    end
    return nothing
end
```

### Answers

#### Q1: What is wrong with Kernel A in terms of warp execution? How much could this slow down performance?

**Problem: Warp Divergence (Branch Divergence)**

In CUDA, threads are grouped into **warps** of 32 threads that execute in lockstep (SIMT - Single Instruction, Multiple Thread). When threads in the same warp take different branches:

1. **Both branches must be executed serially**
2. Threads not taking a branch are **masked** (idle but still consume cycles)
3. The warp effectively runs at 50% efficiency

**In Kernel A:**
- Even-indexed threads (0, 2, 4, ...) execute `sin()`
- Odd-indexed threads (1, 3, 5, ...) execute `cos()`
- Within each warp of 32 threads, 16 execute sin() while 16 wait, then 16 execute cos() while 16 wait

**Performance Impact:**
- **Theoretical slowdown: 2x** (both paths executed sequentially)
- Actual slowdown may be less if `sin` and `cos` have similar latencies
- If branches had very different compute costs, slowdown could be worse

**Better approach:**
```julia
# Process even and odd indices in separate kernels or
# reorganize data so consecutive threads take the same branch
function non_divergent_kernel(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # All threads in a warp do the same thing
    A[i] = sin(A[i]) + cos(A[i])  # No divergence
    return nothing
end
```

#### Q2: Explain what SIMT means and how Kernel A violates this execution model

**SIMT = Single Instruction, Multiple Thread**

**SIMT Characteristics:**
1. **32 threads form a warp** - the basic unit of execution
2. **All threads in a warp execute the same instruction simultaneously**
3. Threads can have different data (different registers, memory addresses)
4. **Divergence occurs when threads need different instructions**

**How Kernel A violates SIMT:**

```
Warp 0 (threads 0-31):
  Thread 0:  if (0 % 2 == 0) → TRUE  → sin()
  Thread 1:  if (1 % 2 == 0) → FALSE → cos()
  Thread 2:  if (2 % 2 == 0) → TRUE  → sin()
  Thread 3:  if (3 % 2 == 0) → FALSE → cos()
  ...
```

**Execution timeline:**
```
Cycle 1-N:   Execute sin() | Threads 0,2,4,6... active, 1,3,5,7... masked
Cycle N+1-M: Execute cos() | Threads 1,3,5,7... active, 0,2,4,6... masked
```

**The SIMT model expects:** All 32 threads doing the same instruction  
**What happens:** Half do `sin`, half wait; then half do `cos`, half wait

#### Q3: What memory access problem does Kernel B have? What is "coalesced" memory access?

**Problem: Non-coalesced (strided) memory access**

**Coalesced Memory Access:**
- When threads in a warp access **consecutive memory addresses**, the GPU can combine these into a single memory transaction
- Ideal: Thread 0 accesses address N, Thread 1 accesses N+1, Thread 2 accesses N+2, etc.
- The memory controller fetches one large chunk (128 bytes) serving all 32 threads

**In Kernel B:**
```julia
B[i] = A[i * stride]  # If stride = 100
```

**Memory access pattern (stride=100):**
```
Thread 0: A[0]
Thread 1: A[100]
Thread 2: A[200]
Thread 3: A[300]
...
```

**Problems:**
1. **32 threads need 32 separate memory transactions** instead of 1
2. Each 128-byte cache line fetched serves only 1 useful element (8 bytes)
3. **Memory bandwidth utilization: ~6%** (8/128)
4. **Potential slowdown: 10-32x** compared to coalesced access

**Coalesced pattern (ideal):**
```julia
B[i] = A[i]  # Sequential access
# Thread 0: A[0], Thread 1: A[1], Thread 2: A[2]...
# All served by 1-4 memory transactions
```

#### Q4: Rewrite the thread ID calculation to be more readable

**Original:**
```julia
i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
```

**Readable version with explanation:**
```julia
function explained_kernel(A)
    # Block index (which block am I in?) - 1-indexed in Julia
    block_id = blockIdx().x - 1        # 0, 1, 2, ... (0-indexed for math)
    
    # Threads per block (how many threads in each block?)
    threads_per_block = blockDim().x   # e.g., 256
    
    # Thread index within this block - 1-indexed in Julia
    thread_in_block = threadIdx().x    # 1, 2, ..., 256
    
    # Global thread ID (unique across all blocks)
    # = (block_id × threads_per_block) + thread_in_block
    global_id = block_id * threads_per_block + thread_in_block
    
    # Example with 256 threads/block:
    # Block 0, Thread 1  → 0 × 256 + 1 = 1
    # Block 0, Thread 256 → 0 × 256 + 256 = 256
    # Block 1, Thread 1  → 1 × 256 + 1 = 257
    # Block 1, Thread 256 → 1 × 256 + 256 = 512
    
    @inbounds if global_id <= length(A)
        A[global_id] = A[global_id] * 2
    end
    return nothing
end
```

**Visual representation:**
```
Grid of Blocks:
┌─────────────┬─────────────┬─────────────┐
│  Block 0    │  Block 1    │  Block 2    │
│ Threads 1-256│ Threads 257-512│ Threads 513-768│
└─────────────┴─────────────┴─────────────┘

Global ID = Block × BlockSize + LocalThread
```

---

## Problem 3: Broadcasting vs Libraries

### Code Analysis

**Approach A (Multiple operations):**
```julia
x = CUDA.randn(10000)
y = x .^ 2
z = sin.(y)
w = z .+ 1
```

**Approach B (Fused operation):**
```julia
x = CUDA.randn(10000)
w = @. sin(x^2) + 1
```

**Approach C (Library function):**
```julia
A = CUDA.randn(2000, 2000)
B = CUDA.randn(2000, 2000)
C = A * B  # Uses CUBLAS
```

### Answers

#### Q1: Which approach (A or B) will be faster and why? How many kernel launches does each require?

**Approach B is faster, approximately 2-3x faster than Approach A.**

**Kernel launches:**

| Approach | Kernel Launches | Memory Operations |
|----------|-----------------|-------------------|
| A | 3 kernels | 6 global memory accesses (3 read + 3 write) |
| B | 1 kernel | 2 global memory accesses (1 read + 1 write) |

**Approach A breakdown:**
```julia
y = x .^ 2      # Kernel 1: Read x, compute, write y (temp array)
z = sin.(y)     # Kernel 2: Read y, compute, write z (temp array)
w = z .+ 1      # Kernel 3: Read z, compute, write w
```
- Creates 2 temporary arrays (y, z)
- 3 kernel launch overheads (~5-10 μs each)
- 6 global memory operations

**Approach B breakdown:**
```julia
w = @. sin(x^2) + 1  # Single fused kernel
```
- No temporary arrays
- 1 kernel launch
- 2 global memory operations (read x, write w)

**Why B is faster:**
1. **Reduced kernel launch overhead** (1 vs 3 launches)
2. **Reduced memory bandwidth** (2 vs 6 memory operations)
3. **Better cache utilization** (data stays in registers)
4. **No temporary array allocation**

#### Q2: What is "kernel fusion" and how does broadcasting achieve it automatically?

**Kernel Fusion:**
> Combining multiple operations into a single GPU kernel to reduce memory transfers and kernel launch overhead.

**Without fusion (3 kernels):**
```
GPU Memory → Registers → Compute x² → GPU Memory (temp1)
GPU Memory → Registers → Compute sin() → GPU Memory (temp2)  
GPU Memory → Registers → Compute +1 → GPU Memory (result)
```

**With fusion (1 kernel):**
```
GPU Memory → Registers → Compute x² → sin() → +1 → GPU Memory (result)
```

**How Julia broadcasting achieves this:**

1. **Lazy evaluation:** The `@.` macro creates a fused broadcast expression
2. **Compiler optimization:** Julia's broadcast machinery recognizes chained operations
3. **Single kernel generation:** CUDA.jl generates one kernel for the entire expression

```julia
# This expression:
w = @. sin(x^2) + 1

# Becomes a single kernel like:
function fused_kernel(w, x)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds w[i] = sin(x[i]^2) + 1
    return nothing
end
```

**Key benefits:**
- Data loaded once, all operations applied, result written once
- Intermediate values stay in fast registers (not slow global memory)
- Reduced memory bandwidth by 3x in this example

#### Q3: Why would Approach C (CUBLAS) be much faster than implementing matrix multiplication with custom kernels?

**CUBLAS can be 10-100x faster than naive custom kernels for several reasons:**

1. **Highly optimized algorithms:**
   - Uses Strassen's algorithm for large matrices
   - Optimal tiling strategies for cache hierarchy
   - Vectorized operations (using Tensor Cores on modern GPUs)

2. **Architecture-specific tuning:**
   - Tuned for each GPU architecture (Volta, Ampere, Hopper)
   - Optimal thread block dimensions
   - Optimal shared memory usage

3. **Memory access optimization:**
   - Perfect coalesced memory access patterns
   - Shared memory tiling to reduce global memory access
   - Register blocking for data reuse

4. **Hardware feature utilization:**
   - **Tensor Cores:** Specialized hardware for matrix operations (8x-16x speedup)
   - Warp-level matrix operations (WMMA)
   - Optimized memory prefetching

5. **Years of expert optimization:**
   - NVIDIA engineers have spent thousands of hours optimizing
   - Benchmarked across millions of matrix sizes
   - Heuristics for choosing optimal algorithms

**Performance comparison (2000×2000 matrix multiply):**

| Implementation | GFLOPS | Relative Speed |
|----------------|--------|----------------|
| Naive triple loop (CPU) | ~2 | 1x |
| Naive GPU kernel | ~50-200 | 25-100x |
| Optimized tiled kernel | ~500-1000 | 250-500x |
| CUBLAS (FP32) | ~5000-15000 | 2500-7500x |
| CUBLAS + Tensor Cores (FP16) | ~50000+ | 25000x+ |

#### Q4: Library Selection for Different Problems

**(a) 2D Image Filtering**
**Answer: CUFFT (cuFFT)**

- 2D image filtering in frequency domain: $G = F \cdot H$ (element-wise multiply of FFTs)
- Convolution theorem: Convolution in spatial domain = multiplication in frequency domain
- Steps: FFT(image) → multiply with FFT(filter) → IFFT
- For large kernels, FFT-based filtering is O(N log N) vs O(N×K) for direct convolution
- Alternative: cuDNN for standard CNN convolutions (optimized for small kernels)

**(b) Graph Algorithms on Sparse Matrices**
**Answer: CUSPARSE**

- Sparse matrices have mostly zero elements (e.g., adjacency matrices)
- CUSPARSE provides:
  - CSR/CSC/COO sparse matrix formats
  - Sparse matrix-vector multiplication (SpMV) - key for PageRank, BFS
  - Sparse matrix-matrix multiplication (SpGEMM)
- Memory efficient: Only stores non-zero elements
- Algorithmic efficiency: O(nnz) instead of O(N²)

**(c) Dense Neural Network Training**
**Answer: CUBLAS (through cuDNN)**

- Neural networks rely heavily on dense matrix operations:
  - Forward pass: Y = XW + b (matrix multiply)
  - Backward pass: Gradient computation (matrix multiply)
  - Weight updates: W -= α∇W (BLAS operations)
- cuDNN (which uses CUBLAS internally) provides:
  - Optimized convolution algorithms
  - Batch normalization
  - Activation functions
  - Tensor Core acceleration
- Most deep learning frameworks (PyTorch, TensorFlow) use cuDNN/CUBLAS

**Summary Table:**

| Problem | Best Library | Why |
|---------|--------------|-----|
| 2D Image Filtering | CUFFT | FFT-based convolution efficient for large filters |
| Sparse Graph Algorithms | CUSPARSE | Efficient sparse matrix operations |
| Dense Neural Networks | CUBLAS/cuDNN | Optimized dense matrix multiply + DL operations |

---

## Summary

### Key Takeaways

1. **Minimize CPU-GPU transfers** - The PCIe bus is 50-150x slower than GPU memory
2. **Avoid warp divergence** - Threads in a warp should execute the same instructions
3. **Use coalesced memory access** - Adjacent threads should access adjacent memory
4. **Fuse operations with broadcasting** - Reduces kernel launches and memory traffic
5. **Use optimized libraries** - CUBLAS, CUSPARSE, CUFFT are highly optimized

### Performance Hierarchy

```
Fastest ──────────────────────────────────────────────── Slowest
Registers → Shared Memory → L2 Cache → Global Memory → PCIe → CPU Memory
   ~TB/s       ~12 TB/s       ~2 TB/s     ~1 TB/s     ~30 GB/s  ~50 GB/s
```

### Golden Rules of GPU Programming

1. **Maximize parallelism** - Use thousands of threads
2. **Minimize data movement** - Keep data on GPU
3. **Optimize memory access** - Coalesced, aligned, cached
4. **Avoid divergence** - Same code path for warp
5. **Use libraries** - Don't reinvent the wheel
