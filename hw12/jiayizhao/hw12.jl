using CUDA
using BenchmarkTools
using LinearAlgebra
using Statistics

println("=== Homework 12 Execution Started ===")

# ==============================================================================
# Problem 1: Performance Anti-patterns
# ==============================================================================
println("\n--- Problem 1: Host-Device Transfers Analysis ---")

function snippet_A()
    x_cpu = randn(10000)
    # Anti-pattern: Allocation & Transfer inside loop
    for i in 1:100
        x_gpu = CuArray(x_cpu)
        x_gpu .+= 1
        x_cpu = Array(x_gpu)
    end
    return x_cpu
end

function snippet_B()
    # Golden Rule: Transfer once
    x_gpu = CuArray(randn(10000))
    for i in 1:100
        x_gpu .+= 1
    end
    result = Array(x_gpu)
    return result
end

# Benchmarking
println("Benchmarking Snippet A (Inefficient)...")
@btime snippet_A()
println("Benchmarking Snippet B (Efficient)...")
@btime snippet_B()

# ================= ANALYSIS (Answers) =================
#=
Q1: Speed Difference
Snippet B is expected to be roughly 20x-50x faster than Snippet A.
(Run the code to see exact numbers. My prediction: A ~ ms range, B ~ μs range).

Q2: Performance Problem in A
- PCIe Bottleneck: Transferring data between Host and Device is slow (limited by PCIe bandwidth ~16GB/s).
- Allocation Overhead: Creating `CuArray` inside the loop triggers repeated `cudaMalloc` and Garbage Collection (GC), which adds significant latency.
- Low Arithmetic Intensity: The code spends >99% time moving data and <1% computing.

Q3: The "Golden Rule"
"Data Locality": Keep data on the GPU as long as possible. Upload once, compute many times, download once. Snippet A violates this by moving data every iteration.

Q4: Bandwidth Comparison
- PCIe 4.0: ~16 GB/s
- GPU Memory (HBM/GDDR): ~900 GB/s (on high-end GPUs)
The GPU memory is approx 50x faster than the PCIe bus.
=#


# ==============================================================================
# Problem 2: Kernel Divergence and Memory Access
# ==============================================================================
println("\n--- Problem 2: Kernel Analysis (Theoretical) ---")

# ================= ANALYSIS (Answers) =================
#=
Q1: Warp Divergence in Kernel A
- The `if i % 2 == 0` condition causes threads within the same Warp (32 threads) to diverge.
- Half the warp wants to do `sin`, the other half `cos`. 
- The GPU serializes this: it runs the `sin` threads while others wait, then runs `cos`.
- Performance impact: ~2x slowdown (50% warp utilization).

Q2: SIMT Execution
- SIMT (Single Instruction, Multiple Threads) means all threads in a warp must execute the SAME instruction at the same time.
- Kernel A violates this by requiring different instructions for even/odd threads, forcing serialization.

Q3: Memory Coalescing in Kernel B
- The problem is "Strided Access" (`A[i * stride]`).
- If stride > 1, threads read non-consecutive memory addresses.
- The GPU cannot "coalesce" (combine) these reads into a single transaction.
- Instead of 1 transaction for 32 threads, it might issue 32 separate transactions.
- Effective bandwidth drops significantly (can be <5% of peak).

Q4: Readable Thread ID
Detailed calculation:
i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
     |__________________| |__________|   |__________|
        Global offset      Block Size     Local ID
     (Threads in prior    (Threads per    (Position in
          blocks)            block)       current block)
=#


# ==============================================================================
# Problem 3: Broadcasting vs Libraries
# ==============================================================================
println("\n--- Problem 3: Broadcasting vs Libraries Analysis ---")

function approach_A()
    x = CUDA.randn(10000)
    # Multiple kernels, intermediate allocations
    y = x .^ 2
    z = sin.(y)
    w = z .+ 1
    CUDA.synchronize()
end

function approach_B()
    x = CUDA.randn(10000)
    # Kernel Fusion via broadcasting
    w = @. sin(x^2) + 1
    CUDA.synchronize()
end

function approach_C()
    A = CUDA.randn(2000, 2000)
    B = CUDA.randn(2000, 2000)
    # CUBLAS GEMM
    C = A * B
    CUDA.synchronize()
end

println("Benchmarking Approach A (Multiple Kernels)...")
@btime approach_A()
println("Benchmarking Approach B (Fused)...")
@btime approach_B()

println("Benchmarking Approach C (CUBLAS Matrix Mul)...")
# Note: Matrix mul is O(N^3), just running to show it works. 
# A/B comparisons are O(N), so C will take longer naturally, 
# but we analyze why it's faster than a *custom* loop implementation.
@btime approach_C()


# --- Problem 1: Host-Device Transfers Analysis ---
# Benchmarking Snippet A (Inefficient)...
#   2.764 ms (5781 allocations: 7.86 MiB)
# Benchmarking Snippet B (Efficient)...
#   449.407 μs (3622 allocations: 264.59 KiB)

# --- Problem 2: Kernel Analysis (Theoretical) ---

# --- Problem 3: Broadcasting vs Libraries Analysis ---
# Benchmarking Approach A (Multiple Kernels)...
#   34.251 μs (221 allocations: 6.33 KiB)
# Benchmarking Approach B (Fused)...
#   22.351 μs (125 allocations: 3.55 KiB)
# Benchmarking Approach C (CUBLAS Matrix Mul)...
#   467.049 μs (370 allocations: 8.19 KiB)




# ================= ANALYSIS =================
#=
Q1: Approach A vs B speed.
Based on my benchmark results:
- Approach A (Multiple Kernels): ~34.251 μs
- Approach B (Fused):            ~22.351 μs
Approach B is roughly 1.5x faster. 
Approach A launches 3 separate kernels and writes intermediate arrays (y, z) to global memory. Approach B fuses the logic into a single kernel launch.

Q2: Kernel Fusion.
Kernel Fusion combines multiple element-wise operations into a single GPU kernel. Julia's broadcasting (`@.`) achieves this automatically. It reads `x` once, computes `sin(x^2)+1` in registers, and writes `w` once, avoiding the memory bandwidth cost of reading/writing temporary arrays.

Q3: Why CUBLAS (Approach C) is much faster.
CUBLAS is a vendor-optimized library. It uses:
1. Assembly-level tuning specific to the GPU architecture.
2. Sophisticated Tiling to optimize L1/L2 cache and Shared Memory usage.
3. Tensor Cores (on Volta/Ampere+ GPUs) which provide specialized hardware acceleration for matrix multiplication, far exceeding the capability of standard CUDA cores.

Q4: Library Selection.
(a) 2D Image Filtering: CUFFT (or NPP). FFT reduces convolution complexity from O(N^2) to O(N log N).
(b) Graph Algorithms (Sparse): CUSPARSE. Optimized for CSR/CSC formats to handle irregular memory access.
(c) Dense NN Training: CUBLAS (or cuDNN). Neural networks rely heavily on GEMM (General Matrix Multiply), which CUBLAS optimizes perfectly.
=#