# Homework 12: GPU Programming Analysis
# Mingxu Zhang

using CUDA
using BenchmarkTools

println("=" ^ 60)
println("Homework 12: CUDA GPU Programming Analysis")
println("=" ^ 60)

# Check GPU availability
println("\n📊 GPU Information:")
println("-" ^ 40)
if CUDA.functional()
    println("CUDA is functional: ✓")
    println("GPU Device: ", CUDA.name(CUDA.device()))
    println("Compute Capability: ", CUDA.capability(CUDA.device()))
    println("Total Memory: ", round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=2), " GB")
else
    println("CUDA is not functional!")
    exit(1)
end

#=============================================================================
# Problem 1: Performance Anti-patterns
=============================================================================#
println("\n" * "=" ^ 60)
println("Problem 1: Performance Anti-patterns")
println("=" ^ 60)

function snippet_a()
    x_cpu = randn(10000)
    for i in 1:100
        x_gpu = CuArray(x_cpu)    # Upload
        x_gpu .+= 1
        x_cpu = Array(x_gpu)       # Download
    end
    return x_cpu
end

function snippet_b()
    x_gpu = CuArray(randn(10000))  # Upload once
    for i in 1:100
        x_gpu .+= 1                # All on GPU
    end
    result = Array(x_gpu)          # Download once
    return result
end

# Warm up
CUDA.@sync snippet_a()
CUDA.@sync snippet_b()

# Benchmark
println("\n🔬 Benchmarking Snippet A (repeated transfers)...")
time_a = CUDA.@elapsed for _ in 1:10
    CUDA.@sync snippet_a()
end
time_a_avg = time_a / 10 * 1000  # ms

println("🔬 Benchmarking Snippet B (minimal transfers)...")
time_b = CUDA.@elapsed for _ in 1:10
    CUDA.@sync snippet_b()
end
time_b_avg = time_b / 10 * 1000  # ms

println("\n📈 Results:")
println("-" ^ 40)
println("Snippet A (inefficient): $(round(time_a_avg, digits=3)) ms")
println("Snippet B (efficient):   $(round(time_b_avg, digits=3)) ms")
println("Speedup: $(round(time_a_avg / time_b_avg, digits=1))x")

#=============================================================================
# Problem 2: Kernel Divergence and Memory Access
=============================================================================#
println("\n" * "=" ^ 60)
println("Problem 2: Kernel Divergence and Memory Access")
println("=" ^ 60)

# Kernel A: Divergent kernel
function divergent_kernel!(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        if i % 2 == 0
            A[i] = sin(A[i])    # Half of warp
        else
            A[i] = cos(A[i])    # Other half
        end
    end
    return nothing
end

# Non-divergent version
function non_divergent_kernel!(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        # All threads do the same computation
        A[i] = sin(A[i]) * cos(A[i])
    end
    return nothing
end

# Kernel B: Bad memory access (strided)
function strided_memory_kernel!(B, A, stride)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(B) && (i-1) * stride + 1 <= length(A)
        @inbounds B[i] = A[(i-1) * stride + 1]
    end
    return nothing
end

# Good memory access (coalesced)
function coalesced_memory_kernel!(B, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds B[i] = A[i]
    end
    return nothing
end

# Test divergent vs non-divergent
N = 1024 * 1024
threads = 256
blocks = cld(N, threads)

println("\n🔬 Testing Warp Divergence...")
A_div = CUDA.rand(Float32, N)
A_nodiv = CUDA.rand(Float32, N)

# Warm up
CUDA.@sync @cuda threads=threads blocks=blocks divergent_kernel!(A_div)
CUDA.@sync @cuda threads=threads blocks=blocks non_divergent_kernel!(A_nodiv)

# Benchmark divergent kernel
time_div = CUDA.@elapsed for _ in 1:100
    CUDA.@sync @cuda threads=threads blocks=blocks divergent_kernel!(A_div)
end

# Benchmark non-divergent kernel
time_nodiv = CUDA.@elapsed for _ in 1:100
    CUDA.@sync @cuda threads=threads blocks=blocks non_divergent_kernel!(A_nodiv)
end

println("Divergent kernel:     $(round(time_div * 1000, digits=3)) ms (100 runs)")
println("Non-divergent kernel: $(round(time_nodiv * 1000, digits=3)) ms (100 runs)")
println("Divergence overhead:  $(round(time_div / time_nodiv, digits=2))x")

# Test memory access patterns
println("\n🔬 Testing Memory Access Patterns...")
N_mem = 1024 * 256
stride = 32

A_mem = CUDA.rand(Float32, N_mem * stride)
B_strided = CUDA.zeros(Float32, N_mem)
B_coalesced = CUDA.zeros(Float32, N_mem)
A_coal = CUDA.rand(Float32, N_mem)

threads_mem = 256
blocks_mem = cld(N_mem, threads_mem)

# Warm up
CUDA.@sync @cuda threads=threads_mem blocks=blocks_mem strided_memory_kernel!(B_strided, A_mem, stride)
CUDA.@sync @cuda threads=threads_mem blocks=blocks_mem coalesced_memory_kernel!(B_coalesced, A_coal)

# Benchmark strided access
time_strided = CUDA.@elapsed for _ in 1:100
    CUDA.@sync @cuda threads=threads_mem blocks=blocks_mem strided_memory_kernel!(B_strided, A_mem, stride)
end

# Benchmark coalesced access
time_coalesced = CUDA.@elapsed for _ in 1:100
    CUDA.@sync @cuda threads=threads_mem blocks=blocks_mem coalesced_memory_kernel!(B_coalesced, A_coal)
end

println("Strided access (stride=$stride):  $(round(time_strided * 1000, digits=3)) ms (100 runs)")
println("Coalesced access:                 $(round(time_coalesced * 1000, digits=3)) ms (100 runs)")
println("Strided overhead:                 $(round(time_strided / time_coalesced, digits=2))x slower")

#=============================================================================
# Problem 3: Broadcasting vs Libraries
=============================================================================#
println("\n" * "=" ^ 60)
println("Problem 3: Broadcasting vs Libraries")
println("=" ^ 60)

# Approach A: Multiple operations (no fusion)
function approach_a(x)
    y = x .^ 2
    z = sin.(y)
    w = z .+ 1
    return w
end

# Approach B: Fused operation
function approach_b(x)
    w = @. sin(x^2) + 1
    return w
end

# Test broadcasting fusion
println("\n🔬 Testing Kernel Fusion (Broadcasting)...")
x_test = CUDA.randn(Float32, 100000)

# Warm up
CUDA.@sync approach_a(x_test)
CUDA.@sync approach_b(x_test)

# Benchmark Approach A
time_approach_a = CUDA.@elapsed for _ in 1:100
    CUDA.@sync approach_a(x_test)
end

# Benchmark Approach B
time_approach_b = CUDA.@elapsed for _ in 1:100
    CUDA.@sync approach_b(x_test)
end

println("Approach A (3 separate operations): $(round(time_approach_a * 1000, digits=3)) ms (100 runs)")
println("Approach B (fused @. macro):        $(round(time_approach_b * 1000, digits=3)) ms (100 runs)")
println("Fusion speedup:                     $(round(time_approach_a / time_approach_b, digits=2))x")

# Verify correctness
x_verify = CUDA.randn(Float32, 1000)
result_a = approach_a(x_verify)
result_b = approach_b(x_verify)
println("Results match: $(isapprox(Array(result_a), Array(result_b), rtol=1e-5))")

# Approach C: CUBLAS matrix multiplication
println("\n🔬 Testing CUBLAS Matrix Multiplication...")
sizes = [500, 1000, 2000]

for n in sizes
    A = CUDA.randn(Float32, n, n)
    B = CUDA.randn(Float32, n, n)
    
    # Warm up
    CUDA.@sync C = A * B
    
    # Benchmark
    time_matmul = CUDA.@elapsed for _ in 1:10
        CUDA.@sync C = A * B
    end
    time_matmul_avg = time_matmul / 10 * 1000  # ms
    
    # Calculate GFLOPS (2*n^3 operations for matrix multiply)
    flops = 2.0 * n^3
    gflops = flops / (time_matmul_avg / 1000) / 1e9
    
    println("Matrix size $(n)x$(n): $(round(time_matmul_avg, digits=3)) ms, $(round(gflops, digits=1)) GFLOPS")
end

#=============================================================================
# Summary
=============================================================================#
println("\n" * "=" ^ 60)
println("Summary of Results")
println("=" ^ 60)

println("""
┌─────────────────────────────────────────────────────────────┐
│ Problem 1: Data Transfer Overhead                           │
├─────────────────────────────────────────────────────────────┤
│ • Minimizing CPU-GPU transfers provides ~$(round(time_a_avg / time_b_avg, digits=0))x speedup       │
│ • Golden Rule: Move data once, compute many times           │
├─────────────────────────────────────────────────────────────┤
│ Problem 2: Kernel Optimization                              │
├─────────────────────────────────────────────────────────────┤
│ • Warp divergence overhead: ~$(round(time_div / time_nodiv, digits=1))x                           │
│ • Strided memory access overhead: ~$(round(time_strided / time_coalesced, digits=1))x                   │
│ • Use coalesced access and avoid branch divergence          │
├─────────────────────────────────────────────────────────────┤
│ Problem 3: Kernel Fusion & Libraries                        │
├─────────────────────────────────────────────────────────────┤
│ • Broadcasting fusion speedup: ~$(round(time_approach_a / time_approach_b, digits=1))x                     │
│ • Use @. macro to fuse operations automatically             │
│ • CUBLAS provides highly optimized matrix operations        │
└─────────────────────────────────────────────────────────────┘
""")

println("\n✅ All experiments completed successfully!")
