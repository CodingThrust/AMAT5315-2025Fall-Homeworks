# Homework 12: CUDA GPU Programming Demonstration
# Mingxu Zhang
# 
# This code demonstrates the concepts discussed in the homework.
# Note: Requires CUDA.jl and a CUDA-capable GPU to run.

using CUDA

# Check if CUDA is available
if CUDA.functional()
    println("CUDA is available!")
    println("GPU: ", CUDA.name(CUDA.device()))
    println()
else
    println("CUDA is not available. This is a demonstration code.")
    println("The analysis in answer.md does not require running this code.")
    exit(0)
end

println("=" ^ 60)
println("Problem 1: Performance Anti-patterns Demonstration")
println("=" ^ 60)

# Snippet A: Inefficient (repeated transfers)
function snippet_a()
    x_cpu = randn(10000)
    for i in 1:100
        x_gpu = CuArray(x_cpu)
        x_gpu .+= 1
        x_cpu = Array(x_gpu)
    end
    return x_cpu
end

# Snippet B: Efficient (minimal transfers)
function snippet_b()
    x_gpu = CuArray(randn(10000))
    for i in 1:100
        x_gpu .+= 1
    end
    result = Array(x_gpu)
    return result
end

# Warmup
snippet_a()
snippet_b()
CUDA.synchronize()

# Benchmark
println("\nBenchmarking Snippet A (inefficient)...")
time_a = CUDA.@elapsed begin
    for _ in 1:10
        snippet_a()
    end
    CUDA.synchronize()
end
println("Snippet A time: $(time_a/10 * 1000) ms per run")

println("\nBenchmarking Snippet B (efficient)...")
time_b = CUDA.@elapsed begin
    for _ in 1:10
        snippet_b()
    end
    CUDA.synchronize()
end
println("Snippet B time: $(time_b/10 * 1000) ms per run")

println("\nSpeedup: $(round(time_a/time_b, digits=1))x")

println("\n" * "=" ^ 60)
println("Problem 2: Kernel Examples")
println("=" ^ 60)

# Divergent kernel (for demonstration - not optimal)
function divergent_kernel!(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        if i % 2 == 0
            A[i] = sin(A[i])
        else
            A[i] = cos(A[i])
        end
    end
    return nothing
end

# Non-divergent kernel (better)
function non_divergent_kernel!(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        # All threads do the same operation
        A[i] = sin(A[i]) * cos(A[i])
    end
    return nothing
end

# Test divergent kernel
A = CUDA.randn(10000)
threads = 256
blocks = cld(length(A), threads)

println("\nLaunching divergent kernel...")
CUDA.@sync @cuda threads=threads blocks=blocks divergent_kernel!(A)
println("Divergent kernel completed")

println("\nLaunching non-divergent kernel...")
B = CUDA.randn(10000)
CUDA.@sync @cuda threads=threads blocks=blocks non_divergent_kernel!(B)
println("Non-divergent kernel completed")

println("\n" * "=" ^ 60)
println("Problem 3: Broadcasting vs Separate Operations")
println("=" ^ 60)

# Approach A: Multiple operations (3 kernels)
function approach_a(x)
    y = x .^ 2
    z = sin.(y)
    w = z .+ 1
    return w
end

# Approach B: Fused operation (1 kernel)
function approach_b(x)
    w = @. sin(x^2) + 1
    return w
end

x = CUDA.randn(1_000_000)

# Warmup
approach_a(x)
approach_b(x)
CUDA.synchronize()

# Benchmark
println("\nBenchmarking Approach A (separate operations)...")
time_a = CUDA.@elapsed begin
    for _ in 1:100
        approach_a(x)
    end
    CUDA.synchronize()
end
println("Approach A time: $(time_a/100 * 1000) ms per run")

println("\nBenchmarking Approach B (fused operation)...")
time_b = CUDA.@elapsed begin
    for _ in 1:100
        approach_b(x)
    end
    CUDA.synchronize()
end
println("Approach B time: $(time_b/100 * 1000) ms per run")

println("\nSpeedup from fusion: $(round(time_a/time_b, digits=2))x")

# Approach C: CUBLAS matrix multiplication
println("\n" * "-" ^ 40)
println("Matrix Multiplication with CUBLAS")
println("-" ^ 40)

A_mat = CUDA.randn(2000, 2000)
B_mat = CUDA.randn(2000, 2000)

# Warmup
C_mat = A_mat * B_mat
CUDA.synchronize()

# Benchmark
println("\nBenchmarking CUBLAS matrix multiply (2000x2000)...")
time_cublas = CUDA.@elapsed begin
    for _ in 1:10
        C_mat = A_mat * B_mat
    end
    CUDA.synchronize()
end
println("CUBLAS time: $(time_cublas/10 * 1000) ms per multiply")

# Calculate GFLOPS
flops = 2 * 2000^3  # 2N³ for matrix multiply
gflops = flops / (time_cublas/10) / 1e9
println("Performance: $(round(gflops, digits=1)) GFLOPS")

println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
println("✓ Minimizing CPU-GPU transfers provides ~$(round(time_a/time_b, digits=0))x speedup")
println("✓ Kernel fusion provides ~$(round(time_a/time_b, digits=1))x speedup")
println("✓ CUBLAS achieves $(round(gflops, digits=0)) GFLOPS for matrix multiply")
println("=" ^ 60)
