# Homework 12 Solutions
# Author: tengxianglin
#
# Run with: julia --project=hw12 hw12/tengxianglin/hw12.jl

using Printf

println("\n" * "="^70)
println("HOMEWORK 12 - GPU Performance & CUDA.jl Analysis")
println("="^70)

# ============================================================================
# Problem 1: Performance Anti-patterns
# ============================================================================
println("\nProblem 1: Code Analysis - Performance Anti-patterns")
println("="^70)

println("\n1.1 Which code will run faster and by approximately how much?")
println("-"^70)
println("Answer: Snippet B will be much faster (approximately 10-100× faster).")
println("\nReason:")
println("  - Snippet A: 100 uploads + 100 downloads over PCIe")
println("  - Snippet B: 1 upload + 1 download + 100 GPU-internal operations")
println("  - PCIe transfers have fixed overheads that accumulate")

println("\n1.2 Identify the performance problem in Snippet A")
println("-"^70)
println("Answer: Excessive CPU↔GPU data transfers (host-device transfers).")
println("  - Each small computation requires upload and download")
println("  - Performance dominated by high-latency, lower-bandwidth PCIe bus")
println("  - GPU compute power stays underutilized")

println("\n1.3 What is the 'golden rule'?")
println("-"^70)
println("Answer: Keep data on GPU as long as possible, minimize host-device transfers.")
println("  - Upload once, do as much work as possible on GPU, download once")
println("  - Snippet B follows this rule; Snippet A violates it")

println("\n1.4 Estimate bandwidth difference")
println("-"^70)
println("Answer:")
println("  - GPU device memory bandwidth: ~300-1000 GB/s (GDDR6X/HBM)")
println("  - PCIe bandwidth: ~16-32 GB/s (PCIe 3.0/4.0 x16)")
println("  - Ratio: GPU memory bandwidth is ~10× higher than PCIe")
println("  - Including latency overhead, penalty is even worse")

# ============================================================================
# Problem 2: Kernel Divergence and Memory Access
# ============================================================================
println("\n\n" * "="^70)
println("Problem 2: Kernel Divergence and Memory Access")
println("="^70)

println("\n2.1 What is wrong with Kernel A in terms of warp execution?")
println("-"^70)
println("Answer: Warp divergence - threads in same warp take different branches.")
println("  - Half threads execute sin branch, half execute cos branch")
println("  - Warp must serialize execution of both branches")
println("  - Performance penalty: approximately 2× slower")
println("  - Only half of threads active at any given time")

println("\n2.2 Explain SIMT and how Kernel A violates it")
println("-"^70)
println("Answer:")
println("  - SIMT (Single Instruction, Multiple Threads):")
println("    * All threads in a warp execute same instruction simultaneously")
println("    * Ideal: all threads follow same control flow, differ only in data")
println("  - Kernel A violation:")
println("    * Different threads take different branches (sin vs cos)")
println("    * Hardware must mask threads and serialize branch execution")
println("    * Breaks SIMT ideal of 'one instruction stream per warp'")

println("\n2.3 What memory access problem does Kernel B have?")
println("-"^70)
println("Answer: Non-coalesced (strided) memory access.")
println("  - Threads access A[i*stride] with stride > 1")
println("  - Adjacent threads access non-contiguous addresses")
println("  - GPU cannot coalesce accesses into large memory transactions")
println("  - Results in poor effective memory bandwidth")
println("\nCoalesced memory access:")
println("  - Threads in warp access consecutive/near-consecutive addresses")
println("  - Hardware can merge requests into few wide transactions")
println("  - Key to achieving peak global memory bandwidth")

println("\n2.4 Rewrite thread ID calculation")
println("-"^70)
println("More readable version:")
println("""
    function bad_memory_kernel(A, B, stride)
        # blockIdx().x      - which block (along x), 1-based
        # blockDim().x      - number of threads per block (along x)
        # threadIdx().x     - index of thread within block, 1-based
        
        global_block_id   = blockIdx().x - 1      # 0-based block index
        threads_per_block = blockDim().x           # threads in each block
        local_thread_id   = threadIdx().x          # 1-based index within block
        
        i = global_block_id * threads_per_block + local_thread_id
        
        @inbounds if i <= length(A)
            B[i] = A[i * stride]
        end
        return nothing
    end
""")

# ============================================================================
# Problem 3: Broadcasting vs Libraries
# ============================================================================
println("\n\n" * "="^70)
println("Problem 3: Broadcasting vs Libraries")
println("="^70)

println("\n3.1 Which approach (A or B) will be faster?")
println("-"^70)
println("Answer: Approach B will be faster.")
println("\nKernel launches:")
println("  - Approach A: 3 kernel launches (x.^2, sin.(y), z.+1)")
println("  - Approach B: 1 kernel launch (fused @. sin(x^2) + 1)")
println("\nWhy B is faster:")
println("  - Fewer kernel launch overheads (1 vs 3)")
println("  - Single kernel performs all operations in registers")
println("  - No intermediate global memory writes/reads")
println("  - Better cache and register utilization")

println("\n3.2 What is 'kernel fusion'?")
println("-"^70)
println("Answer:")
println("  - Kernel fusion: merging multiple operations into one larger kernel")
println("  - Benefits:")
println("    * Fewer kernel launches")
println("    * Fewer intermediate global memory operations")
println("    * Better cache and register utilization")
println("\nHow broadcasting achieves fusion:")
println("  - Broadcast expressions like @. sin(x^2) + 1 compile to single kernel")
println("  - Kernel performs: read x[i] → compute x[i]^2 → sin → +1 → write")
println("  - No intermediate arrays, automatic kernel fusion")

println("\n3.3 Why is Approach C (CUBLAS) much faster?")
println("-"^70)
println("Answer: CUBLAS is highly optimized NVIDIA BLAS library.")
println("  - Sophisticated tiling and caching strategies")
println("  - Architecture-specific SIMT/vectorization tuning")
println("  - Uses Tensor Cores and specialized hardware")
println("  - Aggressive loop unrolling, prefetching, register allocation")
println("  - Hand-optimized for various matrix shapes and data types")
println("\nNaive custom kernels typically:")
println("  - Simple nested loops with basic thread mapping")
println("  - Poor shared memory/cache utilization")
println("  - Non-coalesced memory access patterns")
println("  - Don't exploit instruction pipelines or specialized hardware")
println("\nResult: CUBLAS is orders of magnitude faster than naive kernels")

println("\n3.4 Choose between CUBLAS, CUSPARSE, CUFFT")
println("-"^70)
println("(a) 2D image filtering:")
println("  → CUFFT (Fast Fourier Transform)")
println("  - Many filters use FFT: transform → multiply → inverse transform")
println("\n(b) Graph algorithms on sparse matrices:")
println("  → CUSPARSE")
println("  - Graph adjacency matrices are typically very sparse")
println("  - CUSPARSE optimized for sparse matrix operations")
println("\n(c) Dense neural network training:")
println("  → CUBLAS")
println("  - Core operation is large-scale dense matrix multiplication (GEMM)")
println("  - Often via cuDNN or deep learning frameworks that use CUBLAS")

# ============================================================================
# Summary
# ============================================================================
println("\n\n" * "="^70)
println("Summary")
println("="^70)
println("✓ Problem 1: Performance anti-patterns analyzed")
println("✓ Problem 2: Kernel divergence and memory access issues explained")
println("✓ Problem 3: Broadcasting fusion and library selection discussed")
println("="^70)

