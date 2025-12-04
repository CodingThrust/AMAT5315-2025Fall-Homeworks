# Homework 12

## Problem 1
```
julia> using CUDA

julia> using BenchmarkTools

julia> function Snippet()
           x_cpu = randn(10000)
           for i in 1:100
               x_gpu = CuArray(x_cpu)    # Upload
               x_gpu .+= 1
               x_cpu = Array(x_gpu)       # Download
           end
           return nothing
       end
Snippet (generic function with 1 method)

julia> function Snippet_2()
           x_gpu = CuArray(randn(10000))  # Upload once
           for i in 1:100
               x_gpu .+= 1                # All on GPU
           end
           result = Array(x_gpu)          # Download once
           return nothing
       end
Snippet_2 (generic function with 1 method)

julia> CUDA.@profile Snippet
Profiler ran for 1.67 µs, capturing 2 events.

No host-side activity was recorded.

No device-side activity was recorded.


julia> CUDA.@profile Snippet()
Profiler ran for 123.36 ms, capturing 81803 events.

Host-side activity: calling CUDA APIs took 27.71 ms (22.47% of the trace)
┌──────────┬────────────┬───────┬───────────────────────────────────────┬─────────────────────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                     │ Name                    │
├──────────┼────────────┼───────┼───────────────────────────────────────┼─────────────────────────┤
│   62.95% │   77.65 ms │   200 │ 388.27 µs ± 561.42 (   0.0 ‥ 2123.36) │ cuStreamSynchronize     │
│   16.76% │   20.68 ms │   100 │ 206.77 µs ± 115.01 ( 51.26 ‥ 668.05)  │ cuMemcpyDtoHAsync       │
│    0.97% │     1.2 ms │   100 │  11.99 µs ± 7.49   (  8.82 ‥ 79.15)   │ cuMemcpyHtoDAsync       │
│    0.89% │     1.1 ms │   100 │  11.03 µs ± 7.45   (  6.68 ‥ 77.96)   │ cuLaunchKernel          │
│    0.40% │   489.0 µs │   100 │   4.89 µs ± 3.14   (  1.43 ‥ 15.97)   │ cuMemAllocFromPoolAsync │
└──────────┴────────────┴───────┴───────────────────────────────────────┴─────────────────────────┘

Device-side activity: GPU was busy for 20.45 ms (16.58% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name                                                                                                 ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────
│    7.68% │    9.47 ms │   100 │  94.72 µs ± 43.98  ( 26.23 ‥ 180.01) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tu ⋯
│    5.85% │    7.22 ms │   100 │  72.21 µs ± 92.94  ( 26.94 ‥ 415.56) │ [copy pageable to device memory]                                                                     ⋯
│    3.05% │    3.76 ms │   100 │  37.57 µs ± 36.48  ( 22.89 ‥ 195.03) │ [copy device to pageable memory]                                                                     ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                               1 column omitted


julia> CUDA.@profile Snippet_2()
Profiler ran for 12.75 ms, capturing 3314 events.

Host-side activity: calling CUDA APIs took 11.17 ms (87.67% of the trace)
┌──────────┬────────────┬───────┬─────────────────────────────────────┬─────────────────────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                   │ Name                    │
├──────────┼────────────┼───────┼─────────────────────────────────────┼─────────────────────────┤
│   87.67% │   11.17 ms │     2 │   5.59 ms ± 7.9    (   0.0 ‥ 11.17) │ cuStreamSynchronize     │
│    3.45% │  439.88 µs │   100 │    4.4 µs ± 2.97   (   3.1 ‥ 33.62) │ cuLaunchKernel          │
│    1.88% │  239.85 µs │     1 │                                     │ cuMemcpyDtoHAsync       │
│    0.14% │    17.4 µs │     1 │                                     │ cuMemcpyHtoDAsync       │
│    0.08% │   10.73 µs │     1 │                                     │ cuMemAllocFromPoolAsync │
└──────────┴────────────┴───────┴─────────────────────────────────────┴─────────────────────────┘

Device-side activity: GPU was busy for 2.75 ms (21.60% of the trace)
┌──────────┬────────────┬───────┬─────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Time distribution                   │ Name                                                                                                  ⋯
├──────────┼────────────┼───────┼─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────
│   21.19% │     2.7 ms │   100 │  27.01 µs ± 4.11   ( 26.23 ‥ 67.71) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tup ⋯
│    0.22% │   28.13 µs │     1 │                                     │ [copy pageable to device memory]                                                                      ⋯
│    0.18% │   23.13 µs │     1 │                                     │ [copy device to pageable memory]                                                                      ⋯
└──────────┴────────────┴───────┴─────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                               1 column omitted
                                                                                                                                                
```
Snippet A: 123.36ms.
Snippet B: 12.75ms.

It is clearly to see Snippet B is faster. The problem of Snippet A is that it needs unnecessary CPU - GPU data transferring. The "golden rule" that Snippet B follows but Snippet A violate is:  Minimize CPU-GPU transfers! Keep data on GPU as long as possible. 
Snippet B is 10 times faster than A, so the bandwidth difference between PCIe transfer and GPU memory operations is about 10 times.

## Problem2

Kernel A problem:

Threads in the same warp take different paths (sin vs. cos), forcing the GPU to serialize execution, it violates SIMT, which means Single Instruction. Threads are grouped into  “warps” (typically 32 threads) that execute the same instruction at the same time. 

Kernel B problem:

Accessing global memory with a stride (A[i * stride]) means threads access non-contiguous data locations. Requires many more memory transactions, drastically reducing memory bandwidth.

Fix:

```
function coalesced_sin_cos_kernel(A)
    # Calculate the 1-based global index 'i'
    # Assuming CUDA.jl conventions where blockIdx().x and threadIdx().x are 1-based.
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Check if the current Block's index is odd or even
    # This check is performed once per block (not per thread)
    is_even_block = (blockIdx().x % 2 == 0)

    @inbounds if i <= length(A)
        if is_even_block
            # Even Blocks (all threads) execute sin
            A[i] = sin(A[i])
        else
            # Odd Blocks (all threads) execute cos
            A[i] = cos(A[i])
        end
    end
    return nothing
end

```
Rewrite the thread ID calculation to be more readable, explaining each component:
```
 global_id = (block_id - 1) × threads_per_block + local_thread_id

```

blockIdx().x - The Block's Position in the grid (e.g., Block 1, 2, 3...).

blockDim().x - The Size of the Block (how many threads are in the block).

threadIdx().x - The Thread's Position within its block (e.g., Thread 1, 2, 3...).

## Problem 3

Approach B is faster.

REPL

```
julia> function Approach_A()
           x = CUDA.randn(10000)
           y = x .^ 2
           z = sin.(y)
           w = z .+ 1
       end
Approach_A (generic function with 1 method)

julia> function Approach_B()
           x = CUDA.randn(10000)
           w = @. sin(x^2) + 1
       end
Approach_B (generic function with 1 method)

julia> CUDA.@profile Approach_A()
Profiler ran for 1.55 s, capturing 901 events.

Host-side activity: calling CUDA APIs took 66.3 ms (4.27% of the trace)
┌──────────┬────────────┬───────┬───────────────────────────────────────┬─────────────────────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                     │ Name                    │
├──────────┼────────────┼───────┼───────────────────────────────────────┼─────────────────────────┤
│    2.13% │   33.05 ms │     2 │  16.53 ms ± 6.96   ( 11.61 ‥ 21.45)   │ cudaLaunchKernel        │
│    1.59% │   24.66 ms │     4 │   6.17 ms ± 10.15  (  1.03 ‥ 21.39)   │ cuModuleLoadDataEx      │
│    0.27% │     4.2 ms │     1 │                                       │ cudaDeviceSynchronize   │
│    0.12% │    1.88 ms │     1 │                                       │ cudaMalloc              │
│    0.03% │  463.25 µs │     1 │                                       │ cudaFree                │
│    0.03% │  436.54 µs │     4 │ 109.14 µs ± 57.45  ( 53.88 ‥ 189.07)  │ cuLaunchKernel          │
│    0.02% │  296.35 µs │     4 │  74.09 µs ± 46.38  ( 42.92 ‥ 141.86)  │ cuModuleGetFunction     │
│    0.02% │  233.89 µs │     1 │                                       │ cudaGetDevice           │
│    0.01% │  127.79 µs │     1 │                                       │ cuMemcpyDtoDAsync       │
│    0.01% │   81.78 µs │     6 │  13.63 µs ± 7.23   (  3.81 ‥ 26.23)   │ cuMemAllocFromPoolAsync │
│    0.00% │   40.29 µs │     4 │  10.07 µs ± 2.27   (  8.11 ‥ 13.35)   │ cuCtxSynchronize        │
│    0.00% │   36.95 µs │     1 │                                       │ cuInit                  │
│    0.00% │   29.09 µs │     2 │  14.54 µs ± 4.38   ( 11.44 ‥ 17.64)   │ cuMemFreeAsync          │
│    0.00% │   27.66 µs │     1 │                                       │ cuDeviceGetName         │
│    0.00% │   17.17 µs │     1 │                                       │ cuDeviceTotalMem        │
│    0.00% │    15.5 µs │     4 │   3.87 µs ± 6.48   (  0.48 ‥ 13.59)   │ cudaGetLastError        │
│    0.00% │    6.91 µs │     1 │                                       │ cuCtxSetCurrent         │
│    0.00% │    1.19 µs │     2 │ 596.05 ns ± 842.94 (   0.0 ‥ 1192.09) │ cuCtxPopCurrent         │
│    0.00% │  715.26 ns │     3 │ 238.42 ns ± 238.42 (   0.0 ‥ 476.84)  │ cuDeviceGet             │
│    0.00% │  715.26 ns │     3 │ 238.42 ns ± 238.42 (   0.0 ‥ 476.84)  │ cudaDeviceGetAttribute  │
│    0.00% │  715.26 ns │     3 │ 238.42 ns ± 238.42 (   0.0 ‥ 476.84)  │ cuCtxGetDevice          │
│    0.00% │  238.42 ns │     1 │                                       │ cuDriverGetVersion      │
│    0.00% │  238.42 ns │     1 │                                       │ cuModuleGetLoadingMode  │
│    0.00% │  238.42 ns │     1 │                                       │ cuDeviceGetUuid         │
│    0.00% │  238.42 ns │     2 │ 119.21 ns ± 168.59 (   0.0 ‥ 238.42)  │ cuCtxPushCurrent        │
│    0.00% │     0.0 ns │     2 │    0.0 ns ± 0.0    (   0.0 ‥ 0.0)     │ cuDeviceGetCount        │
└──────────┴────────────┴───────┴───────────────────────────────────────┴─────────────────────────┘

Device-side activity: GPU was busy for 3.93 ms (0.25% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Name                                                                                                                                        ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│    0.21% │    3.19 ms │     1 │ void generate_seed_pseudo<rng_config<curandStateXORWOW, (curandOrdering)101>>(unsigned long long, unsigned long long, unsigned long long, c ⋯
│    0.02% │  312.33 µs │     1 │ void gen_sequenced<curandStateXORWOW, float2, normal_args_st, &float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_a ⋯
│    0.01% │  119.45 µs │     1 │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicS ⋯
│    0.01% │  117.54 µs │     1 │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicS ⋯
│    0.01% │  114.44 µs │     1 │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicS ⋯
│    0.00% │   54.36 µs │     1 │ gpu_getindex_kernel(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicSize, Dyn ⋯
│    0.00% │   22.65 µs │     1 │ [copy device to device memory]                                                                                                              ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                               1 column omitted
                                                                                                                                                    
julia> CUDA.@profile Approach_B()
Profiler ran for 248.9 ms, capturing 169 events.

Host-side activity: calling CUDA APIs took 1.76 ms (0.71% of the trace)
┌──────────┬────────────┬───────┬───────────────────────────────────────┬─────────────────────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                     │ Name                    │
├──────────┼────────────┼───────┼───────────────────────────────────────┼─────────────────────────┤
│    0.62% │    1.55 ms │     1 │                                       │ cuModuleLoadDataEx      │
│    0.03% │   68.19 µs │     2 │  34.09 µs ± 30.68  (  12.4 ‥ 55.79)   │ cuLaunchKernel          │
│    0.01% │   30.99 µs │     1 │                                       │ cuMemcpyDtoDAsync       │
│    0.01% │   30.76 µs │     1 │                                       │ cuModuleGetFunction     │
│    0.01% │   27.89 µs │     1 │                                       │ cudaLaunchKernel        │
│    0.00% │    12.4 µs │     1 │                                       │ cuCtxSynchronize        │
│    0.00% │   10.01 µs │     4 │    2.5 µs ± 1.11   (  1.43 ‥ 3.58)    │ cuMemAllocFromPoolAsync │
│    0.00% │    5.96 µs │     1 │                                       │ cuMemFreeAsync          │
│    0.00% │    2.86 µs │     2 │   1.43 µs ± 1.69   (  0.24 ‥ 2.62)    │ cuCtxPopCurrent         │
│    0.00% │    1.19 µs │     2 │ 596.05 ns ± 842.94 (   0.0 ‥ 1192.09) │ cudaGetLastError        │
│    0.00% │  715.26 ns │     2 │ 357.63 ns ± 168.59 (238.42 ‥ 476.84)  │ cuCtxPushCurrent        │
│    0.00% │  238.42 ns │     2 │ 119.21 ns ± 168.59 (   0.0 ‥ 238.42)  │ cuCtxGetDevice          │
│    0.00% │     0.0 ns │     2 │    0.0 ns ± 0.0    (   0.0 ‥ 0.0)     │ cuDeviceGet             │
└──────────┴────────────┴───────┴───────────────────────────────────────┴─────────────────────────┘

Device-side activity: GPU was busy for 331.88 µs (0.13% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Name                                                                                                                                        ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│    0.07% │  185.01 µs │     1 │ void gen_sequenced<curandStateXORWOW, float2, normal_args_st, &float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_a ⋯
│    0.05% │  118.49 µs │     1 │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicS ⋯
│    0.01% │   15.74 µs │     1 │ gpu_getindex_kernel(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRange<1, DynamicSize, Dyn ⋯
│    0.01% │   12.64 µs │     1 │ [copy device to device memory]                                                                                                              ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                               1 column omitted
```
Approach B is faster due to Kernel Fusion and minimal Global Memory Traffic.

Approach A requires four separate kernel launches:

y = x .^ 2

z = sin.(y)

w = z .+ 1

Implicitly, one launch for x = CUDA.randn(...)

Approach B uses Julia's dot-call syntax (@.) which automatically triggers Kernel Fusion. All three mathematical operations are combined into a single expression, resulting in one single kernel launch:

w = @. sin(x^2) + 1

Kernel fusion is a compiler optimization technique used in parallel programming environments.

In Julia, the mechanism of broadcasting, is often the trigger for automatic kernel fusion when running on a GPU backend.

Approach C (CUBLAS) is much faster than implementing matrix multiplication with custom kernels, because it has advanced Tiling and Memory Management. CUBLAS contains many different kernels (algorithms) for the same operation and automatically selects the fastest one based on the input matrix sizes, leading and trailing dimensions.

(a) 2D image filtering
CUFFT. Image filtering often involves using FFT. CUFFT (CUDA Fast Fourier Transform) is the fastest way to perform the necessary Forward FFT.

(b) Graph algorithms on sparse matrices
CUSPARSE. Graph algorithms are often implemented using Sparse Matrix-Vector (SpMV) or Sparse Matrix-Matrix (SpMM) multiplication. CUSPARSE is specifically optimized for these operations.

(c) Dense neural network training
CUBLAS. Neural network training is dominated by  dense matrix-matrix multiplication (GEMM) operations. 

