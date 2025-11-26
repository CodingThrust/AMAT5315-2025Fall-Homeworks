# Homework 12

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. **(Code Analysis: Performance Anti-patterns)** Analyze the following code snippets and predict their performance characteristics.

   **Snippet A:**
   ```julia
   x_cpu = randn(10000)
   for i in 1:100
       x_gpu = CuArray(x_cpu)    # Upload
       x_gpu .+= 1
       x_cpu = Array(x_gpu)       # Download
   end
   ```

   **Snippet B:**
   ```julia
   x_gpu = CuArray(randn(10000))  # Upload once
   for i in 1:100
       x_gpu .+= 1                # All on GPU
   end
   result = Array(x_gpu)          # Download once
   ```

   **Questions:**
   - Which code will run faster and by approximately how much?
   - Identify the performance problem in Snippet A and explain why it occurs
   - What is the "golden rule" that Snippet B follows but Snippet A violates?
   - Estimate the bandwidth difference between PCIe transfer and GPU memory operations

2. **(Code Analysis: Kernel Divergence and Memory Access)** Examine these CUDA kernel implementations and identify potential performance issues.

   **Kernel A:**
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

   **Kernel B:**
   ```julia
   function bad_memory_kernel(A, B, stride)
       i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
       @inbounds if i <= length(A)
           B[i] = A[i * stride]  # Non-sequential access!
       end
       return nothing
   end
   ```

   **Questions:**
   - What is wrong with Kernel A in terms of warp execution? How much could this slow down performance?
   - Explain what SIMT means and how Kernel A violates this execution model
   - What memory access problem does Kernel B have? What is "coalesced" memory access?
   - Rewrite the thread ID calculation to be more readable, explaining each component

3. **(Code Analysis: Broadcasting vs Libraries)** Compare these different approaches to the same computation and predict their relative performance.

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

   **Questions:**
   - Which approach (A or B) will be faster and why? How many kernel launches does each require?
   - What is "kernel fusion" and how does broadcasting achieve it automatically?
   - Why would Approach C (CUBLAS) be much faster than implementing matrix multiplication with custom kernels?
   - If you had to choose between CUBLAS, CUSPARSE, and CUFFT for the following problems, which would you pick: (a) 2D image filtering, (b) Graph algorithms on sparse matrices, (c) Dense neural network training?
