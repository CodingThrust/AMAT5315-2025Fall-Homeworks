# Homework 12

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


**Answer:**

- **Which code will run faster and by approximately how much?**  
  Snippet B, about **10–100×** faster.

- **Identify the performance problem in Snippet A and explain why it occurs:**  
  Repeated CPU↔GPU transfers (100 uploads + 100 downloads); transfer cost dominates tiny kernels.

- **What is the "golden rule" that Snippet B follows but Snippet A violates?**  
  Keep data on the GPU; minimize host–device transfers.

- **Estimate the bandwidth difference between PCIe transfer and GPU memory operations:**  
  PCIe ≈ **10–30 GB/s** vs GPU DRAM **hundreds of GB/s** (≈10–50×+ gap).

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

**Answer:**

- **What is wrong with Kernel A in terms of warp execution? How much could this slow down performance?**  
  Warp divergence → branches serialize; ~**2×** slowdown (or worse with expensive ops).

- **Explain what SIMT means and how Kernel A violates this execution model:**  
  SIMT: a warp executes one instruction stream; divergent branches are serialized.

- **What memory access problem does Kernel B have? What is "coalesced" memory access?**  
  Non-coalesced (strided) loads. Coalesced = consecutive threads access consecutive addresses.

- **Rewrite the thread ID calculation to be more readable, explaining each component:**
  ```julia
  block_offset  = (blockIdx().x - 1) * blockDim().x
  thread_offset = threadIdx().x - 1
  i = block_offset + thread_offset + 1
  ```

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

**Answer:**

- **Which approach (A or B) will be faster and why? How many kernel launches does each require?**  
  **B** is faster; A launches **3** kernels, B launches **1** (fusion).

- **What is "kernel fusion" and how does broadcasting achieve it automatically?**  
  Combine elementwise ops into one kernel; broadcasting (`@.`) does this automatically.

- **Why would Approach C (CUBLAS) be much faster than implementing matrix multiplication with custom kernels?**  
  Architecture-tuned GEMM (tiling, cache, tensor cores); far faster than simple custom kernels.

- **Pick the best library:**  
  (a) **CUFFT** (image filtering), (b) **CUSPARSE** (graphs/sparse), (c) **CUBLAS** (dense NN).
