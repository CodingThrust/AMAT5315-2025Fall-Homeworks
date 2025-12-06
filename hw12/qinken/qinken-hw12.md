# Homework12

## Task1
	•	Snippet B runs faster—approximately 1–2 orders of magnitude (≈20–100×) depending on hardware, because it transfers data only twice (once upload, once download) while Snippet A transfers every iteration.
	•	Anti-pattern in Snippet A: Repeated host↔GPU PCIe transfers inside the loop cause high latency and low bandwidth usage relative to GPU memory operations, dominating runtime.
	•	Golden rule violated: Minimize CPU–GPU data movement; maximize compute where data resides (GPU).
	•	Bandwidth estimate:
	•	PCIe 3.0 x16: ~12–16 GB/s
	•	GPU memory ops: ~300–1000 GB/s (≈20–60× higher)

## Task2
1. 
	•	Within each warp, odd and even i go down different branches (sin vs cos).
	•	Since warps must execute one branch at a time, the hardware runs the if arm with half the threads active, then the else arm with the other half.
	•	Effective throughput is cut roughly in half: ~2× slowdown in this two-way divergent case (more generally, up to “number of distinct branch paths” times slower).

⸻

2. 
	•	SIMT stands for Single Instruction, Multiple Threads: a warp is meant to issue the same instruction for all its threads simultaneously.
	•	Kernel A violates this by introducing control-flow divergence inside the warp. The even threads want one instruction sequence, the odd threads want another, so SIMT degenerates into sequential execution of each path with partial masks.

⸻

3. 
	•	Problem: A[i * stride] means thread i touches element i * stride. For stride > 1, adjacent threads access addresses that are far apart, causing scattered loads.
	•	This defeats coalesced memory access, where consecutive threads should access consecutive or nearby memory locations so the GPU can service them with a single (or few) wide memory transactions. Non-coalesced patterns increase memory latency and reduce effective bandwidth.

4. 
  block_idx = blockIdx().x - 1
  block_idx = blockIdx().x - 1
  block_idx = blockIdx().x - 1
  block_idx = blockIdx().x - 1

## Task3 
1. 
	•	Approach B is faster.
	•	A does:
	•	y = x .^ 2 → 1 kernel
	•	z = sin.(y) → 1 kernel
	•	w = z .+ 1 → 1 kernel
→ 3 kernel launches + 2 intermediate arrays on GPU.
	•	B (@. sin(x^2) + 1) fuses the whole expression into 1 broadcast kernel, with no intermediate arrays.
→ Less launch overhead, less global memory traffic → faster.

⸻

2. 
	•	Kernel fusion: combining several elementwise operations into one GPU kernel so that each element is:
	1.	Loaded from memory once,
	2.	All operations applied in registers,
	3.	Written back once.
	•	Julia’s broadcasting (.@ or dotted syntax) analyzes the whole expression and generates a single fused kernel for compatible elementwise operations, giving fusion “for free” when you write vectorized code correctly.

⸻

3. 
	•	CUBLAS is NVIDIA’s highly optimized BLAS library:
	•	Uses blocked algorithms, shared memory, register tiling, SIMD, and often tensor cores.
	•	Tuned per-architecture by GPU experts.
	•	A naive custom kernel usually:
	•	Loads data inefficiently,
	•	Misses coalescing and tiling,
	•	Underutilizes compute units.
→ CUBLAS achieves near-peak GPU performance, whereas naive matmul can be orders of magnitude slower.

⸻

4. 
	•	(a) 2D image filtering → typically convolution / frequency-domain ops → CUFFT
	•	(b) Graph algorithms on sparse matrices → sparse linear algebra → CUSPARSE
	•	(c) Dense neural network training → dominated by dense GEMMs → CUBLAS