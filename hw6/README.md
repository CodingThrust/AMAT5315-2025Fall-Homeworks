# Homework 6

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. **(Sparse Matrix Construction)** Find the correct values of `rowindices`, `colindices`, and `data` to reproduce the following sparse matrix in CSC format:

   ```julia
   julia> sp = sparse(rowindices, colindices, data, 5, 5);

   julia> sp.colptr
   6-element Vector{Int64}:
    1
    2
    3
    5
    6
    6

   julia> sp.rowval
   5-element Vector{Int64}:
    3
    1
    1
    4
    5

   julia> sp.nzval
   5-element Vector{Float64}:
    0.799
    0.942
    0.848
    0.164
    0.637

   julia> sp.m
   5

   julia> sp.n
   5
   ```

   **Requirements:**
   - Determine the correct input arrays that produce this exact CSC structure
   - Show your reasoning for how you derived the values

2. **(Graph Spectral Analysis)** The following code generates a random 3-regular graph with 100,000 nodes. Find the number of connected components by analyzing the eigenvalues of the Laplacian matrix using `KrylovKit.jl`:

   ```julia
   using Graphs, Random, KrylovKit
   Random.seed!(42)
   g = random_regular_graph(100000, 3)
   # your code here
   ```

   **Requirements:**
   - Use the relationship between zero eigenvalues and connected components
   - Implement eigenvalue computation with `KrylovKit.jl`
   - Report the number of connected components found

3. **(Restarting Lanczos Algorithm)** Implement a restarting Lanczos algorithm to find the largest eigenvalue of a Hermitian matrix. The algorithm works as follows:

   1. Generate $q_2,\ldots,q_s \in \mathbb{C}^{n}$ via the Lanczos algorithm
   2. Form $T_s = ( q_1 \mid \ldots \mid q_s)^\dagger A ( q_1 \mid \ldots \mid q_s)$, an s-by-s matrix
   3. Compute orthogonal matrix $U = ( u_1 \mid \ldots\mid u_s)$ such that $U^\dagger T_s U = \text{diag}(\theta_1, \ldots, \theta_s)$ with $\theta_1\geq \ldots \geq \theta_s$
   4. Set $q_1^{(\text{new})} = ( q_1 \mid \ldots \mid q_s)u_1$

   **Requirements:**
   - Implement the restarting Lanczos tridiagonalization as a Julia function
   - Include a test demonstrating your implementation works correctly
   - Document your function with clear comments explaining each step