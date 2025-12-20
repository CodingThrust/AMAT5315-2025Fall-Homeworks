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

Answer:

- `colptr = [1, 2, 3, 5, 6, 6]` means: column 1 has 1 nonzero (index 1), column 2 has 1 nonzero (index 2), column 3 has 2 nonzeros (indices 3–4), column 4 has 1 nonzero (index 5), column 5 has 0 nonzeros.
- Reading `rowval` and `nzval` in that order gives the entries by column:
  - col 1: row 3, value 0.799
  - col 2: row 1, value 0.942
  - col 3: row 1, value 0.848; row 4, value 0.164
  - col 4: row 5, value 0.637
- So one valid input (already sorted by column then row) is:

```julia
rowindices = [3, 1, 1, 4, 5]
colindices = [1, 2, 3, 3, 4]
data       = [0.799, 0.942, 0.848, 0.164, 0.637]
```

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

Answer:

```julia
using Graphs, Random, KrylovKit
Random.seed!(42)

n = 100000
k = 3

g = random_regular_graph(n, k)
L = laplacian_matrix(g)

# Smallest real eigenvalues; count how many are (numerically) zero.
q1 = randn(n)
vals, _, _ = eigsolve(L, q1, 6, :SR)
num_components = count(v -> abs(v) < 1e-8, vals)
```
```
Smallest eigenvalues: [-2.469722951009431e-15, 0.17173162534019834, 0.17216583440834246, 0.17267946627042527, 0.17299781348833632, 0.17311238806286774, 0.17333612780416027, 0.1734802244920756, 0.17363689665481138]
Connected components (zero eigvals): 1
```

With the fixed seed, the graph is connected, so the number of connected components is `1`.

3. **(Restarting Lanczos Algorithm)** Implement a restarting Lanczos algorithm to find the largest eigenvalue of a Hermitian matrix. The algorithm works as follows:

   1. Generate $q_2,\ldots,q_s \in \mathbb{C}^{n}$ via the Lanczos algorithm
   2. Form $T_s = ( q_1 \mid \ldots \mid q_s)^\dagger A ( q_1 \mid \ldots \mid q_s)$, an s-by-s matrix
   3. Compute orthogonal matrix $U = ( u_1 \mid \ldots\mid u_s)$ such that $U^\dagger T_s U = \text{diag}(\theta_1, \ldots, \theta_s)$ with $\theta_1\geq \ldots \geq \theta_s$
   4. Set $q_1^{(\text{new})} = ( q_1 \mid \ldots \mid q_s)u_1$

   **Requirements:**
   - Implement the restarting Lanczos tridiagonalization as a Julia function
   - Include a test demonstrating your implementation works correctly
   - Document your function with clear comments explaining each step

Answer:

See `examples/answer.jl`, which includes:
- `lanczos_tridiag` for the tridiagonalization,
- `restarted_lanczos_largest` for restarting,
- `test_restarted_lanczos` as a correctness test.

```
Computed largest eigenvalue: 8.721659642991888
Reference largest eigenvalue: 8.721659642991886
```