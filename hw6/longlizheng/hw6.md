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
  
  Ans:
  ```julia
   julia> colval = [1,2,3,3,4];

   julia> rowval = [3,1,1,4,5];

   julia> nzval = [0.799, 0.942, 0.848, 0.164, 0.637];

   julia> colval = [1, 2, 3, 3, 4];

   julia> rowval = [3, 1, 1, 4, 5];

   julia> nzval = [0.799, 0.942, 0.848, 0.164, 0.637];

   julia> sp = sparse(rowval, colval, nzval, 5, 5)
   5×5 SparseMatrixCSC{Float64, Int64} with 5 stored entries:
     ⋅     0.942  0.848   ⋅      ⋅ 
     ⋅      ⋅      ⋅      ⋅      ⋅ 
    0.799   ⋅      ⋅      ⋅      ⋅ 
     ⋅      ⋅     0.164   ⋅      ⋅ 
     ⋅      ⋅      ⋅     0.637   ⋅ 
  ```

  By definition of colptr, the rowval can be partitioned into
  ```
   | 3 | 1 | 1 4 | 5 | |
  ```
  Thus we have
  ```
  colval = [1, 2, 3, 3, 4]
  ```

2. **(Graph Spectral Analysis)** The following code generates a random 3-regular graph with 100,000 nodes. Find the number of connected components by analyzing the eigenvalues of the Laplacian matrix using `KrylovKit.jl`:

   ```julia
   using Graphs, Random, KrylovKit
   Random.seed!(42)
   g = random_regular_graph(100000, 3)
   
   A = laplacian_matrix(g)
   q = randn(nv(g))
   vals, _, _ = eigsolve(A, q, 5, :SR)
   ```

   **Requirements:**
   - Use the relationship between zero eigenvalues and connected components
   - Implement eigenvalue computation with `KrylovKit.jl`
   - Report the number of connected components found

Ans:
code is implemented as above, which gives results:
```julia
julia> vals
5-element Vector{Float64}:
 -2.469722951009431e-15
  0.1717316253401988
  0.17216583440834107
  0.17267946627399564
  0.17299784606514
```
Thus, there is one connected component.

2. **(Restarting Lanczos Algorithm)** Implement a restarting Lanczos algorithm to find the largest eigenvalue of a Hermitian matrix. The algorithm works as follows:

   1. Generate $q_2,\ldots,q_s \in \mathbb{C}^{n}$ via the Lanczos algorithm
   2. Form $T_s = ( q_1 \mid \ldots \mid q_s)^\dagger A ( q_1 \mid \ldots \mid q_s)$, an s-by-s matrix
   3. Compute orthogonal matrix $U = ( u_1 \mid \ldots\mid u_s)$ such that $U^\dagger T_s U = \text{diag}(\theta_1, \ldots, \theta_s)$ with $\theta_1\geq \ldots \geq \theta_s$
   4. Set $q_1^{(\text{new})} = ( q_1 \mid \ldots \mid q_s)u_1$

   **Requirements:**
   - Implement the restarting Lanczos tridiagonalization as a Julia function
   - Include a test demonstrating your implementation works correctly
   - Document your function with clear comments explaining each step

Ans:
```julia
using SparseArrays, LinearAlgebra

"""
restarting_lanczos(A, q::AbstractVector{T}; s::Int=20, maxiter::Int=20,abstol::Real=1e-10 ) -> (Real,Vector)
Restarting Lanczos algorithm to find the largest eigen-value of a Hermitian matrix.
The restatring lanczos algorithm builds an orthonormal basis for the Krylov subspace
K_m(A, q) = span{q, Aq, A²q, ..., A^(m-1)q} and produces a tridiagonal
matrix T that approximates A in this subspace.

# Arguments:
- `A`: Hermitian matrix (can be sparse)
- `q::AbstractVector`: Initial vector
- `s::Int`: Number of Lanczos iteration steps per restart (default: 20)
- `maxiter::Int`: Maximum number of iterations
- `abstol::Real`: Absolute tolerance (default: 1e-10)

# Returns:
- maximal eigenvalue of A

"""
function restarting_lanczos(A, q::AbstractVector{T}; s::Int=20, maxiter::Int=100, abstol::Real=1e-8) where T
    _s = min(length(q), s)
    normalize!(q)

    basis = zeros(T, length(q), _s)
    basis[:,1] .= q
    α = zeros(T, _s)
    α[1] = q' * A * q

    rk = A * q - α[1] * q
    β = zeros(T, _s - 1)
    nrk = norm(rk)
    β[1] = nrk

    for iter in 1:maxiter
        for k = 2:_s
            basis[:,k] .= rk ./ β[k-1]
            Aq_k = A * basis[:,k]
            α[k] = basis[:,k]' * Aq_k
            rk = Aq_k -  α[k] * basis[:,k] - β[k-1] * basis[:,k-1]
            nrk = norm(rk)
            if abs(nrk) < abstol
                return eigen(SymTridiagonal(α, β)).values[end]
            end
            if k < _s
                β[k] = nrk
            end
        end
        if iter == maxiter
            return eigen(SymTridiagonal(α, β)).values[end]
        end
        Ts =  basis' * A * basis
        u1 = eigen(Ts).vectors[:,end]
        basis[:,1] = normalize(basis * u1)
        α[1] = basis[:,1]' * (A * basis[:,1])
        rk = A * basis[:,1] - α[1] * basis[:,1]
        β[1] = norm(rk)
    end

    return eigen(SymTridiagonal(α, β)).values[end]
end


using Test
@testset "restarting_lanczos" begin
    A = Symmetric(rand(100,100))
    q1 = randn(100)
    @test restarting_lanczos(A,q1) ≈ eigen(A).values[end]
end
```