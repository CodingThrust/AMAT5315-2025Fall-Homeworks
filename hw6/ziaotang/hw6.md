## Problem 1

Given 

```julia
colptr = [1, 2, 3, 5, 6, 6]
rowval = [3, 1, 1, 4, 5]
nzval  = [0.799, 0.942, 0.848, 0.164, 0.637]
m = 5
n = 5
```

Let’s break this down step by step.

------

## 1. Understanding CSC format

In the **Compressed Sparse Column (CSC)** format:

- `colptr`tells where each column starts in `rowval`and `nzval`.
- `rowval`gives the row index for each stored element.
- `nzval`gives the value for each stored element.

The structure is:

- `colptr[j]`= index in `rowval`/`nzval`of the first element of column `j`
- `colptr[j+1] - 1`= index in `rowval`/`nzval`of the last element of column `j`
- `rowval[k]`= row index of the k-th stored element
- `nzval[k]`= value of the k-th stored element

------

## 2. Given data

```
colptr = [1, 2, 3, 5, 6, 6]
rowval = [3, 1, 1, 4, 5]
nzval  = [0.799, 0.942, 0.848, 0.164, 0.637]
```

Matrix size: 5×5.

------

## 3. Interpret `colptr`

`colptr`has length 6 (for 5 columns + 1).

- Column 1: indices in `rowval`/`nzval`from `colptr[1] = 1`to `colptr[2] - 1 = 2 - 1 = 1`→ only index 1.
- Column 2: from `colptr[2] = 2`to `colptr[3] - 1 = 3 - 1 = 2`→ only index 2.
- Column 3: from `colptr[3] = 3`to `colptr[4] - 1 = 5 - 1 = 4`→ indices 3 and 4.
- Column 4: from `colptr[4] = 5`to `colptr[5] - 1 = 6 - 1 = 5`→ only index 5.
- Column 5: from `colptr[5] = 6`to `colptr[6] - 1 = 6 - 1 = 5`→ no elements (empty range because 6 > 5).

So:

- Col 1: 1 entry
- Col 2: 1 entry
- Col 3: 2 entries
- Col 4: 1 entry
- Col 5: 0 entries

  Rearrange to have

- Column 1: (3, 1, 0.799)

- Column 2: (1, 2, 0.942)

- Column 3: (1, 3, 0.848) and (4, 3, 0.164)

- Column 4: (5, 4, 0.637)

- Column 5: none

Hence,

```julia
rowindices = [3, 1, 1, 4, 5]
colindices = [1, 2, 3, 3, 4]
data       = [0.799, 0.942, 0.848, 0.164, 0.637]
```

Check the output is the same:

```julia
julia> using SparseArrays

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

julia> sp.n
5

julia> sp.m
5
```

## Problem 2

```julia
using Graphs, Random, KrylovKit
Random.seed!(42)
g = random_regular_graph(100000, 3)

# Get the Laplacian matrix
L = laplacian_matrix(g)

# Find the smallest eigenvalues (including zeros)
# The number of zero eigenvalues equals the number of connected components
vals, vecs = eigsolve(L, 5, :SR, tol=1e-6)  # Find 10 smallest eigenvalues


julia> vals
5-element Vector{Float64}:
 -0.4334888244625619
  1.5401585127639006e-5
  0.17180192146081122
  0.172236977113317
  0.17268079092363003
```

So the graph has one connected component.





## Problem 3

```julia
using LinearAlgebra
using Test
using Random

"""
    restarting_lanczos(A, s, max_restarts=10; tol=1e-10)

Restarting Lanczos algorithm to find the largest eigenvalue of a Hermitian matrix A.

# Arguments
- `A`: Hermitian matrix
- `s`: Number of Lanczos vectors to generate
- `max_restarts`: Maximum number of restart iterations
- `tol`: Convergence tolerance

# Returns
- `λ_max`: Largest eigenvalue
- `v_max`: Corresponding eigenvector
"""
function restarting_lanczos(A, s, max_restarts=10; tol=1e-10)
    n = size(A, 1)
    
    # Initialize random starting vector q₁
    Random.seed!(123)
    q_old = randn(n)
    q_old = q_old / norm(q_old)
    
    for restart in 1:max_restarts
        # Step i: Generate Lanczos vectors q₂,...,qₛ
        Q = zeros(n, s)
        α = zeros(s)
        β = zeros(s-1)
        
        # First vector
        Q[:, 1] = q_old
        v = zeros(n)
        # Lanczos iteration
        for j in 1:s
            if j == 1
                v = A * Q[:, 1]
                α[1] = dot(Q[:, 1], v)
                v = v - α[1] * Q[:, 1]
            else
                β[j-1] = norm(v)
                
                if β[j-1] < tol
                    # Breakdown - use random vector
                    q_new = randn(n)
                    for k in 1:j-1
                        q_new = q_new - dot(Q[:, k], q_new) * Q[:, k]
                    end
                    Q[:, j] = q_new / norm(q_new)
                else
                    Q[:, j] = v / β[j-1]
                end
                
                v = A * Q[:, j]
                α[j] = dot(Q[:, j], v)
                v = v - α[j] * Q[:, j] - (j > 1 ? β[j-1] * Q[:, j-1] : 0)
            end
        end
        
        # Step ii: Form tridiagonal matrix Tₛ
        T = zeros(s, s)
        for i in 1:s
            T[i, i] = α[i]
            if i < s
                T[i, i+1] = β[i]
                T[i+1, i] = β[i]
            end
        end
        
        # Step iii: Compute eigenvalues and eigenvectors of Tₛ
        F = eigen(Symmetric(T))
        θ = F.values
        U = F.vectors
        
        # Sort in descending order
        perm = sortperm(θ, rev=true)
        θ = θ[perm]
        U = U[:, perm]
        
        # Step iv: Compute new starting vector q₁^(new)
        u₁ = U[:, 1]
        q_new = zeros(n)
        for i in 1:s
            q_new += u₁[i] * Q[:, i]
        end
        q_new = q_new / norm(q_new)
        
        # Check convergence
        if restart > 1
            # Recompute eigenvalue estimate with new vector
            λ_new = dot(q_new, A * q_new)
            λ_prev = dot(q_old, A * q_old)
            
            if abs(λ_new - λ_prev) < tol
                return λ_new, q_new
            end
        end
        
        q_old = q_new
    end
    
    # If max restarts reached, return best estimate
    λ_final = dot(q_old, A * q_old)
    return λ_final, q_old
end






@testset "Restarting_Lanczos_Test" begin
    A = [4.0 1.0 0.5;
         1.0 3.0 0.2;
         0.5 0.2 2.0]
    
    λ_max, v_max = restarting_lanczos(A, 3, 10, tol=1e-12)
    λ_exact = maximum(eigvals(Symmetric(A)))
    @test abs(λ_max - λ_exact) < 1e-8
end
```





The test result gives

```julia
Test Summary:           | Pass  Total  Time
Restarting_Lanczos_Test |    1      1  0.5s
Test.DefaultTestSet("Restarting_Lanczos_Test", Any[], 1, false, false, true, 1.763606708145e9, 1.763606708684e9, false, "D:\\juliahw\\hw6\\hw6p3.jl")

```

