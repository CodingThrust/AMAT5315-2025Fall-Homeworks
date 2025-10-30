# Homework 6

## Problem 1

colptr[1] = 1, colptr[2] - 1 = 2 - 1 = 1. 1 nonzero value
rowval[1] = 3，nzval[1] = 0.799
Hence, matrix (3,1) = 0.799

colptr[2] = 2, colptr[3] - 1 = 3 - 1 = 2.
rowval[2] = 1，nzval[2] = 0.942
Hence, matrix (1,2) = 0.942

colptr[3] = 3, colptr[4] - 1 = 5 - 1 = 4
rowval[3] = 1，nzval[3] = 0.848
Hence, matrix (1,3) = 0.848
rowval[4] = 4，nzval[4] = 0.164
Hence, matrix (4,3) = 0.164

colptr[4] = 5
colptr[5] - 1 = 6 - 1 = 5
rowval[5] = 5，nzval[5] = 0.637
Hence, matrix (5,4) = 0.637

colptr[5] = 6
colptr[6] - 1 = 6 - 1 = 5
No nonzero element.

Test,
```
julia> rowindices = [3, 1, 1, 4, 5]
5-element Vector{Int64}:
 3
 1
 1
 4
 5

julia> colindices = [1, 2, 3, 3, 4]
5-element Vector{Int64}:
 1
 2
 3
 3
 4

julia> data = [0.799, 0.942, 0.848, 0.164, 0.637]
5-element Vector{Float64}:
 0.799
 0.942
 0.848
 0.164
 0.637

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

## Problem 2
```

julia> using Graphs, Random, KrylovKit

julia> using LinearAlgebra, SparseArrays

julia> using LinearMaps

julia> Random.seed!(42)
TaskLocalRNG()

julia> g = random_regular_graph(100000, 3)
{100000, 150000} undirected simple Int64 graph

julia> L = laplacian_matrix(g)
100000×100000 SparseMatrixCSC{Int64, Int64} with 400000 stored entries:
⎡⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎤
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎢⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⎥
⎣⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠇⎦

julia> L_sparse = LinearMap(L)
100000×100000 LinearMaps.WrappedMap{Int64} of
  100000×100000 SparseMatrixCSC{Int64, Int64} with 400000 stored entries

julia> eig_result = eigsolve(L_sparse,randn(size(L_sparse, 1)), 1, :SR)
(ComplexF64[3.0527217673814588e-15 + 0.0im], Vector{ComplexF64}[[0.003162277660168541 + 0.0im, 0.003162277660166926 + 0.0im, 0.0031622776601690043 + 0.0im, 0.0031622776601681556 + 0.0im, 0.003162277660167221 + 0.0im, 0.0031622776601677297 + 0.0im, 0.0031622776601682246 + 0.0im, 0.003162277660167078 + 0.0im, 0.00316227766016655 + 0.0im, 0.003162277660167209 + 0.0im  …  0.003162277660168576 + 0.0im, 0.003162277660170128 + 0.0im, 0.0031622776601698843 + 0.0im, 0.003162277660169044 + 0.0im, 0.00316227766016834 + 0.0im, 0.0031622776601702837 + 0.0im, 0.0031622776601693977 + 0.0im, 0.0031622776601706237 + 0.0im, 0.0031622776601677163 + 0.0im, 0.003162277660168092 + 0.0im]], ConvergenceInfo: one converged value after 7 iterations and 102 applications of the linear map;
norms of residuals are given by (2.62e-13).)

julia> eigenvalues = real(eig_result[1])
1-element Vector{Float64}:
 3.0527217673814588e-15

julia> sorted_eigenvalues = sort(eigenvalues)
1-element Vector{Float64}:
 3.0527217673814588e-15

julia> zero_tolerance = 1e-10
1.0e-10

julia> num_zero_eigenvalues = count(x -> abs(x) < zero_tolerance, sorted_eigenvalues)
1

julia> println("Method 2 - Spectral analysis: ", num_zero_eigenvalues, " connected components")
Method 2 - Spectral analysis: 1 connected components

```

## Problem 3

```

using LinearAlgebra

function restarting_lanczos(A, q1::AbstractVector{T}; s::Int, max_restarts::Int=100) where T
    """
    Parameter:
    A - Hermitian Matrix
    q1 - initial vector
    s - subspace size
    max_restarts - max restarts times
    
    return:
    λ - estimated the largest eigenvalue
    v - eigenvector
    """
    
    # normalization
    q_current = normalize(q1)
    
    for restart in 1:max_restarts
        q_basis = [q_current]  
        α = [real(dot(q_current, A * q_current))]  
        β = T[]  
        
        # Lanczos iteration
        for k in 2:s
            if k == 2
                # Compute first residual: r₁ = Aq₁ - α₁q₁
                Aq1 = A * q_basis[1]
                rk = Aq1 .- α[1] .* q_basis[1]
            else
                # Compute diagonal element: α_k = q_k' * A * q_k
                Aqk = A * q_basis[k-1]
                rk = Aqk .- α[k-1] .* q_basis[k-1] .- β[k-2] * q_basis[k-2]
            end
            
            
            βk = norm(rk)
            push!(β, βk)
            
            
            push!(q_basis, rk ./ βk)
            
          
            Aqk_new = A * q_basis[k]
            push!(α, real(dot(q_basis[k], Aqk_new)))
        end
        
        
        T_matrix = SymTridiagonal(α, β)
        
        
        F = eigen(Matrix(T_matrix))
        θ = real(F.values)
        U = F.vectors
        
        perm = sortperm(θ, rev=true)
        θ_sorted = θ[perm]
        U_sorted = U[:, perm]
        
        Q_matrix = hcat(q_basis...)  
        q_new = Q_matrix * U_sorted[:, 1]
        q_new = normalize(q_new)
        
        q_current = q_new
    end
    
    v_final = q_current
    λ_final = real(dot(v_final, A * v_final))
    
    return λ_final, v_final
end

```

Create Hermitian matrix and do test

```

julia> n = 1000
1000

julia> A = randn(n, n) + im * randn(n, n)
1000×1000 Matrix{ComplexF64}:
 -0.666447+0.413921im  1.61656-1.31965im  0.475324-0.575816im  0.589697-0.283349im  …  1.10376+1.30216im  -0.600375-0.347927im  0.707873-2.03934im
          ⋮                                                                         ⋱

julia> A = A + A'
1000×1000 Matrix{ComplexF64}:
 -1.33289+0.0im  1.89152-2.2426im  0.960961-1.03512im  -0.401084-1.73379im  …  1.8872+1.82705im  0.596103-0.0375922im  1.21955-1.64407im
         ⋮                                                                  ⋱

julia> q1 = randn(ComplexF64, n)
1000-element Vector{ComplexF64}:
 0.7374501478038081 + 0.09733122539869257im
                    ⋮
julia> λ_max, v_max = restarting_lanczos(A, q1; s=30, max_restarts=20)
(125.88222642319371, ComplexF64[0.012937757736650533 + 0.03122731476120804im, -0.032238218254156834 - 0.008077201912374337im, 0.005309192075305651 - 0.024225282989868275im, -0.04587207428916289 + 0.031210602948287185im, 0.01950536968503028 + 0.00985115318988048im, -0.015044430483518742 - 0.0382947186312158im, -0.016267529017644104 + 0.0024727438637545747im, 0.020605460397957468 - 0.0007011544566295497im, 0.019959665560657043 + 0.016312817045791482im, 0.004855361522865508 + 0.054785604486515996im  …  -0.0023466390380702472 - 0.010894362447696818im, -0.012266480437556963 + 0.012294774115225346im, 0.013177262373766712 + 0.003477359690032761im, -0.0016397510496784136 - 0.004085991889756534im, -0.03327939220500726 + 0.016968078104692166im, 0.0228048746462079 - 0.03209172558146101im, -0.0022875683681803944 - 0.014614803008504994im, 0.004587848110745969 + 0.001367310755691433im, 0.02409299507526221 - 0.005588456157832663im, -0.018273874727398484 - 0.010736287399816614im])
663im, -0.018273874727398484 - 0.010736287399816614im])

julia> println("The largest eigenvalue: ", λ_max)
The largest eigenvalue: 125.88222642319371
```
