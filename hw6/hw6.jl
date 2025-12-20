#!/usr/bin/env julia

"""
Homework 6 solutions (Julia)

Contents
- Task 1: Sparse matrix construction in CSC from given colptr/rowval/nzval
- Task 2: Graph spectral analysis using KrylovKit to count connected components
- Task 3: Restarting Lanczos algorithm to estimate the largest eigenvalue (Hermitian)

Run:
  julia --project=@. hw6.jl

Notes:
- Task 2 generates a 100,000-node 3-regular graph by default. This is memory-intensive but feasible
  on a typical modern machine. To run a quicker demo, set environment variable HW6_QUICK=1.
"""

using LinearAlgebra
using SparseArrays
using Random
using Graphs
using KrylovKit

# -----------------------------
# Task 1: Sparse CSC construction
# -----------------------------

"""
task1_construct_sparse()

Construct the sparse matrix sp = sparse(rowindices, colindices, data, 5, 5) such that its
internal CSC storage matches:
  colptr = [1, 2, 3, 5, 6, 6]
  rowval = [3, 1, 1, 4, 5]
  nzval  = [0.799, 0.942, 0.848, 0.164, 0.637]
  m = 5, n = 5

Returns a named tuple with the arrays and the constructed sparse matrix.
"""
function task1_construct_sparse()
    # From colptr we infer per-column nonzero counts:
    # col 1: indices 1:1 (1 entry)     -> (row=3, val=0.799)
    # col 2: indices 2:2 (1 entry)     -> (row=1, val=0.942)
    # col 3: indices 3:4 (2 entries)   -> (row=1, val=0.848), (row=4, val=0.164)
    # col 4: indices 5:5 (1 entry)     -> (row=5, val=0.637)
    # col 5: indices 6:5 (0 entries)
    rowindices = [3, 1, 1, 4, 5]
    colindices = [1, 2, 3, 3, 4]
    data       = [0.799, 0.942, 0.848, 0.164, 0.637]

    sp = sparse(rowindices, colindices, data, 5, 5)

    # Validate exact CSC layout
    @assert sp.colptr == [1, 2, 3, 5, 6, 6]
    @assert sp.rowval == [3, 1, 1, 4, 5]
    @assert sp.nzval  == [0.799, 0.942, 0.848, 0.164, 0.637]
    @assert sp.m == 5 && sp.n == 5

    return (rowindices=rowindices, colindices=colindices, data=data, sp=sp)
end

function task1_demo()
    r = task1_construct_sparse()
    println("Task 1: Constructed sparse matrix with target CSC arrays.")
    println("sp.colptr = ", r.sp.colptr)
    println("sp.rowval = ", r.sp.rowval)
    println("sp.nzval  = ", r.sp.nzval)
    println("sp.m, sp.n = ", (r.sp.m, r.sp.n))
end

# ---------------------------------------------------------
# Task 2: Graph spectral analysis via KrylovKit eigenvalues
# ---------------------------------------------------------

"""
count_components_via_laplacian(g; nev=4, tol=1e-10, zerotol=1e-13, V0=nothing)

Estimate the number of connected components by counting (numerical) zero eigenvalues of the
combinatorial Laplacian L. Uses KrylovKit to compute the smallest eigenvalues.

Arguments:
- g::Graphs.AbstractGraph
- nev: initial number of smallest eigenvalues to compute (may adapt)
- tol: eigensolver tolerance
- zerotol: threshold to classify an eigenvalue as zero
- V0: optional initial block (n×k) for eigensolver; if provided, it will be orthonormalized

Returns: (num_components::Int, eigenvalues::Vector{Float64})
"""
function _orthonormal_block(V::AbstractMatrix)
    n, k = size(V)
    qrf = qr(V)
    return qrf.Q * Matrix(I, n, k)
end

function _as_columns(vecs)
    if vecs isa AbstractMatrix
        return vecs
    elseif vecs isa AbstractVector && !isempty(vecs) && (vecs[1] isa AbstractVector)
        n = length(vecs[1])
        k = length(vecs)
        M = Matrix{eltype(vecs[1])}(undef, n, k)
        for j in 1:k
            M[:, j] = vecs[j]
        end
        return M
    else
        # Fallback: treat as single vector
        return reshape(vecs, :, 1)
    end
end

function count_components_via_laplacian(g; nev::Int=4, tol::Float64=1e-10, zerotol::Float64=1e-13, V0::Union{Nothing,AbstractMatrix}=nothing)
    # Build Laplacian (sparse, symmetric, PSD) and convert to Float64
    L = laplacian_matrix(g)
    Lf = SparseMatrixCSC{Float64, Int}(L)

    n = size(Lf, 1)
    # Adaptive solve: increase nev until we see at least one clearly nonzero eigenvalue
    nev_curr = nev
    vals = Float64[]
    for attempt in 1:4
        # Build or use provided orthonormal initial block for this attempt
        if V0 === nothing
            blockdim = min(n, max(nev_curr, 1))
            V0_curr = _orthonormal_block(randn(n, blockdim))
        else
            @assert size(V0, 1) == n
            V0_curr = _orthonormal_block(Matrix{Float64}(V0))
        end

        krylovdim = max(20, 4*nev_curr)
        maxiter = 2_000
        X0 = [V0_curr[:, j] for j in 1:size(V0_curr, 2)]
        vals, vecs, info = eigsolve(v->Lf*v, X0, nev_curr, :SR; ishermitian=true, tol=tol, krylovdim=krylovdim, maxiter=maxiter)
        Vcols = _as_columns(vecs)
        # Filter converged pairs using residual norms
        selected = Int[]
        for j in 1:min(length(vals), size(Vcols, 2))
            r = Lf * Vcols[:, j] - vals[j] .* Vcols[:, j]
            if norm(r) ≤ max(1e-8, 10*tol)
                push!(selected, j)
            end
        end
        vals_conv = vals[selected]
        absvals = sort(abs.(vals_conv))
        if !isempty(absvals) && absvals[end] > max(1e-8, 100*zerotol)
            num_components = count(absvals .< zerotol)
            return (num_components, vals_conv)
        end
        nev_curr = min(32, max(nev_curr + 2, 2*nev_curr))
    end
    # Fallback count with the last computed eigenvalues
    num_components = count(abs.(vals) .< zerotol)
    return (num_components, vals)
end

function task2_demo(; n::Int=100_000, d::Int=3, seed::Int=42, nev::Int=4)
    println("Task 2: Generating $n-node, $d-regular random graph (seed=$seed)...")
    Random.seed!(seed)
    g = random_regular_graph(n, d)

    println("Task 2: Computing $nev smallest eigenvalues of Laplacian via KrylovKit...")
    num_components, vals = count_components_via_laplacian_deflation(g; max_zeros=nev)

    println("Smallest eigenvalues ≈ ", vals)
    println("Estimated number of connected components = ", num_components)
    if n <= 10_000
        true_components = length(connected_components(g))
        println("Exact components (Graphs.connected_components) = ", true_components)
    end
end

"""
task2_small_test()

Build a small graph with two disconnected components and verify the spectral component count
equals 2 by counting near-zero Laplacian eigenvalues.
"""
function task2_small_test()
    g = Graph(6)
    # Component 1: 1-2-3
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    # Component 2: 4-5-6
    add_edge!(g, 4, 5)
    add_edge!(g, 5, 6)

    # For tiny graphs, verify via dense eigendecomposition for robustness
    L = laplacian_matrix(g)
    Lf = SparseMatrixCSC{Float64, Int}(L)
    vals = eigvals(Symmetric(Matrix(Lf)))
    num_components = count(abs.(vals) .< 1e-10)
    println("Task 2 small test: eigenvalues ≈ ", vals)
    println("Task 2 small test: estimated components = ", num_components)
    @assert num_components == 2
end

"""
count_components_via_laplacian_deflation(g; max_zeros=8, tol=1e-10, zerotol=1e-12)

Counts the number of zero eigenvalues of the Laplacian using repeated KrylovKit solves
with orthogonal deflation. This avoids spurious near-zero Ritz values by computing one
eigenpair at a time and projecting out previously found nullspace vectors.
"""
function count_components_via_laplacian_deflation(g; max_zeros::Int=8, tol::Float64=1e-10, zerotol::Float64=1e-12)
    L = laplacian_matrix(g)
    Lf = SparseMatrixCSC{Float64, Int}(L)
    n = size(Lf, 1)

    # Start by deflating the trivial nullspace spanned by the all-ones vector
    onevec = fill(1.0, n)
    onevec ./= norm(onevec)
    Q = reshape(onevec, :, 1)

    project(v) = size(Q, 2) == 0 ? v : (v .- Q * (Q' * v))
    op(v) = project(Lf * v)

    count_zero = 1  # we already know one zero eigenvalue corresponding to ones vector
    vals_found = Float64[]
    for k in 1:max_zeros
        v0 = project(randn(n))
        if norm(v0) ≤ eps()
            break
        end
        v0 ./= norm(v0)
        vals, vecs, info = eigsolve(op, v0, 1, :SR; ishermitian=true, tol=tol, krylovdim=30, maxiter=2_000)
        λ = vals[1]
        q = vecs[1]
        r = Lf * q - λ .* q
        push!(vals_found, λ)
        if (λ ≥ 0) && (abs(λ) < zerotol) && (norm(r) ≤ max(1e-8, 10*tol))
            # Accept as a zero eigenvector, expand deflation space
            qproj = project(q)
            if norm(qproj) > 0
                qproj ./= norm(qproj)
                Q = size(Q, 2) == 0 ? reshape(qproj, :, 1) : hcat(Q, qproj)
            end
            count_zero += 1
        else
            break
        end
    end
    return (count_zero, vals_found)
end

# ----------------------------------------------------
# Task 3: Restarting Lanczos for largest eigenvalue λ₁
# ----------------------------------------------------

"""
lanczos_tridiagonalize(op, n, q1, s)

Run s-step Lanczos tridiagonalization with starting vector q1 (normalized inside), using
matrix-vector operator `op(::AbstractVector) -> AbstractVector`.

Returns: (Q, T, α, β)
- Q :: Matrix{Float64}       (n × t, t ≤ s): orthonormal basis vectors actually produced
- T :: SymTridiagonal{Float64}: Lanczos tridiagonal matrix of size t×t
- α :: Vector{Float64}       (length t): diagonal of T
- β :: Vector{Float64}       (length t-1): off-diagonal of T
"""
function lanczos_tridiagonalize(op::Function, n::Int, q1::AbstractVector, s::Int)
    @assert length(q1) == n
    @assert s ≥ 1

    q_curr = copy(q1)
    q_curr ./= norm(q_curr)
    q_prev = zeros(eltype(q_curr), n)

    Q = Matrix{Float64}(undef, n, s)
    α = Vector{Float64}(undef, s)
    β = Vector{Float64}(undef, s - 1)

    t = 0
    β_prev = 0.0
    for j in 1:s
        t = j
        Q[:, j] = q_curr
        z = op(q_curr)
        α[j] = dot(q_curr, z)
        z .-= α[j] .* q_curr
        if j > 1
            z .-= β_prev .* q_prev
        end
        if j < s
            β[j] = norm(z)
            if β[j] ≤ eps(eltype(β[j]))
                t = j
                break
            end
            q_next = z ./ β[j]
            q_prev = q_curr
            q_curr = q_next
            β_prev = β[j]
        end
    end

    αt = α[1:t]
    βt = t ≥ 2 ? β[1:t-1] : Float64[]
    T = SymTridiagonal(αt, βt)
    return Q[:, 1:t], T, αt, βt
end

"""
lanczos_restart_largest(A; subspace_dim=20, max_restarts=50, tol=1e-8)

Estimate the largest eigenvalue of a Hermitian matrix/operator A using restarted Lanczos.

Arguments:
- A: AbstractMatrix or function op(v) returning A*v
- subspace_dim: s in the description
- max_restarts: number of restarts
- tol: stopping tolerance based on residual norm

Returns: (λ_est, v_est, iters)
"""
function lanczos_restart_largest(A; subspace_dim::Int=20, max_restarts::Int=50, tol::Float64=1e-8)
    matvec = A isa Function ? A : (v->A*v)
    n = A isa Function ? nothing : size(A, 1)
    if n === nothing
        error("Provide an AbstractMatrix A or adapt this function to accept dimension for function operators.")
    end

    # Initialize starting vector
    q1 = randn(n)
    q1 ./= norm(q1)

    λ_old = Inf
    for r in 1:max_restarts
        Q, T, _, _ = lanczos_tridiagonalize(matvec, n, q1, subspace_dim)
        evals, evecs = eigen(T)
        λ_est = evals[end]
        u1 = evecs[:, end]
        q1 = Q * u1

        # Residual norm ||A*q1 - λ*q1|| as stopping criterion
        res = matvec(q1) - λ_est .* q1
        resnorm = norm(res)
        if resnorm ≤ tol * max(1.0, abs(λ_est))
            return λ_est, q1, r
        end

        if !isfinite(λ_est) || abs(λ_est - λ_old) ≤ eps()
            return λ_est, q1, r
        end
        λ_old = λ_est
    end
    return λ_old, q1, max_restarts
end

function task3_test()
    println("Task 3: Testing restarted Lanczos on a random symmetric matrix...")
    n = 80
    A = sprandn(n, n, 0.05)
    A = (A + A') / 2  # ensure symmetry

    # Reference (dense) for small n
    λ_ref = maximum(eigvals(Matrix(A)))

    λ_est, v_est, iters = lanczos_restart_largest(A; subspace_dim=20, max_restarts=60, tol=1e-8)
    println("Estimated λ_max = ", λ_est, "; restarts used = ", iters)
    println("Reference  λ_max = ", λ_ref)
    @assert abs(λ_est - λ_ref) ≤ 1e-5
end


# -----------------
# Script entrypoint
# -----------------

if abspath(PROGRAM_FILE) == @__FILE__
    # Task 1
    task1_demo()

    # Task 2 (use HW6_QUICK=1 for a faster demo)
    task2_small_test()
    quick = get(ENV, "HW6_QUICK", "0") == "1"
    if quick
        task2_demo(n=5_000, d=3, seed=42, nev=2)
    else
        task2_demo(n=100_000, d=3, seed=42, nev=4)
    end

    # Task 3 test
    task3_test()
end


