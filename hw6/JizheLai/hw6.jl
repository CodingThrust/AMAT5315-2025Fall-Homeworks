############################################################
# Homework 6 — Julia solutions (.jl)
############################################################

############################################################
# Problem 1: Sparse Matrix Construction (CSC format)
############################################################

using SparseArrays

# Arrays that reproduce the given CSC structure
rowindices = [3, 1, 1, 4, 5]
colindices = [1, 2, 3, 3, 4]
data       = [0.799, 0.942, 0.848, 0.164, 0.637]

sp = sparse(rowindices, colindices, data, 5, 5)

println("=== Problem 1: Verification of CSC structure ===")
println("sp.colptr = ", sp.colptr)
println("sp.rowval = ", sp.rowval)
println("sp.nzval  = ", sp.nzval)
println("sp.m, sp.n = ", (sp.m, sp.n))
println()


############################################################
# Problem 2: Graph Spectral Analysis with KrylovKit
############################################################
#
# 说明：
#   - 这一题需要 Graphs.jl 和 KrylovKit.jl。
#   - 如果本机没装它们，可以先运行：
#       import Pkg
#       Pkg.add("Graphs")
#       Pkg.add("KrylovKit")
#
# 为了避免在“没装包”的环境下一跑整份 hw6.jl 就报错，
# 我们把真正用到 Graphs 的代码放在一个 let-block 里，
# 并在前面加一个 try/catch，捕获缺包的情况，只打印提示。

let
    try
        using Graphs
        using Random
        using KrylovKit
        using LinearAlgebra

        println("=== Problem 2: Graph Spectral Analysis ===")
        println("(如果你看到这一段输出，说明 Graphs & KrylovKit 已成功加载)")

        # Fix random seed for reproducibility
        Random.seed!(42)

        # Generate a random 3-regular graph with 100,000 nodes
        n_vertices = 100_000
        degree = 3
        g = random_regular_graph(n_vertices, degree)

        # Build sparse adjacency and Laplacian L = 3I - A
        A = adjacency_matrix(g)      # sparse adjacency matrix
        A = Float64.(A)              # ensure floating-point arithmetic
        n = size(A, 1)
        L = 3.0 * I - A              # sparse Laplacian (3-regular => D = 3I)

        # Use KrylovKit to find the smallest-magnitude eigenvalues.
        # For Laplacian, the number of zero eigenvalues = # of connected components.
        nev = 10                     # number of eigenvalues to compute
        v0 = randn(n)                # random initial vector

        λs, vecs, info = eigsolve(L, v0, nev, :SR; issymmetric = true)

        println("Approximate smallest eigenvalues of L:")
        println(λs)

        tol = 1e-6
        num_components = count(abs.(λs) .< tol)

        println("Estimated number of connected components = $num_components")
        println("(对这个随机 3-正则图来说，通常是 1 个连通分量)")
        println()

    catch err
        # 如果 Graphs 或 KrylovKit 未安装，这里给出提示，不让整个文件报错退出
        println("=== Problem 2: Graph Spectral Analysis ===")
        println("未能加载 Graphs 或 KrylovKit：$err")
        println("如果要运行本题代码，请先在 REPL 中执行：")
        println("    import Pkg")
        println("    Pkg.add(\"Graphs\")")
        println("    Pkg.add(\"KrylovKit\")")
        println()
    end
end


############################################################
# Problem 3: Restarting Lanczos Algorithm (largest eigenvalue)
############################################################

using LinearAlgebra
using Random

"""
    lanczos_run(A, q1, m)

Run m steps of (unrestarted) Lanczos on Hermitian matrix `A`
starting from normalized vector `q1`.

Returns:
    Q :: Matrix{T}       -- n×s matrix of orthonormal Lanczos vectors
    α :: Vector{Float64} -- diagonal entries of T_s (length s)
    β :: Vector{Float64} -- off-diagonal entries of T_s (length s-1)

where s ≤ m is the actual number of Lanczos vectors generated
(if a breakdown occurs).
"""
function lanczos_run(A::AbstractMatrix, q1::AbstractVector, m::Int)
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"
    @assert length(q1) == n "q1 has incompatible dimension"
    @assert m >= 1 "m must be at least 1"

    # Ensure q1 is normalized
    q = copy(q1)
    q ./= norm(q)

    # Storage (allocate maximum size; we may truncate if early termination)
    Q = zeros(eltype(A), n, m)
    α = zeros(Float64, m)
    β = zeros(Float64, m - 1)

    q_prev = zeros(eltype(A), n)
    for j in 1:m
        Q[:, j] .= q
        w = A * q
        # Rayleigh quotient (real for Hermitian A)
        α[j] = real(dot(conj(q), w))

        # Orthogonalize against previous Lanczos vectors (3-term recurrence)
        if j > 1
            w .-= β[j - 1] .* q_prev
        end
        w .-= α[j] .* q

        # Next β_j
        β_j = norm(w)
        if j == m
            # No need to form next q if we reached target dimension
            break
        end

        if β_j == 0.0
            # Exact Krylov subspace: terminate early
            β = β[1:j-1]
            α = α[1:j]
            Q = Q[:, 1:j]
            return Q, α, β
        end

        # Prepare for next iteration
        β[j] = β_j
        q_next = w ./ β_j
        q_prev .= q
        q .= q_next
    end

    # If we reached full m without breakdown, return all entries
    return Q, α, β
end

"""
    restarted_lanczos_largest_eig(A; m=20, max_restarts=20, tol=1e-8, v0=nothing)

Approximate the largest eigenvalue and corresponding eigenvector
of a Hermitian matrix `A` using a restarted Lanczos algorithm
with subspace size `m`.

Arguments:
    A            :: AbstractMatrix (Hermitian or real symmetric)
    m            :: Int        - Lanczos subspace dimension per restart
    max_restarts :: Int        - maximum number of restart cycles
    tol          :: Real       - tolerance for residual norm
    v0           :: AbstractVector or `nothing`
                                  initial guess for eigenvector.
                                  If `nothing`, a random vector is used.

Returns:
    λ_approx :: Float64         - approximation to largest eigenvalue
    v_approx :: Vector{T}       - approximate eigenvector (‖v_approx‖₂ = 1)
"""
function restarted_lanczos_largest_eig(
    A::AbstractMatrix;
    m::Int = 20,
    max_restarts::Int = 20,
    tol::Real = 1e-8,
    v0 = nothing,
)
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"
    @assert m >= 1 "m must be at least 1"
    @assert max_restarts >= 1 "max_restarts must be at least 1"

    T = eltype(A)

    # Initial vector
    if v0 === nothing
        Random.seed!(1234)  # for reproducibility
        v = randn(T, n)
    else
        @assert length(v0) == n "v0 has incompatible dimension"
        v = copy(v0)
    end
    v ./= norm(v)

    λ_old = -Inf

    for restart in 1:max_restarts
        # 1. Run Lanczos with starting vector v (treated as q1)
        Q, α, β = lanczos_run(A, v, m)
        s = size(Q, 2)

        # 2. Form tridiagonal Ritz matrix T_s
        if s == 1
            # Only one vector: Rayleigh quotient is the eigenvalue approx.
            λ = α[1]
            v = Q[:, 1]
        else
            T_s = SymTridiagonal(α[1:s], β[1:s-1])

            # 3. Eigen-decomposition of T_s
            evals, evecs = eigen(T_s)
            # eigen returns ascending order; largest is last
            λ = evals[end]
            u1 = evecs[:, end]

            # 4. Form new starting vector q₁(new) = Q * u₁
            v = Q * u1
        end

        # Normalize Ritz vector (should already be ~1)
        v ./= norm(v)

        # Compute residual norm to test convergence
        r = A * v .- λ .* v
        resnorm = norm(r)

        println("Restart $restart: λ ≈ $λ, ‖r‖₂ ≈ $resnorm")

        if resnorm <= tol
            println("Converged with residual ≤ $tol.")
            return λ, v
        end

        # Optional: also monitor change in eigenvalue
        if isfinite(λ_old) && abs(λ - λ_old) <= tol * max(1.0, abs(λ))
            println("Eigenvalue stabilized within tolerance.")
            return λ, v
        end
        λ_old = λ
    end

    println("Maximum number of restarts reached without meeting tolerance.")
    return λ_old, v
end

############################################################
# Test for Problem 3
############################################################

println("=== Problem 3: Restarted Lanczos Test ===")

# Create a random symmetric matrix
Random.seed!(2024)
n_test = 50
M = randn(n_test, n_test)
A_test = (M + M') / 2   # make it symmetric real

# True largest eigenvalue (dense computation)
λ_true = maximum(eigvals(Symmetric(A_test)))

# Approximate via restarted Lanczos
λ_approx, v_approx = restarted_lanczos_largest_eig(
    A_test;
    m = 15,
    max_restarts = 30,
    tol = 1e-8,
)

println("True largest eigenvalue    = $λ_true")
println("Lanczos-approximated value = $λ_approx")
println("Absolute error             = ", abs(λ_true - λ_approx))

if isapprox(λ_approx, λ_true; rtol = 1e-6, atol = 1e-6)
    println("Test PASSED: Lanczos approximation is accurate.")
else
    println("Test FAILED: Lanczos approximation is not accurate enough.")
end


# === Problem 1: Verification of CSC structure ===
# sp.colptr = [1, 2, 3, 5, 6, 6]
# sp.rowval = [3, 1, 1, 4, 5]
# sp.nzval  = [0.799, 0.942, 0.848, 0.164, 0.637]
# sp.m, sp.n = (5, 5)

# === Problem 2: Graph Spectral Analysis ===
# (如果你看到这一段输出，说明 Graphs & KrylovKit 已成功加载)
# ┌ Warning: Lanczos eigsolve stopped without convergence after 100 iterations:
# │ * 1 eigenvalues converged
# │ * norm of residuals = (0.00e+00, 6.56e-11, 4.28e-09, 4.32e-07, 3.51e-05, 1.21e-04, 8.70e-04, 1.83e-03, 5.77e-04, 2.74e-03)
# │ * number of operations = 1218
# └ @ KrylovKit ~/.julia/packages/KrylovKit/ZcdRg/src/eigsolve/lanczos.jl:142
# Approximate smallest eigenvalues of L:
# [-1.6646519531261627e-15, 0.17173162534019926, 0.17216583440834127, 0.17267946627399602, 0.17299784606514018, 0.17333702314959812, 0.17361451820294457, 0.17371315826177844, 0.17397936255914978, 0.17463428632807945]
# Estimated number of connected components = 1
# (对这个随机 3-正则图来说，通常是 1 个连通分量)

# === Problem 3: Restarted Lanczos Test ===
# Restart 1: λ ≈ 9.783704966910726, ‖r‖₂ ≈ 0.002464309813412003
# Restart 2: λ ≈ 9.783705906886063, ‖r‖₂ ≈ 5.897939764936505e-7
# Restart 3: λ ≈ 9.783705906886109, ‖r‖₂ ≈ 1.2337992294516778e-10
# Converged with residual ≤ 1.0e-8.
# True largest eigenvalue    = 9.78370590688611
# Lanczos-approximated value = 9.783705906886109
# Absolute error             = 1.7763568394002505e-15
# Test PASSED: Lanczos approximation is accurate.