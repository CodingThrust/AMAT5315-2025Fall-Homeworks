using LinearAlgebra
using SparseArrays
using Random
import Pkg

println("Running Linear Algebra Script...")
println("--------------------------------\n")

# ==============================================================================
# Part 1: Verification of CSC Structure
# ==============================================================================
println("=== Part 1: CSC structure ===")

# Arrays that reproduce the given CSC structure
rowindices = [3, 1, 1, 4, 5]
colindices = [1, 2, 3, 3, 4]
data       = [0.799, 0.942, 0.848, 0.164, 0.637]

# Construct the sparse matrix
sp = sparse(rowindices, colindices, data, 5, 5)

# Verification Output
println("sp.colptr = ", sp.colptr)
println("sp.rowval = ", sp.rowval)
println("sp.nzval  = ", sp.nzval)
println("sp.m, sp.n = ", (sp.m, sp.n))
println() 


# ==============================================================================
# Part 2: Graph Spectral Analysis with KrylovKit
# ==============================================================================
# Note: Wrapped in a try-catch-let block for safety and performance.

let
    try
        using Graphs
        using KrylovKit
        
        println("=== Part 2: Graph Spectral Analysis ===")
        println("(Graphs & KrylovKit loaded, starting computation...)")

        # 1. Fix random seed
        Random.seed!(42)

        # 2. Generate a random 3-regular graph with 100,000 nodes
        n_vertices = 100_000
        degree = 3
        g = random_regular_graph(n_vertices, degree)

        # 3. Build sparse adjacency and Laplacian L = 3I - A
        A = adjacency_matrix(g)      # sparse adjacency matrix
        A = Float64.(A)              # ensure floating-point arithmetic
        L = 3.0 * I - A              # sparse Laplacian

        # 4. Use KrylovKit to find smallest eigenvalues
        nev = 10                     
        v0 = randn(n_vertices)       

        # Lanczos execution
        lambdas, vecs, info = eigsolve(L, v0, nev, :SR; issymmetric = true)

        println("Approximate smallest eigenvalues of L:")
        println(lambdas)

        # 5. Estimate connected components
        tol = 1e-6
        num_components = count(abs.(lambdas) .< tol)

        println("Estimated number of connected components = $num_components")
        println()

    catch err
        println("=== Part 2: Graph Spectral Analysis ===")
        println("Warning: Graphs or KrylovKit not installed.")
        println("Error: $err")
        println()
    end
end


# ==============================================================================
# Part 3: Restarted Lanczos Algorithm
# ==============================================================================

# -------------------------------------------------------
# Helper Function: Unrestarted Lanczos Step
# -------------------------------------------------------
function lanczos_run(A::AbstractMatrix, q1::AbstractVector, m::Int)
    n = size(A, 1)
    
    # Ensure q1 is normalized
    q = copy(q1)
    q ./= norm(q)

    # Storage
    Q = zeros(eltype(A), n, m)
    alpha = zeros(Float64, m)
    beta = zeros(Float64, m - 1)

    q_prev = zeros(eltype(A), n)
    
    # Main Lanczos Loop
    for j in 1:m
        Q[:, j] .= q
        w = A * q
        
        # Rayleigh quotient
        alpha[j] = real(dot(conj(q), w))

        # Orthogonalize
        if j > 1
            w .-= beta[j - 1] .* q_prev
        end
        w .-= alpha[j] .* q

        # Next beta_j
        beta_j = norm(w)
        
        if j == m
            break
        end

        # Check for breakdown
        if beta_j == 0.0
            beta = beta[1:j-1]
            alpha = alpha[1:j]
            Q = Q[:, 1:j]
            return Q, alpha, beta
        end

        # Prepare for next iteration
        beta[j] = beta_j
        q_next = w ./ beta_j
        q_prev .= q
        q .= q_next
    end

    return Q, alpha, beta
end

# -------------------------------------------------------
# Main Function: Restarted Lanczos
# -------------------------------------------------------
function restarted_lanczos_largest_eig(
    A::AbstractMatrix;
    m::Int = 20,
    max_restarts::Int = 20,
    tol::Real = 1e-8,
    v0 = nothing,
)
    n = size(A, 1)
    T = eltype(A)

    # Initial vector setup
    if v0 === nothing
        Random.seed!(1234)
        v = randn(T, n)
    else
        v = copy(v0)
    end
    v ./= norm(v)

    lambda_old = -Inf

    # Restart Loop
    for restart in 1:max_restarts
        # 1. Run Lanczos
        Q, alpha, beta = lanczos_run(A, v, m)
        s = size(Q, 2)

        # 2. Form tridiagonal Ritz matrix and solve
        if s == 1
            lambda = alpha[1]
            v = Q[:, 1]
        else
            T_s = SymTridiagonal(alpha[1:s], beta[1:s-1])
            evals, evecs = eigen(T_s)
            
            # Largest eigenvalue is the last one
            lambda = evals[end]
            u1 = evecs[:, end]

            # 3. Form new starting vector
            v = Q * u1
        end

        # Normalize Ritz vector
        v ./= norm(v)

        # 4. Check convergence
        r = A * v .- lambda .* v
        resnorm = norm(r)

        println("Restart $restart: lambda ≈ $lambda, res ≈ $resnorm")

        if resnorm <= tol
            println(">> Converged with residual <= $tol.")
            return lambda, v
        end

        # Optional: check eigenvalue stabilization
        if isfinite(lambda_old) && abs(lambda - lambda_old) <= tol * max(1.0, abs(lambda))
            println(">> Eigenvalue stabilized within tolerance.")
            return lambda, v
        end
        lambda_old = lambda
    end

    println("!! Maximum number of restarts reached without meeting tolerance.")
    return lambda_old, v
end

# -------------------------------------------------------
# Test Script for Part 3
# -------------------------------------------------------
println("=== Part 3: Restarted Lanczos Test ===")

# 1. Create a random symmetric matrix
Random.seed!(2024)
n_test = 50
M = randn(n_test, n_test)
A_test = (M + M') / 2   

# 2. Compute True largest eigenvalue
lambda_true = maximum(eigvals(Symmetric(A_test)))

# 3. Compute Approximate via Restarted Lanczos
lambda_approx, v_approx = restarted_lanczos_largest_eig(
    A_test;
    m = 15,
    max_restarts = 30,
    tol = 1e-8,
)

# 4. Compare Results
println("\n--- Results Comparison ---")
println("True largest eigenvalue    = $lambda_true")
println("Lanczos-approximated value = $lambda_approx")
println("Absolute error             = ", abs(lambda_true - lambda_approx))

if isapprox(lambda_approx, lambda_true; rtol = 1e-6, atol = 1e-6)
    println("\n[Test PASSED]: Lanczos approximation is accurate.")
else
    println("\n[Test FAILED]: Lanczos approximation is not accurate enough.")
end

println("\n==========================================================")
println("Done.")
println("==========================================================")











# === Part 1: CSC structure ===
# sp.colptr = [1, 2, 3, 5, 6, 6]
# sp.rowval = [3, 1, 1, 4, 5]
# sp.nzval  = [0.799, 0.942, 0.848, 0.164, 0.637]
# sp.m, sp.n = (5, 5)

# === Part 2: Graph Spectral Analysis ===
# (Graphs & KrylovKit loaded, starting computation...)
# ┌ Warning: Lanczos eigsolve stopped without convergence after 100 iterations:
# │ * 1 eigenvalues converged
# │ * norm of residuals = (0.00e+00, 6.56e-11, 4.28e-09, 4.32e-07, 3.51e-05, 1.21e-04, 8.70e-04, 1.83e-03, 5.77e-04, 2.74e-03)
# │ * number of operations = 1218
# └ @ KrylovKit ~/.julia/packages/KrylovKit/ZcdRg/src/eigsolve/lanczos.jl:142
# Approximate smallest eigenvalues of L:
# [-4.03067974302993e-15, 0.17173162534019812, 0.17216583440834132, 0.1726794662739956, 0.1729978460651408, 0.17333702314959834, 0.17361451820294493, 0.17371315826177905, 0.17397936255915036, 0.17463428632808]
# Estimated number of connected components = 1

# === Part 3: Restarted Lanczos Test ===
# Restart 1: lambda ≈ 9.783704966910726, res ≈ 0.002464309813412632
# Restart 2: lambda ≈ 9.783705906886063, res ≈ 5.897939754901747e-7
# Restart 3: lambda ≈ 9.783705906886103, res ≈ 1.2337858645503117e-10
# >> Converged with residual <= 1.0e-8.

# --- Results Comparison ---
# True largest eigenvalue    = 9.783705906886105
# Lanczos-approximated value = 9.783705906886103
# Absolute error             = 1.7763568394002505e-15

# [Test PASSED]: Lanczos approximation is accurate.

# ==========================================================
# Done.
# ==========================================================