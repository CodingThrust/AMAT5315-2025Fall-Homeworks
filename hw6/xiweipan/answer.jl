using LinearAlgebra
using SparseArrays
using Random
using KrylovKit
using Graphs

function csc_inputs()
    # CSC input arrays (sorted by column, then row).
    rowindices = [3, 1, 1, 4, 5]
    colindices = [1, 2, 3, 3, 4]
    data = [0.799, 0.942, 0.848, 0.164, 0.637]
    return rowindices, colindices, data
end

function count_components_by_eigs(; n=100000, d=3, seed=42, nev=6, tol=1e-8)
    # Build graph and Laplacian.
    Random.seed!(seed)
    g = random_regular_graph(n, d)
    L = laplacian_matrix(g)
    # Smallest eigenvalues: count how many are numerically zero.
    q1 = randn(n)
    vals, _, info = eigsolve(L, q1, nev, :SR; tol=tol / 10, maxiter=400)
    # If not all eigenpairs converged, fall back to a conservative threshold.
    thresh = info.converged == nev ? tol : max(tol, 1e-6)
    num_zero = count(v -> abs(v) < thresh, vals)
    return num_zero, vals
end

function lanczos_tridiag(A, q1, s; reorth=true, tol=1e-12)
    # Initialize the first vector and storage.
    n = length(q1)
    q = q1 / norm(q1)
    q_prev = zeros(eltype(q1), n)
    beta_prev = zero(real(eltype(q1)))

    Q = zeros(eltype(q1), n, s)
    alpha = zeros(real(eltype(q1)), s)
    beta = zeros(real(eltype(q1)), s - 1)

    for j in 1:s
        Q[:, j] = q
        # Matrix-vector multiply and three-term recurrence.
        w = A * q
        if j > 1
            w -= beta_prev * q_prev
        end
        alpha[j] = real(dot(q, w))
        w -= alpha[j] * q

        if reorth
            # Modified Gram-Schmidt for better numerical stability.
            for i in 1:j
                coeff = dot(Q[:, i], w)
                w -= coeff * Q[:, i]
            end
        end

        if j < s
            # Next Lanczos vector.
            beta_j = norm(w)
            if beta_j < tol
                Q = Q[:, 1:j]
                alpha = alpha[1:j]
                beta = beta[1:j-1]
                return Q, alpha, beta
            end
            beta[j] = beta_j
            q_prev = q
            q = w / beta_j
            beta_prev = beta_j
        end
    end

    return Q, alpha, beta
end

function restarted_lanczos_largest(A; s=20, max_restarts=20, tol=1e-8, rng=Random.GLOBAL_RNG)
    # Start from a random unit vector.
    n = size(A, 1)
    q = randn(rng, eltype(A), n)
    q /= norm(q)

    theta_prev = Inf
    theta = theta_prev

    for _ in 1:max_restarts
        # Run Lanczos, then solve the small tridiagonal problem.
        Q, alpha, beta = lanczos_tridiag(A, q, s; reorth=true, tol=tol / 10)
        T = SymTridiagonal(alpha, beta)
        eig = eigen(T)
        idx = argmax(eig.values)
        theta = eig.values[idx]
        u1 = eig.vectors[:, idx]

        # Restart with the dominant Ritz vector.
        q = Q * u1
        q /= norm(q)

        if abs(theta - theta_prev) < tol * max(1, abs(theta))
            break
        end
        theta_prev = theta
    end

    return theta, q
end

function test_restarted_lanczos()
    println("Running restarted Lanczos test...")
    Random.seed!(123)
    n = 40
    A = randn(n, n)
    A = (A + A') / 2

    theta, _ = restarted_lanczos_largest(A; s=15, max_restarts=25, tol=1e-10)
    lambda_max = maximum(eigvals(Symmetric(A)))

    println("Computed largest eigenvalue: ", theta)
    println("Reference largest eigenvalue: ", lambda_max)
    @assert isapprox(theta, lambda_max; rtol=1e-6, atol=1e-8)
    println("Test passed.")
    return theta, lambda_max
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("CSC inputs: ", csc_inputs())
    println("Estimating connected components from Laplacian eigenvalues...")
    num_zero, vals = count_components_by_eigs()
    println("Smallest eigenvalues: ", vals)
    println("Connected components (zero eigvals): ", num_zero)
    test_restarted_lanczos()
end
