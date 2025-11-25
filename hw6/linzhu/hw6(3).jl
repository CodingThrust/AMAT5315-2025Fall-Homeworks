
using LinearAlgebra, Random, SparseArrays

# Lanczos tridiagonalization


function lanczos_tridiagonalization(A, q1, s)
    n = size(A, 1)

    Q = zeros(eltype(A), n, s)
    α = zeros(real(eltype(A)), s)
    β = zeros(real(eltype(A)), s-1)

    Q[:, 1] = q1
    v = A * q1
    α[1] = real(q1' * v)
    v -= α[1] * q1

    for j = 2:s
        β[j-1] = norm(v)

        # Lucky breakdown: Krylov space is invariant
        if β[j-1] < 1e-14
            Q = Q[:, 1:j-1]
            α = α[1:j-1]
            β = β[1:j-2]
            break
        end

        Q[:, j] = v / β[j-1]
        v = A * Q[:, j]
        α[j] = real(Q[:, j]' * v)

        # Lanczos recurrence
        v -= α[j] * Q[:,j] + β[j-1] * Q[:,j-1]

        # (Optional) full reorthogonalization
        for k = 1:j
            v -= (Q[:, k]' * v) * Q[:, k]
        end
    end

    k = length(α)
    T = zeros(real(eltype(A)), k, k)

    for i = 1:k
        T[i,i] = α[i]
        if i < k
            T[i, i+1] = β[i]
            T[i+1, i] = β[i]
        end
    end

    return Q, T
end


# Restarting Lanczos Algorithm

function restarting_lanczos(A, q1_init, s, max_restarts; tol=1e-10, verbose=true)
    q1 = q1_init / norm(q1_init)
    prev_val = 0.0

    for r = 1:max_restarts
        # Step i: Lanczos tridiagonalization
        Q, _ = lanczos_tridiagonalization(A, q1, s)

        # Step ii: form Ts = Q' A Q
        Ts = Q' * (A * Q)

        # Step iii: diagonalize Ts as Hermitian
        eigvals, eigvecs = eigen(Hermitian(Ts))   # <--- FIXED HERE!
        idx = sortperm(eigvals, rev=true)
        θ = eigvals[idx]
        U = eigvecs[:, idx]

        # Step iv: restart vector q1 = Q * u1
        q1 = Q * U[:,1]
        q1 /= norm(q1)

        λ = θ[1]
        verbose && println("Restart $r: λ ≈ $λ")

        if r > 1 && abs(λ - prev_val) < tol
            verbose && println("Converged after $r restarts.")
            return λ, q1
        end

        prev_val = λ
    end

    verbose && println("Hit max restarts without strict convergence.")
    return prev_val, q1
end


# Tests

println("Testing Restarting Lanczos Algorithm")
println("="^60)

n = 100
Random.seed!(123)

M = randn(n, n) + im * randn(n, n)
A_test = (M + M') / 2    # Hermitian

q1_init = randn(ComplexF64, n)

s = 20
max_restarts = 20

λ_approx, v_approx = restarting_lanczos(A_test, q1_init, s, max_restarts)

# Compare to exact eigenvalue
eig_test = eigen(Hermitian(A_test))
λ_exact = maximum(eig_test.values)

println("\nDense Matrix Test:")
println("Lanczos largest eigenvalue: $λ_approx")
println("Exact largest eigenvalue  : $λ_exact")
println("Absolute error            : ", abs(λ_exact - λ_approx))
println("Residual norm             : ", norm(A_test * v_approx - λ_approx * v_approx))


println("\n" * "="^60)
println("Testing on large sparse Hermitian matrix")
println("="^60)

n_large = 1000
Random.seed!(456)

A_sparse = sprand(n_large, n_large, 0.01)
A_sparse = (A_sparse + A_sparse') / 2 + 10I   # symmetric & positive definite

q1_large = randn(n_large)

λ_sparse, v_sparse = restarting_lanczos(A_sparse, q1_large, 30, 15, verbose=false)

println("Largest eigenvalue (sparse): $λ_sparse")
println("Residual norm             : ", norm(A_sparse * v_sparse - λ_sparse * v_sparse))
