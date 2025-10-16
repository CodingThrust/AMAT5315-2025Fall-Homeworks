# Problem 3: Restarting Lanczos Algorithm
# Implement a restarting Lanczos algorithm to find the largest eigenvalue

using LinearAlgebra, Random, SparseArrays

"""
    lanczos_tridiagonalization(A, q1, s)

Perform Lanczos tridiagonalization starting with vector q1 for s iterations.

# Arguments
- `A`: Hermitian matrix
- `q1`: Initial unit vector
- `s`: Number of iterations

# Returns
- `Q`: Matrix of Lanczos vectors (n × s)
- `T`: Tridiagonal matrix (s × s)
"""
function lanczos_tridiagonalization(A, q1, s)
    n = size(A, 1)
    Q = zeros(ComplexF64, n, s)
    α = zeros(Float64, s)
    β = zeros(Float64, s-1)
    
    Q[:, 1] = q1
    v = A * q1
    α[1] = real(dot(q1, v))
    v = v - α[1] * q1
    
    for j = 2:s
        β[j-1] = norm(v)
        if β[j-1] < 1e-14
            # Lucky breakdown - we've found an invariant subspace
            Q = Q[:, 1:j-1]
            α = α[1:j-1]
            β = β[1:j-2]
            break
        end
        
        Q[:, j] = v / β[j-1]
        v = A * Q[:, j]
        α[j] = real(dot(Q[:, j], v))
        v = v - α[j] * Q[:, j] - β[j-1] * Q[:, j-1]
        
        # Reorthogonalization to maintain numerical stability
        for k = 1:j
            v = v - dot(Q[:, k], v) * Q[:, k]
        end
    end
    
    # Construct tridiagonal matrix T
    s_actual = length(α)
    T = zeros(Float64, s_actual, s_actual)
    for i = 1:s_actual
        T[i, i] = α[i]
        if i < s_actual
            T[i, i+1] = β[i]
            T[i+1, i] = β[i]
        end
    end
    
    return Q, T
end

"""
    restarting_lanczos(A, q1_init, s, max_restarts)

Restarting Lanczos algorithm to find the largest eigenvalue of a Hermitian matrix.

# Algorithm:
1. Generate q2, ..., qs via Lanczos algorithm
2. Form Ts = Q' * A * Q, an s-by-s matrix
3. Compute orthogonal matrix U such that U' * Ts * U = diag(θ1, ..., θs) with θ1 ≥ ... ≥ θs
4. Set q1_new = Q * u1

# Arguments
- `A`: Hermitian matrix
- `q1_init`: Initial unit vector
- `s`: Number of Lanczos vectors to generate before restarting
- `max_restarts`: Maximum number of restarts

# Returns
- `largest_eigenvalue`: Approximation of the largest eigenvalue
- `eigenvector`: Corresponding eigenvector
"""
function restarting_lanczos(A, q1_init, s, max_restarts)
    q1 = copy(q1_init)
    q1 = q1 / norm(q1)  # Ensure unit norm
    
    largest_eig = 0.0
    eigvec = q1
    prev_eig = 0.0
    
    for restart = 1:max_restarts
        # Step 1: Generate q2, ..., qs via Lanczos algorithm
        Q, T = lanczos_tridiagonalization(A, q1, s)
        
        # Step 2: T is already formed (T = Q' * A * Q)
        
        # Step 3: Compute eigenvectors of T
        # Get eigenvalues and eigenvectors, sorted in descending order
        eig_vals, eig_vecs = eigen(T)
        sorted_indices = sortperm(eig_vals, rev=true)
        θ = eig_vals[sorted_indices]
        U = eig_vecs[:, sorted_indices]
        
        # Step 4: Set q1_new = Q * u1
        u1 = U[:, 1]
        q1_new = Q * u1
        q1_new = q1_new / norm(q1_new)  # Ensure unit norm
        
        # Store largest eigenvalue and eigenvector
        largest_eig = θ[1]
        eigvec = q1_new
        
        # Check convergence
        if restart > 1
            eig_change = abs(largest_eig - prev_eig)
            if eig_change < 1e-10
                println("Converged after $restart restarts")
                break
            end
        end
        prev_eig = largest_eig
        
        # Restart with new q1
        q1 = q1_new
    end
    
    return largest_eig, eigvec
end

# Test the implementation
println("Testing Restarting Lanczos Algorithm")
println("=" ^ 60)

# Create a test Hermitian matrix with known eigenvalues
n = 100
Random.seed!(123)
A_test = randn(n, n) + im * randn(n, n)
A_test = (A_test + A_test') / 2  # Make it Hermitian

# Random initial vector
q1_init = randn(ComplexF64, n)

# Run restarting Lanczos
s = 20  # Subspace size
max_restarts = 10
largest_eig_approx, eigvec_approx = restarting_lanczos(A_test, q1_init, s, max_restarts)

# Compare with exact solution using Julia's eigen
eig_exact = eigen(Hermitian(A_test))
largest_eig_exact = maximum(eig_exact.values)

println("\nResults:")
println("Largest eigenvalue (Restarting Lanczos): $largest_eig_approx")
println("Largest eigenvalue (Exact):              $largest_eig_exact")
println("Absolute error:                          $(abs(largest_eig_approx - largest_eig_exact))")

# Verify eigenvector
residual = norm(A_test * eigvec_approx - largest_eig_approx * eigvec_approx)
println("Residual ||A*v - λ*v||:                  $residual")

# Test on a larger sparse matrix
println("\n" * "=" ^ 60)
println("Testing on a larger sparse symmetric matrix")
println("=" ^ 60)

n_large = 1000
Random.seed!(456)
# Create a sparse symmetric positive definite matrix
A_sparse = sprand(n_large, n_large, 0.01)
A_sparse = (A_sparse + A_sparse') / 2 + 10 * I  # Make symmetric and positive definite

q1_init_large = randn(n_large)
largest_eig_sparse, _ = restarting_lanczos(A_sparse, q1_init_large, 30, 15)

println("\nLargest eigenvalue (Restarting Lanczos): $largest_eig_sparse")
println("This demonstrates the algorithm works on large sparse matrices.")

