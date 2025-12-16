# HW6
## Task1
```
using SparseArrays

# Define input arrays derived from CSC structure
rowindices = [3, 1, 1, 4, 5]  # Row indices of non-zeros (matches rowval)
colindices = [1, 2, 3, 3, 4]  # Column indices of non-zeros
data = [0.799, 0.942, 0.848, 0.164, 0.637]  # Values of non-zeros (matches nzval)

# Construct 5x5 sparse matrix in CSC format
sp = sparse(rowindices, colindices, data, 5, 5)

# Verify results
println("=== Verification ===")
println("colptr: ", sp.colptr, " (expected: [1,2,3,5,6,6])")
println("rowval: ", sp.rowval, " (expected: [3,1,1,4,5])")
println("nzval: ", sp.nzval, " (expected: [0.799,0.942,0.848,0.164,0.637])")
println("Rows (m): ", sp.m, " (expected: 5)")
println("Cols (n): ", sp.n, " (expected: 5)")
```

## Task2
```
using Graphs, Random, KrylovKit

# Generate random 3-regular graph with 100,000 nodes
Random.seed!(42)
g = random_regular_graph(100000, 3)

# Compute sparse Laplacian matrix (memory-efficient for large graphs)
L = laplacian_matrix(g, T=Float64)

# Find smallest-magnitude eigenvalues (focus on near-zero values)
# :SM = smallest magnitude; 10 eigenvalues suffice for component detection
eigenvalues, _, _ = eigsolve(L, 10, :SM; tol=1e-10, maxiter=1000)

# Count effective zero eigenvalues (account for numerical error)
zero_threshold = 1e-8
num_components = count(λ -> abs(λ) < zero_threshold, eigenvalues)

# Result: number of connected components
println("Number of connected components: ", num_components)
```

## Task3
```
using LinearAlgebra
using Random

function restarting_lanczos(A, s; max_restarts=100, tol=1e-6)
    n = size(A, 1)
    s = min(s, n)  # Ensure restart length ≤ matrix size

    # Initialize with random normalized vector (complex to handle general Hermitian matrices)
    q = randn(ComplexF64, n)
    q /= norm(q)

    prev_λ = Inf  # Track previous estimate for convergence check

    for restart in 1:max_restarts
    
        # Step 1: Generate s Lanczos vectors and tridiagonal matrix T_s
        Q = Matrix{ComplexF64}(undef, n, s)  # Lanczos vectors [q₁ ... qₛ]
        α = Vector{Float64}(undef, s)        # Diagonal entries of T_s (real for Hermitian A)
        β = Vector{Float64}(undef, s-1)      # Sub/superdiagonal entries of T_s (real for Hermitian A)

        # First Lanczos vector
        Q[:, 1] = q
        v = A * Q[:, 1]
        α[1] = real(dot(Q[:, 1]', v))  # q₁† A q₁ (real for Hermitian A)
        v .-= α[1] * Q[:, 1]

        # Generate remaining s-1 vectors
        for j in 2:s
            β[j-1] = norm(v)
            β[j-1] < eps() && break  # Early exit if no new orthogonal vectors

            Q[:, j] = v / β[j-1]
            v = A * Q[:, j]
            α[j] = real(dot(Q[:, j]', v))  # qⱼ† A qⱼ
            v .-= α[j] * Q[:, j] + β[j-1] * Q[:, j-1]
        end

        # Step 2: Form tridiagonal matrix T_s (s×s)
        T = Tridiagonal(β, α[1:length(β)+1], β)  # Symmetric for Hermitian A

        # Step 3: Diagonalize T_s to get sorted eigenvalues/vectors
        eig_decomp = eigen(Matrix(T))  # Convert to dense for eigen decomposition
        perm = sortperm(eig_decomp.values, rev=true)  # Sort descending
        θ = eig_decomp.values[perm]    # Eigenvalues of T_s
        U = eig_decomp.vectors[:, perm]  # Eigenvectors of T_s

        # Step 4: Restart with new initial vector q₁ = Q * u₁
        q = Q * U[:, 1]
        q /= norm(q)  # Re-normalize

        # Check convergence (change in largest eigenvalue estimate)
        if abs(θ[1] - prev_λ) < tol
            return θ[1]
        end
        prev_λ = θ[1]
    end

    @warn "Maximum restarts ($max_restarts) reached. Returning last estimate."
    return prev_λ
end


Random.seed!(42)

# Test 1: Diagonal matrix (known largest eigenvalue = 10.0)
A_diag = Diagonal([10.0, 8.0, 5.0, 3.0, 1.0])
λ_diag = restarting_lanczos(A_diag, 3)  # Restart length = 3
println("Test 1 - Diagonal matrix:")
println("  Expected largest eigenvalue: 10.0")
println("  Computed: ", λ_diag, "\n")

# Test 2: Random real symmetric matrix (Hermitian)
n = 10
A_rand = randn(n, n)
A_rand = A_rand + A_rand'  # Make symmetric (Hermitian)
true_max = maximum(eigvals(A_rand))  # True largest eigenvalue
λ_rand = restarting_lanczos(A_rand, 5)  # Restart length = 5
println("Test 2 - Random symmetric matrix:")
println("  True largest eigenvalue: ", true_max)
println("  Computed: ", λ_rand)
```