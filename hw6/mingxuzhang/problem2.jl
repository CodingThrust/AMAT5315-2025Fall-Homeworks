# Problem 2: Graph Spectral Analysis
# Find the number of connected components by analyzing eigenvalues of the Laplacian

using Graphs, Random, KrylovKit, LinearAlgebra, SparseArrays

Random.seed!(42)
g = random_regular_graph(100000, 3)

# First, check connectivity using a simpler method (for verification)
println("Quick check: Is the graph connected? ", is_connected(g))
num_components_direct = length(connected_components(g))
println("Number of connected components (direct method): $num_components_direct")

# Now use spectral analysis with KrylovKit
println("\n" * "="^60)
println("Spectral Analysis using KrylovKit")
println("="^60)

# Construct the Laplacian matrix
# L = D - A where D is degree matrix and A is adjacency matrix
A = adjacency_matrix(g)
n = nv(g)
degrees = vec(sum(A, dims=2))
L = spdiagm(0 => degrees) - A

# The number of connected components equals the multiplicity of the zero eigenvalue
# For a large sparse matrix, we use KrylovKit to find the smallest eigenvalues

# Find the smallest eigenvalues (closest to 0)
println("Computing smallest eigenvalues using KrylovKit.eigsolve...")
n_eigs = 5  # Number of eigenvalues to compute
vals, vecs, info = eigsolve(L, n_eigs, :SR, issymmetric=true, krylovdim=20, maxiter=200, tol=1e-4)

println("\nSmallest eigenvalues of Laplacian:")
for (i, val) in enumerate(vals)
    println("λ[$i] = $(round(val, sigdigits=8))")
end

# Count eigenvalues that are approximately zero (within numerical tolerance)
tolerance = 1e-4
num_components_spectral = count(abs.(vals) .< tolerance)

println("\n" * "="^60)
println("Results:")
println("Number of connected components (spectral method): $num_components_spectral")
println("Number of connected components (direct method):   $num_components_direct")
println("="^60)

