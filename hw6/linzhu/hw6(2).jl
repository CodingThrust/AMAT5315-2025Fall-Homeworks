using Graphs, Random, KrylovKit, LinearAlgebra, SparseArrays

Random.seed!(42)

println("Generating 100,000-node 3-regular graph...")
g = random_regular_graph(100000, 3)

# Direct check (verification)
println("\n" * "="^60)
println("Direct Connectivity Check (for verification)")
println("="^60)

isconn = is_connected(g)
num_components_direct = length(connected_components(g))

println("Is the graph connected? $isconn")
println("Number of connected components (direct): $num_components_direct")

# Spectral analysis (main task)
println("\n" * "="^60)
println("Spectral Analysis Using Laplacian Eigenvalues")
println("="^60)

A = adjacency_matrix(g)
degrees = vec(sum(A, dims=2))             # degree vector
L = spdiagm(0 => degrees) - A             # Laplacian matrix (sparse)

println("Computing smallest eigenvalues with KrylovKit...")
n_eigs = 5   # compute 5 smallest eigenvalues

vals, vecs, info = eigsolve(
    L, 
    n_eigs, 
    :SR; 
    issymmetric = true,
    krylovdim = 30,
    maxiter = 300,
    tol = 1e-6
)

println("\nSmallest eigenvalues of the Laplacian:")
for (i, λ) in enumerate(vals)
    println("λ[$i] = $(round(λ, sigdigits=8))")
end

# Number of connected components = multiplicity of λ = 0
tolerance = 1e-5
num_components_spectral = count(abs.(vals) .< tolerance)

println("\n" * "="^60)
println("Final Results")
println("="^60)
println("Connected components (spectral method): $num_components_spectral")
println("Connected components (direct method):   $num_components_direct")
println("="^60)
