using Graphs, Random, KrylovKit
using LinearAlgebra, SparseArrays
using LinearMaps

Random.seed!(42)

g = random_regular_graph(100000, 3)
L = laplacian_matrix(g)
L_sparse = LinearMap(L)
eig_result = eigsolve(L_sparse,randn(size(L_sparse, 1)), 1, :SR)
eigenvalues = real(eig_result[1])
sorted_eigenvalues = sort(eigenvalues)
zero_tolerance = 1e-10
num_zero_eigenvalues = count(x -> abs(x) < zero_tolerance, sorted_eigenvalues)
println("Method 2 - Spectral analysis: ", num_zero_eigenvalues, " connected components")