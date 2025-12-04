using Graphs, Random, KrylovKit
Random.seed!(42)
g = random_regular_graph(100000, 3)

A = laplacian_matrix(g)

q1 = randn(100000)

vals, vecs, info = eigsolve(A, q1, 2, :SR)
