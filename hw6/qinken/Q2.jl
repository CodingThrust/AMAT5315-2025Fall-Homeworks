using Graphs, Random, KrylovKit, LinearAlgebra

Random.seed!(42)
g = random_regular_graph(100000, 3)

function laplacian_mul(v)
    res = similar(v)
    for i in 1:nv(g)
        s = 3 * v[i]  
        for nbr in neighbors(g, i)
            s -= v[nbr]
        end
        res[i] = s
    end
    return res
end

L_op = LinearMap(laplacian_mul, nv(g)) 

vals, vecs, info = eigsolve(L_op, randn(nv(g)), 10, :SR)  # SR = smallest real

num_components = count(abs.(vals) .< 1e-8)

println("Number of connected components: $num_components")