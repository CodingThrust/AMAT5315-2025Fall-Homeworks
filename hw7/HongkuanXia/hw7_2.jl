
using ProblemReductions, Graphs, Printf, SparseArrays, KrylovKit, Plots

function transition_matrix(model::SpinGlass, beta::T) where T
    N = num_variables(model)
    P = spzeros(T, 2^N, 2^N)  # P[i, j] = probability of transitioning from j to i
    readbit(cfg, i::Int) = (cfg >> (i - 1)) & 1  # read the i-th bit of cfg
    int2cfg(cfg::Int) = [readbit(cfg, i) for i in 1:N]
    for j in 1:2^N
        for i in 1:2^N
            if count_ones((i-1) ⊻ (j-1)) == 1  # Hamming distance is 1
                P[i, j] = 1/N * min(one(T), exp(-beta * (energy(model, int2cfg(i-1)) - energy(model, int2cfg(j-1)))))
            end
        end
        P[j, j] = 1 - sum(P[:, j])  # rejected transitions
    end
    return P
end

using LinearAlgebra: eigvals

function spectral_gap(P)
    vals, vecs, info = eigsolve(P, 2, :LM,tol=1e-12, maxiter=50000,krylovdim=80)
    return 1.0 - real(vals[2])
end

function triangular_grid_graph(m, n)
    # 1. Start with a standard rectangular grid (m x n)
    g = Graphs.grid([m, n])
    
    # 2. Add diagonal edges
    # In a grid, vertex (i, j) usually has index: i + (j-1)*m
    # We want to connect (i, j) to (i+1, j+1)
    
    for j in 1:(n-1)
        for i in 1:(m-1)
            # Calculate current vertex index
            u = i + (j-1)*m
            
            # Calculate the diagonal neighbor (i+1, j+1)
            # This is: (i+1) + ((j+1)-1)*m = i + 1 + j*m
            v = (i + 1) + j*m
            
            add_edge!(g, u, v)
        end
    end
    
    return g
end

function diamond_graph(m)
    g = SimpleGraph(3*m+1)
    for i in 1:3*m
        if i%3 == 1
            add_edge!(g,i,i+1)
            add_edge!(g,i,i+2)
        elseif i%3 == 2
            add_edge!(g,i,i+2)
        else
            add_edge!(g,i,i+1)
        end
    end
    return g
end 



########   different T  ##########


T_list = [0.1:0.1:2.0;]
gap_square = zeros(length(T_list))
gap_triangle = zeros(length(T_list))
gap_diamond = zeros(length(T_list))

# initilaize a 4*2 site Ising model on a Square lattice
graph_square = Graphs.grid([4,2])
model_square = SpinGlass(graph_square, -ones(ne(graph_square)), zeros(nv(graph_square)))
for i in eachindex(T_list)
    T = T_list[i]
    gap_square[i] = spectral_gap(transition_matrix(model_square, 1/T))
    #println("T: $T, Spectral gap: $gap")
end  


# initilaize a 4*2 site Ising model on a Triangles lattice
graph_triangle = triangular_grid_graph(4,2)
model_triangle = SpinGlass(graph_triangle, -ones(ne(graph_triangle)), zeros(nv(graph_triangle)))
for i in eachindex(T_list)
    T = T_list[i]
    gap_triangle[i] = spectral_gap(transition_matrix(model_triangle, 1/T))
    #println("T: $T, Spectral gap: $gap")
end     

# initilaize a 10 site Ising model on a diamond lattice
graph_diamond = diamond_graph(3)
model_diamond = SpinGlass(graph_diamond, -ones(ne(graph_diamond)), zeros(nv(graph_diamond)))
for i in eachindex(T_list)
    T = T_list[i]
    gap_diamond[i] = spectral_gap(transition_matrix(model_diamond, 1/T))
    #println("T: $T, Spectral gap: $gap")
end   

fig = plot([0.1:0.1:2.0;],gap_square,xlabel = "T",ylabel = "gap",title = "spectral gap v.s. at different temperature",label = "square")
plot!([0.1:0.1:2.0;],gap_triangle,title = "spectral gap v.s. at different temperature",label = "triangle")
plot!([0.1:0.1:2.0;],gap_diamond,title = "spectral gap v.s. at different temperature",label = "diamond")
savefig("./spectral_gap_vs_T.png")



######## different system size ##########

T = 0.1
gap_square = zeros(8)
gap_triangle = zeros(8)
gap_diamond = zeros(5)

# initilaize a 4*2 site Ising model on a Square lattice

for i in eachindex([4:2:18;])
    n = [4:2:18;][i]
    graph_square = Graphs.grid([n÷2,2])
    model_square = SpinGlass(graph_square, -ones(ne(graph_square)), zeros(nv(graph_square)))
    gap_square[i] = spectral_gap(transition_matrix(model_square, 1/T))
    #println("T: $T, Spectral gap: $gap")
end  


# initilaize a 4*2 site Ising model on a Triangles lattice

for i in eachindex([4:2:18;])
    n = [4:2:18;][i]
    graph_triangle = triangular_grid_graph(n÷2,2)
    model_triangle = SpinGlass(graph_triangle, -ones(ne(graph_triangle)), zeros(nv(graph_triangle)))
    gap_triangle[i] = spectral_gap(transition_matrix(model_triangle, 1/T))
    #println("T: $T, Spectral gap: $gap")
end     

# initilaize a 10 site Ising model on a diamond lattice

for i in eachindex([4:3:18;])
    n = [4:3:18;][i]
    graph_diamond = diamond_graph(n÷3)
    model_diamond = SpinGlass(graph_diamond, -ones(ne(graph_diamond)), zeros(nv(graph_diamond)))
    gap_diamond[i] = spectral_gap(transition_matrix(model_diamond, 1/T))
    #println("T: $T, Spectral gap: $gap")
end   

fig = plot([4:2:18;],gap_square,xlabel = "sites",ylabel = "gap",title = "spectral gap v.s. at different size",label = "square")
plot!([4:2:18;],gap_triangle,label = "triangle")
plot!([4:3:18;],gap_diamond,label = "diamond")
savefig("./spectral_gap_vs_sites.png")