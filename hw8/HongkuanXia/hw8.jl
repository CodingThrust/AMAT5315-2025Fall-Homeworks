using Graphs

function fullerene()  # construct the fullerene graph in 3D space
    th = (1+sqrt(5))/2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x,y,z), (y,z,x), (z,x,y))
            for loc in ((a,b,c), (a,b,-c), (a,-b,c), (a,-b,-c), (-a,b,c), (-a,b,-c), (-a,-b,c), (-a,-b,-c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

function partition_function_exact(sg::Spinglass, β)
    Z = 0.0
    for σ in Iterators.product(fill([-1, 1], nv(sg.graph))...)
        E = 0.0
        for (e, Jij) in zip(edges(sg.graph), sg.J)
            srcv, dstv = src(e), dst(e)
            E -= Jij * σ[srcv] * σ[dstv]
        end
        for (v, hi) in zip(vertices(sg.graph), sg.h)
            E += hi * σ[v]
        end
        Z += exp(-β * E)
    end
    return Z
end

fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5)) # construct the unit disk graph

spin_glass = SpinGlass(
            fullerene_graph,   # graph
           -ones(Int, ne(fullerene_graph)),     # J, in order of edges
           zeros(Int, nv(fullerene_graph))     # h, in order of vertices
       )
β_list = [0.1:0.1:2.0;]
Z = zeros(length(β_list))

for i in eachindex(β_list)
    Z[i] = partition_function_exact(spin_glass,β_list[i])
end

