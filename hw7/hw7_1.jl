using Graphs, ProblemReductions
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

function annealing(nrun::Int, prob::SpinGlassSA, tempscales::Vector{Float64}, num_update_each_temp::Int)
    
end

fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5)); # construct the unit disk graph

spin_glass = SpinGlass(
            fullerene_graph,   # graph
           -ones(Int, ne(fullerene_graph)),     # J, in order of edges
           zeros(Int, nv(fullerene_graph))     # h, in order of vertices
       )

