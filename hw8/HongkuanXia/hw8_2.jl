using Graphs, Plots, ProblemReductions
using OMEinsum, OMEinsumContractionOrders

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



fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5)) # construct the unit disk graph

code = EinCode([[e.src, e.dst] for e in edges(fullerene_graph)], Int[])
optcode = optimize_code(code, uniformsize(code, 2), TreeSA())
sitetensors(β::Real) = [[exp(-β) exp(β); exp(β) exp(-β)] for _ in 1:ne(fullerene_graph)]
partition_func(β::Real) = only(optcode(sitetensors(β)...))

Z = [[partition_func(β) for β in 0.1:0.1:2.0]]