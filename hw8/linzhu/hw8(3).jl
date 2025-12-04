using OMEinsum
using Graphs
using ProblemReductions   # only for UnitDiskGraph / fullerene construction

############################
# 1. Fullerene graph
############################

function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th),
                      (1.0, 2 + th, 2th),
                      (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x,y,z), (y,z,x), (z,x,y))
            for loc in ((a,b,c), (a,b,-c), (a,-b,c), (a,-b,-c),
                        (-a,b,c), (-a,b,-c), (-a,-b,c), (-a,-b,-c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5))
@info "fullerene_graph" nv(fullerene_graph) ne(fullerene_graph)
# should be 60 vertices, 90 edges

############################
# 2. Build einsum code
############################

# Each edge (i,j) is a 2×2 tensor W_{σ_i, σ_j}.
# The Einstein code uses the vertex indices as labels.
edges_list = collect(edges(fullerene_graph))

code = EinCode([[src(e), dst(e)] for e in edges_list], Int[])
optcode = optimize_code(code, uniformsize(code, 2), TreeSA())

############################
# 3. AFM Ising edge tensor
############################

# AFM Ising, J = -1, zero field:
#
# H_edge = σ_i σ_j
# weight w(σ_i,σ_j) = exp(-β H_edge)
#
# σ_i σ_j = +1 (aligned)      -> exp(-β)
# σ_i σ_j = -1 (anti-aligned) -> exp(+β)
#
# With spin basis ( +1, -1 ) ≡ (1, 2) indices, we get
#     W = [ e^{-β}  e^{β}
#           e^{β}   e^{-β} ]

function edge_tensor(β)
    return [exp(-β)  exp(β);
            exp(β)   exp(-β)]
end

# For this homogeneous model every edge uses the same tensor.
sitetensors(β) = fill(edge_tensor(β), ne(fullerene_graph))

############################
# 4. Partition function Z(β)
############################

partition_func(β) = only(optcode(sitetensors(β)...))

βs = 0.1:0.1:2.0
Zs = [partition_func(β) for β in βs]

println("β\tZ")
for (β, Z) in zip(βs, Zs)
    println("$(round(β, digits=1))\t$Z")
end
