##############################
# hw8.jl  —  AMAT5315 Homework 8
##############################

using LinearAlgebra
using Graphs
using OMEinsum
using OMEinsumContractionOrders

##############################
# Problem 1 – Einsum notation
##############################

# (a) C = A Bᵀ
#     A_{i j}, B_{k j} → C_{i k} = ∑_j A_{i j} B_{k j}
#     Einsum: "ij,kj->ik"
#
# (b) ∑_{i,j} A_{i,j}
#     Einsum: "ij->"
#
# (c) D = A ⊙ B ⊙ C
#     D_{i j} = A_{i j} B_{i j} C_{i j}
#     Einsum: "ij,ij,ij->ij"
#
# (d) D = A ⊗ B ⊗ C
#     A_{i j}, B_{k ℓ}, C_{m n}
#     D_{i k m, j ℓ n} = A_{i j} B_{k ℓ} C_{m n}
#     Einsum: "ij,kl,mn->ikm,jln"


##############################
# Problem 2 – Contraction order (text answer)
##############################
#
# Contraction order for the 4‑tensor network:
#
#   1. Contract T2 and T3 over their common internal indices → X
#   2. Contract X with T1 over their common indices          → Y
#   3. Contract Y with T4 over the remaining internal indices→ Z
#
# Bracket notation:
#       Z = (((T2 * T3) * T1) * T4)
#
# where “*” denotes contraction over shared indices.


##############################
# Fullerene construction
##############################

"""
    fullerene_points() -> Vector{NTuple{3,Float64}}

Generate the 60 coordinates of the C60 fullerene (same as in HW7).
"""
function fullerene_points()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3 * th),
                      (1.0, 2 + th, 2 * th),
                      (th, 2.0, 2 * th + 1.0))
        for (a, b, c) in ((x, y, z), (y, z, x), (z, x, y))
            for loc in ((a,  b,  c),
                        (a,  b, -c),
                        (a, -b,  c),
                        (a, -b, -c),
                        (-a,  b,  c),
                        (-a,  b, -c),
                        (-a, -b,  c),
                        (-a, -b, -c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

"""
    fullerene_graph() -> SimpleGraph

Construct the fullerene graph as a unit disk graph with radius √5.
"""
function fullerene_graph()
    pts = fullerene_points()
    n   = length(pts)
    g   = SimpleGraph(n)

    r2 = 5.0          # (√5)^2

    @inbounds for i in 1:n-1
        xi, yi, zi = pts[i]
        for j in i+1:n
            xj, yj, zj = pts[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            if dx*dx + dy*dy + dz*dz < r2 + 1e-8
                add_edge!(g, i, j)
            end
        end
    end

    return g
end

"""
    ising_fullerene_graph() -> (N, edges)

Return (N, edges) for the C60 fullerene graph.
edges is a vector of (i, j) with i < j, each edge once.
"""
function ising_fullerene_graph()
    g = fullerene_graph()
    N = nv(g)
    es = Tuple{Int,Int}[]
    for e in edges(g)
        i = src(e); j = dst(e)
        if j < i
            i, j = j, i
        end
        push!(es, (i, j))
    end
    sort!(es)
    unique!(es)
    return N, es
end


##############################
# Problem 4 – Ising partition function on C60
##############################

const SPINS = (+1, -1)

"""
    interaction_tensor(β; J = -1.0)

2×2 edge interaction tensor for AFM Ising:
    W[s_i, s_j] = exp(β * J * s_i * s_j)
with J = -1.
"""
function interaction_tensor(β::Float64; J::Float64 = -1.0)
    W = zeros(Float64, 2, 2)
    for (a, σi) in enumerate(SPINS), (b, σj) in enumerate(SPINS)
        W[a, b] = exp(β * J * σi * σj)
    end
    return W
end

"""
    build_ising_einsum(N, edges)

Return (ixs, iy, size) specification for the Ising partition function.
"""
function build_ising_einsum(N::Int, edges::Vector{Tuple{Int,Int}})
    ixs  = [Int[i, j] for (i, j) in edges]
    iy   = Int[]                      # scalar output
    size = Dict(i => 2 for i in 1:N)  # spin dimension = 2
    return (ixs = ixs, iy = iy, size = size)
end

"""
    build_tensors_for_ising(edges, β)

Return the list of edge tensors for a given β.
"""
function build_tensors_for_ising(edges::Vector{Tuple{Int,Int}}, β::Real)
    W = interaction_tensor(Float64(β))
    return [W for _ in edges]
end

"""
    optimize_contraction(ixs, iy, size)

Use OMEinsumContractionOrders/OMEinsum to obtain an optimized contraction order.

Per OMEinsum help:

    code  = EinCode(ixs, iy)
    sizeD = Dict(label => dim, ...)
    opt   = optimize_code(code, sizeD, optimizer)

Here optimizer = TreeSA().
"""
function optimize_contraction(ixs, iy, size)
    code = EinCode(ixs, iy)                 # DynamicEinCode
    optimizer = TreeSA()                    # default parameters
    opt_code = optimize_code(code, size, optimizer)
    return opt_code                         # NestedEinsum, callable
end

"""
    partition_function_fullerene(βs)

Compute Z(β) for each β in βs for the AFM Ising model on the C60 graph.
Return Dict{Float64,Float64} mapping β → Z(β).
"""
function partition_function_fullerene(βs::AbstractVector{<:Real})
    N, edges = ising_fullerene_graph()
    spec = build_ising_einsum(N, edges)

    opt_code = optimize_contraction(spec.ixs, spec.iy, spec.size)

    Zs = Dict{Float64,Float64}()
    for β in βs
        tensors = build_tensors_for_ising(edges, β)

        # NestedEinsum returns a 0‑dim Array when the result is scalar.
        # Extract the scalar with `only`.
        Z_tensor = opt_code(tensors...)
        Z = only(Z_tensor)

        Zs[float(β)] = Z
    end
    return Zs
end

"""
    run_problem4()

Scan β from 0.1 to 2.0 with step 0.1 and print Z(β).
"""
function run_problem4()
    βs = 0.1:0.1:2.0
    Zs = partition_function_fullerene(collect(βs))
    println("# beta\tZ(beta)")
    for β in βs
        println("$(round(β, digits = 2))\t", Zs[float(β)])
    end
end


##############################
# Problem 6 – baseline optimizer
##############################

"""
    my_contraction_order_optimizer(ixs, iy, size) -> NestedEinsum

Baseline optimizer using TreeSA.
"""
function my_contraction_order_optimizer(ixs, iy, size)
    code = EinCode(ixs, iy)
    optimizer = TreeSA()
    return optimize_code(code, size, optimizer)
end


##############################
# Main
##############################

if abspath(PROGRAM_FILE) == @__FILE__
    println("Homework 8 script")
    println("Problem 4: AFM Ising on C60, beta = 0.1:0.1:2.0")
    try
        run_problem4()
    catch e
        @error "run_problem4() failed" exception = e
    end
end

# Homework 8 script
# Problem 4: AFM Ising on C60, beta = 0.1:0.1:2.0
# # beta  Z(beta)
# 0.1     1.80661094039768e18
# 0.2     6.87566576214655e18
# 0.3     6.151122581721418e19
# 0.4     1.2302123108278566e21
# 0.5     5.144558716182041e22
# 0.6     4.1116282893989694e24
# 0.7     5.612702683289911e26
# 0.8     1.1622488726804853e29
# 0.9     3.2853302335009414e31
# 1.0     1.1667408970547078e34
# 1.1     4.898099153436166e36
# 1.2     2.3273948792397584e39
# 1.3     1.2136776638732638e42
# 1.4     6.793958994977612e44
# 1.5     4.017291439227066e47
# 1.6     2.479402296712645e50
# 1.7     1.5828842349953103e53
# 1.8     1.038099917657124e56
# 1.9     6.956429030632339e58
# 2.0     4.7430829554740573e61