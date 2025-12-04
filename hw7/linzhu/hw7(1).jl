###########################
# Ground state of Ising model on the Fullerene graph
###########################

using Graphs, ProblemReductions, Random

# -----------------------------
# 1. Fullerene graph construction
# -----------------------------
function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th),
                      (1.0, 2 + th, 2th),
                      (th, 2.0, 2th + 1.0))
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

function build_fullerene_graph()
    coords = fullerene()
    g_geom = UnitDiskGraph(coords, sqrt(5.0))  # geometric/weighted graph
    g = SimpleGraph(g_geom)                    # plain unweighted graph
    return g
end

# -----------------------------
# 2. Ising energy helpers
# -----------------------------

"""
    energy(g, σ)

Compute H(σ) = ∑_{(i,j)∈E} σ[i] σ[j]
for graph g and spin configuration σ ∈ {±1}^n.
"""
function energy(g::AbstractGraph, σ::AbstractVector{<:Integer})
    E = 0
    for e in edges(g)
        i, j = src(e), dst(e)
        E += σ[i] * σ[j]
    end
    return E
end

"""
    deltaE(g, σ, v)

Energy change ΔE = H(σ') - H(σ) when flipping spin at vertex v.
"""
function deltaE(g::AbstractGraph, σ::AbstractVector{<:Integer}, v::Integer)
    sv = σ[v]
    dE = 0
    for u in neighbors(g, v)
        # edge (v,u): sv*σ[u] → (-sv)*σ[u], so ΔE_edge = -2 sv σ[u]
        dE += -2 * sv * σ[u]
    end
    return dE
end

# -----------------------------
# 3. Simulated annealing
# -----------------------------

"""
    simulated_annealing_ising(g; iters, T0, alpha, rng)

Metropolis simulated annealing to minimize
H(σ) = ∑ σ_i σ_j on graph g.

Returns (bestE, bestσ).
"""
function simulated_annealing_ising(
    g::AbstractGraph;
    iters::Int = 200_000,
    T0::Float64 = 2.0,
    alpha::Float64 = 0.9995,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    n = nv(g)

    # random initial spins ±1
    σ = Vector{Int8}(undef, n)
    for i in 1:n
        σ[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end

    E = energy(g, σ)
    bestE = E
    bestσ = copy(σ)
    T = T0

    for t in 1:iters
        v = rand(rng, 1:n)
        dE = deltaE(g, σ, v)
        if dE <= 0 || rand(rng) < exp(-dE / T)
            σ[v] = -σ[v]
            E += dE
            if E < bestE
                bestE = E
                bestσ .= σ
            end
        end
        T *= alpha
    end

    return bestE, bestσ
end

# -----------------------------
# 4. Driver / multi-start search
# -----------------------------
function main()
    Random.seed!(42)

    g = build_fullerene_graph()
    @info "fullerene_graph" nv(g) ne(g)  # should be 60, 90

    num_runs = 20
    global_bestE = typemax(Int)
    global_bestσ = Int8[]

    for r in 1:num_runs
        E, σ = simulated_annealing_ising(
            g;
            iters = 200_000,
            T0 = 2.0,
            alpha = 0.9995,
        )
        println("Run $r: E = $E")
        if E < global_bestE
            global_bestE = E
            global_bestσ = σ
        end
    end

    println("\nBest energy found over $num_runs runs: H_min = $global_bestE")
    return global_bestE, global_bestσ
end

# Always run main when this file is executed
main()
