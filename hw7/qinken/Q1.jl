# simulated_annealing_fullerene.jl
using Random, LinearAlgebra
using Graphs
using ProblemReductions  # for UnitDiskGraph (as in the problem statement)


# build fullerene graph

function fullerene()
    th = (1+sqrt(5))/2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
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

# Create the fullerene graph (as in problem statement).
fullerene_points = fullerene()
fullerene_graph = UnitDiskGraph(fullerene_points, sqrt(5))
# make sure it's a simple Graph object
g = SimpleGraph(fullerene_graph)

# Quick graph stats
n = nv(g)
m = ne(g)
println("Graph: n = $n, m = $m (expected n=60, m=90)")


# helpers: energy & flip

# compute H = sum_{(i,j) in E} sigma_i * sigma_j
# sigma is a Vector{Int} with entries ±1
function energy(g::Graph, sigma::AbstractVector{<:Integer})
    E = edges(g)
    s = 0
    for e in E
        s += sigma[src(e)] * sigma[dst(e)]
    end
    return s
end

# local energy change when flipping spin at i:

function delta_energy_flip(g::Graph, sigma::AbstractVector{<:Integer}, i::Int)
    ssum = 0
    for j in neighbors(g, i)
        ssum += sigma[j]
    end
    return -2 * sigma[i] * ssum
end

# count cut edges: edges with σ_i != σ_j
function cut_size(g::Graph, sigma::AbstractVector{<:Integer})
    c = 0
    for e in edges(g)
        c += (sigma[src(e)] != sigma[dst(e)]) ? 1 : 0
    end
    return c
end

# Simulated annealing (single run)

function simulated_annealing(g::Graph; T0=2.0, Tf=1e-3, steps=200000, rng=GLOBAL_RNG, record_every=5000)
    n = nv(g)
    # random initial configuration ±1
    sigma = rand(rng, [-1, 1], n)
    curE = energy(g, sigma)
    best_sigma = copy(sigma)
    bestE = curE

    # geometric cooling schedule
    # temperature at step t: T(t) = T0 * (Tf/T0)^(t/steps)
    history = Float64[]
    for t in 1:steps
        T = T0 * (Tf/T0) ^ (t/steps)
        # random site
        i = rand(rng, 1:n)
        Δ = delta_energy_flip(g, sigma, i)
        # acceptance: if energy decreases accept; else accept with exp(-Δ/T) (Δ > 0)
        if Δ <= 0 || rand(rng) < exp(-Δ / T)
            sigma[i] = -sigma[i]  # perform flip
            curE += Δ
            if curE < bestE
                bestE = curE
                best_sigma .= sigma
            end
        end

        if (t % record_every) == 0
            push!(history, curE)
        end
    end
    return best_sigma, bestE, history
end


# Multi-restart annealing

function run_sa_restarts(g::Graph; restarts=20, seed=42, kwargs...)
    rng = MersenneTwister(seed)
    bestE_overall = Inf
    best_sigma_overall = zeros(Int, nv(g))
    details = []
    for r in 1:restarts
        # make independent rng for each restart for reproducibility
        rng_r = MersenneTwister(rand(rng, UInt))
        sigma_r, E_r, hist = simulated_annealing(g; rng=rng_r, kwargs...)
        println("Restart $r: E = $E_r, cut = $( (m - E_r) ÷ 2 )")
        push!(details, (E=E_r, cut=(m - E_r) ÷ 2, history=hist))
        if E_r < bestE_overall
            bestE_overall = E_r
            best_sigma_overall .= sigma_r
        end
    end
    return best_sigma_overall, bestE_overall, details
end

# example run settings

res_sigma, res_E, res_details = run_sa_restarts(g; restarts=30,
                                               seed=12345,
                                               T0=2.0, Tf=1e-4,
                                               steps=300_000,
                                               record_every=50_000)

println("Best energy found: H = $res_E")
println("Corresponding cut size = $( (m - res_E) ÷ 2 ) out of $m edges")
