###########################################################
# greedy_mis_experiment.jl
#
# (Greedy Algorithm) Maximum Independent Set on
# random 3-regular graphs and approximation ratio scaling.
###########################################################

using Graphs
using Random
using JuMP
using HiGHS
import MathOptInterface as MOI

###########################################################
# 1. Greedy MIS algorithm
###########################################################

"""
    greedy_max_independent_set(g::SimpleGraph, rng::AbstractRNG) -> Vector{Int}

Simple random greedy algorithm for MIS:

    S = ∅
    R = V(G)
    while R ≠ ∅:
        pick a random v ∈ R
        add v to S
        remove v and all neighbors N(v) from R

Returns the independent set S as a vector of vertex indices.
"""
function greedy_max_independent_set(g::SimpleGraph, rng::AbstractRNG)
    remaining = Set(vertices(g))
    indep = Int[]

    while !isempty(remaining)
        v = rand(rng, collect(remaining))  # pick random vertex from remaining set
        push!(indep, v)

        delete!(remaining, v)
        for u in neighbors(g, v)
            delete!(remaining, u)
        end
    end

    return indep
end

###########################################################
# 2. Exact MIS size via MILP (JuMP + HiGHS)
###########################################################

"""
    exact_mis_size(g::SimpleGraph) -> Int

Exact maximum independent set size via 0–1 MILP:

    maximize  ∑ x_i
    subject to x_i + x_j ≤ 1  for each edge (i,j),
              x_i ∈ {0,1}.
"""
function exact_mis_size(g::SimpleGraph)
    n = nv(g)

    model = Model(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    @variable(model, x[1:n], Bin)
    @objective(model, Max, sum(x[i] for i in 1:n))

    for e in edges(g)
        i, j = src(e), dst(e)
        @constraint(model, x[i] + x[j] <= 1)
    end

    optimize!(model)
    return Int(round(objective_value(model)))
end

###########################################################
# 3. Experiment driver
###########################################################

"""
    experiment_greedy_MIS(; n_trials=5)

For n = 10,20,...,200 (where a 3-regular graph exists), generate
`n_trials` random 3-regular graphs, compute:

  - greedy_size = |MIS_greedy|
  - opt_size    = |MIS_opt|

and print averages and approximation ratio:

  avg_ratio = avg_greedy / avg_opt
"""
function experiment_greedy_MIS(; n_trials::Int = 5)
    global_rng = MersenneTwister(5315)

    println("n\tavg_greedy\tavg_opt\tapprox_ratio")

    for n in 10:10:200
        # For a d-regular graph to exist, n*d must be even. For d=3, that means n must be even.
        if isodd(3n)
            continue
        end

        total_greedy = 0.0
        total_opt    = 0.0

        for _ in 1:n_trials
            # Each trial uses a fresh seed to keep graphs different
            seed = rand(global_rng, 1:10^9)
            rng  = MersenneTwister(seed)

            g = random_regular_graph(n, 3; seed=seed)

            greedy_set  = greedy_max_independent_set(g, rng)
            greedy_size = length(greedy_set)

            opt_size = exact_mis_size(g)

            total_greedy += greedy_size
            total_opt    += opt_size
        end

        avg_greedy = total_greedy / n_trials
        avg_opt    = total_opt / n_trials
        ratio      = avg_greedy / avg_opt

        println("$(lpad(n,3))\t$(round(avg_greedy, digits=2))\t\t$(round(avg_opt, digits=2))\t$(round(ratio, digits=3))")
    end
end

###########################################################
# 4. Main
###########################################################

function main()
    println("Greedy MIS on random 3-regular graphs")
    println("=====================================")
    experiment_greedy_MIS(n_trials = 5)
end

# Run immediately when this file is executed
main()
