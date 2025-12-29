#!/usr/bin/env julia

using Random
using Statistics
using Printf
using JuMP
import MathOptInterface as MOI
using HiGHS

# ---------------- Half-adder Ising model ----------------
const HALF_ADDER_SPINS = [:A, :B, :S, :C]

struct PairwiseIsing
    spins::Vector{Symbol}
    fields::Dict{Symbol,Float64}
    couplings::Dict{Tuple{Symbol,Symbol},Float64}
    constant::Float64
end

function half_adder_model()
    h = Dict(:A => 1.0, :B => 1.0, :S => -1.0, :C => -2.0)
    J = Dict(
        (:A, :B) => 1.0,
        (:A, :S) => -1.0,
        (:A, :C) => -2.0,
        (:B, :S) => -1.0,
        (:B, :C) => -2.0,
        (:S, :C) => 2.0,
    )
    return PairwiseIsing(HALF_ADDER_SPINS, h, J, 3.0)
end

spin_from_bit(bit::Int) = bit == 0 ? 1 : -1
bit_from_spin(spin::Int) = spin == 1 ? 0 : 1

function local_field(model::PairwiseIsing, spins::Dict{Symbol,Int}, var::Symbol)
    field = model.fields[var]
    for ((a, b), J) in model.couplings
        if var == a
            field += J * spins[b]
        elseif var == b
            field += J * spins[a]
        end
    end
    return field
end

function energy(model::PairwiseIsing, assignment::Dict{Symbol,Int})
    e = model.constant
    for s in model.spins
        e += model.fields[s] * assignment[s]
    end
    for ((a, b), J) in model.couplings
        e += J * assignment[a] * assignment[b]
    end
    return e
end

function enumerate_ground_states(model::PairwiseIsing)
    best_energy = Inf
    configs = Vector{Dict{Symbol,Int}}()
    values = (-1, 1)
    for (sa, sb, ss, sc) in Iterators.product(values, values, values, values)
        assignment = Dict(:A => sa, :B => sb, :S => ss, :C => sc)
        e = energy(model, assignment)
        if e < best_energy - 1e-9
            best_energy = e
            empty!(configs)
            push!(configs, assignment)
        elseif abs(e - best_energy) < 1e-9
            push!(configs, assignment)
        end
    end
    return best_energy, configs
end

function zero_temp_glauber(model::PairwiseIsing; fixed_spins=Dict{Symbol,Int}(), sweeps=50, restarts=100, rng = Random.default_rng())
    vars = model.spins
    best_energy = Inf
    best_config = Dict{Symbol,Int}()
    for _ in 1:restarts
        config = Dict{Symbol,Int}()
        for v in vars
            if haskey(fixed_spins, v)
                config[v] = fixed_spins[v]
            else
                config[v] = rand(rng, (-1, 1))
            end
        end
        for _ in 1:sweeps
            changed = false
            for v in Random.shuffle(rng, vars)
                if haskey(fixed_spins, v)
                    continue
                end
                field = local_field(model, config, v)
                desired = field > 0 ? -1 : (field < 0 ? 1 : config[v])
                if desired != config[v]
                    config[v] = desired
                    changed = true
                end
            end
            changed || break
        end
        e = energy(model, config)
        if e < best_energy - 1e-9
            best_energy = e
            best_config = deepcopy(config)
        end
    end
    return best_energy, best_config
end

function report_half_adder()
    println("--- Half-adder Ising reduction ---")
    model = half_adder_model()
    E0, configs = enumerate_ground_states(model)
    println("Ground energy: $E0")
    println("Ground-state assignments (spin -> bit):")
    for cfg in configs
        bits = Dict(k => bit_from_spin(cfg[k]) for k in model.spins)
        println(cfg, " | bits ", bits)
    end
    println("\n--- Spin dynamics with S=0, C=1 pinned ---")
    fixed = Dict(:S => spin_from_bit(0), :C => spin_from_bit(1))
    energy_sol, config = zero_temp_glauber(model; fixed_spins=fixed, sweeps=30, restarts=200)
    bits = Dict(k => bit_from_spin(config[k]) for k in model.spins)
    println("Best constrained energy: $energy_sol")
    println("Recovered spin configuration: $config")
    println("Recovered bits: $bits")
end

# ---------------- Maximum independent set experiments ----------------

function random_regular_graph(n::Int, d::Int; rng = Random.default_rng(), max_attempts::Int = 10_000)
    if n * d % 2 != 0
        error("n*d must be even")
    end
    for attempt in 1:max_attempts
        stubs = repeat(collect(1:n), inner=d)
        Random.shuffle!(rng, stubs)
        adj = [Int[] for _ in 1:n]
        valid = true
        for i in 1:2:length(stubs)
            u = stubs[i]
            v = stubs[i + 1]
            if u == v || v in adj[u]
                valid = false
                break
            end
            push!(adj[u], v)
            push!(adj[v], u)
        end
        if valid && all(length(adj[v]) == d for v in 1:n)
            return adj
        end
    end
    error("Failed to generate a simple $d-regular graph after $max_attempts attempts")
end

function greedy_mis(adj::Vector{Vector{Int}}; rng = Random.default_rng())
    n = length(adj)
    alive = trues(n)
    independent_set = Int[]
    while any(alive)
        mindeg = typemax(Int)
        candidates = Int[]
        for v in 1:n
            alive[v] || continue
            deg = count(u -> alive[u], adj[v])
            if deg < mindeg
                mindeg = deg
                empty!(candidates)
                push!(candidates, v)
            elseif deg == mindeg
                push!(candidates, v)
            end
        end
        v = rand(rng, candidates)
        push!(independent_set, v)
        alive[v] = false
        for u in adj[v]
            alive[u] = false
        end
    end
    sort!(independent_set)
    return independent_set
end

function exact_mis_size(adj::Vector{Vector{Int}})
    n = length(adj)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:n], Bin)
    for v in 1:n
        for u in adj[v]
            if u > v
                @constraint(model, x[v] + x[u] <= 1)
            end
        end
    end
    @objective(model, Max, sum(x))
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("Solver terminated with status $status")
    return round(Int, objective_value(model))
end

function mis_scaling(; sizes = 10:10:200, samples_per_size::Int = 3, seed::Int = 42)
    rng = MersenneTwister(seed)
    results = Vector{Dict{Symbol,Any}}()
    header = @sprintf("%5s %10s %10s %12s %12s", "n", "avg_ratio", "std_ratio", "avg_greedy", "avg_opt")
    println("\n--- Greedy MIS scaling on random 3-regular graphs ---")
    println(header)
    println(repeat('-', length(header)))
    for n in sizes
        ratios = Float64[]
        greedy_sizes = Float64[]
        opt_sizes = Float64[]
        for _ in 1:samples_per_size
            adj = random_regular_graph(n, 3; rng=rng)
            greedy = length(greedy_mis(adj; rng=rng))
            opt = exact_mis_size(adj)
            push!(greedy_sizes, greedy)
            push!(opt_sizes, opt)
            push!(ratios, greedy / opt)
        end
        avg_ratio = mean(ratios)
        std_ratio = std(ratios)
        avg_greedy = mean(greedy_sizes)
        avg_opt = mean(opt_sizes)
        push!(results, Dict(
            :n => n,
            :avg_ratio => avg_ratio,
            :std_ratio => std_ratio,
            :avg_greedy => avg_greedy,
            :avg_opt => avg_opt,
        ))
        println(@sprintf("%5d %10.3f %10.3f %12.2f %12.2f", n, avg_ratio, std_ratio, avg_greedy, avg_opt))
    end
    return results
end

function main()
    report_half_adder()
    mis_results = mis_scaling()
    open("mis_results.csv", "w") do io
        println(io, "n,avg_ratio,std_ratio,avg_greedy,avg_opt")
        for row in mis_results
            println(io, @sprintf("%d,%.6f,%.6f,%.4f,%.4f", row[:n], row[:avg_ratio], row[:std_ratio], row[:avg_greedy], row[:avg_opt]))
        end
    end
    println("\nSaved MIS summary to mis_results.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
