using ProblemReductions
using Graphs
using Random
using LinearAlgebra
using CairoMakie

half_adder = @circuit begin
    s = a ⊻ b
    c = a ∧ b
end

sat = CircuitSAT(half_adder; use_constraints=true)

reduce_path = reduction_paths(CircuitSAT, SpinGlass)
res = reduceto(reduce_path[1], sat)
sg = target_problem(res)

function assign_variable!(circuit::Circuit, variable::Symbol, value::Bool)
    push!(circuit.exprs, Assignment([variable],BooleanExpr(value)))
end

function assign_variables!(circuit::Circuit, variables::Vector{Symbol}, values::Vector{Bool})
    for (variable, value) in zip(variables, values)
        assign_variable!(circuit, variable, value)
    end
end

assign_variables!(half_adder, [:s, :c], [false, true])

sat_assigned = CircuitSAT(half_adder; use_constraints=true)
res_assigned = reduceto(reduce_path[1], sat_assigned)
sg_assigned = target_problem(res_assigned)

function spin_energy(sg::SpinGlass, spins::Vector{Int})
    e = 0.0
    for (edge, weight) in zip(edges(sg.graph), sg.J)
        e += weight * spins[src(edge)] * spins[dst(edge)]
    end
    @inbounds for i in 1:nv(sg.graph)
        e += sg.h[i] * spins[i]
    end
    return e
end

function spinglass_coupling_matrix(sg::SpinGlass)
    n = nv(sg.graph)
    Jmat = zeros(Float64, n, n)
    for (edge, weight) in zip(edges(sg.graph), sg.J)
        i = src(edge)
        j = dst(edge)
        Jmat[i, j] = weight
        Jmat[j, i] = weight
    end
    return Jmat
end

function greedy_spin_dynamics(sg::SpinGlass; nstart::Int=64, max_sweeps::Int=2000, seed::Int=1)
    rng = MersenneTwister(seed)
    n = nv(sg.graph)
    Jmat = spinglass_coupling_matrix(sg)
    best_spins = fill(1, n)
    best_energy = Inf
    for _ in 1:nstart
        spins = rand(rng, [-1, 1], n)
        improved = true
        sweep = 0
        while improved && sweep < max_sweeps
            improved = false
            sweep += 1
            for v in 1:n
                local_field = sg.h[v] + dot(Jmat[v, :], spins)
                ΔE = -2 * spins[v] * local_field
                if ΔE < 0
                    spins[v] = -spins[v]
                    improved = true
                end
            end
        end
        e = spin_energy(sg, spins)
        if e < best_energy
            best_energy = e
            best_spins = copy(spins)
        end
    end
    return best_energy, best_spins
end

function mis_greedy_local_search(g::SimpleGraph; nstart::Int=20, max_sweeps::Int=2000, penalty::Float64=2.0, seed::Int=1)
    rng = MersenneTwister(seed)
    n = nv(g)
    best_bits = zeros(Bool, n)
    best_energy = Inf
    for _ in 1:nstart
        bits = rand(rng, Bool, n)
        improved = true
        sweep = 0
        while improved && sweep < max_sweeps
            improved = false
            sweep += 1
            for v in 1:n
                neigh_on = count(u -> bits[u], neighbors(g, v))
                if bits[v]
                    ΔE = 1 - penalty * neigh_on
                else
                    ΔE = -1 + penalty * neigh_on
                end
                if ΔE < 0
                    bits[v] = !bits[v]
                    improved = true
                end
            end
        end
        energy = -count(identity, bits) + penalty * count(e -> bits[src(e)] && bits[dst(e)], edges(g))
        if energy < best_energy
            best_energy = energy
            best_bits = copy(bits)
        end
    end
    return best_bits
end

function mis_size(bits::Vector{Bool})
    return count(identity, bits)
end

function exact_mis_size_branch_and_bound(g::SimpleGraph)
    n = nv(g)
    if n > 60
        error("exact_mis_size_branch_and_bound is intended for n <= 60.")
    end
    neighbor_masks = zeros(UInt64, n)
    for v in 1:n
        mask = UInt64(0)
        for u in neighbors(g, v)
            mask |= UInt64(1) << (u - 1)
        end
        neighbor_masks[v] = mask
    end

    function mis_recursive(remaining::UInt64)
        remaining == 0 && return 0
        v = trailing_zeros(remaining) + 1
        without_v = remaining & ~(UInt64(1) << (v - 1))
        best = mis_recursive(without_v)
        with_v = without_v & ~neighbor_masks[v]
        best_with = 1 + mis_recursive(with_v)
        return max(best, best_with)
    end

    return mis_recursive((UInt64(1) << n) - 1)
end

function mis_approx_ratio_scaling(ns::Vector{Int}; exact_limit::Int=40, ntrials::Int=20, seed::Int=1)
    rng = MersenneTwister(seed)
    ratios = Dict{Int, Float64}()
    for n in ns
        ratios_n = Float64[]
        for _ in 1:ntrials
            g = random_regular_graph(n, 3; rng=rng)
            greedy_bits = mis_greedy_local_search(g; seed=rand(rng, 1:10^9))
            greedy_size = mis_size(greedy_bits)
            if n <= exact_limit
                opt_size = exact_mis_size_branch_and_bound(g)
                push!(ratios_n, greedy_size / opt_size)
            end
        end
        if !isempty(ratios_n)
            ratios[n] = sum(ratios_n) / length(ratios_n)
        end
    end
    return ratios
end

function plot_mis_ratio_scaling(ns::Vector{Int}; exact_limit::Int=40, ntrials::Int=20, seed::Int=1, outpath::String="examples/hw9_mis_ratio.png")
    ratios = mis_approx_ratio_scaling(ns; exact_limit=exact_limit, ntrials=ntrials, seed=seed)
    xs = sort(collect(keys(ratios)))
    ys = [ratios[x] for x in xs]
    fig = Figure(size=(700, 420))
    ax = Axis(fig[1, 1], xlabel="n", ylabel="Approximation ratio", title="Greedy MIS ratio on 3-regular graphs")
    lines!(ax, xs, ys)
    scatter!(ax, xs, ys)
    save(outpath, fig)
    return fig
end

function main()
    energy_assigned, spins_assigned = greedy_spin_dynamics(sg_assigned)
    bits_assigned = map(s -> s < 0 ? 1 : 0, spins_assigned)
    extracted = extract_solution(res_assigned, bits_assigned)
    assigned_vars = Dict(sat_assigned.symbols .=> Bool.(extracted))
    @info "Fixed outputs: s=false, c=true"
    @info "Recovered inputs: a=$(assigned_vars[:a]), b=$(assigned_vars[:b])"

    ns = collect(10:10:200)
    fig = plot_mis_ratio_scaling(ns; exact_limit=40, ntrials=20, seed=1, outpath="examples/mis_ratio.png")
    display(fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
