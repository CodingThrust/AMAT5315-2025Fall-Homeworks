# ============================================================================
# Homework 7 - Huicheng Zhang
# Complete solution for all three problems
# ============================================================================

# 输出重定向已移至main函数中，可选启用
# log_file = open("output.log", "w")
# redirect_stdout(log_file)
# redirect_stderr(log_file)


using Graphs
using SparseArrays
using LinearAlgebra
using Arpack
using Plots
using Random
using Printf
println("Starting the script...")
flush(stdout)  # 强制刷新输出流

# Try to load GenericTensorNetworks if available
try
    using GenericTensorNetworks
    using GenericTensorNetworks.Graphs
    global GENERIC_TENSOR_AVAILABLE = true
catch
    global GENERIC_TENSOR_AVAILABLE = false
    # Define SpinGlass struct if GenericTensorNetworks is not available
    struct SpinGlass
        graph::SimpleGraph
        J::Vector{Int}  # coupling
        h::Vector{Int}  # bias
    end
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
Construct a unit disk graph from coordinates
Points within distance `radius` are connected
"""
function UnitDiskGraph(coordinates::Vector{NTuple{3,Float64}}, radius::Float64)
    n = length(coordinates)
    g = SimpleGraph(n)
    
    for i in 1:n
        for j in (i+1):n
            dist = sqrt(sum((coordinates[i][k] - coordinates[j][k])^2 for k in 1:3))
            if dist <= radius + 1e-10  # small tolerance for numerical errors
                add_edge!(g, i, j)
            end
        end
    end
    
    return g
end

# ============================================================================
# PROBLEM 1: Ground State Energy of Anti-ferromagnetic Ising Model
# ============================================================================

"""
Construct the fullerene graph in 3D space
"""
function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x, y, z), (y, z, x), (z, x, y))
            for loc in ((a, b, c), (a, b, -c), (a, -b, c), (a, -b, -c), 
                       (-a, b, c), (-a, b, -c), (-a, -b, c), (-a, -b, -c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

"""
Calculate energy of the anti-ferromagnetic Ising model
H = Σ_{ij ∈ E} σ_i σ_j
"""
function ising_energy(graph::SimpleGraph, spins::Vector{Int})
    energy = 0
    for edge in edges(graph)
        i, j = src(edge), dst(edge)
        energy += spins[i] * spins[j]
    end
    return energy
end

"""
Local Simulated Annealing to improve local optimization
"""
function local_simulated_annealing(spins, graph, T, steps=100)
    n = length(spins)
    for _ in 1:steps
        i = rand(1:n)
        ΔE = 0
        for neighbor in neighbors(graph, i)
            ΔE -= 2 * spins[i] * spins[neighbor]
        end
        
        if ΔE < 0 || rand() < exp(-ΔE / T)
            spins[i] *= -1
        end
    end
    return spins
end

"""
Simulated Annealing Algorithm for Problem 1
"""
function simulated_annealing(
    graph::SimpleGraph;
    T_init::Float64=10.0,
    T_final::Float64=0.01,
    cooling_rate::Float64=0.95,
    steps_per_temp::Int=1000,
    n_runs::Int=10
)
    n = nv(graph)
    best_energy = Inf
    best_spins = nothing
    
    for run in 1:n_runs
        spins = rand([-1, 1], n)
        current_energy = ising_energy(graph, spins)
        
        T = T_init
        
        while T > T_final
            for _ in 1:steps_per_temp
                i = rand(1:n)
                ΔE = 0
                for neighbor in neighbors(graph, i)
                    ΔE -= 2 * spins[i] * spins[neighbor]  # Change: -= not +=
                end
                
                if ΔE < 0 || rand() < exp(-ΔE / T)
                    spins[i] *= -1
                    current_energy += ΔE
                end
            end
            T *= cooling_rate
        end
        
        # Verify and update best solution
        final_energy = ising_energy(graph, spins)
        if final_energy < best_energy
            best_energy = final_energy
            best_spins = copy(spins)
        end
        
        # Local search (Simulated Annealing) to refine the result
        spins = local_simulated_annealing(spins, graph, T_init)
        
        final_energy = ising_energy(graph, spins)
        if final_energy < best_energy
            best_energy = final_energy
            best_spins = copy(spins)
        end
        
        println("  Run $run: Energy = $final_energy")
    end
    
    return best_energy, best_spins
end

"""
Solve Problem 1: Ground state energy of Fullerene graph
"""
function solve_problem1()
    println("\n" * "=" ^ 70)
    println("PROBLEM 1: Ground State Energy of Fullerene Graph")
    println("=" ^ 70)
    
    fullerene_coords = fullerene()
    fullerene_graph = UnitDiskGraph(fullerene_coords, sqrt(5))
    
    println("Fullerene graph: $(nv(fullerene_graph)) vertices, $(ne(fullerene_graph)) edges")
    println("\nRunning Simulated Annealing...")
    
    best_energy, best_spins = simulated_annealing(
        fullerene_graph,
        T_init=20.0,      # Higher initial temperature
        T_final=0.001,
        cooling_rate=0.98, # Slower cooling
        steps_per_temp=3000,
        n_runs=30
    )
    
    println("\n✅ Ground State Energy: $best_energy")
    verify_energy = ising_energy(fullerene_graph, best_spins)
    println("   Verified: $verify_energy")
    
    n_up = count(s -> s == 1, best_spins)
    n_down = count(s -> s == -1, best_spins)
    println("   Spin distribution: ↑=$n_up, ↓=$n_down")
    
    return best_energy, best_spins, fullerene_graph
end

# ============================================================================
# PROBLEM 2: Spectral Gap Analysis
# ============================================================================

"""
Construct a Metropolis single-spin-flip transition matrix P(T)
Row-stochastic: choose a spin uniformly at random, attempt a flip with
acceptance probability min(1, exp(-ΔE/T)).
"""
function metropolis_transition_matrix(graph::SimpleGraph, T::Float64)
    n = nv(graph)
    N = 2^n
    
    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float64[]
    
    for config in 0:(N - 1)
        # Decode spins: bit 0 corresponds to vertex 1
        spins = [(config >> i) & 1 == 0 ? -1 : 1 for i in 0:(n - 1)]
        row_sum = 0.0
        
        # Propose flipping each spin i with probability 1/n
        for i in 1:n
            ΔE = 0.0
            for nb in neighbors(graph, i)
                ΔE -= 2 * spins[i] * spins[nb]
            end
            p_acc = min(1.0, exp(-ΔE / T))
            p = p_acc / n
            if p > 0
                dest = config ⊻ (1 << (i - 1))  # flip bit (i-1)
                push!(I_vals, config + 1)
                push!(J_vals, dest + 1)
                push!(V_vals, p)
                row_sum += p
            end
        end
        # Self-transition to make the row sum to 1
        push!(I_vals, config + 1)
        push!(J_vals, config + 1)
        push!(V_vals, max(0.0, 1.0 - row_sum))
    end
    
    return sparse(I_vals, J_vals, V_vals, N, N)
end

"""
Calculate spectral gap of the Metropolis chain: 1 - |λ₂|
λ₁ = 1 for an ergodic Markov chain.
"""
function spectral_gap(graph::SimpleGraph, T::Float64)
    P = metropolis_transition_matrix(graph, T)
    
    if size(P, 1) <= 2048
        ev = eigvals(Matrix(P))
        ev = sort(abs.(ev), rev=true)
    else
        ev, _ = eigs(P, nev=2, which=:LM, maxiter=10000, tol=1e-6)  # largest magnitudes
        ev = sort(abs.(ev), rev=true)
    end
    
    # Ensure the leading eigenvalue ~ 1.0
    λ1 = ev[1]
    λ2 = length(ev) >= 2 ? ev[2] : 0.0
    return max(0.0, 1.0 - abs(λ2))
end

"""
Generate different graph topologies
"""
function generate_topologies()
    topologies = Dict{String, Vector{SimpleGraph}}()
    
    topologies["Path"] = [path_graph(n) for n in 2:18]
    topologies["Cycle"] = [cycle_graph(n) for n in 3:18]
    topologies["Complete"] = [complete_graph(n) for n in 2:10]
    topologies["Star"] = [star_graph(n) for n in 2:18]
    topologies["Wheel"] = [wheel_graph(n) for n in 4:18]
    
    return topologies
end

"""
Task 1: Analyze spectral gap vs. temperature
"""
function analyze_gap_vs_temperature()
    println("\n--- Task 1: Spectral Gap vs. Temperature ---")
    
    temperatures = 0.1:0.2:2.0  # Reduced points for faster testing
    topologies = generate_topologies()
    
    plots_dict = Dict()
    
    for (name, graphs) in topologies
        println("Processing $name topology...")
        
        if name == "Complete" && length(graphs) >= 7
            graph = graphs[7]
        elseif length(graphs) >= 8
            graph = graphs[8]
        else
            graph = graphs[end]
        end
        
        n = nv(graph)
        println("  Graph size: $n vertices, $(ne(graph)) edges")
        
        gaps = Float64[]
        for T in temperatures
            gap = spectral_gap(graph, T)
            push!(gaps, gap)
        end
        
        plots_dict[name] = (temperatures, gaps, n)
    end
    
    p = plot(
        title="Spectral Gap vs. Temperature",
        xlabel="Temperature T",
        ylabel="Spectral Gap",
        legend=:topright,
        size=(800, 600)
    )
    
    for (name, (temps, gaps, n)) in plots_dict
        plot!(p, temps, gaps, label="$name (n=$n)", marker=:circle, linewidth=2)
    end
    
    savefig(p, "gap_vs_temperature.png")
    println("✅ Plot saved: gap_vs_temperature.png")
    
    return plots_dict
end

"""
Task 2: Analyze spectral gap vs. system size at T = 0.1
"""
function analyze_gap_vs_size()
    println("\n--- Task 2: Spectral Gap vs. System Size ---")
    
    T = 0.1
    topologies = generate_topologies()
    
    plots_dict = Dict()
    
    for (name, graphs) in topologies
        println("Processing $name topology...")
        
        sizes = Int[]
        gaps = Float64[]
        
        for graph in graphs
            n = nv(graph)
            
            if n > 19  # Reduced for faster testing
                println("  Skipping n=$n (too large)")
                continue
            end
            
            gap = spectral_gap(graph, T)
            push!(sizes, n)
            push!(gaps, gap)
            println("  n = $n, Gap = $gap")
        end
        
        plots_dict[name] = (sizes, gaps)
    end
    
    p = plot(
        title="Spectral Gap vs. System Size (T = 0.1)",
        xlabel="System Size N",
        ylabel="Spectral Gap",
        legend=:topright,
        size=(800, 600)
    )
    
    for (name, (sizes, gaps)) in plots_dict
        plot!(p, sizes, gaps, label=name, marker=:circle, linewidth=2)
    end
    
    savefig(p, "gap_vs_size.png")
    println("✅ Plot saved: gap_vs_size.png")
    
    return plots_dict
end

"""
Solve Problem 2: Spectral gap analysis
"""
function solve_problem2()
    println("\n" * "=" ^ 70)
    println("PROBLEM 2: Spectral Gap Analysis")
    println("=" ^ 70)
    
    temp_results = analyze_gap_vs_temperature()
    size_results = analyze_gap_vs_size()
    
    println("\n✅ Problem 2 completed!")
    
    return temp_results, size_results
end

# ============================================================================
# PROBLEM 3: Parallel Tempering for Spin Glass (Challenge)
# ============================================================================

"""
Strong product of two graphs
"""
function strong_product(g1, g2)
    vs = [(v1, v2) for v1 in vertices(g1), v2 in vertices(g2)]
    graph = SimpleGraph(length(vs))
    for (i, vi) in enumerate(vs), (j, vj) in enumerate(vs)
        if (vi[1] == vj[1] && has_edge(g2, vi[2], vj[2])) ||
                (vi[2] == vj[2] && has_edge(g1, vi[1], vj[1])) ||
                (has_edge(g1, vi[1], vj[1]) && has_edge(g2, vi[2], vj[2]))
            add_edge!(graph, i, j)
        end
    end
    return graph
end

"""
Strong power of a graph
"""
strong_power(g, k::Int) = k == 1 ? g : strong_product(g, strong_power(g, k - 1))

"""
Create spin glass problem on cycle graph
"""
function spin_glass_c(n::Int, k::Int)
    g1 = Graphs.cycle_graph(n)
    g = strong_power(g1, k)
    coupling = fill(1, ne(g))
    bias = 1 .- degree(g)
    return SpinGlass(g, coupling, bias)
end

"""
Calculate energy of a spin configuration for SpinGlass problem
"""
function energy(sg::SpinGlass, spins::Vector{Int})
    E = 0
    for (idx, edge) in enumerate(edges(sg.graph))
        i, j = src(edge), dst(edge)
        E += sg.J[idx] * spins[i] * spins[j]
    end
    for i in 1:length(spins)
        E += sg.h[i] * spins[i]
    end
    return E
end

"""
Parallel Tempering (Replica Exchange Monte Carlo) Algorithm
"""
function parallel_tempering(
    sg::SpinGlass;
    n_replicas::Int=16,
    T_min::Float64=0.01,
    T_max::Float64=5.0,
    n_sweeps::Int=10000,
    exchange_interval::Int=10,
    thermalization::Int=1000,
    show_progress::Bool=true,
    progress_name::String="PT",
    temperature_schedule::Symbol=:beta_power,
    schedule_gamma::Float64=2.0,
    quench_interval::Int=0
)
    n = nv(sg.graph)
    
    # Precompute neighbor lists and edge indices for fast ΔE updates
    edges_vec = collect(edges(sg.graph))
    edge_index = Dict{Tuple{Int,Int},Int}()
    for (idx, e) in enumerate(edges_vec)
        i, j = src(e), dst(e)
        edge_index[(min(i,j), max(i,j))] = idx
    end
    neighbors_list = Vector{Vector{Int}}(undef, n)
    neighbor_edge_indices = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        nbs = collect(neighbors(sg.graph, i))
        neighbors_list[i] = nbs
        neighbor_edge_indices[i] = [edge_index[(min(i,nb), max(i,nb))] for nb in nbs]
    end

    # Temperature ladder generator (densify near low T by default)
    function make_temperatures(Tmin::Float64, Tmax::Float64, R::Int, schedule::Symbol, gamma::Float64)
        if R < 2
            return [Tmax]
        end
        if schedule === :geometric
            return [Tmax * (Tmin / Tmax)^(i / (R - 1)) for i in 0:(R - 1)]
        elseif schedule === :beta_linear
            βmin = 1.0 / Tmax
            βmax = 1.0 / Tmin
            βs = [βmin + (βmax - βmin) * (i / (R - 1)) for i in 0:(R - 1)]
            return [1.0 / β for β in βs]
        else # :beta_power
            βmin = 1.0 / Tmax
            βmax = 1.0 / Tmin
            βs = [βmin + (βmax - βmin) * ((i / (R - 1))^gamma) for i in 0:(R - 1)]
            return [1.0 / β for β in βs]
        end
    end
    temperatures = make_temperatures(T_min, T_max, n_replicas, temperature_schedule, schedule_gamma)
    
    # Seed a few replicas with structured initial states for better coverage
    replicas = Vector{Vector{Int}}(undef, n_replicas)
    replicas[1] = fill(-1, n)
    if n_replicas >= 2
        replicas[2] = fill(1, n)
    end
    if n_replicas >= 3
        # Align with local fields: s_i = -sign(h_i) (treat 0 as +1)
        replicas[3] = [sg.h[i] <= 0 ? 1 : -1 for i in 1:n]
    end
    for r in 4:n_replicas
        replicas[r] = rand([-1, 1], n)
    end
    energies = [energy(sg, replica) for replica in replicas]
    
    best_energy = minimum(energies)
    best_spins = copy(replicas[argmin(energies)])
    
    n_accepted = zeros(Int, n_replicas)
    n_exchanges = zeros(Int, n_replicas - 1)
    n_exchange_attempts = zeros(Int, n_replicas - 1)
    
    # progress helper
    function bar(p::Float64; width::Int=40)
        k = clamp(Int(round(p * width)), 0, width)
        return "[" * repeat("=", k) * repeat(" ", width - k) * "]"
    end
    next_tick = 1
    tick_every = max(1, Int(ceil(n_sweeps / 100)))  # ~100 updates

    for sweep in 1:n_sweeps
        for r in 1:n_replicas
            T = temperatures[r]
            
            # Random-order sweep: consider each spin once per sweep in random order
            for i in randperm(n)
                
                ΔE = 0.0
                nbs = neighbors_list[i]
                eidxs = neighbor_edge_indices[i]
                @inbounds for k in eachindex(nbs)
                    nb = nbs[k]
                    edge_idx = eidxs[k]
                    ΔE -= 2 * sg.J[edge_idx] * replicas[r][i] * replicas[r][nb]
                end
                ΔE -= 2 * sg.h[i] * replicas[r][i]
                
                if ΔE < 0 || rand() < exp(-ΔE / T)
                    replicas[r][i] *= -1
                    energies[r] += ΔE
                    n_accepted[r] += 1
                end
            end
        end
        
        if sweep % exchange_interval == 0
            for r in 1:(n_replicas-1)
                T1, T2 = temperatures[r], temperatures[r+1]
                E1, E2 = energies[r], energies[r+1]
                
                # ΔE is the log Metropolis ratio for the swap up to sign:
                # ΔE = (E2 - E1) * (β1 - β2), where β = 1/T.
                # Acceptance: min(1, exp(-ΔE))
                ΔE = (E2 - E1) * (1/T1 - 1/T2)
                
                n_exchange_attempts[r] += 1
                
                if ΔE <= 0 || rand() < exp(-ΔE)
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                    energies[r], energies[r+1] = energies[r+1], energies[r]
                    n_exchanges[r] += 1
                end
            end
            # Periodic greedy quench on the coldest replica to intensify search
            if quench_interval > 0 && sweep > thermalization && sweep % quench_interval == 0
                cold_idx = n_replicas
                E_quench = greedy_quench!(sg, replicas[cold_idx]; max_passes=5)
                if E_quench < energies[cold_idx]
                    energies[cold_idx] = E_quench
                end
            end
        end
        
        if sweep > thermalization
            min_idx = argmin(energies)
            if energies[min_idx] < best_energy
                best_energy = energies[min_idx]
                best_spins = copy(replicas[min_idx])
            end
        end
        
        if show_progress && sweep >= next_tick
            p = sweep / n_sweeps
            attempted_flips = n_replicas * sweep * n
            acc_rate = attempted_flips > 0 ? sum(n_accepted) / attempted_flips : 0.0
            exch_rate = sum(n_exchange_attempts) > 0 ? sum(n_exchanges) / sum(n_exchange_attempts) : 0.0
            println("  $progress_name ", bar(p), @sprintf(" %6.2f%% ", p*100),
                    "Best=", best_energy, 
                    " Min=", minimum(energies),
                    " Acc=", @sprintf("%.3f", acc_rate),
                    " Exch=", @sprintf("%.3f", exch_rate),
                    " Sweep=", sweep, "/", n_sweeps)
            flush(stdout)
            next_tick += tick_every
        end
    end
    
    return best_spins, best_energy
end

# ============================================================================
# PROBLEM 3: Parallel Tempering for Spin Glass (Challenge)
# ============================================================================

"""
Strong product of two graphs
"""
function strong_product(g1, g2)
    vs = [(v1, v2) for v1 in vertices(g1), v2 in vertices(g2)]
    graph = SimpleGraph(length(vs))
    for (i, vi) in enumerate(vs), (j, vj) in enumerate(vs)
        if (vi[1] == vj[1] && has_edge(g2, vi[2], vj[2])) ||
                (vi[2] == vj[2] && has_edge(g1, vi[1], vj[1])) ||
                (has_edge(g1, vi[1], vj[1]) && has_edge(g2, vi[2], vj[2]))
            add_edge!(graph, i, j)
        end
    end
    return graph
end

"""
Strong power of a graph
"""
strong_power(g, k::Int) = k == 1 ? g : strong_product(g, strong_power(g, k - 1))

"""
Create spin glass problem on cycle graph
"""
function spin_glass_c(n::Int, k::Int)
    g1 = Graphs.cycle_graph(n)
    g = strong_power(g1, k)
    coupling = fill(1, ne(g))
    bias = 1 .- degree(g)
    return SpinGlass(g, coupling, bias)
end

"""
Calculate energy of a spin configuration for SpinGlass problem
"""
function energy(sg::SpinGlass, spins::Vector{Int})
    E = 0
    for (idx, edge) in enumerate(edges(sg.graph))
        i, j = src(edge), dst(edge)
        E += sg.J[idx] * spins[i] * spins[j]
    end
    for i in 1:length(spins)
        E += sg.h[i] * spins[i]
    end
    return E
end

"""
Parallel Tempering (Replica Exchange Monte Carlo) Algorithm
"""
function parallel_tempering(
    sg::SpinGlass;
    n_replicas::Int=16,
    T_min::Float64=0.01,
    T_max::Float64=5.0,
    n_sweeps::Int=10000,
    exchange_interval::Int=10,
    thermalization::Int=1000,
    show_progress::Bool=true,
    progress_name::String="PT",
    temperature_schedule::Symbol=:beta_power,
    schedule_gamma::Float64=2.0,
    quench_interval::Int=0
)
    n = nv(sg.graph)
    
    # Precompute neighbor lists and edge indices for fast ΔE updates
    edges_vec = collect(edges(sg.graph))
    edge_index = Dict{Tuple{Int,Int},Int}()
    for (idx, e) in enumerate(edges_vec)
        i, j = src(e), dst(e)
        edge_index[(min(i,j), max(i,j))] = idx
    end
    neighbors_list = Vector{Vector{Int}}(undef, n)
    neighbor_edge_indices = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        nbs = collect(neighbors(sg.graph, i))
        neighbors_list[i] = nbs
        neighbor_edge_indices[i] = [edge_index[(min(i,nb), max(i,nb))] for nb in nbs]
    end

    # Temperature ladder generator (densify near low T by default)
    function make_temperatures(Tmin::Float64, Tmax::Float64, R::Int, schedule::Symbol, gamma::Float64)
        if R < 2
            return [Tmax]
        end
        if schedule === :geometric
            return [Tmax * (Tmin / Tmax)^(i / (R - 1)) for i in 0:(R - 1)]
        elseif schedule === :beta_linear
            βmin = 1.0 / Tmax
            βmax = 1.0 / Tmin
            βs = [βmin + (βmax - βmin) * (i / (R - 1)) for i in 0:(R - 1)]
            return [1.0 / β for β in βs]
        else # :beta_power
            βmin = 1.0 / Tmax
            βmax = 1.0 / Tmin
            βs = [βmin + (βmax - βmin) * ((i / (R - 1))^gamma) for i in 0:(R - 1)]
            return [1.0 / β for β in βs]
        end
    end
    temperatures = make_temperatures(T_min, T_max, n_replicas, temperature_schedule, schedule_gamma)
    
    # Seed a few replicas with structured initial states for better coverage
    replicas = Vector{Vector{Int}}(undef, n_replicas)
    replicas[1] = fill(-1, n)
    if n_replicas >= 2
        replicas[2] = fill(1, n)
    end
    if n_replicas >= 3
        # Align with local fields: s_i = -sign(h_i) (treat 0 as +1)
        replicas[3] = [sg.h[i] <= 0 ? 1 : -1 for i in 1:n]
    end
    for r in 4:n_replicas
        replicas[r] = rand([-1, 1], n)
    end
    energies = [energy(sg, replica) for replica in replicas]
    
    best_energy = minimum(energies)
    best_spins = copy(replicas[argmin(energies)])
    
    n_accepted = zeros(Int, n_replicas)
    n_exchanges = zeros(Int, n_replicas - 1)
    n_exchange_attempts = zeros(Int, n_replicas - 1)
    
    # progress helper
    function bar(p::Float64; width::Int=40)
        k = clamp(Int(round(p * width)), 0, width)
        return "[" * repeat("=", k) * repeat(" ", width - k) * "]"
    end
    next_tick = 1
    tick_every = max(1, Int(ceil(n_sweeps / 100)))  # ~100 updates

    for sweep in 1:n_sweeps
        for r in 1:n_replicas
            T = temperatures[r]
            
            # Random-order sweep: consider each spin once per sweep in random order
            for i in randperm(n)
                
                ΔE = 0.0
                nbs = neighbors_list[i]
                eidxs = neighbor_edge_indices[i]
                @inbounds for k in eachindex(nbs)
                    nb = nbs[k]
                    edge_idx = eidxs[k]
                    ΔE -= 2 * sg.J[edge_idx] * replicas[r][i] * replicas[r][nb]
                end
                ΔE -= 2 * sg.h[i] * replicas[r][i]
                
                if ΔE < 0 || rand() < exp(-ΔE / T)
                    replicas[r][i] *= -1
                    energies[r] += ΔE
                    n_accepted[r] += 1
                end
            end
        end
        
        if sweep % exchange_interval == 0
            for r in 1:(n_replicas-1)
                T1, T2 = temperatures[r], temperatures[r+1]
                E1, E2 = energies[r], energies[r+1]
                
                # ΔE is the log Metropolis ratio for the swap up to sign:
                # ΔE = (E2 - E1) * (β1 - β2), where β = 1/T.
                # Acceptance: min(1, exp(-ΔE))
                ΔE = (E2 - E1) * (1/T1 - 1/T2)
                
                n_exchange_attempts[r] += 1
                
                if ΔE <= 0 || rand() < exp(-ΔE)
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                    energies[r], energies[r+1] = energies[r+1], energies[r]
                    n_exchanges[r] += 1
                end
            end
            # Periodic greedy quench on the coldest replica to intensify search
            if quench_interval > 0 && sweep > thermalization && sweep % quench_interval == 0
                cold_idx = n_replicas
                E_quench = greedy_quench!(sg, replicas[cold_idx]; max_passes=5)
                if E_quench < energies[cold_idx]
                    energies[cold_idx] = E_quench
                end
            end
        end
        
        if sweep > thermalization
            min_idx = argmin(energies)
            if energies[min_idx] < best_energy
                best_energy = energies[min_idx]
                best_spins = copy(replicas[min_idx])
            end
        end
        
        if show_progress && sweep >= next_tick
            p = sweep / n_sweeps
            attempted_flips = n_replicas * sweep * n
            acc_rate = attempted_flips > 0 ? sum(n_accepted) / attempted_flips : 0.0
            exch_rate = sum(n_exchange_attempts) > 0 ? sum(n_exchanges) / sum(n_exchange_attempts) : 0.0
            println("  $progress_name ", bar(p), @sprintf(" %6.2f%% ", p*100),
                    "Best=", best_energy, 
                    " Min=", minimum(energies),
                    " Acc=", @sprintf("%.3f", acc_rate),
                    " Exch=", @sprintf("%.3f", exch_rate),
                    " Sweep=", sweep, "/", n_sweeps)
            flush(stdout)
            next_tick += tick_every
        end
    end
    
    return best_spins, best_energy
end

"""
Solve Problem 3: Spin glass with parallel tempering
"""
function solve_problem3()
    println("\n" * "=" ^ 70)
    println("PROBLEM 3: Parallel Tempering for Spin Glass (Challenge)")
    println("=" ^ 70)
    
    # Test case 1
    println("\n--- Test Case 1: spin_glass_c(5, 2) ---")
    sg1 = spin_glass_c(5, 2)
    solution1 = my_ground_state_solver(sg1)
    energy1 = energy(sg1, solution1)
    println("\n✅ Test 1 Energy: $energy1 (Expected: -85)")
    test1_pass = energy1 == -85
    println("   Test 1: ", test1_pass ? "PASSED ✓" : "FAILED ✗")
    
    # Test case 2
    println("\n--- Test Case 2: spin_glass_c(7, 4) ---")
    println("⚠️  This may take 20-60 minutes...")
    sg2 = spin_glass_c(7, 4)
    solution2 = my_ground_state_solver(sg2)
    energy2 = energy(sg2, solution2)
    println("\n✅ Test 2 Energy: $energy2 (Expected: < -93855)")
    test2_pass = energy2 < -93855
    println("   Test 2: ", test2_pass ? "PASSED ✓" : "FAILED ✗")
    
    return energy1, energy2, test1_pass, test2_pass
end

"""
My ground state solver using parallel tempering + advanced local search
"""
function my_ground_state_solver(sg::SpinGlass; show_progress::Bool=true, progress_name::String="PT")
    n = nv(sg.graph)
    println("  Vertices: $n, Edges: $(ne(sg.graph))")
    
    if n <= 30
        n_replicas = 12
        n_sweeps = 5000
        T_max = 3.0
        n_runs = 5
    elseif n <= 100
        n_replicas = 16
        n_sweeps = 10000
        T_max = 5.0
        n_runs = 5
    else
        # Large problem: increase everything
        n_replicas = 80  # Increased from 56
        n_sweeps = 50000  # Increased from 30000
        T_max = 200.0
        n_runs = 10  # Increased from 6
    end
    
    best_solution = nothing
    best_energy = Inf
    population = Vector{Vector{Int}}()  # Store good solutions for crossover
    
    for run in 1:n_runs
        println("\n  --- Run $run/$n_runs ---")
        
        # Periodic quench interval
        quench_interval = max(50, n_sweeps ÷ 100)

        spins, E = parallel_tempering(
            sg,
            n_replicas=n_replicas,
            T_min=0.0003,  # Lower minimum temperature
            T_max=T_max,
            n_sweeps=n_sweeps,
            exchange_interval=1,
            thermalization=n_sweeps ÷ 5,
            show_progress=show_progress,
            progress_name=progress_name,
            temperature_schedule=:beta_power,
            schedule_gamma=3.5,  # Increased from 3.2
            quench_interval=quench_interval
        )
        
        # Apply advanced local search techniques
        println("  Applying advanced local search...")
        
        # 1. Greedy quench
        E_quench = greedy_quench!(sg, spins; max_passes=15)
        if E_quench < E
            E = E_quench
        end
        
        # 2. Cluster flips
        for _ in 1:5
            improved, delta = cluster_flip!(sg, spins; n_attempts=20)
            if improved
                E += delta
                println("    Cluster flip: ΔE = $delta, New E = $E")
            end
        end
        
        # 3. k-opt search (try k=2,3,4)
        for k in [2, 3, 4]
            improved = k_opt_search!(sg, spins; k=k, n_attempts=100)
            if improved
                E_new = energy(sg, spins)
                println("    $k-opt improved: E = $E_new")
                E = E_new
            end
        end
        
        # 4. Final greedy quench
        E = greedy_quench!(sg, spins; max_passes=10)
        
        # Update best solution
        if E < best_energy
            best_energy = E
            best_solution = copy(spins)
            println("  ★ NEW BEST: $best_energy")
        end
        
        # Add to population if good enough
        if E <= best_energy + 100  # Within 100 of best
            push!(population, copy(spins))
        end
        
        println("  Run $run Energy: $E")
        
        # Every few runs, try crossover if we have multiple solutions
        if run >= 3 && length(population) >= 2 && run % 2 == 0
            println("  Attempting crossover...")
            for _ in 1:3
                p1 = rand(population)
                p2 = rand(population)
                child = crossover(sg, p1, p2)
                
                # Apply advanced search to child
                E_child = energy(sg, child)
                cluster_flip!(sg, child; n_attempts=15)
                k_opt_search!(sg, child; k=3, n_attempts=50)
                E_child = greedy_quench!(sg, child; max_passes=10)
                
                if E_child < best_energy
                    best_energy = E_child
                    best_solution = copy(child)
                    println("  ★★ CROSSOVER BEST: $best_energy")
                end
            end
        end
        
        # Intelligent restart: occasionally restart from perturbed best solution
        if run >= 4 && run % 3 == 0 && best_solution !== nothing
            println("  Intelligent restart from best solution...")
            restart_spins = intelligent_restart(best_solution, 0.15)
            E_restart = energy(sg, restart_spins)
            
            # Quick refinement
            cluster_flip!(sg, restart_spins; n_attempts=20)
            E_restart = greedy_quench!(sg, restart_spins; max_passes=15)
            
            if E_restart < best_energy
                best_energy = E_restart
                best_solution = copy(restart_spins)
                println("  ★★★ RESTART BEST: $best_energy")
            end
        end
    end
    
    # Final intensive local search on best solution
    if best_solution !== nothing
        println("\n  Final intensive optimization on best solution...")
        final_spins = copy(best_solution)
        
        for round in 1:3
            println("    Round $round/3")
            cluster_flip!(sg, final_spins; n_attempts=30)
            for k in [2, 3, 4, 5]
                k_opt_search!(sg, final_spins; k=k, n_attempts=150)
            end
            E_final = greedy_quench!(sg, final_spins; max_passes=20)
            
            if E_final < best_energy
                best_energy = E_final
                best_solution = copy(final_spins)
                println("    Final round improvement: $best_energy")
            end
        end
    end
    
    println("\n  ✅ FINAL BEST ENERGY: $best_energy")
    
    return best_solution
end
"""
Cluster flip: identify and flip connected components of spins
This helps escape local minima by making larger moves
"""
function cluster_flip!(sg::SpinGlass, spins::Vector{Int}; n_attempts::Int=10)
    n = length(spins)
    best_delta = 0.0
    best_cluster = Int[]
    
    for _ in 1:n_attempts
        # Start a random walk to build a cluster
        cluster_size = rand(2:min(8, n÷10 + 2))
        cluster = Set{Int}([rand(1:n)])
        
        # Grow cluster by adding neighbors
        for _ in 1:(cluster_size-1)
            if isempty(cluster)
                break
            end
            current = rand(collect(cluster))
            nbs = neighbors(sg.graph, current)
            if !isempty(nbs)
                push!(cluster, rand(nbs))
            end
        end
        
        cluster = collect(cluster)
        
        # Calculate energy change if we flip this cluster
        delta = 0.0
        # Edge terms
        for (idx, edge) in enumerate(edges(sg.graph))
            i, j = src(edge), dst(edge)
            i_in = i ∈ cluster
            j_in = j ∈ cluster
            
            if i_in && j_in
                # Both in cluster: no change in interaction
                continue
            elseif i_in || j_in
                # One in cluster: interaction flips sign
                delta -= 2 * sg.J[idx] * spins[i] * spins[j]
            end
        end
        # Bias terms
        for i in cluster
            delta -= 2 * sg.h[i] * spins[i]
        end
        
        if delta < best_delta
            best_delta = delta
            best_cluster = cluster
        end
    end
    
    # Apply best cluster flip if it improves
    if best_delta < -1e-10
        for i in best_cluster
            spins[i] *= -1
        end
        return true, best_delta
    end
    return false, 0.0
end

"""
k-opt local search: try flipping k spins simultaneously
For k=2, this is 2-opt; we implement a stochastic version
"""
function k_opt_search!(sg::SpinGlass, spins::Vector{Int}; k::Int=2, n_attempts::Int=50)
    n = length(spins)
    improved = false
    
    edges_vec = collect(edges(sg.graph))
    edge_index = Dict{Tuple{Int,Int},Int}()
    for (idx, e) in enumerate(edges_vec)
        i, j = src(e), dst(e)
        edge_index[(min(i,j), max(i,j))] = idx
    end
    
    for _ in 1:n_attempts
        # Random k spins
        indices = randperm(n)[1:min(k, n)]
        
        # Calculate delta E for flipping all k spins
        delta = 0.0
        
        # Edge contributions
        for i in indices
            for nb in neighbors(sg.graph, i)
                key = (min(i,nb), max(i,nb))
                if haskey(edge_index, key)
                    idx = edge_index[key]
                    if nb ∈ indices
                        # Both flipped: no change
                        continue
                    else
                        # Only i flipped
                        delta -= 2 * sg.J[idx] * spins[i] * spins[nb]
                    end
                end
            end
        end
        
        # Bias contributions
        for i in indices
            delta -= 2 * sg.h[i] * spins[i]
        end
        
        # Apply if improves
        if delta < -1e-10
            for i in indices
                spins[i] *= -1
            end
            improved = true
        end
    end
    
    return improved
end

"""
Solution crossover: combine two solutions to create a new one
Uses partition crossover - randomly select spins from each parent
"""
function crossover(sg::SpinGlass, parent1::Vector{Int}, parent2::Vector{Int})
    n = length(parent1)
    child = similar(parent1)
    
    # Random crossover: each spin has 50% chance from each parent
    for i in 1:n
        child[i] = rand() < 0.5 ? parent1[i] : parent2[i]
    end
    
    # Apply greedy quench to the child
    greedy_quench!(sg, child; max_passes=3)
    
    return child
end

"""
Intelligent restart: perturb the best solution with controlled noise
"""
function intelligent_restart(spins::Vector{Int}, flip_prob::Float64=0.1)
    perturbed = copy(spins)
    n = length(spins)
    
    # Flip each spin with probability flip_prob
    for i in 1:n
        if rand() < flip_prob
            perturbed[i] *= -1
        end
    end
    
    return perturbed
end

"""
Greedy quench: deterministically flip any spin that lowers the energy.
Runs up to `max_passes` full passes or until no improvement.
"""
function greedy_quench!(sg::SpinGlass, spins::Vector{Int}; max_passes::Int=2)
    n = length(spins)
    # Precompute neighbor lists and edge indices
    edges_vec = collect(edges(sg.graph))
    edge_index = Dict{Tuple{Int,Int},Int}()
    for (idx, e) in enumerate(edges_vec)
        i, j = src(e), dst(e)
        edge_index[(min(i,j), max(i,j))] = idx
    end
    neighbors_list = Vector{Vector{Int}}(undef, n)
    neighbor_edge_indices = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        nbs = collect(neighbors(sg.graph, i))
        neighbors_list[i] = nbs
        neighbor_edge_indices[i] = [edge_index[(min(i,nb), max(i,nb))] for nb in nbs]
    end

    E = energy(sg, spins)
    for _ in 1:max_passes
        improved = false
        for i in 1:n
            ΔE = 0.0
            nbs = neighbors_list[i]
            eidxs = neighbor_edge_indices[i]
            @inbounds for k in eachindex(nbs)
                nb = nbs[k]
                edge_idx = eidxs[k]
                ΔE -= 2 * sg.J[edge_idx] * spins[i] * spins[nb]
            end
            ΔE -= 2 * sg.h[i] * spins[i]
            if ΔE < 0
                spins[i] *= -1
                E += ΔE
                improved = true
            end
        end
        if !improved
            break
        end
    end
    return E
end



# ============================================================================
# MAIN EXECUTION
# ============================================================================

"""
Main function to run all problems
"""
function main(; run_problem1=true, run_problem2=true, run_problem3=true)
    println("\n")
    println("█" ^ 70)
    println(" " ^ 15 * "HOMEWORK 7 - HUICHENG ZHANG")
    println("█" ^ 70)
    
    Random.seed!(42)
    
    results = Dict()
    
    # Problem 1
    if run_problem1
        try
            energy1, spins1, graph1 = solve_problem1()
            results["problem1"] = (energy=energy1, spins=spins1, graph=graph1)
        catch e
            println("\n❌ Problem 1 failed: $e")
            results["problem1"] = nothing
        end
    end
    
    # Problem 2
    if run_problem2
        try
            temp_res, size_res = solve_problem2()
            results["problem2"] = (temp=temp_res, size=size_res)
        catch e
            println("\n❌ Problem 2 failed: $e")
            results["problem2"] = nothing
        end
    end
    
    # Problem 3
    if run_problem3
        try
            e1, e2, t1, t2 = solve_problem3()
            results["problem3"] = (energy1=e1, energy2=e2, test1=t1, test2=t2)
        catch e
            println("\n❌ Problem 3 failed: $e")
            results["problem3"] = nothing
        end
    end
    
    # Summary
    println("\n")
    println("█" ^ 70)
    println(" " ^ 25 * "SUMMARY")
    println("█" ^ 70)
    
    if run_problem1 && haskey(results, "problem1") && results["problem1"] !== nothing
        println("\n✅ Problem 1: Ground State Energy = $(results["problem1"].energy)")
    end
    
    if run_problem2 && haskey(results, "problem2") && results["problem2"] !== nothing
        println("✅ Problem 2: Plots generated (gap_vs_temperature.png, gap_vs_size.png)")
    end
    
    if run_problem3 && haskey(results, "problem3") && results["problem3"] !== nothing
        r = results["problem3"]
        println("✅ Problem 3:")
        println("   Test 1: Energy = $(r.energy1), ", r.test1 ? "PASSED ✓" : "FAILED ✗")
        println("   Test 2: Energy = $(r.energy2), ", r.test2 ? "PASSED ✓" : "FAILED ✗")
    end
    
    println("\n" * "█" ^ 70)
    println()
    
    return results
end

# Run main if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # For quick testing, you can disable problem 3 (which takes a long time)
    # main(run_problem1=true, run_problem2=true, run_problem3=false)
    
    # Full run (uncomment the line below)
    main(run_problem1=true, run_problem2=true, run_problem3=true)
    flush(stdout)
end
