using LinearAlgebra
using SparseArrays
using Arpack
using LightGraphs
using Plots
using Printf

# Energy function for anti-ferromagnetic Ising model
function energy(config::Vector{Int}, graph)
    N = nv(graph)
    E = 0.0
    for e in edges(graph)
        i, j = src(e), dst(e)
        E += config[i] * config[j]
    end
    return E
end

# Convert integer to spin configuration (-1, +1)
function int_to_spins(x::Int, N::Int)
    spins = zeros(Int, N)
    for i in 1:N
        spins[i] = (x >> (i-1)) & 1 == 1 ? 1 : -1
    end
    return spins
end

# Build generator matrix for continuous-time Glauber dynamics
function build_glauber_generator(graph, β::Float64)
    N = nv(graph)
    n_states = 1 << N
    
    # Precompute energies for all states
    energies = zeros(Float64, n_states)
    for s in 0:n_states-1
        config = int_to_spins(s, N)
        energies[s+1] = energy(config, graph)
    end
    
    # Build sparse matrix
    I_vec = Int[]
    J_vec = Int[]
    V_vec = Float64[]
    
    for s in 0:n_states-1
        E_s = energies[s+1]
        total_rate = 0.0
        
        for i in 1:N
            # Flip the i-th spin to get new state
            s_prime = s ⊻ (1 << (i-1))
            E_sp = energies[s_prime+1]
            
            # Heat bath transition rate
            rate = exp(-β * E_sp) / (exp(-β * E_s) + exp(-β * E_sp))
            
            push!(I_vec, s_prime + 1)
            push!(J_vec, s + 1)
            push!(V_vec, rate)
            total_rate += rate
        end
        
        # Diagonal element
        push!(I_vec, s + 1)
        push!(J_vec, s + 1)
        push!(V_vec, -total_rate)
    end
    
    L = sparse(I_vec, J_vec, V_vec, n_states, n_states)
    return L
end

# Calculate spectral gap
function spectral_gap(graph, β::Float64; nev=4)
    N = nv(graph)
    if N > 8
        return estimate_spectral_gap(graph, β)
    end
    
    L = build_glauber_generator(graph, β)
    
    try
        λs = eigs(L, which=:SR, nev=min(nev, size(L,1)), ritzvec=false, maxiter=1000)
        sorted_λs = sort(λs[1], by=real)
        
        # Find first non-zero eigenvalue
        gap = 0.0
        for λ in sorted_λs
            if abs(real(λ)) > 1e-12
                gap = abs(real(λ))
                break
            end
        end
        return gap
    catch e
        println("Eigenvalue calculation error: ", e)
        return NaN
    end
end

# Estimate spectral gap for large systems
function estimate_spectral_gap(graph, β::Float64)
    N = nv(graph)
    T = 1/β
    
    if N <= 4
        L = build_glauber_generator(graph, β)
        λs = eigen(Matrix(L))
        sorted_λs = sort(real.(λs.values))
        for λ in sorted_λs
            if abs(λ) > 1e-12
                return abs(λ)
            end
        end
        return 0.0
    else
        # Simplified estimation based on graph type and temperature
        is_bipartite = is_bipartite_graph(graph)
        if is_bipartite
            return 0.1 + 0.5 * exp(-2/T)
        else
            return exp(-4/T) + 0.01 * exp(-1/T)
        end
    end
end

# Graph topology definitions
function create_complete_graph(N::Int)
    return complete_graph(N)
end

function create_chain_graph(N::Int)
    return path_graph(N)
end

function create_cycle_graph(N::Int)
    return cycle_graph(N)
end

# Task 1: Spectral gap vs temperature (T = 0.1 to 2.0)
function analyze_temperature_dependence()
    println("="^50)
    println("Task 1: Spectral Gap vs Temperature Analysis")
    println("="^50)
    
    temps = 0.1:0.1:2.0
    graph_types = ["Complete", "Chain", "Cycle"]
    N = 4  # Fixed system size
    
    results = Dict{String, Vector{Float64}}()
    
    for graph_type in graph_types
        println("\nAnalyzing $graph_type graph (N=$N)...")
        gaps = Float64[]
        
        for (i, T) in enumerate(temps)
            @printf "  T = %.1f: " T
            β = 1.0 / T
            
            # Create graph
            if graph_type == "Complete"
                g = create_complete_graph(N)
            elseif graph_type == "Chain"
                g = create_chain_graph(N)
            else
                g = create_cycle_graph(N)
            end
            
            gap = spectral_gap(g, β)
            push!(gaps, gap)
            @printf "gap = %.6e\n" gap
        end
        results[graph_type] = gaps
    end
    
    # Plot temperature dependence
    p = plot(xlabel="Temperature T", ylabel="Spectral Gap", 
             title="Spectral Gap vs Temperature (N=$N)", 
             legend=:topright, size=(800, 600), dpi=300)
    
    colors = [:red, :blue, :green]
    markers = [:circle, :square, :diamond]
    
    for (i, graph_type) in enumerate(graph_types)
        plot!(p, temps, results[graph_type], 
              label=graph_type, 
              linewidth=2, marker=markers[i], markersize=4,
              color=colors[i])
    end
    
    savefig(p, "spectral_gap_vs_temperature.png")
    println("\nSaved plot: spectral_gap_vs_temperature.png")
    
    # Save data
    open("temperature_data.txt", "w") do file
        println(file, "Temperature Dependence Data (N=$N)")
        println(file, "T\tComplete\tChain\tCycle")
        println(file, "-"^50)
        for (i, T) in enumerate(temps)
            @printf file "%.1f\t%.6e\t%.6e\t%.6e\n" T results["Complete"][i] results["Chain"][i] results["Cycle"][i]
        end
    end
    println("Saved data: temperature_data.txt")
    
    return results
end

# Task 2: Spectral gap vs system size (T = 0.1)
function analyze_size_dependence()
    println("\n" * "="^50)
    println("Task 2: Spectral Gap vs System Size Analysis")
    println("="^50)
    
    T = 0.1
    β = 1.0 / T
    sizes = 3:6  # System size range
    graph_types = ["Complete", "Chain", "Cycle"]
    
    results = Dict{String, Vector{Float64}}()
    
    for graph_type in graph_types
        println("\nAnalyzing $graph_type graph (T=$T)...")
        gaps = Float64[]
        
        for N in sizes
            @printf "  N = %d: " N
            
            # Create graph
            if graph_type == "Complete"
                g = create_complete_graph(N)
            elseif graph_type == "Chain"
                g = create_chain_graph(N)
            else
                g = create_cycle_graph(N)
            end
            
            gap = spectral_gap(g, β)
            push!(gaps, gap)
            @printf "gap = %.6e\n" gap
        end
        results[graph_type] = gaps
    end
    
    # Plot size dependence
    p = plot(xlabel="System Size N", ylabel="Spectral Gap", 
             title="Spectral Gap vs System Size (T=$T)", 
             legend=:topright, size=(800, 600), dpi=300)
    
    colors = [:red, :blue, :green]
    markers = [:circle, :square, :diamond]
    
    for (i, graph_type) in enumerate(graph_types)
        plot!(p, sizes, results[graph_type], 
              label=graph_type, 
              linewidth=2, marker=markers[i], markersize=6,
              color=colors[i])
    end
    
    savefig(p, "spectral_gap_vs_size.png")
    println("\nSaved plot: spectral_gap_vs_size.png")
    
    # Save data
    open("size_data.txt", "w") do file
        println(file, "Size Dependence Data (T=$T)")
        println(file, "N\tComplete\tChain\tCycle")
        println(file, "-"^50)
        for (i, N) in enumerate(sizes)
            @printf file "%d\t%.6e\t%.6e\t%.6e\n" N results["Complete"][i] results["Chain"][i] results["Cycle"][i]
        end
    end
    println("Saved data: size_data.txt")
    
    return (sizes, results)
end

# Main function
function main()
    println("Anti-ferromagnetic Ising Model Spectral Gap Analysis")
    println("="^50)
    
    # Task 1: Temperature dependence analysis
    temp_results = analyze_temperature_dependence()
    
    # Task 2: System size dependence analysis  
    size_results = analyze_size_dependence()
    
    println("\n" * "="^50)
    println("Analysis completed!")
    println("Generated files:")
    println("  - spectral_gap_vs_temperature.png")
    println("  - spectral_gap_vs_size.png")
    println("  - temperature_data.txt")
    println("  - size_data.txt")
    println("="^50)
    
    return temp_results, size_results
end

# Run analysis
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
