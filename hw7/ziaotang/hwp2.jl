using SparseArrays
using LinearAlgebra
using Arpack
using Random
using Plots

function triangle_graph(n::Int)
    g = SimpleGraph(n)
    for i in 1:n-2
        add_edge!(g, i, i+1)
        add_edge!(g, i, i+2)
    end
    if n >= 3
        add_edge!(g, n-1, n)
    end
    return g
end

function square_graph(n::Int)
    g = SimpleGraph(n)
    for i in 1:n-1
        add_edge!(g, i, i+1)
    end
    for i in 1:2:n-2
        add_edge!(g, i, i+2)
    end
    return g
end

function diamond_graph(n::Int)
    g = SimpleGraph(n)
    for i in 1:n-1
        add_edge!(g, i, i+1)
    end
    for i in 1:n-2
        add_edge!(g, i, i+2)
    end
    return g
end

function create_graphs(n::Int)
    return triangle_graph(n), square_graph(n), diamond_graph(n)
end

function calc_energy_diff(state, site, neighbor_lists)
    ΔE = 0.0
    spin_i = state[site]
    for neighbor in neighbor_lists[site]
        ΔE += 2 * spin_i * state[neighbor]
    end
    return ΔE
end

function transition_matrix_sparse(g::SimpleGraph, β::Float64)
    N = nv(g)
    total_states = 2^N
    I = Vector{Int}(undef, total_states * (N + 1))
    J = Vector{Int}(undef, total_states * (N + 1))
    V = Vector{Float64}(undef, total_states * (N + 1))
    
    idx = 1
    readbit(cfg, i::Int) = (cfg >> (i - 1)) & 1
    int2cfg(cfg::Int) = [2*readbit(cfg, i) - 1 for i in 1:N]
    
    neighbor_lists = [neighbors(g, i) |> collect for i in 1:N]

    for j_col in 0:total_states-1
        state_j = int2cfg(j_col)
        sum_prob = 0.0

        for k in 1:N
            i_row = j_col ⊻ (1 << (k - 1))
            
            ΔE = 0.0
            for neighbor in neighbor_lists[k]
                ΔE += 2 * state_j[k] * state_j[neighbor]
            end
            
            prob = min(1.0, exp(-β * ΔE)) / N
            
            I[idx] = i_row + 1
            J[idx] = j_col + 1
            V[idx] = prob
            idx += 1
            sum_prob += prob
        end

        I[idx] = j_col + 1
        J[idx] = j_col + 1
        V[idx] = 1.0 - sum_prob
        idx += 1
    end

    return sparse(I[1:idx-1], J[1:idx-1], V[1:idx-1], total_states, total_states)
end

function spectral_gap(P::SparseMatrixCSC)
    λ = eigs(P, nev=2, which=:LR)[1]
    return 1.0 - abs(λ[2])
end

function run_vs_temperature(g::SimpleGraph, β_list::Vector{Float64})
    gaps = Vector{Float64}(undef, length(β_list))
    for i in eachindex(β_list)
        β = β_list[i]
        P = transition_matrix_sparse(g, β)
        gaps[i] = spectral_gap(P)
        println("β = $(round(β, digits=2)) → gap = $(round(gaps[i], digits=6))")
    end
    return gaps
end

function run_vs_size(graph_constructor, N_list, β)
    gaps = Vector{Float64}(undef, length(N_list))
    for i in eachindex(N_list)
        N = N_list[i]
        g = graph_constructor(N)
        P = transition_matrix_sparse(g, β)
        gaps[i] = spectral_gap(P)
        println("N = $N → gap = $(round(gaps[i], digits=6))")
    end
    return gaps
end

function save_spectral_gap_plots()
    # Plot 1: Spectral Gap vs Temperature
    println("Generating Spectral Gap vs Temperature plot...")
    
    N = 8
    β_list = collect(0.1:0.2:2.0)
    g_tri, g_sq, g_dia = create_graphs(N)

    gaps_tri = run_vs_temperature(g_tri, β_list)
    gaps_sq = run_vs_temperature(g_sq, β_list)
    gaps_dia = run_vs_temperature(g_dia, β_list)

    # Create temperature plot
    p1 = plot(β_list, gaps_tri, label="Triangle", 
              xlabel="β (1/Temperature)", ylabel="Spectral Gap", 
              linewidth=2, marker=:circle, markersize=4,
              title="Spectral Gap vs Temperature (N=$N)",
              legend=:topright, grid=true)
    plot!(p1, β_list, gaps_sq, label="Square", 
          linewidth=2, marker=:square, markersize=4)
    plot!(p1, β_list, gaps_dia, label="Diamond", 
          linewidth=2, marker=:diamond, markersize=4)
    
    savefig(p1, "D:/juliahw/hw7/spectral_gap_vs_temperature.png")
    println("Saved: spectral_gap_vs_temperature.png")

    # Plot 2: Spectral Gap vs System Size
    println("\nGenerating Spectral Gap vs System Size plot...")
    
    β = 1.0
    N_list = collect(4:2:12)

    gaps_tri_size = run_vs_size(triangle_graph, N_list, β)
    gaps_sq_size = run_vs_size(square_graph, N_list, β)
    gaps_dia_size = run_vs_size(diamond_graph, N_list, β)

    # Create size plot
    p2 = plot(N_list, gaps_tri_size, label="Triangle", 
              xlabel="System Size (N)", ylabel="Spectral Gap", 
              linewidth=2, marker=:circle, markersize=4,
              title="Spectral Gap vs System Size (β=$β)",
              legend=:topright, grid=true)
    plot!(p2, N_list, gaps_sq_size, label="Square", 
          linewidth=2, marker=:square, markersize=4)
    plot!(p2, N_list, gaps_dia_size, label="Diamond", 
          linewidth=2, marker=:diamond, markersize=4)
    
    savefig(p2, "D:/juliahw/hw7/spectral_gap_vs_size.png")
    println("Saved: spectral_gap_vs_size.png")

    # Summary
    println("\n" * "="^50)
    println("PLOTS GENERATED SUCCESSFULLY!")
    println("="^50)
    println("Saved files:")
    println("1. spectral_gap_vs_temperature.png")
    println("   - Shows spectral gap vs β (1/Temperature) for N=8")
    println("   - Compares triangle, square, and diamond graph topologies")
    println("2. spectral_gap_vs_size.png")
    println("   - Shows spectral gap vs system size for β=1.0")
    println("   - System sizes: N = 4, 6, 8, 10, 12")
    
    return p1, p2
end

# Run the analysis and save plots
p1, p2 = save_spectral_gap_plots()