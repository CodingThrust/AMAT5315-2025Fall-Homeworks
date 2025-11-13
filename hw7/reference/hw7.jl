using Graphs, Random, LinearAlgebra, ProblemReductions, SparseArrays, Arpack
ENV["GKSwstype"] = "100"
using CairoMakie

# Problem 1
function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3*th), (1.0, 2 + th, 2*th), (th, 2.0, 2*th + 1.0))
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

function ising_energy(graph, spins)
    energy = 0
    for e in edges(graph)
        energy += spins[src(e)] * spins[dst(e)]
    end
    return energy
end

function simulated_annealing(graph; steps=10^6, T0=10.0, Tf=0.01)
    n = nv(graph)
    spins = rand([-1, 1], n)
    energy = ising_energy(graph, spins)
    best_energy = energy
    best_spins = copy(spins)

    neighbor_lists = [neighbors(graph, i) |> collect for i in 1:n]

    for step in 1:steps
        T = T0 * (Tf / T0)^(step / steps)
        site = rand(1:n)
        
        flip_Δenergy = 0
        for neighbor in neighbor_lists[site]
            flip_Δenergy -= 2 * spins[site] * spins[neighbor]
        end
        
        if flip_Δenergy < 0 || exp(-flip_Δenergy / T) > rand()
            spins[site] *= -1
            energy += flip_Δenergy
            
            if energy < best_energy
                best_energy = energy
                copy!(best_spins, spins)
            end
        end
    end
    return best_energy, best_spins
end

# Problem 2
function triangle_graph(n::Int)
    g = Graph(n)
    for i in 1:n-2
        add_edge!(g, i, i+1)
        add_edge!(g, i, i+2)
    end
    return g
end

function square_graph(n::Int)
    g = Graph(n)
    for i in 1:n-1
        add_edge!(g, i, i+1)
    end
    for i in 1:2:n-2
        add_edge!(g, i, i+2)
    end
    return g
end

function diamond_graph(n::Int)
    g = Graph(n)
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

function transition_matrix_sparse(g::Graph, β::Float64)
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

function run_vs_temperature(g::Graph, β_list::Vector{Float64})
    gaps = Vector{Float64}(undef, length(β_list))
    Threads.@threads for i in eachindex(β_list)
        β = β_list[i]
        P = transition_matrix_sparse(g, β)
        gaps[i] = spectral_gap(P)
        println("β = $(round(β, digits=2)) → gap = $(round(gaps[i], digits=6))")
    end
    return gaps
end

function run_vs_size(graph_constructor, N_list, β)
    gaps = Vector{Float64}(undef, length(N_list))
    Threads.@threads for i in eachindex(N_list)
        N = N_list[i]
        g = graph_constructor(N)
        P = transition_matrix_sparse(g, β)
        gaps[i] = spectral_gap(P)
        println("N = $N → gap = $(round(gaps[i], digits=6))")
    end
    return gaps
end

function plot_and_save(x, y_list, labels, xlabel, ylabel, title, filename)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = ylabel, title = title)

    for (i, y) in enumerate(y_list)
        scatter!(ax, x, y, markersize=6, label=labels[i])
        lines!(ax, x, y)
    end

    axislegend(ax, position=:lt)
    CairoMakie.save(filename, fig)
end

function main()
    # Fullerene part
    fullerene_pos = fullerene()
    fullerene_graph = UnitDiskGraph(fullerene_pos, sqrt(5))
    ground_energy, _ = simulated_annealing(fullerene_graph)
    @info "Ground state energy estimate: $ground_energy" # -66

    N = 18                                  
    β_list = collect(0.1:0.2:2.0)                   
    g_tri, g_sq, g_dia = create_graphs(N)

    @info "Spectral Gap vs Temperature"
    gaps_tri = run_vs_temperature(g_tri, β_list)
    gaps_sq = run_vs_temperature(g_sq, β_list)
    gaps_dia = run_vs_temperature(g_dia, β_list)

    plot_and_save(
        β_list,
        [gaps_tri, gaps_sq, gaps_dia],
        ["Triangle", "Square", "Diamond"],
        "β (1/Temperature)",
        "Spectral Gap",
        "hw7/reference/Spectral Gap vs Temperature",
        "hw7/reference/spectral_gap_vs_temperature.png"  
    )

    @info "Spectral Gap vs System Size (β=1.0)"
    β = 1.0                                 
    N_list = collect(4:2:18)                      

    gaps_tri_size = run_vs_size(triangle_graph, N_list, β)
    gaps_sq_size = run_vs_size(square_graph, N_list, β)
    gaps_dia_size = run_vs_size(diamond_graph, N_list, β)

    plot_and_save(
        N_list,
        [gaps_tri_size, gaps_sq_size, gaps_dia_size],
        ["Triangle", "Square", "Diamond"],
        "System Size (N)",
        "Spectral Gap",
        "Spectral Gap vs System Size (β=1.0)",
        "spectral_gap_vs_size.png"  
    )
end

main()
