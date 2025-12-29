# Homework 7 Solutions
# Author: tengxianglin
#
# Run with: julia --project=hw7 hw7/tengxianglin/hw7.jl

using Graphs
using LinearAlgebra, SparseArrays
using Random
using Printf

# ============================================================================
# Problem 1: Ground State Energy of Anti-ferromagnetic Ising Model
# ============================================================================

"""
Construct a unit disk graph from positions and radius
"""
function UnitDiskGraph(positions::Vector{NTuple{3,Float64}}, radius::Float64)
    n = length(positions)
    g = SimpleGraph(n)
    
    for i in 1:n
        for j in (i+1):n
            # Calculate Euclidean distance
            dist = sqrt(sum((positions[i][k] - positions[j][k])^2 for k in 1:3))
            if dist <= radius
                add_edge!(g, i, j)
            end
        end
    end
    
    return g
end

"""
Construct the fullerene graph in 3D space
"""
function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x, y, z), (y, z, x), (z, x, y))
            for loc in ((a, b, c), (a, b, -c), (a, -b, c), (a, -b, -c), (-a, b, c), (-a, b, -c), (-a, -b, c), (-a, -b, -c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

"""
Calculate the energy of a spin configuration
"""
function ising_energy(graph::SimpleGraph, spins::Vector{Int})
    energy = 0
    for e in edges(graph)
        energy += spins[src(e)] * spins[dst(e)]
    end
    return energy
end

"""
Simulated annealing to find ground state
"""
function simulated_annealing(graph::SimpleGraph; 
                             T_init=10.0, 
                             T_final=0.01, 
                             cooling_rate=0.95,
                             steps_per_temp=1000,
                             seed=42)
    Random.seed!(seed)
    n = nv(graph)
    
    # Random initial configuration
    spins = rand([-1, 1], n)
    current_energy = ising_energy(graph, spins)
    
    best_spins = copy(spins)
    best_energy = current_energy
    
    T = T_init
    while T > T_final
        for _ in 1:steps_per_temp
            # Randomly flip a spin
            i = rand(1:n)
            spins[i] *= -1
            new_energy = ising_energy(graph, spins)
            
            # Accept or reject
            ΔE = new_energy - current_energy
            if ΔE < 0 || rand() < exp(-ΔE / T)
                current_energy = new_energy
                if current_energy < best_energy
                    best_energy = current_energy
                    best_spins = copy(spins)
                end
            else
                # Revert the flip
                spins[i] *= -1
            end
        end
        T *= cooling_rate
    end
    
    return best_energy, best_spins
end

# Solve Problem 1
function solve_problem1()
    println("\n" * "="^70)
    println("Problem 1: Ground State Energy of Fullerene AFM Ising Model")
    println("="^70)
    
    fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5))
    println("Fullerene graph: $(nv(fullerene_graph)) vertices, $(ne(fullerene_graph)) edges")
    
    # Run simulated annealing multiple times to ensure we find the ground state
    best_energy = Inf
    best_config = nothing
    
    println("\nRunning simulated annealing (10 trials)...")
    for trial in 1:10
        # Use different seed for each trial to ensure variation
        # Also vary cooling schedule slightly
        seed_val = trial * 1000 + 42
        Random.seed!(seed_val)
        
        # Vary parameters slightly to get different paths
        cooling_rate = 0.95 + (trial % 3) * 0.01  # Slight variation
        energy, config = simulated_annealing(fullerene_graph, 
                                            seed=seed_val,
                                            cooling_rate=cooling_rate,
                                            steps_per_temp=1000 + trial * 10)
        if energy < best_energy
            best_energy = energy
            best_config = config
        end
        print("Trial $trial: E = $energy  ")
        trial % 3 == 0 && println()
    end
    
    println("\n\n" * "="^70)
    println("Ground State Energy: $best_energy")
    println("="^70)
    
    return best_energy, best_config
end

# ============================================================================
# Problem 2: Spectral Gap Analysis
# ============================================================================

"""
Construct the Glauber dynamics transition matrix for anti-ferromagnetic Ising model
The spectral gap is defined as gap = λ_max - λ_2, where λ_max = 1 (stationary state)
"""
function glauber_transition_matrix(graph::SimpleGraph, T::Float64=1.0)
    n = nv(graph)
    N = 2^n
    β = 1.0 / T
    
    # Build the transition rate matrix (sparse)
    W = spzeros(Float64, N, N)
    
    # Iterate over all configurations
    for state in 0:(N-1)
        # Convert state to spin configuration
        spins = [2 * ((state >> i) & 1) - 1 for i in 0:(n-1)]
        
        # Calculate energy of current state
        E_current = 0.0
        for e in edges(graph)
            E_current += spins[src(e)] * spins[dst(e)]
        end
        
        # For each possible spin flip
        total_rate = 0.0
        for i in 1:n
            # Flip spin i
            spins_new = copy(spins)
            spins_new[i] *= -1
            
            # Calculate energy after flip
            E_new = 0.0
            for e in edges(graph)
                E_new += spins_new[src(e)] * spins_new[dst(e)]
            end
            
            # Energy difference
            ΔE = E_new - E_current
            
            # Glauber transition rate (heat bath dynamics)
            # rate = 1/(1 + exp(β*ΔE))
            rate = 1.0 / (1.0 + exp(β * ΔE))
            
            # Convert new spin configuration back to state index
            new_state = 0
            for j in 1:n
                if spins_new[j] == 1
                    new_state += (1 << (j-1))
                end
            end
            
            # Add transition rate to off-diagonal
            W[new_state+1, state+1] += rate / n
            total_rate += rate / n
        end
        
        # Diagonal element: -sum of outgoing rates
        W[state+1, state+1] = -total_rate
    end
    
    return W
end

"""
Compute the spectral gap of the Glauber dynamics transition matrix
Gap = 0 - λ_2, where λ_2 is the second largest eigenvalue (largest is 0)
"""
function compute_spectral_gap(graph::SimpleGraph, T::Float64=1.0)
    W = glauber_transition_matrix(graph, T)
    
    # For small systems, compute all eigenvalues
    if nv(graph) <= 12
        eigenvalues = eigvals(Matrix(W))
        # Take real part (should be real for this matrix, imaginary part is numerical noise)
        eigenvalues_real = real.(eigenvalues)
        # Sort in descending order (largest first)
        sort!(eigenvalues_real, rev=true)
        # The largest eigenvalue should be ~0 (stationary state)
        # The second largest determines the relaxation rate
        # Spectral gap = -λ_2 (since λ_1 ≈ 0)
        gap = -eigenvalues_real[2]
        return gap
    else
        # For larger systems, would need Arpack/KrylovKit
        eigenvalues = eigvals(Matrix(W))
        eigenvalues_real = real.(eigenvalues)
        sort!(eigenvalues_real, rev=true)
        gap = -eigenvalues_real[2]
        return gap
    end
end

"""
Generate small test graphs
"""
function generate_test_graphs()
    graphs = Dict{String, SimpleGraph}()
    
    # Path graph
    for n in [4, 6, 8, 10, 12, 14, 16, 18]
        graphs["Path_$n"] = path_graph(n)
    end
    
    # Cycle graph
    for n in [4, 6, 8, 10, 12, 14, 16, 18]
        graphs["Cycle_$n"] = cycle_graph(n)
    end
    
    # Complete graph
    for n in [4, 6, 8, 10, 12]
        graphs["Complete_$n"] = complete_graph(n)
    end
    
    # Grid graph
    for (m, n) in [(2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (2, 5)]
        graphs["Grid_$(m)x$(n)"] = grid([m, n])
    end
    
    return graphs
end

function solve_problem2_temperature()
    println("\n" * "="^70)
    println("Problem 2a: Spectral Gap vs Temperature")
    println("="^70)
    
    # Use a small graph for demonstration (cycle graph with 8 vertices)
    graph = cycle_graph(8)
    temperatures = 0.1:0.1:2.0
    
    println("\nAnalyzing Cycle Graph (n=8)")
    println("Vertices: $(nv(graph)), Edges: $(ne(graph))")
    println("\nTemperature vs Spectral Gap:")
    println("-" * "-"^50)
    
    gaps = Float64[]
    for T in temperatures
        gap = compute_spectral_gap(graph, T)
        push!(gaps, gap)
        @printf("T = %.1f: Gap = %.6f\n", T, gap)
    end
    
    println("\nObservation:")
    println("  - Spectral gap increases with temperature")
    println("  - At low T (< 0.4), gap ≈ 0: system is 'frozen', slow mixing")
    println("  - At high T, gap grows: faster relaxation to equilibrium")
    println("  - Relaxation time τ ≈ 1/gap")
    
    return collect(temperatures), gaps
end

function solve_problem2_size()
    println("\n" * "="^70)
    println("Problem 2b: Spectral Gap vs System Size")
    println("="^70)
    
    T = 1.0  # Use higher temperature to see clearer trends
    sizes = [4, 6, 8, 10, 12]  # Limited by 2^n computation
    
    println("\nAnalyzing Cycle Graphs at T = $T")
    println("System Size vs Spectral Gap:")
    println("-" * "-"^50)
    
    gaps = Float64[]
    for n in sizes
        graph = cycle_graph(n)
        gap = compute_spectral_gap(graph, T)
        push!(gaps, gap)
        @printf("n = %2d: Gap = %.6f\n", n, gap)
    end
    
    println("\nObservation:")
    println("  - Spectral gap decreases as system size increases")
    println("  - Larger systems have slower relaxation (smaller gap)")
    println("  - This is expected for 1D chains at finite temperature")
    
    return sizes, gaps
end

# ============================================================================
# Problem 3 (Challenge): Parallel Tempering for Spin Glass
# ============================================================================

"""
SpinGlass type to match GenericTensorNetworks interface
"""
struct SpinGlass
    graph::SimpleGraph
    coupling::Vector{Float64}  # J_ij for each edge
    bias::Vector{Float64}      # h_i for each vertex
end

"""
Compute energy for a spin glass configuration
"""
function energy(sg::SpinGlass, spins::Vector{Int})
    E = 0.0
    # Coupling terms
    for (idx, e) in enumerate(edges(sg.graph))
        E += sg.coupling[idx] * spins[src(e)] * spins[dst(e)]
    end
    # Bias terms
    for i in 1:length(spins)
        E += sg.bias[i] * spins[i]
    end
    return E
end

"""
Parallel Tempering algorithm for spin glass ground state finding
"""
function parallel_tempering(sg::SpinGlass;
                           n_replicas=16,
                           T_min=0.1,
                           T_max=10.0,
                           n_sweeps=10000,
                           swap_interval=10,
                           seed=42)
    Random.seed!(seed)
    n = nv(sg.graph)
    
    # Temperature ladder (geometric spacing)
    temperatures = [T_min * (T_max/T_min)^(i/(n_replicas-1)) for i in 0:(n_replicas-1)]
    βs = 1.0 ./ temperatures
    
    # Initialize random configurations for each replica
    replicas = [rand([-1, 1], n) for _ in 1:n_replicas]
    energies = [energy(sg, replicas[i]) for i in 1:n_replicas]
    
    # Track best configuration
    best_energy = minimum(energies)
    best_config = copy(replicas[argmin(energies)])
    
    # Statistics
    n_swaps_attempted = 0
    n_swaps_accepted = 0
    
    for sweep in 1:n_sweeps
        # Perform Monte Carlo updates for each replica
        for r in 1:n_replicas
            β = βs[r]
            spins = replicas[r]
            E = energies[r]
            
            # Multiple single-spin flips per sweep
            for _ in 1:n
                # Random spin flip
                i = rand(1:n)
                spins[i] *= -1
                E_new = energy(sg, spins)
                
                # Metropolis criterion
                ΔE = E_new - E
                if ΔE < 0 || rand() < exp(-β * ΔE)
                    E = E_new
                    if E < best_energy
                        best_energy = E
                        best_config = copy(spins)
                    end
                else
                    # Reject: flip back
                    spins[i] *= -1
                end
            end
            
            energies[r] = E
        end
        
        # Attempt replica exchanges
        if sweep % swap_interval == 0
            for r in 1:(n_replicas-1)
                # Try to swap replicas r and r+1
                β1, β2 = βs[r], βs[r+1]
                E1, E2 = energies[r], energies[r+1]
                
                # Exchange probability
                Δβ = β2 - β1
                ΔE = E2 - E1
                accept_prob = exp(Δβ * ΔE)
                
                n_swaps_attempted += 1
                if rand() < accept_prob
                    # Swap configurations
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                    energies[r], energies[r+1] = energies[r+1], energies[r]
                    n_swaps_accepted += 1
                end
            end
        end
        
        # Progress reporting
        if sweep % 1000 == 0 || sweep == n_sweeps
            swap_rate = n_swaps_accepted / max(n_swaps_attempted, 1)
            println("  Sweep $sweep: Best E = $best_energy, Swap rate = $(round(swap_rate, digits=3))")
        end
    end
    
    println("\nFinal statistics:")
    println("  Total swaps attempted: $n_swaps_attempted")
    println("  Total swaps accepted: $n_swaps_accepted")
    println("  Overall swap rate: $(round(n_swaps_accepted/n_swaps_attempted, digits=3))")
    
    return best_config, best_energy
end

"""
Helper functions from README for constructing test cases
"""
function strong_product(g1::SimpleGraph, g2::SimpleGraph)
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

strong_power(g::SimpleGraph, k::Int) = k == 1 ? g : strong_product(g, strong_power(g, k - 1))

function spin_glass_c(n::Int, k::Int)
    g1 = cycle_graph(n)
    g = strong_power(g1, k)
    coupling = fill(1.0, ne(g))
    bias = [1.0 - degree(g, i) for i in vertices(g)]
    return SpinGlass(g, coupling, bias)
end

function solve_problem3()
    println("\n" * "="^70)
    println("Problem 3 (Challenge): Parallel Tempering for Spin Glass")
    println("="^70)
    
    # Test case 1: smaller problem for verification
    println("\n[Test 1] Spin glass on C5^2 (5-cycle to 2nd power)")
    println("-"^70)
    sg1 = spin_glass_c(5, 2)
    println("Graph: $(nv(sg1.graph)) vertices, $(ne(sg1.graph)) edges")
    println("Target energy: -85")
    
    config1, E1 = parallel_tempering(sg1, n_replicas=12, T_min=0.5, T_max=15.0,
                                     n_sweeps=5000, swap_interval=10)
    
    println("\nResult: Energy = $E1")
    println(E1 == -85 ? "✓ Test 1 PASSED!" : "✗ Test 1 FAILED (expected -85)")
    
    # Test case 2: challenge problem
    println("\n\n[Test 2] Spin glass on C7^4 (7-cycle to 4th power) - CHALLENGE")
    println("-"^70)
    sg2 = spin_glass_c(7, 4)
    println("Graph: $(nv(sg2.graph)) vertices, $(ne(sg2.graph)) edges")
    println("Target energy: < -93855")
    
    config2, E2 = parallel_tempering(sg2, n_replicas=20, T_min=0.3, T_max=20.0,
                                     n_sweeps=20000, swap_interval=10)
    
    println("\nResult: Energy = $E2")
    println(E2 < -93855 ? "✓ Test 2 PASSED! A+ achieved!" : "✗ Test 2 FAILED (need < -93855)")
    
    return (E1, E2)
end

# ============================================================================
# Main execution
# ============================================================================

function main()
    println("\n")
    println("╔" * "="^68 * "╗")
    println("║" * " "^15 * "HOMEWORK 7 - Julia Solutions" * " "^25 * "║")
    println("╚" * "="^68 * "╝")
    
    # Problem 1
    ground_energy, ground_config = solve_problem1()
    
    # Problem 2a
    temps, gaps_temp = solve_problem2_temperature()
    
    # Problem 2b
    sizes, gaps_size = solve_problem2_size()
    
    # Problem 3 (Challenge)
    E1, E2 = solve_problem3()
    
    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    println("✓ Problem 1: Ground state energy found (E = $ground_energy)")
    println("✓ Problem 2a: Spectral gap vs temperature analyzed")
    println("✓ Problem 2b: Spectral gap vs system size analyzed")
    println("✓ Problem 3: Parallel tempering implemented")
    println("    Test 1 (C5^2): E = $E1 " * (E1 == -85 ? "✓" : "✗"))
    println("    Test 2 (C7^4): E = $E2 " * (E2 < -93855 ? "✓ A+!" : "✗"))
    println("="^70)
end

# Run if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
