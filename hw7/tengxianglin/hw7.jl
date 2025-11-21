# Homework 7 Solutions
# Author: tengxianglin

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
        energy, config = simulated_annealing(fullerene_graph, seed=trial)
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
Construct the Ising Hamiltonian matrix for anti-ferromagnetic model
"""
function ising_hamiltonian(graph::SimpleGraph, T::Float64=1.0)
    n = nv(graph)
    N = 2^n
    β = 1.0 / T
    
    # Build the Hamiltonian matrix (sparse)
    H = spzeros(Float64, N, N)
    
    # Iterate over all configurations
    for state in 0:(N-1)
        # Convert state to spin configuration
        spins = [2 * ((state >> i) & 1) - 1 for i in 0:(n-1)]
        
        # Diagonal element: energy of this configuration
        energy = 0.0
        for e in edges(graph)
            energy += spins[src(e)] * spins[dst(e)]
        end
        H[state+1, state+1] = energy
    end
    
    return H
end

"""
Compute the spectral gap (difference between two smallest eigenvalues)
"""
function compute_spectral_gap(graph::SimpleGraph, T::Float64=1.0)
    H = ising_hamiltonian(graph, T)
    
    # For small systems, compute all eigenvalues
    if nv(graph) <= 10
        eigenvalues = eigvals(Matrix(H))
        sort!(eigenvalues)
        return eigenvalues[2] - eigenvalues[1]
    else
        # For larger systems, use sparse eigenvalue solver
        # This would require Arpack or KrylovKit
        eigenvalues = eigvals(Matrix(H))
        sort!(eigenvalues)
        return eigenvalues[2] - eigenvalues[1]
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
    
    return collect(temperatures), gaps
end

function solve_problem2_size()
    println("\n" * "="^70)
    println("Problem 2b: Spectral Gap vs System Size")
    println("="^70)
    
    T = 0.1
    sizes = [4, 6, 8, 10, 12]  # Limited by 2^n computation
    
    println("\nAnalyzing Cycle Graphs at T = $T")
    println("System Size vs Spectral Gap:")
    println("-" * "-"^50)
    
    gaps = Float64[]
    for n in sizes
        graph = cycle_graph(n)
        gap = compute_spectral_gap(graph, T)
        push!(gaps, gap)
        println("n = $n: Gap = $gap")
    end
    
    return sizes, gaps
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
    
    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    println("✓ Problem 1: Ground state energy found")
    println("✓ Problem 2a: Spectral gap vs temperature analyzed")
    println("✓ Problem 2b: Spectral gap vs system size analyzed")
    println("\nNote: Problem 3 (Challenge) requires implementing parallel tempering.")
    println("This is an advanced algorithm for A+ credit.")
    println("="^70)
end

# Run if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
