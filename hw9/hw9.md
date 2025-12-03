# HW9

## Task1
To reduce the half-adder circuit SAT problem to a spin glass ground state problem:

1. Map Variables to Spins**: Let \( A, B, S, C \) correspond to spins \( \sigma_A, \sigma_B, \sigma_S, \sigma_C \in \{\pm 1\} \) (1 = logic 1, -1 = logic 0).

2. Define Hamiltonian for Gate Constraints**:
   \[
   H = (\sigma_S + \sigma_A \sigma_B)^2 + (\sigma_C - \sigma_A \sigma_B)^2
   \]
   - \( (\sigma_S + \sigma_A \sigma_B)^2 = 0 \) iff \( \sigma_S = -\sigma_A \sigma_B \) (satisfies XOR gate \( S = A \oplus B \)).
   - \( (\sigma_C - \sigma_A \sigma_B)^2 = 0 \) iff \( \sigma_C = \sigma_A \sigma_B \) (satisfies AND gate \( C = A \land B \)).

3. Relate to Ground State**: The half-adder is satisfiable if and only if the spin glass has a ground state energy of 0.

## Task2 
```
using Random, LinearAlgebra

# Hamiltonian with fixed S=0 (σ_S=-1) and C=1 (σ_C=1)
function hamiltonian(σ_A, σ_B; σ_S=-1, σ_C=1, λ=100.0)
    term1 = (σ_S + σ_A * σ_B)^2          # XOR constraint
    term2 = (σ_C - σ_A * σ_B)^2          # AND constraint
    term3 = λ * ((σ_S + 1)^2 + (σ_C - 1)^2)  # Fix S and C
    return term1 + term2 + term3
end

# Langevin spin dynamics to find ground state
function spin_dynamics(; β=10.0, dt=0.01, steps=10000)
    Random.seed!(42)
    σ_A = rand([-1, 1])  # Initialize input spins
    σ_B = rand([-1, 1])
    σ_S, σ_C = -1, 1     # Fixed outputs
    
    for _ in 1:steps
        # Current energy
        E = hamiltonian(σ_A, σ_B, σ_S=σ_S, σ_C=σ_C)
        
        # Propose flip for σ_A
        σ_A_new = -σ_A
        E_new = hamiltonian(σ_A_new, σ_B, σ_S=σ_S, σ_C=σ_C)
        ΔE = E_new - E
        if ΔE < 0 || rand() < exp(-β * ΔE)
            σ_A = σ_A_new
        end
        
        # Propose flip for σ_B
        σ_B_new = -σ_B
        E_new = hamiltonian(σ_A, σ_B_new, σ_S=σ_S, σ_C=σ_C)
        ΔE = E_new - E
        if ΔE < 0 || rand() < exp(-β * ΔE)
            σ_B = σ_B_new
        end
    end
    return σ_A, σ_B
end

# Run simulation and output input configuration
σ_A, σ_B = spin_dynamics()
println("Input spin configuration: σ_A = $σ_A, σ_B = $σ_B")
println("Logical input: A = $(σ_A == 1 ? 1 : 0), B = $(σ_B == 1 ? 1 : 0)")
```

## Task3
```
using Graphs, Random, Statistics, Plots

# Greedy algorithm for maximum independent set: select minimum-degree vertex iteratively
function greedy_mis(g::Graph)
    g_copy = copy(g)
    mis_size = 0
    while !isempty(g_copy)
        # Select vertex with smallest degree
        degrees = degree(g_copy)
        min_deg = minimum(degrees)
        v = findfirst(==(min_deg), degrees)  # Pick first min-degree vertex
        
        # Add to MIS and remove v + its neighbors
        mis_size += 1
        neighbors_v = neighbors(g_copy, v)
        rem_vertices!(g_copy, vcat(v, neighbors_v))  # Remove from graph
    end
    return mis_size
end

# Exact MIS calculation (for small graphs, using recursion with pruning)
function exact_mis(g::Graph)
    isempty(g) && return 0
    v = first(vertices(g))  # Pick first vertex
    
    # Option 1: Include v, remove v and its neighbors
    g1 = copy(g)
    rem_vertices!(g1, vcat(v, neighbors(g1, v)))
    res1 = 1 + exact_mis(g1)
    
    # Option 2: Exclude v
    g2 = copy(g)
    rem_vertices!(g2, v)
    res2 = exact_mis(g2)
    
    return max(res1, res2)
end

# Main experiment: test on random 3-regular graphs
function main()
    Random.seed!(42)
    ns = 10:10:200  # Graph sizes (even numbers, required for 3-regular graphs)
    num_samples = 5  # Number of samples per size to reduce randomness
    
    # Store results: [greedy sizes, exact sizes] for each n
    results = [(Float64[], Float64[]) for _ in ns]
    
    for (i, n) in enumerate(ns)
        for _ in 1:num_samples
            # Generate random 3-regular graph
            g = random_regular_graph(n, 3)
            
            # Greedy result
            greedy = greedy_mis(g)
            push!(results[i][1], greedy)
            
            # Exact result (only for small n; too slow for large n)
            if n ≤ 30
                exact = exact_mis(g)
                push!(results[i][2], exact)
            else
                push!(results[i][2], NaN)  # Mark large n as uncomputed
            end
        end
    end
    
    # Calculate average approximation ratios
    avg_ratios = [mean(g ./ e) for (g, e) in results]
    
    # Plot scaling of approximation ratio
    plot(ns, avg_ratios, 
         marker=:circle, 
         xlabel="Number of vertices (n)", 
         ylabel="Approximation Ratio", 
         title="Greedy MIS Approximation Ratio on 3-Regular Graphs", 
         label="Greedy Algorithm",
         ylim=(0, 1))
    hline!([mean(skipmissing(avg_ratios))], 
           linestyle=:dash, 
           color=:red, 
           label="Average Ratio")
    savefig("mis_approximation_scaling.png")
    
    println("Average approximation ratio across all sizes: ", mean(skipmissing(avg_ratios)))
end

# Run experiment
main()
```