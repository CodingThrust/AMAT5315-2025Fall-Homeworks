# HW7

## task 1
```
using Graphs, ProblemReductions, Random, LinearAlgebra

# 1. Construct the Fullerene (C60) graph 
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

# Build the graph and extract edges (60 nodes, 90 edges for C60)
fullerene_points = fullerene()
fullerene_graph = UnitDiskGraph(fullerene_points, sqrt(5))
edges = collect(edges(fullerene_graph))  # Convert edges to a list for energy calculation
n_nodes = nv(fullerene_graph)  # Should be 60
@assert n_nodes == 60 "Fullerene graph must have 60 nodes"
@assert ne(fullerene_graph) == 90 "Fullerene graph must have 90 edges"

# 2. Ising Model Energy Calculation
"""
Calculate the energy of a spin configuration for the Ising Hamiltonian H = Σ(ij∈E) σ_iσ_j
- spin_config: Vector of ±1, length = number of nodes
- edges: List of edges (each edge is a tuple of two node indices)
"""
function ising_energy(spin_config, edges)
    energy = 0.0
    for e in edges
        i, j = src(e), dst(e)
        energy += spin_config[i] * spin_config[j]
    end
    return energy
end

# 3. Simulated Annealing Implementation

function simulated_annealing(edges, n_nodes; T0=10.0, T_min=1e-6, cooling_rate=0.95, max_steps_per_T=1000, seed=42)
    Random.seed!(seed)
    # Initialize spin configuration (random ±1)
    spin_config = rand([-1, 1], n_nodes)
    current_energy = ising_energy(spin_config, edges)
    ground_energy = current_energy
    best_config = copy(spin_config)
    
    T = T0
    while T > T_min
        for _ in 1:max_steps_per_T
            # Generate new state: flip a random spin
            flip_idx = rand(1:n_nodes)
            new_config = copy(spin_config)
            new_config[flip_idx] *= -1  # Flip spin
            
            # Calculate energy difference
            new_energy = ising_energy(new_config, edges)
            ΔE = new_energy - current_energy
            
            # Accept new state (Metropolis criterion)
            if ΔE < 0 || rand() < exp(-ΔE / T)
                spin_config = new_config
                current_energy = new_energy
                
                # Update ground state if current state is better
                if current_energy < ground_energy
                    ground_energy = current_energy
                    best_config = copy(spin_config)
                end
            end
        end
        # Cool down temperature
        T *= cooling_rate
    end
    return ground_energy, best_config
end

# 4. Run Simulated Annealing and Output Result
println("Starting Simulated Annealing for Fullerene Ising Model...")
ground_energy, best_config = simulated_annealing(edges, n_nodes; 
                                                 T0=20.0, 
                                                 T_min=1e-8, 
                                                 cooling_rate=0.98, 
                                                 max_steps_per_T=2000, 
                                                 seed=42)

println("="^50)
println("Ground State Energy of Anti-Ferromagnetic Ising Model on Fullerene Graph:")
println("Energy = ", ground_energy)
println("="^50)
```

## task 2
```
using Graphs, SparseArrays, KrylovKit, LinearAlgebra, Random, CairoMakie

# ------------------------------
# 1. Generate graph topologies (includes common types: linear chain, cycle, star, complete graph)
# ------------------------------
function generate_topologies(max_N::Int)
    topologies = Dict(
        :linear => [path_graph(n) for n in 4:2:max_N],  # Linear chain (even nodes to avoid small systems)
        :cycle => [cycle_graph(n) for n in 4:2:max_N],   # Cycle graph
        :star => [star_graph(n) for n in 4:2:max_N],     # Star graph (1 central node connected to all others)
        :complete => [complete_graph(n) for n in 4:2:8]  # Complete graph (small N to avoid state space explosion)
    )
    return topologies
end

# ------------------------------
# 2. Convert between spin states and integers (N spins ↔ integer in 0~2^N-1)
# ------------------------------
spin_to_int(σ::Vector{Int}) = sum((σ .== -1) .* (2 .^ (0:length(σ)-1)))  # σ=-1 → 1 (binary bit)

function int_to_spin(s::Int, N::Int)
    bits = digits(s, base=2, pad=N)  # Binary digits (least significant bit first)
    return [b == 0 ? 1 : -1 for b in bits]  # 0→1, 1→-1 for spin values
end

# ------------------------------
# 3. Calculate energy H(σ) for a given spin configuration
# ------------------------------
function ising_energy(σ::Vector{Int}, g::Graph)
    E = 0.0
    for e in edges(g)
        i, j = src(e), dst(e)
        E += σ[i] * σ[j]  # Sum over adjacent spin products (Hamiltonian)
    end
    return E
end

# ------------------------------
# 4. Build Glauber dynamics transfer matrix (sparse matrix for efficiency)
# ------------------------------
function build_transfer_matrix(g::Graph, β::Float64)
    N = nv(g)
    n_states = 2^N  # Total number of spin states (2^N for N spins)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    
    for s in 0:n_states-1  # Iterate over all possible states
        σ = int_to_spin(s, N)
        current_E = ising_energy(σ, g)
        total_flip_prob = 0.0  # Accumulate probabilities of flipping any spin
        
        # Calculate transition probability for flipping each spin
        for i in 1:N
            σ_new = copy(σ)
            σ_new[i] *= -1  # Flip the i-th spin
            s_new = spin_to_int(σ_new)
            new_E = ising_energy(σ_new, g)
            ΔE = new_E - current_E  # Energy change from flip
            
            # Glauber flip probability
            prob = (1.0 / (2N)) * exp(-β * ΔE / 2) / cosh(β * ΔE / 2)
            push!(rows, s+1)  # Matrix indices start at 1 in Julia
            push!(cols, s_new+1)
            push!(vals, prob)
            total_flip_prob += prob
        end
        
        # Probability of staying in the current state
        stay_prob = 1.0 - total_flip_prob
        push!(rows, s+1)
        push!(cols, s+1)
        push!(vals, stay_prob)
    end
    
    return sparse(rows, cols, vals, n_states, n_states)  # Sparse matrix to save memory
end

# ------------------------------
# 5. Calculate spectral gap (1 - second-largest eigenvalue)
# ------------------------------
function spectral_gap(T::Float64, g::Graph)
    β = 1.0 / T  # Inverse temperature
    T_mat = build_transfer_matrix(g, β)
    # Find 2 largest-magnitude eigenvalues (:LM = largest magnitude)
    λs, _, _ = eigsolve(T_mat, 2, :LM; tol=1e-6, maxiter=1000)
    λ1, λ2 = real.(λs)  # Eigenvalues of transfer matrix should be real
    return 1.0 - λ2  # Spectral gap = 1 - second-largest eigenvalue
end

# ------------------------------
# 6. Task 1: Spectral gap vs temperature T ∈ [0.1, 2.0]
# ------------------------------
function task1(topologies; max_N=10)
    Ts = range(0.1, 2.0, length=10)  # Temperature range
    fig = Figure(resolution=(1000, 800))
    ax = Axis(fig[1, 1], xlabel="Temperature T", ylabel="Spectral Gap", title="Gap vs Temperature")
    
    for (name, graphs) in topologies
        N = nv(graphs[1])  # Use smallest system for each topology (faster computation)
        gaps = [spectral_gap(T, graphs[1]) for T in Ts]
        lines!(ax, Ts, gaps, marker=:circle, label="$(name), N=$N")
    end
    
    axislegend(ax)
    save("gap_vs_temperature.png", fig)
    println("Task 1: Gap-temperature plot saved as gap_vs_temperature.png")
end

# ------------------------------
# 7. Task 2: Spectral gap vs system size N (at T=0.1)
# ------------------------------
function task2(topologies; T=0.1)
    fig = Figure(resolution=(1000, 800))
    ax = Axis(fig[1, 1], xlabel="System Size N", ylabel="Spectral Gap", title="Gap vs System Size (T=$T)")
    
    for (name, graphs) in topologies
        Ns = [nv(g) for g in graphs]  # System sizes for current topology
        gaps = [spectral_gap(T, g) for g in graphs]
        lines!(ax, Ns, gaps, marker=:square, label="$name")
    end
    
    axislegend(ax)
    save("gap_vs_size.png", fig)
    println("Task 2: Gap-system size plot saved as gap_vs_size.png")
end

# ------------------------------
# Run the program
# ------------------------------
max_N = 10  # Test with small N first (increase to 18 for full results)
topologies = generate_topologies(max_N)

task1(topologies)
task2(topologies)
```
