using Graphs, Random, Statistics, ProblemReductions

function fullerene()
    th = (1+sqrt(5))/2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x,y,z), (y,z,x), (z,x,y))
            for loc in ((a,b,c), (a,b,-c), (a,-b,c), (a,-b,-c), (-a,b,c), (-a,b,-c), (-a,-b,c), (-a,-b,-c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

function ising_energy(spins, graph)
    energy = 0.0
    for edge in edges(graph)
        i, j = src(edge), dst(edge)
        energy += spins[i] * spins[j]
    end
    return energy
end

function simulated_annealing(graph; num_sweeps=10000, initial_temp=10.0, final_temp=0.01)
    n = nv(graph)
    spins = rand([-1, 1], n)
    current_energy = ising_energy(spins, graph)
    best_spins = copy(spins)
    best_energy = current_energy
    
    temp = initial_temp
    cooling_rate = (final_temp / initial_temp)^(1/num_sweeps)
    
    for sweep in 1:num_sweeps
        # Propose a spin flip
        site = rand(1:n)
        spins[site] *= -1
        new_energy = ising_energy(spins, graph)
        
        # Metropolis criterion
        ΔE = new_energy - current_energy
        if ΔE < 0 || rand() < exp(-ΔE/temp)
            current_energy = new_energy
            if current_energy < best_energy
                best_energy = current_energy
                best_spins = copy(spins)
            end
        else
            # Reject the move
            spins[site] *= -1
        end
        
        temp *= cooling_rate
    end
    
    return best_energy, best_spins
end

# Main execution
fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5))
println("Fullerene graph has $(nv(fullerene_graph)) vertices and $(ne(fullerene_graph)) edges")

# Run simulated annealing multiple times to ensure convergence
energies = Float64[]
for run in 1:10
    energy, spins = simulated_annealing(fullerene_graph)
    push!(energies, energy)
    println("Run $run: Energy = $energy")
end

ground_state_energy = minimum(energies)
println("\nGround state energy: $ground_state_energy")


#Ground state energy: -66.0