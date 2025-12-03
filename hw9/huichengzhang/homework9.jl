# Homework 9 - Huicheng Zhang
using Random, Graphs, Plots, Printf, Statistics

println("="^70)
println("HOMEWORK 9 - Huicheng Zhang")
println("="^70)

# Problem 1: Half Adder → Spin Glass
println("\n" * "="^70)
println("PROBLEM 1: Half Adder → Spin Glass")
println("="^70)

bool_to_spin(x::Int) = 2*x - 1
spin_to_bool(s::Int) = (s + 1) ÷ 2

function xor_penalty(A, B, S)
    target = (A == B) ? 0 : 1
    return (S - target)^2
end

function and_penalty(A, B, C)
    target = (A == 1 && B == 1) ? 1 : 0
    return (C - target)^2
end

function halfadder_energy(sA::Int, sB::Int, sS::Int, sC::Int)
    A, B, S, C = spin_to_bool(sA), spin_to_bool(sB), spin_to_bool(sS), spin_to_bool(sC)
    return xor_penalty(A, B, S) + and_penalty(A, B, C)
end

println("Verification:")
global valid_count = 0
for a in [0,1], b in [0,1], s in [0,1], c in [0,1]
    sA, sB, sS, sC = bool_to_spin(a), bool_to_spin(b), bool_to_spin(s), bool_to_spin(c)
    E = halfadder_energy(sA, sB, sS, sC)
    valid = (s == (a ⊻ b)) && (c == (a & b))
    if valid
        global valid_count += 1
        @printf("  %d %d %d %d | E=%.0f ✓\n", a, b, s, c, E)
    end
end
println("Ground states: $valid_count")

# Problem 2: Spin Dynamics
println("\n" * "="^70)
println("PROBLEM 2: Spin Dynamics for S=0, C=1")
println("="^70)

function spin_dynamics_SA(fixed_S::Int, fixed_C::Int; n_runs=10, steps=5000)
    best_spins = nothing
    best_energy = Inf
    
    for run in 1:n_runs
        spins = Dict(:A => rand([-1, 1]), :B => rand([-1, 1]), :S => fixed_S, :C => fixed_C)
        current_E = halfadder_energy(spins[:A], spins[:B], spins[:S], spins[:C])
        
        for step in 1:steps
            T = 2.0 * (0.001 / 2.0)^(step / steps)
            flip_var = rand([:A, :B])
            old_val = spins[flip_var]
            spins[flip_var] = -old_val
            new_E = halfadder_energy(spins[:A], spins[:B], spins[:S], spins[:C])
            
            if new_E < current_E || rand() < exp(-(new_E - current_E) / T)
                current_E = new_E
            else
                spins[flip_var] = old_val
            end
        end
        
        if current_E < best_energy
            best_energy = current_E
            best_spins = copy(spins)
        end
        if current_E == 0; break; end
    end
    return best_spins, best_energy
end

result_spins, result_energy = spin_dynamics_SA(bool_to_spin(0), bool_to_spin(1))
A_result = spin_to_bool(result_spins[:A])
B_result = spin_to_bool(result_spins[:B])

println("Found (A,B) = ($A_result, $B_result), Energy = $result_energy")
if (A_result ⊻ B_result) == 0 && (A_result & B_result) == 1
    println("\u2713 CORRECT!")
end

# Problem 3: Greedy MIS (Extended)
println("\n" * "="^70)
println("PROBLEM 3: Greedy MIS (n=10-200)")
println("="^70)

function greedy_MIS(g::SimpleGraph)
    remaining = Set(vertices(g))
    indep_set = Set{Int}()
    while !isempty(remaining)
        v = rand(collect(remaining))
        push!(indep_set, v)
        delete!(remaining, v)
        for neighbor in neighbors(g, v)
            delete!(remaining, neighbor)
        end
    end
    return indep_set
end

function generate_3regular_graph(n::Int)
    for attempt in 1:100
        try
            return random_regular_graph(n, 3)
        catch; continue; end
    end
    error("Failed")
end

ns = collect(10:10:200)
ratios = Float64[]

println("Running tests...")
for n in ns
    results = [length(greedy_MIS(generate_3regular_graph(n))) for _ in 1:5]
    avg_greedy = mean(results)
    ratio = avg_greedy / (n / 3.2)
    push!(ratios, ratio)
    if n % 50 == 0
        @printf("n=%3d: ratio=%.3f\n", n, ratio)
    end
end

println("\nStatistics:")
println("  Avg ratio: $(round(mean(ratios), digits=3))")
println("  Min: $(round(minimum(ratios), digits=3))")
println("  Max: $(round(maximum(ratios), digits=3))")

p = plot(ns, ratios, marker=:circle, xlabel="n", ylabel="Ratio",
         title="Greedy MIS Ratio (n=10-200)", legend=false, linewidth=2)
hline!([1.0], linestyle=:dash, color=:red)
savefig(p, "mis_ratio_final.png")
println("✓ Plot saved")

println("\n" * "="^70)
println("ALL PROBLEMS COMPLETED!")
println("="^70)
