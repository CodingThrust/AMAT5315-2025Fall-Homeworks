# Import required packages
using Random, Graphs, Plots, Printf, Statistics

println("="^70)
println("HOMEWORK 9 - Songming Liu (Integrating Huicheng Zhang's work)")
println("="^70)

# ================================================================
# PROBLEM 1: Half Adder → Spin Glass
# ================================================================
println("\n" * "="^70)
println("PROBLEM 1: Half Adder → Spin Glass")
println("="^70)

bit_from_spin(σ::Int) = (1 - σ) ÷ 2
spin_from_bit(b::Int) = b == 0 ? 1 : -1

function H_half_adder(σA::Int, σB::Int, σS::Int, σC::Int)
    return 1 +
           0.25 * σA * σB * σC   -  # 3-body ABC
           0.5  * σA * σB * σS   -  # 3-body ABS
           0.25 * σA * σC        -  # 2-body AC
           0.25 * σB * σC        -  # 2-body BC
           0.25 * σC               # field on C
end

println("Ground states (energy 0):")
spins = (-1, +1)
valid_count = 0
for σA in spins, σB in spins, σS in spins, σC in spins
    E = H_half_adder(σA, σB, σS, σC)
    if isapprox(E, 0.0; atol=1e-9)
        A = bit_from_spin(σA)
        B = bit_from_spin(σB)
        S = bit_from_spin(σS)
        C = bit_from_spin(σC)
        println("  spins = ($σA, $σB, $σS, $σC),  bits = (A=$A, B=$B, S=$S, C=$C)")
        valid_count += 1
    end
end
println("Total ground states found: $valid_count")

function xor_penalty(A, B, S)
    target = (A == B) ? 0 : 1
    return (S - target)^2
end

function and_penalty(A, B, C)
    target = (A == 1 && B == 1) ? 1 : 0
    return (C - target)^2
end

function halfadder_energy(sA::Int, sB::Int, sS::Int, sC::Int)
    A, B, S, C = bit_from_spin(sA), bit_from_spin(sB), bit_from_spin(sS), bit_from_spin(sC)
    return xor_penalty(A, B, S) + and_penalty(A, B, C)
end

println("\nVerification using original penalty function:")
println("  All ground states should have E=0 for both functions.")
for σA in spins, σB in spins, σS in spins, σC in spins
    E1 = H_half_adder(σA, σB, σS, σC)
    E2 = halfadder_energy(σA, σB, σS, σC)
    if isapprox(E1, 0.0; atol=1e-9)
        @printf("  spins=(%2d,%2d,%2d,%2d) | E_H=%.2f, E_orig=%.2f ✓\n", σA, σB, σS, σC, E1, E2)
    end
end


# ================================================================
# PROBLEM 2: Spin Dynamics for S=0, C=1
# ================================================================
println("\n" * "="^70)
println("PROBLEM 2: Spin Dynamics for S=0, C=1")
println("="^70)

"""
    spin_dynamics_SA(; T_start=2.0, T_end=0.1, nsteps=50_000, nrestarts=20)
Metropolis simulated annealing for the half-adder spin system.
Outputs are fixed to S = 0, C = 1 (bits). We only flip A and B.
Returns `(A_bit, B_bit, best_energy)`.
"""
function spin_dynamics_SA(; T_start=2.0, T_end=0.1, nsteps=50_000, nrestarts=20)
    rng = MersenneTwister(5315)

    # Fix outputs: S=0, C=1  (in bits → spins)
    σS_fixed = spin_from_bit(0)   # +1
    σC_fixed = spin_from_bit(1)   # -1

    bestE = Inf
    best_σA = 1
    best_σB = 1

    # geometric cooling schedule
    function temperature(t)
        if nsteps == 1
            return T_end
        else
            α = (T_end / T_start)^(1 / (nsteps - 1))
            return T_start * α^(t - 1)
        end
    end

    for restart in 1:nrestarts
        # random initial spins for A,B
        σA = rand(rng, Bool) ? 1 : -1
        σB = rand(rng, Bool) ? 1 : -1

        E = H_half_adder(σA, σB, σS_fixed, σC_fixed)

        for t in 1:nsteps
            T = temperature(t)

            # propose flipping either A or B
            if rand(rng, Bool)
                σA_trial = -σA
                σB_trial = σB
            else
                σA_trial = σA
                σB_trial = -σB
            end

            E_trial = H_half_adder(σA_trial, σB_trial, σS_fixed, σC_fixed)
            ΔE = E_trial - E

            if ΔE <= 0 || rand(rng) < exp(-ΔE / T)
                σA, σB, E = σA_trial, σB_trial, E_trial
            end

            if E < bestE
                bestE = E
                best_σA = σA
                best_σB = σB
            end
        end
    end

    A_bit = bit_from_spin(best_σA)
    B_bit = bit_from_spin(best_σB)

    println("Spin dynamics with outputs fixed S=0, C=1")
    println("Best energy found: $bestE")
    println("Recovered inputs:")
    println("  A = $A_bit, B = $B_bit")

    # Check if solution is correct
    if (A_bit ⊻ B_bit) == 0 && (A_bit & B_bit) == 1
        println("  \u2713 CORRECT! (A,B) = ($A_bit, $B_bit) satisfies S=0, C=1")
    else
        println("  ❌ INCORRECT!")
    end

    return A_bit, B_bit, bestE
end

A_result, B_result, result_energy = spin_dynamics_SA(nsteps=5000, nrestarts=10) # Reduced steps for demo


# ================================================================
# PROBLEM 3: Greedy MIS (Extended)
# ================================================================
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
    error("Failed to generate 3-regular graph for n=$n")
end

ns = collect(10:10:200)
ratios = Float64[]

println("Running tests...")
for n in ns
    # Ensure n is even for 3-regular graph (3*n must be even)
    if isodd(n)
        push!(ratios, NaN) # Placeholder for odd n
        continue
    end

    results = [length(greedy_MIS(generate_3regular_graph(n))) for _ in 1:5]
    avg_greedy = mean(results)
    # Theoretical ratio based on n / 3.2 (from first code)
    theoretical = n / 3.2
    ratio = avg_greedy / theoretical
    push!(ratios, ratio)
    if n % 50 == 0
        @printf("n=%3d: avg_size=%.2f, ratio=%.3f\n", n, avg_greedy, ratio)
    end
end

valid_ns = [n for (i, n) in enumerate(ns) if !isnan(ratios[i])]
valid_ratios = [r for r in ratios if !isnan(r)]

println("\nStatistics (excluding odd n):")
println("  Avg ratio: $(round(mean(valid_ratios), digits=3))")
println("  Min: $(round(minimum(valid_ratios), digits=3))")
println("  Max: $(round(maximum(valid_ratios), digits=3))")

p = plot(valid_ns, valid_ratios, marker=:circle, xlabel="n", ylabel="Ratio (|MIS| / (n / 3.2))",
         title="Greedy MIS Ratio vs. Graph Size (n=10-200)", legend=false, linewidth=2)
hline!([1.0], linestyle=:dash, color=:red, label="Theoretical Line (y=1)")
savefig(p, "mis_ratio_final.png")
println("✓ Plot saved as mis_ratio_final.png")

println("\n" * "="^70)
println("ALL PROBLEMS COMPLETED!")
println("="^70)