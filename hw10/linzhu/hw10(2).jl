###############################
# half_adder_spin_dynamics.jl #
###############################

using Random

###############################
# 1. Bit–spin mapping helpers
###############################
# We use spins σ ∈ {+1, -1}:
#   bit 0 ↔ σ = +1
#   bit 1 ↔ σ = -1

bit_from_spin(σ::Int8) = (1 - σ) ÷ 2          # σ → {0,1}
spin_from_bit(b::Int)  = b == 0 ? Int8(1) : Int8(-1)  # {0,1} → σ

###############################
# 2. Half-adder Hamiltonian
###############################
# Spins σA, σB, σS, σC ∈ {+1, -1}.
#
# Hamiltonian (derived from the XOR and AND constraints):
#
# H = 1
#     + (1/4) σA σB σC
#     - (1/2) σA σB σS
#     - (1/4) σA σC
#     - (1/4) σB σC
#     - (1/4) σC
#
# Ground states (H=0) ↔ valid half-adder truth-table rows.

function H_half_adder(σA::Int8, σB::Int8, σS::Int8, σC::Int8)::Float64
    return 1.0 +
           0.25 * σA * σB * σC   -  # 3-body ABC
           0.50 * σA * σB * σS   -  # 3-body ABS
           0.25 * σA * σC        -  # 2-body AC
           0.25 * σB * σC        -  # 2-body BC
           0.25 * σC               # field on C
end

###############################
# 3. Enumerate all ground states
###############################

function list_ground_states()
    spins = (Int8(-1), Int8(1))  # {-1, +1}
    println("Ground states (energy 0):")
    for σA in spins, σB in spins, σS in spins, σC in spins
        E = H_half_adder(σA, σB, σS, σC)
        if isapprox(E, 0.0; atol=1e-9)
            A = bit_from_spin(σA)
            B = bit_from_spin(σB)
            S = bit_from_spin(σS)
            C = bit_from_spin(σC)
            println("spins = ($σA, $σB, $σS, $σC),  bits = (A=$A, B=$B, S=$S, C=$C)")
        end
    end
end

###############################
# 4. Spin dynamics with S=0, C=1 fixed
###############################

"""
    spin_dynamics_half_adder(; T_start=2.0, T_end=0.1,
                             nsteps=50_000, nrestarts=20)

Metropolis simulated annealing for the half-adder spin system.
Outputs are fixed to S = 0, C = 1 (bits). We only flip A and B.

Returns `(A_bit, B_bit, best_energy)`.
"""
function spin_dynamics_half_adder(; T_start=2.0, T_end=0.1,
                                  nsteps=50_000, nrestarts=20)
    rng = MersenneTwister(5315)

    # Fix outputs: S=0, C=1  (in bits → spins)
    σS_fixed = spin_from_bit(0)   # +1
    σC_fixed = spin_from_bit(1)   # -1

    bestE = Inf
    best_σA = Int8(1)
    best_σB = Int8(1)

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
        σA = rand(rng, Bool) ? Int8(1) : Int8(-1)
        σB = rand(rng, Bool) ? Int8(1) : Int8(-1)

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

    println("\nSpin dynamics with outputs fixed S=0, C=1")
    println("Best energy found: $bestE")
    println("Recovered inputs:")
    println("  A = $A_bit, B = $B_bit")

    return A_bit, B_bit, bestE
end

###############################
# 5. Main: run everything
###############################

function main()
    println("=== Half-adder spin-glass ground states ===")
    list_ground_states()

    println("\n=== Spin dynamics (Metropolis) ===")
    spin_dynamics_half_adder()
end

main()
