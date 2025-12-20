# Map between bits and spins:
# bit 0 ↔ spin +1, bit 1 ↔ spin -1
bit_from_spin(σ::Int) = (1 - σ) ÷ 2

# Spin-glass energy for the half adder
function H_half_adder(σA::Int, σB::Int, σS::Int, σC::Int)
    return 1 +
           0.25 * σA * σB * σC   -  # 3-body ABC
           0.5  * σA * σB * σS   -  # 3-body ABS
           0.25 * σA * σC        -  # 2-body AC
           0.25 * σB * σC        -  # 2-body BC
           0.25 * σC               # field on C
end

# Enumerate all 4-spin configurations and print the ground states
spins = (-1, +1)

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
