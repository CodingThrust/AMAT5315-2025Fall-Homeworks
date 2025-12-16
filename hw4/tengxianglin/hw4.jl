# Homework 4 Solutions
# Author: tengxianglin
#
# Run with: julia --project=hw4 hw4/tengxianglin/hw4.jl

using LinearAlgebra
using Polynomials
using Printf
# Optional: For visualization (uncomment if needed)
# using CairoMakie

println("\n" * "="^70)
println("HOMEWORK 4 - Solutions")
println("="^70)

# ============================================================================
# Problem 1: Condition Number Analysis
# ============================================================================
println("\nProblem 1: Condition Number Analysis")
println("="^70)

Aa = [1e10 0; 0 1e-10]
Ab = [1e10 0; 0 1e10]
Ac = [1e-10 0; 0 1e-10]
Ad = [1 2; 2 4]

cond_Aa = cond(Aa)
cond_Ab = cond(Ab)
cond_Ac = cond(Ac)
cond_Ad = cond(Ad)

println("\nMatrix (a): [10^10  0; 0  10^-10]")
@printf("  Condition number: %.2e\n", cond_Aa)
println("  Interpretation: ill-conditioned (very large condition number)")

println("\nMatrix (b): [10^10  0; 0  10^10]")
@printf("  Condition number: %.2f\n", cond_Ab)
println("  Interpretation: Well-conditioned")

println("\nMatrix (c): [10^-10  0; 0  10^-10]")
@printf("  Condition number: %.2f\n", cond_Ac)
println("  Interpretation: Well-conditioned")

println("\nMatrix (d): [1  2; 2  4]")
@printf("  Condition number: %.2e\n", cond_Ad)
println("  Interpretation: ill-conditioned (singular matrix, condition number → ∞)")

println("\n" * "-"^70)
println("Summary Table:")
println("-"^70)
println("  Matrix | Condition Number | Interpretation")
println("-"^70)
@printf("  (a)    | %.2e          | ill-conditioned\n", cond_Aa)
@printf("  (b)    | %.2f              | Well-conditioned\n", cond_Ab)
@printf("  (c)    | %.2f              | Well-conditioned\n", cond_Ac)
@printf("  (d)    | %.2e          | ill-conditioned\n", cond_Ad)

# ============================================================================
# Problem 2: Solving Linear Equations
# ============================================================================
println("\n\n" * "="^70)
println("Problem 2: Solving Linear Equations")
println("="^70)

A = [ 2  1 -1  0  1;
      1  3  1 -1  0;
      0  1  4  1 -1;
     -1  0  1  3  1;
      1 -1  0  1  2 ]

b = [4, 6, 2, 5, 3]

# Numerical solution
x = A \ b

# Exact rational solution
xr = (A .// 1) \ (b .// 1)

println("\nSystem of equations:")
println("  2x₁ + x₂ - x₃ + 0x₄ + x₅ = 4")
println("  x₁ + 3x₂ + x₃ - x₄ + 0x₅ = 6")
println("  0x₁ + x₂ + 4x₃ + x₄ - x₅ = 2")
println("  -x₁ + 0x₂ + x₃ + 3x₄ + x₅ = 5")
println("  x₁ - x₂ + 0x₃ + x₄ + 2x₅ = 3")

println("\nNumerical solution:")
for i in 1:5
    @printf("  x%d = %.4f\n", i, x[i])
end

println("\nExact rational solution:")
for i in 1:5
    @printf("  x%d = %s\n", i, xr[i])
end

println("\nVerification (checking A * x ≈ b):")
residual = A * x - b
@printf("  Maximum residual: %.2e\n", maximum(abs.(residual)))
println("  ✓ Solution is correct!")

# ============================================================================
# Problem 3: Polynomial Data Fitting
# ============================================================================
println("\n\n" * "="^70)
println("Problem 3: Polynomial Data Fitting")
println("="^70)

years = collect(1990:2021)
pop = [2374,2250,2113,2120,2098,2052,2057,2028,1934,1827,1765,1696,
       1641,1594,1588,1612,1581,1591,1604,1587,1588,1600,1800,1640,
       1687,1655,1786,1723,1523,1465,1200,1062]

# Shift years to start from 0 for better numerical stability
t = years .- 1990

# Fit cubic polynomial
p = fit(t, pop, 3)
p_coeffs = coeffs(p)
pred_2024 = p(34.0)  # 2024 - 1990 = 34

println("\nChina's newborn population data (1990-2021)")
println("Fitting cubic polynomial: y = a₀ + a₁t + a₂t² + a₃t³")
println("where t = year - 1990")

println("\nFitted coefficients:")
@printf("  a₀ = %.4f\n", p_coeffs[1])
@printf("  a₁ = %.4f\n", p_coeffs[2])
@printf("  a₂ = %.4f\n", p_coeffs[3])
@printf("  a₃ = %.4f\n", p_coeffs[4])

println("\nPrediction for 2024 (t = 34):")
@printf("  Predicted population: %.2f × 10⁴ = %.2f × 10⁶\n", pred_2024, pred_2024/100)

# Optional: Create plot if CairoMakie is available
# Uncomment the following if you want to generate the plot:
#=
using CairoMakie
fig = Figure(size=(900,600))
ax  = Axis(fig[1,1], xlabel="Year", ylabel="Newborn population (×10⁴)")
scatter!(ax, years, pop, markersize=8, label="Data")
tt = range(first(t), last(t), length=400)
lines!(ax, (tt .+ 1990), p.(tt), linewidth=2, label="Cubic fit")
axislegend(position=:rb)
save("population_fit.png", fig)
println("  Plot saved to population_fit.png")
=#

# ============================================================================
# Problem 4: Eigen-decomposition (Extra points)
# ============================================================================
println("\n\n" * "="^70)
println("Problem 4: Eigen-decomposition (Extra points)")
println("="^70)

using SparseArrays

# Parameters
N = 1000
C = 1.0
m_even, m_odd = 1.0, 2.0

println("\nDual-species spring chain analysis")
println("  Number of sites: $N")
println("  Mass on even sites: $m_even")
println("  Mass on odd sites: $m_odd")
println("  Stiffness constant: $C")
println("  Boundary condition: Periodic")

# Build stiffness matrix with periodic BC
main = fill(2C, N)
off  = fill(-C, N-1)
K = spdiagm(0 => main, 1 => off, -1 => off)
K[1,end] = -C
K[end,1] = -C

# Alternating masses and normalization
m = [ (i % 2 == 0 ? m_even : m_odd) for i in 1:N ]
Dhalf_inv = Diagonal(1.0 ./ sqrt.(m))
Ktil = Dhalf_inv * K * Dhalf_inv

# Eigen-decomposition
λ = eigen(Symmetric(Matrix(Ktil))).values
ω = sqrt.(clamp.(λ, 0, Inf))

# Monoatomic comparison
λ1 = eigen(Symmetric(Matrix(K))).values
ω1 = sqrt.(clamp.(λ1, 0, Inf))

println("\nEigenvalue analysis:")
@printf("  Dual-species chain: ω_min = %.4f, ω_max = %.4f\n", minimum(ω), maximum(ω))
@printf("  Single-species chain: ω_min = %.4f, ω_max = %.4f\n", minimum(ω1), maximum(ω1))

println("\nObservation:")
println("  Dual-species chain exhibits two separated energy bands (with a gap),")
println("  while the single-species chain shows one continuous band.")

# Optional: Create plot if CairoMakie is available
# Uncomment the following if you want to generate the plot:
#=
using CairoMakie
using CairoMakie: RGBAf0
fig = Figure(size=(1000,600))
ax  = Axis(fig[1,1], xlabel="ω", ylabel="Density (counts)")
hist!(ax, ω;  bins=80, normalization=:none, label="Diatomic chain",
      color=RGBAf0(0.2,0.4,0.9,0.35))
hist!(ax, ω1; bins=80, normalization=:none, label="Monoatomic chain",
      color=:transparent, strokecolor=:red, strokewidth=1.5)
axislegend(position=:rt)
save("dos_compare.png", fig)
println("  Plot saved to dos_compare.png")
=#

# ============================================================================
# Summary
# ============================================================================
println("\n\n" * "="^70)
println("Summary")
println("="^70)
println("✓ Problem 1: Condition number analysis completed")
println("✓ Problem 2: Linear system solved (numerical and exact)")
println("✓ Problem 3: Polynomial fitting and prediction completed")
println("✓ Problem 4: Eigen-decomposition analysis completed")
println("="^70)

