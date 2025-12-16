############################################################
# Homework 4 – Julia 实现（最终版）
############################################################

using Pkg

# 如果包没装就自动 add
function ensure_pkg(name::String)
    pkgs = keys(Pkg.project().dependencies)
    if !(name in pkgs)
        println("Package $name not found. Installing...")
        Pkg.add(name)
    end
end

ensure_pkg("LinearAlgebra")
ensure_pkg("Statistics")
ensure_pkg("CairoMakie")
ensure_pkg("Polynomials")
ensure_pkg("Colors")

using LinearAlgebra
using Statistics
using CairoMakie
using Polynomials
using Colors          # 为了用 RGBA

############################################################
# 1. Condition Number Analysis
############################################################

println("===== Problem 1: Condition Numbers =====")

A1a = [1e10 0.0;
       0.0 1e-10]

A1b = [1e10 0.0;
       0.0 1e10]

A1c = [1e-10 0.0;
       0.0 1e-10]

A1d = [1.0 2.0;
       2.0 4.0]

conds = [
    :a => cond(A1a),
    :b => cond(A1b),
    :c => cond(A1c),
    :d => cond(A1d)
]

for (label, κ) in conds
    println("Matrix ($label): cond₂ = $κ")
end

function classify_condition(κ; tol = 1e8)
    if !isfinite(κ)
        return "singular / 极度 ill-conditioned"
    elseif κ < tol
        return "well-conditioned"
    else
        return "ill-conditioned"
    end
end

println("\nClassification:")
for (label, κ) in conds
    println("Matrix ($label): ", classify_condition(κ))
end

############################################################
# 2. Solve Linear System (5×5)
############################################################

println("\n===== Problem 2: Solve Linear System =====")

A2 = [
    2  1  -1  0   1;
    1  3   1 -1   0;
    0  1   4  1  -1;
   -1  0   1  3   1;
    1 -1   0  1   2
] .|> float

b2 = [4, 6, 2, 5, 3] .|> float

x2 = A2 \ b2

println("Solution x = ")
println(x2)

A2r = Rational.(A2)
b2r = Rational.(b2)
xr = A2r \ b2r
println("Exact rational solution = ")
println(xr)

############################################################
# 3. Polynomial Data Fitting (Newborn population in China)
############################################################

println("\n===== Problem 3: Polynomial Regression (degree 3) =====")

years = collect(1990:2021)
population = [
    2374, 2250, 2113, 2120, 2098, 2052, 2057, 2028,
    1934, 1827, 1765, 1696, 1641, 1594, 1588, 1612,
    1581, 1591, 1604, 1587, 1588, 1600, 1800, 1640,
    1687, 1655, 1786, 1723, 1523, 1465, 1200, 1062
] .|> float

@assert length(years) == length(population)

x = years .- 1990         # 平移后的自变量
deg = 3
poly_fit = fit(x, population, deg)

println("Fitted polynomial in t = year - 1990:")
println(poly_fit)

t_2024 = 2024 - 1990
pop_2024_pred = poly_fit(t_2024)
println("Predicted newborn population in 2024 (×10⁴) = $pop_2024_pred")

fig3 = Figure(size = (800, 600))
ax3 = Axis(fig3[1, 1],
           xlabel = "Year",
           ylabel = "Newborn population (×10⁴)",
           title = "China Newborn Population: Data and Cubic Fit")

scatter!(ax3, years, population;
         color = :blue, marker = :circle, markersize = 10,
         label = "Data")

years_fine = range(first(years), last(years) + 5; length = 400)
x_fine = years_fine .- 1990
pop_fine = poly_fit.(x_fine)

lines!(ax3, years_fine, pop_fine;
       color = :red, linewidth = 3,
       label = "Cubic fit")

axislegend(ax3; position = :rt)

save("china_newborn_cubic_fit.png", fig3)
println("Saved plot: china_newborn_cubic_fit.png")

############################################################
# 4. Extra: Eigen-decomposition and Density of States
############################################################

println("\n===== Problem 4 (Extra): DOS of Spring Chains =====")

# 刚度矩阵 K：1D 链 + 周期边界
function spring_chain_K(N::Int; C::Float64 = 1.0)
    K = zeros(Float64, N, N)
    for i in 1:N
        ip = (i == N) ? 1 : i + 1
        im = (i == 1) ? N : i - 1
        K[i, i] += 2C
        K[i, ip] -= C
        K[i, im] -= C
    end
    return K
end

# 双组分质量矩阵：奇数点=2，偶数点=1
function spring_chain_M_dual(N::Int)
    m = [isodd(i) ? 2.0 : 1.0 for i in 1:N]
    return Diagonal(m)
end

# 单组分质量矩阵：全 1
function spring_chain_M_single(N::Int)
    m = ones(Float64, N)
    return Diagonal(m)
end

# 计算广义特征频率
function eigen_frequencies(K::AbstractMatrix, M::Diagonal)
    vals, _ = eigen(Symmetric(K), Symmetric(Matrix(M)))
    vals_clipped = map(x -> max(x, 0.0), vals)
    ω = sqrt.(vals_clipped)
    ω_filtered = filter(x -> x > 1e-8, ω)  # 去掉零模
    return ω_filtered
end

N = 400

K = spring_chain_K(N)
M_dual   = spring_chain_M_dual(N)
M_single = spring_chain_M_single(N)

ω_dual   = eigen_frequencies(K, M_dual)
ω_single = eigen_frequencies(K, M_single)

println("Dual-species: number of modes = $(length(ω_dual))")
println("Single-species: number of modes = $(length(ω_single))")

ω_max = max(maximum(ω_dual), maximum(ω_single))
bins = range(0, ω_max; length = 60)

fig4 = Figure(size = (800, 700))

ax4a = Axis(fig4[1, 1],
            xlabel = "Energy (ω)",
            ylabel = "Population (count)",
            title = "DOS – Dual-species chain (m_even = 1, m_odd = 2)")

hist!(ax4a, ω_dual;
      bins = collect(bins),
      normalization = :none,
      color = RGBA(0.2, 0.4, 0.8, 0.8),
      strokecolor = :black,
      strokewidth = 0.5)

ax4b = Axis(fig4[2, 1],
            xlabel = "Energy (ω)",
            ylabel = "Population (count)",
            title = "DOS – Single-species chain (m = 1)")

hist!(ax4b, ω_single;
      bins = collect(bins),
      normalization = :none,
      color = RGBA(0.8, 0.4, 0.2, 0.8),
      strokecolor = :black,
      strokewidth = 0.5)

# 这里用老接口：不带关键字参数
tightlimits!(ax4a)
tightlimits!(ax4b)

save("spring_chain_DOS_compare.png", fig4)
println("Saved plot: spring_chain_DOS_compare.png")

println("\nAll parts finished.")