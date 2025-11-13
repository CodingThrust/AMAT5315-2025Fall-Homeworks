# Homework 4 解答 (huichengzhang)

using LinearAlgebra

println("=== Homework 4 解答 ===")

# ===== 任务1：条件数分析 (Condition Number Analysis) =====
println("\n1. 条件数分析")
println("="^50)

# 定义4个矩阵
A_a = [1e10 0; 0 1e-10]
A_b = [1e10 0; 0 1e10]
A_c = [1e-10 0; 0 1e-10]
A_d = [1 2; 2 4]

matrices = [A_a, A_b, A_c, A_d]
labels = ["(a)", "(b)", "(c)", "(d)"]

println("计算各矩阵的条件数：")
for (i, (A, label)) in enumerate(zip(matrices, labels))
    cond_num = cond(A)
    classification = cond_num > 1e12 ? "病态 (ill-conditioned)" : "良态 (well-conditioned)"
    println("$label 矩阵:")
    println("   $A")
    println("   条件数: $(cond_num)")
    println("   分类: $classification")
    println()
end

# ===== 任务2：求解线性方程组 (Solving Linear Equations) =====
println("\n2. 求解线性方程组")
println("="^50)

A = [2  1 -1  0  1;
     1  3  1 -1  0;
     0  1  4  1 -1;
    -1  0  1  3  1;
     1 -1  0  1  2]
b = [4, 6, 2, 5, 3]

println("系数矩阵 A:")
display(A)
println("\n右端向量 b: $b")

x = A \ b
println("\n解向量 x = $x")

residual = A * x - b
println("验证: A*x - b = $residual")
println("残差的二范数: $(norm(residual))")
if norm(residual) < 1e-12
    println("✓ 解验证正确！")
else
    println("✗ 解可能不准确")
end

# ===== 任务3：多项式数据拟合 (Polynomial Data Fitting) =====
println("\n3. 多项式数据拟合：中国新生儿人口数据")
println("="^50)

years = collect(1990:2021)
pop = [
    2374, 2250, 2113, 2120, 2098, 2052, 2057, 2028, 1934, 1827,
    1765, 1696, 1641, 1594, 1588, 1612, 1581, 1591, 1604, 1587,
    1588, 1600, 1800, 1640, 1687, 1655, 1786, 1723, 1523, 1465,
    1200, 1062
]

x = Float64.(years .- 1990)
y = Float64.(pop)
X = [ones(length(x)) x x.^2 x.^3]
coeffs = X \ y
a0, a1, a2, a3 = coeffs

println("拟合到三次多项式 y = a0 + a1 x + a2 x^2 + a3 x^3")
println("系数：")
println("  a0 = $(a0)")
println("  a1 = $(a1)")
println("  a2 = $(a2)")
println("  a3 = $(a3)")

x_pred = 2024 - 1990
y_pred = a0 + a1*x_pred + a2*x_pred^2 + a3*x_pred^3
println("2024年预测新生儿人口（万）：$(y_pred)")

y_fit = X * coeffs
residuals = y_fit - y
println("残差二范数：$(norm(residuals))")

ENV["GKSwstype"] = "100"
using Plots
x_plot = range(0, 34, length=400)
y_plot = a0 .+ a1 .* x_plot .+ a2 .* x_plot.^2 .+ a3 .* x_plot.^3
plt = plot(x, y; seriestype=:scatter, markersize=5, label="Data", xlabel="Year (since 1990)", ylabel="Population (1e4)")
plot!(plt, x_plot, y_plot; linewidth=2, color=:red, label="Cubic fit")
savefig(plt, "population_fit.png")
println("已保存图像：population_fit.png")

# ===== 任务4（加分）：双物种弹簧链特征分解 =====
println("\n4. 额外分：双物种弹簧链特征分解与DOS对比")
println("="^50)

function build_ring_stiffness(n::Int; C::Float64=1.0)
    K = zeros(Float64, n, n)
    for i in 1:n
        ip = (i == n) ? 1 : i + 1
        im = (i == 1) ? n : i - 1
        K[i, i] += 2*C
        K[i, ip] -= C
        K[i, im] -= C
    end
    return K
end

function generalized_frequencies(K::AbstractMatrix{<:Real}, masses::AbstractVector{<:Real})
    invsqrtM = Diagonal(1.0 ./ sqrt.(Float64.(masses)))
    S = invsqrtM * (Float64.(K)) * invsqrtM
    λ = eigvals(Symmetric(S))
    λ = clamp.(λ, 0, Inf)
    return sqrt.(λ)
end

N = 400
K = build_ring_stiffness(N; C=1.0)
m_single = ones(Float64, N)
ω_single = generalized_frequencies(K, m_single)
m_dual = [isodd(i) ? 2.0 : 1.0 for i in 1:N]
ω_dual = generalized_frequencies(K, m_dual)

println("单物种频率范围：min=$(minimum(ω_single)), max=$(maximum(ω_single))")
println("双物种频率范围：min=$(minimum(ω_dual)), max=$(maximum(ω_dual))")

plt2 = histogram(ω_single; bins=40, alpha=0.5, label="Single species", xlabel="Energy (freq)", ylabel="Count")
histogram!(plt2, ω_dual; bins=40, alpha=0.5, label="Dual species")
savefig(plt2, "dos_comparison.png")
println("已保存图像：dos_comparison.png")


