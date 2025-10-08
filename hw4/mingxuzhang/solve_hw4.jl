# Homework 4 Solutions
using LinearAlgebra
using Makie, CairoMakie
using Polynomials

# Problem 1: Condition Number Analysis
println("问题1: 条件数分析")
println("=" ^ 50)

# Matrix (a)
A_a = [1e10 0; 0 1e-10]
cond_a = cond(A_a)
println("矩阵(a): [10^10, 0; 0, 10^-10]")
println("条件数: ", cond_a)
println("分类: ", cond_a > 1e12 ? "病态矩阵" : "良态矩阵")
println()

# Matrix (b)
A_b = [1e10 0; 0 1e10]
cond_b = cond(A_b)
println("矩阵(b): [10^10, 0; 0, 10^10]")
println("条件数: ", cond_b)
println("分类: ", cond_b > 1e12 ? "病态矩阵" : "良态矩阵")
println()

# Matrix (c)
A_c = [1e-10 0; 0 1e-10]
cond_c = cond(A_c)
println("矩阵(c): [10^-10, 0; 0, 10^-10]")
println("条件数: ", cond_c)
println("分类: ", cond_c > 1e12 ? "病态矩阵" : "良态矩阵")
println()

# Matrix (d)
A_d = [1 2; 2 4]
cond_d = cond(A_d)
println("矩阵(d): [1, 2; 2, 4]")
println("条件数: ", cond_d)
println("分类: ", cond_d > 1e12 ? "病态矩阵" : "良态矩阵")
println()

# Problem 2: Solving Linear Equations
println("问题2: 求解线性方程组")
println("=" ^ 50)

# Coefficient matrix
A = [2  1 -1  0  1;
     1  3  1 -1  0;
     0  1  4  1 -1;
    -1  0  1  3  1;
     1 -1  0  1  2]

# Right-hand side vector
b = [4, 6, 2, 5, 3]

# Solve the system
x = A \ b

println("线性方程组的解:")
for i in 1:5
    println("x$i = ", x[i])
end
println()

# Verify the solution
residual = A * x - b
println("验证: 残差向量的2范数 = ", norm(residual))
println()

# Problem 3: Polynomial Data Fitting
println("问题3: 多项式数据拟合")
println("=" ^ 50)

# China's newborn population data (1990-2021)
years = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
         2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
         2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
         2020, 2021]

population = [2374, 2250, 2113, 2120, 2098, 2052, 2057, 2028, 1934, 1827,
              1765, 1696, 1641, 1594, 1588, 1612, 1581, 1591, 1604, 1587,
              1588, 1600, 1800, 1640, 1687, 1655, 1786, 1723, 1523, 1465,
              1200, 1062]

# Shift x-axis for numerical stability (years relative to 1990)
x = years .- 1990
y = population

# Set up Vandermonde matrix for cubic polynomial
n = length(x)
A_poly = zeros(n, 4)
for i in 1:n
    A_poly[i, 1] = 1
    A_poly[i, 2] = x[i]
    A_poly[i, 3] = x[i]^2
    A_poly[i, 4] = x[i]^3
end

# Solve for coefficients
coeffs = A_poly \ y
a0, a1, a2, a3 = coeffs

println("三次多项式拟合结果:")
println("y = $a0 + $a1*x + $a2*x² + $a3*x³")
println("其中 x = 年份 - 1990")
println()

# Predict for 2024
x_2024 = 2024 - 1990
y_2024 = a0 + a1*x_2024 + a2*x_2024^2 + a3*x_2024^3
println("2024年预测的新生儿人口: ", round(y_2024, digits=1), " 万人")
println()

# Create plot
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], 
          xlabel="Year", 
          ylabel="Newborn Population (10^4 people)",
          title="China Newborn Population Data and Cubic Polynomial Fit")

# Plot original data
scatter!(ax, years, population, color=:blue, marker=:circle, markersize=10, label="Actual Data")

# Plot fitted curve
x_fit = 0:0.5:35  # From 1990 to 2025
y_fit = a0 .+ a1.*x_fit .+ a2.*x_fit.^2 .+ a3.*x_fit.^3
years_fit = x_fit .+ 1990
lines!(ax, years_fit, y_fit, color=:red, linewidth=2, label="Cubic Polynomial Fit")

# Add prediction point for 2024
scatter!(ax, [2024], [y_2024], color=:green, marker=:star5, markersize=15, label="2024 Prediction")

# Add legend
axislegend(position=:rt)

# Save plot
save("/home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw4/mingxuzhang/population_fit.png", fig)
println("图表已保存为 population_fit.png")
println()

# Problem 4: Eigen-decomposition (Extra points)
println("问题4: 特征分解 - 双种群弹簧链分析")
println("=" ^ 50)

# Parameters
N = 200  # Number of sites (should be even for dual species)
C = 1.0  # Spring constant

# Function to create the dynamical matrix for dual species spring chain
function dual_species_matrix(N, C)
    H = zeros(N, N)
    
    for i in 1:N
        # Mass: 1 for even sites, 2 for odd sites
        m_i = (i % 2 == 0) ? 1.0 : 2.0
        
        # Diagonal term
        H[i, i] = 2*C / m_i
        
        # Off-diagonal terms with periodic boundary conditions
        j_next = (i % N) + 1
        j_prev = (i == 1) ? N : i - 1
        
        m_next = (j_next % 2 == 0) ? 1.0 : 2.0
        m_prev = (j_prev % 2 == 0) ? 1.0 : 2.0
        
        H[i, j_next] = -C / sqrt(m_i * m_next)
        H[i, j_prev] = -C / sqrt(m_i * m_prev)
    end
    
    return H
end

# Function for single species spring chain (for comparison)
function single_species_matrix(N, C)
    H = zeros(N, N)
    
    for i in 1:N
        H[i, i] = 2*C
        
        j_next = (i % N) + 1
        j_prev = (i == 1) ? N : i - 1
        
        H[i, j_next] = -C
        H[i, j_prev] = -C
    end
    
    return H
end

# Calculate eigenvalues
H_dual = dual_species_matrix(N, C)
H_single = single_species_matrix(N, C)

eigenvals_dual = eigvals(H_dual)
eigenvals_single = eigvals(H_single)

# Remove near-zero eigenvalues (numerical artifacts)
eigenvals_dual = eigenvals_dual[eigenvals_dual .> 1e-10]
eigenvals_single = eigenvals_single[eigenvals_single .> 1e-10]

println("双种群弹簧链:")
println("特征值数量: ", length(eigenvals_dual))
println("能量范围: [", minimum(eigenvals_dual), ", ", maximum(eigenvals_dual), "]")
println()

println("单种群弹簧链:")
println("特征值数量: ", length(eigenvals_single))
println("能量范围: [", minimum(eigenvals_single), ", ", maximum(eigenvals_single), "]")
println()

# Create density of states plot
fig2 = Figure(size=(1000, 600))

# Dual species plot
ax1 = Axis(fig2[1, 1], 
          xlabel="Energy", 
          ylabel="Density of States",
          title="Dual Species Spring Chain")

hist!(ax1, eigenvals_dual, bins=30, color=(:blue, 0.7), normalization=:density)

# Single species plot
ax2 = Axis(fig2[1, 2], 
          xlabel="Energy", 
          ylabel="Density of States",
          title="Single Species Spring Chain")

hist!(ax2, eigenvals_single, bins=30, color=(:red, 0.7), normalization=:density)

# Save the plot
save("/home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw4/mingxuzhang/density_of_states.png", fig2)
println("态密度图已保存为 density_of_states.png")
println()

# Comparison plot
fig3 = Figure(size=(800, 600))
ax3 = Axis(fig3[1, 1], 
          xlabel="Energy", 
          ylabel="Density of States",
          title="Comparison: Dual vs Single Species Spring Chain")

hist!(ax3, eigenvals_dual, bins=30, color=(:blue, 0.5), label="Dual Species", normalization=:density)
hist!(ax3, eigenvals_single, bins=30, color=(:red, 0.5), label="Single Species", normalization=:density)

axislegend(position=:rt)

save("/home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw4/mingxuzhang/comparison_density.png", fig3)
println("比较图已保存为 comparison_density.png")
println("所有计算完成！")
