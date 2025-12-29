############################################################
# HW11 - Gradient Based Optimization (Julia)
# Huicheng Zhang
############################################################

using ForwardDiff
using CairoMakie
using Random
using Statistics

############################################################
# 工具函数：通用一阶优化器实现
############################################################

"""
Generic gradient descent:
    x_{k+1} = x_k - α ∇f(x_k)

Inputs:
- f: R^n -> R
- x0: initial point (Vector{Float64})
- α: learning rate
- maxiter: maximum iterations

Returns:
- x: final point
- f_hist: vector of f(x_k)
- x_hist: vector of points (trajectory)
"""
function gradient_descent(f, x0; α=0.01, maxiter=1000)
    x = copy(x0)
    f_hist = Float64[]
    x_hist = Vector{Vector{Float64}}()
    push!(f_hist, f(x))
    push!(x_hist, copy(x))

    for k in 1:maxiter
        g = ForwardDiff.gradient(f, x)
        x .-= α .* g
        push!(f_hist, f(x))
        push!(x_hist, copy(x))
    end
    return x, f_hist, x_hist
end

"""
Momentum:
    v_{k+1} = β v_k + ∇f(x_k)
    x_{k+1} = x_k - α v_{k+1}
"""
function momentum_descent(f, x0; α=0.01, β=0.9, maxiter=2000)
    x = copy(x0)
    v = zeros(length(x))
    f_hist = Float64[]
    push!(f_hist, f(x))

    for k in 1:maxiter
        g = ForwardDiff.gradient(f, x)
        v .= β .* v .+ g
        x .-= α .* v
        push!(f_hist, f(x))
    end
    return x, f_hist
end

"""
Adam optimizer:
    m_t = β1 m_{t-1} + (1-β1) g_t
    v_t = β2 v_{t-1} + (1-β2) g_t^2
    m̂_t = m_t / (1-β1^t)
    v̂_t = v_t / (1-β2^t)
    x_t = x_{t-1} - α m̂_t / (sqrt(v̂_t)+eps)

Returns:
- x: final point
- f_hist: function values
- x_hist: trajectory (optional)
"""
function adam_optimize(f, x0; α=0.01, β1=0.9, β2=0.999,
                       maxiter=2000, record_trajectory=false)
    x = copy(x0)
    m = zeros(length(x))
    v = zeros(length(x))
    eps = 1e-8

    f_hist = Float64[]
    x_hist = record_trajectory ? Vector{Vector{Float64}}() : nothing
    push!(f_hist, f(x))
    if record_trajectory
        push!(x_hist, copy(x))
    end

    for t in 1:maxiter
        g = ForwardDiff.gradient(f, x)
        m .= β1 .* m .+ (1-β1) .* g
        v .= β2 .* v .+ (1-β2) .* (g .^ 2)

        mhat = m ./ (1 - β1^t)
        vhat = v ./ (1 - β2^t)

        x .-= α .* mhat ./ (sqrt.(vhat) .+ eps)

        push!(f_hist, f(x))
        if record_trajectory
            push!(x_hist, copy(x))
        end
    end
    return x, f_hist, x_hist
end

############################################################
# 1. Gradient Descent on Himmelblau's Function
############################################################

"""
Himmelblau's function:
    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
"""
function himmelblau(x::AbstractVector)
    X, Y = x[1], x[2]
    return (X^2 + Y - 11)^2 + (X + Y^2 - 7)^2
end

# 多个不同初始点
starts = [
    [-4.0, -4.0],
    [-4.0, 4.0],
    [4.0, 4.0],
    [0.0, 0.0],
]

α_himmel = 0.01
maxiter_himmel = 5000

println("=== Problem 1: Gradient Descent on Himmelblau's Function ===")
results_p1 = []

for (i, x0) in enumerate(starts)
    x_star, f_hist, x_hist = gradient_descent(himmelblau, x0;
                                              α=α_himmel,
                                              maxiter=maxiter_himmel)
    push!(results_p1, (x0=x0, x_star=x_star, f_hist=f_hist))
    println("Start $i from $x0:")
    println("   Final point: ", x_star)
    println("   f(final)   = ", himmelblau(x_star))
end

# 收敛曲线绘图：f vs iteration
fig1 = Figure(resolution = (800, 600))
ax1 = Axis(fig1[1,1],
           xlabel="Iteration",
           ylabel="f(x_k)",
           title="Gradient Descent on Himmelblau's Function")

for (i, res) in enumerate(results_p1)
    f_hist = res[:f_hist]
    its = 0:(length(f_hist)-1)
    lines!(ax1, its, f_hist, label="start $i")
end
axislegend(ax1)
fig1
save("himmelblau_convergence.png", fig1)
println("Saved plot: himmelblau_convergence.png")

############################################################
# 2. GD vs Momentum vs Adam on Booth Function
############################################################

"""
Booth function:
    f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
Global minimum at (1,3) with f=0.
"""
function booth(x::AbstractVector)
    X, Y = x[1], x[2]
    return (X + 2Y - 7)^2 + (2X + Y - 5)^2
end

x0_booth = [-5.0, -5.0]        # initial point
maxiter_booth = 2000

α_gd = 0.01
α_mom = 0.01
α_adam = 0.1
β_mom = 0.9
β1_adam = 0.9
β2_adam = 0.999

println("\n=== Problem 2: Optimizer Comparison on Booth Function ===")

# Gradient Descent
x_gd, f_hist_gd, _ = gradient_descent(booth, x0_booth;
                                      α=α_gd,
                                      maxiter=maxiter_booth)
println("GD final point: ", x_gd, "  f=", booth(x_gd))

# Momentum
x_mom, f_hist_mom = momentum_descent(booth, x0_booth;
                                     α=α_mom,
                                     β=β_mom,
                                     maxiter=maxiter_booth)
println("Momentum final point: ", x_mom, "  f=", booth(x_mom))

# Adam
x_adam, f_hist_adam, _ = adam_optimize(booth, x0_booth;
                                       α=α_adam,
                                       β1=β1_adam,
                                       β2=β2_adam,
                                       maxiter=maxiter_booth,
                                       record_trajectory=false)
println("Adam final point: ", x_adam, "  f=", booth(x_adam))

# 收敛对比曲线（log-scale y）
fig2 = Figure(resolution = (800, 600))
ax2 = Axis(fig2[1,1],
           xlabel="Iteration",
           ylabel="f(x_k)",
           yscale=log10,
           title="GD vs Momentum vs Adam on Booth Function")

its_gd = 0:(length(f_hist_gd)-1)
its_mom = 0:(length(f_hist_mom)-1)
its_adam = 0:(length(f_hist_adam)-1)

lines!(ax2, its_gd, f_hist_gd, label="GD")
lines!(ax2, its_mom, f_hist_mom, label="Momentum")
lines!(ax2, its_adam, f_hist_adam, label="Adam")
axislegend(ax2)
fig2
save("booth_optimizer_comparison.png", fig2)
println("Saved plot: booth_optimizer_comparison.png")

############################################################
# 3. Logistic Growth Model Fitting with Adam
############################################################

println("\n=== Problem 3: Logistic Growth Parameter Fitting ===")

# Data
t_data = collect(0.0:1.0:10.0)
P_data = [10.0, 15.0, 25.0, 40.0, 60.0, 80.0, 95.0, 105.0, 110.0, 112.0, 113.0]
n_data = length(t_data)

"""
Logistic model:
    P(t) = K / (1 + exp(-r (t - t0)))
θ = [K, r, t0]
"""
function logistic(t, θ)
    K, r, t0 = θ
    return K / (1 + exp(-r * (t - t0)))
end

"""
Mean squared error loss:
    L(θ) = (1/n) ∑ (P_i - P(t_i))^2
"""
function logistic_loss(θ::AbstractVector)
    preds = [logistic(t_data[i], θ) for i in 1:n_data]
    return mean((P_data .- preds).^2)
end

# Initial guess
θ0 = [120.0, 0.5, 5.0]  # [K, r, t0]
α_logistic = 0.05
maxiter_logistic = 5000

θ_star, loss_hist, _ = adam_optimize(logistic_loss, θ0;
                                     α=α_logistic,
                                     β1=0.9,
                                     β2=0.999,
                                     maxiter=maxiter_logistic,
                                     record_trajectory=false)

println("Initial θ0 = ", θ0, "  loss(θ0) = ", logistic_loss(θ0))
println("Fitted θ* = ", θ_star)
println("Final loss = ", logistic_loss(θ_star))

# Plot loss vs iteration
fig3 = Figure(resolution = (800, 600))
ax3 = Axis(fig3[1,1],
           xlabel="Iteration",
           ylabel="Loss",
           yscale=log10,
           title="Adam on Logistic Fitting (MSE)")

its_loss = 0:(length(loss_hist)-1)
lines!(ax3, its_loss, loss_hist)
fig3
save("logistic_loss_convergence.png", fig3)
println("Saved plot: logistic_loss_convergence.png")

# Plot fitted curve vs data
fig4 = Figure(resolution = (800, 600))
ax4 = Axis(fig4[1,1],
           xlabel="t",
           ylabel="P(t)",
           title="Logistic Fit vs Data")

# data points
scatter!(ax4, t_data, P_data, label="Data")

# fitted curve
t_dense = range(0.0, 10.0, length=200)
P_fit = [logistic(t, θ_star) for t in t_dense]
lines!(ax4, t_dense, P_fit, label="Fitted curve")
axislegend(ax4)
fig4
save("logistic_fit.png", fig4)
println("Saved plot: logistic_fit.png")

############################################################
# End of HW11 code
############################################################