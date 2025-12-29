# Homework 11
# Solution: Gradient Based Optimization

using ForwardDiff
using CairoMakie
using LinearAlgebra
using Statistics

# ==============================================================================
# Optimizer Framework
# ==============================================================================

# Abstract parent type for all optimizers
abstract type AbstractOptimizer end

# 1. Gradient Descent
struct GD <: AbstractOptimizer
    alpha::Float64 # Learning rate
end

# 2. Momentum
struct Momentum <: AbstractOptimizer
    alpha::Float64
    beta::Float64
end

# 3. Adam
struct Adam <: AbstractOptimizer
    alpha::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
end
# Default constructor for Adam
Adam(alpha=0.001) = Adam(alpha, 0.9, 0.999, 1e-8)

# --- Step Functions (Multiple Dispatch) ---

# Gradient Descent Update
function step!(opt::GD, x::Vector{Float64}, grads::Vector{Float64}, state=nothing)
    # x_new = x - alpha * gradient
    @. x = x - opt.alpha * grads
    return nothing
end

# Momentum Update
function step!(opt::Momentum, x::Vector{Float64}, grads::Vector{Float64}, state)
    # Initialize velocity if it doesn't exist
    v = (state === nothing) ? zeros(length(x)) : state
    
    # v = beta * v + gradient
    @. v = opt.beta * v + grads
    # x = x - alpha * v
    @. x = x - opt.alpha * v
    
    return v # Return velocity as state
end

# Adam Update
mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

function step!(opt::Adam, x::Vector{Float64}, grads::Vector{Float64}, state)
    if state === nothing
        state = AdamState(zeros(length(x)), zeros(length(x)), 0)
    end
    
    s = state
    s.t += 1
    
    # Update biased moments
    @. s.m = opt.beta1 * s.m + (1 - opt.beta1) * grads
    @. s.v = opt.beta2 * s.v + (1 - opt.beta2) * (grads ^ 2)

    # Bias correction
    m_hat = s.m ./ (1 - opt.beta1 ^ s.t)
    v_hat = s.v ./ (1 - opt.beta2 ^ s.t)

    # Update parameters
    @. x = x - opt.alpha * m_hat / (sqrt(v_hat) + opt.epsilon)
    
    return s
end

# --- Main Optimization Loop ---

function optimize(f, x0::Vector{Float64}, opt::AbstractOptimizer; max_iter=1000)
    x = copy(x0)
    f_hist = Float64[]
    x_hist = Vector{Float64}[]
    
    state = nothing 
    
    push!(f_hist, f(x))
    push!(x_hist, copy(x))

    for i in 1:max_iter
        grads = ForwardDiff.gradient(f, x)
        state = step!(opt, x, grads, state)
        
        push!(f_hist, f(x))
        push!(x_hist, copy(x))
    end

    return (minizer=x, f_vals=f_hist, path=x_hist)
end

# ==============================================================================
# Problem 1: Himmelblau's Function
# ==============================================================================
println("Running Problem 1...")

himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

start_points = [[-4.0, -4.0], [-4.0, 4.0], [4.0, 4.0], [0.0, 0.0]]
gd_solver = GD(0.01)

# Plotting setup
fig1 = Figure(size = (900, 450))
ax1 = Axis(fig1[1, 1], title="Convergence (Function Value)", xlabel="Iteration", ylabel="f(x)")
ax2 = Axis(fig1[1, 2], title="Optimization Trajectory", xlabel="x", ylabel="y")

# Background contours for visualization
xs = range(-6, 6, length=100)
ys = range(-6, 6, length=100)
zs = [himmelblau([x, y]) for x in xs, y in ys]
contour!(ax2, xs, ys, zs, levels=0:10:150, color=:grey80)

for (i, p0) in enumerate(start_points)
    res = optimize(himmelblau, p0, gd_solver; max_iter=200)
    
    # Plot convergence curve
    lines!(ax1, 0:length(res.f_vals)-1, res.f_vals, label="Start $p0", linewidth=2)
    
    # Plot trajectory
    path_mat = hcat(res.path...)
    lines!(ax2, path_mat[1, :], path_mat[2, :], label="Start $i", linewidth=2)
    scatter!(ax2, [res.minizer[1]], [res.minizer[2]], marker=:star5, markersize=15, color=:red)
    
    println("Start: $p0 -> Final: $(round.(res.minizer, digits=3)), f: $(round(res.f_vals[end], digits=5))")
end
axislegend(ax1)
save("p1_himmelblau.png", fig1)

# ==============================================================================
# Problem 2: Optimizer Comparison (Booth Function)
# ==============================================================================
println("\nRunning Problem 2...")

booth(x) = (x[1] + 2x[2] - 7)^2 + (2x[1] + x[2] - 5)^2
x0_booth = [-5.0, -5.0]

# Define optimizers with parameters from problem statement
optimizers = Dict(
    "Gradient Descent" => GD(0.01),
    "Momentum"         => Momentum(0.01, 0.9),
    "Adam"             => Adam(0.1, 0.9, 0.999, 1e-8)
)

fig2 = Figure(size = (600, 500))
ax_comp = Axis(fig2[1, 1], 
    title="Optimizer Comparison on Booth Function", 
    xlabel="Iteration", 
    ylabel="Cost (Log Scale)", 
    yscale=log10)

for (name, opt) in optimizers
    res = optimize(booth, x0_booth, opt; max_iter=2000)
    lines!(ax_comp, res.f_vals, label=name, linewidth=2)
    println("$name final loss: $(res.f_vals[end])")
end

axislegend(ax_comp)
save("p2_comparison.png", fig2)

# ==============================================================================
# Problem 3: Logistic Growth Fitting
# ==============================================================================
println("\nRunning Problem 3...")

t_data = Float64.(0:10)
P_data = [10.0, 15.0, 25.0, 40.0, 60.0, 80.0, 95.0, 105.0, 110.0, 112.0, 113.0]

# Model: P(t) = K / (1 + exp(-r(t-t0)))
# params = [K, r, t0]
logistic_model(t, params) = params[1] / (1 + exp(-params[2] * (t - params[3])))

function mse_loss(params)
    # Calculate predictions for all t_data
    P_pred = logistic_model.(t_data, Ref(params)) 
    return mean((P_data .- P_pred).^2)
end

# Initialization: K=120, r=0.5, t0=5.0
params_init = [120.0, 0.5, 5.0]
adam_opt = Adam(0.1) 

res_fit = optimize(mse_loss, params_init, adam_opt; max_iter=5000)
params_final = res_fit.minizer

println("Fitted Parameters:")
println("K  = $(round(params_final[1], digits=2))")
println("r  = $(round(params_final[2], digits=4))")
println("t0 = $(round(params_final[3], digits=2))")

# Plot results
fig3 = Figure(size = (800, 400))
ax_loss = Axis(fig3[1, 1], title="Training Loss (MSE)", yscale=log10, xlabel="Iteration")
ax_fit  = Axis(fig3[1, 2], title="Model Fit vs Data", xlabel="Time", ylabel="Population")

lines!(ax_loss, res_fit.f_vals, color=:purple)

# Data vs Fit
scatter!(ax_fit, t_data, P_data, label="Data", color=:black)
t_dense = range(0, 10, length=200)
P_dense = logistic_model.(t_dense, Ref(params_final))
lines!(ax_fit, t_dense, P_dense, label="Fitted Model", color=:red, linewidth=3)

axislegend(ax_fit, position=:rb)
save("p3_logistic_fit.png", fig3)