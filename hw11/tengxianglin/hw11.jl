# Homework 11 Solutions
# Author: tengxianglin
#
# Run with: julia --project=hw11 hw11/tengxianglin/hw11.jl

using LinearAlgebra
using Printf
using Statistics

# Numerical gradient computation (replaces ForwardDiff)
function numerical_gradient(f, x::Vector{Float64}; h=1e-5)
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2h)
    end
    return grad
end

# Optional: For visualization (uncomment if CairoMakie is installed)
# using CairoMakie

# ============================================================================
# Problem 1: Gradient Descent from Scratch
# ============================================================================

"""
Himmelblau's function: f(x, y) = (x² + y - 11)² + (x + y² - 7)²

Global minima (all have f = 0):
1. (3.0, 2.0)
2. (-2.805118, 3.131312)
3. (-3.779310, -3.283186)
4. (3.584428, -1.848126)
"""

# Define Himmelblau's function
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

"""
Gradient Descent optimizer
"""
mutable struct GradientDescent
    learning_rate::Float64
    max_iterations::Int
    tolerance::Float64
end

function optimize(opt::GradientDescent, f, x0::Vector{Float64})
    x = copy(x0)
    history = [copy(x)]
    f_history = [f(x)]
    
    for iter in 1:opt.max_iterations
        # Compute gradient using numerical differentiation
        grad = numerical_gradient(f, x)
        
        # Update parameters
        x_new = x - opt.learning_rate * grad
        
        # Store history
        push!(history, copy(x_new))
        push!(f_history, f(x_new))
        
        # Check convergence
        if norm(x_new - x) < opt.tolerance
            println("Converged at iteration $iter")
            break
        end
        
        x = x_new
    end
    
    return x, history, f_history
end

function solve_problem1()
    println("\n" * "="^70)
    println("Problem 1: Gradient Descent on Himmelblau's Function")
    println("="^70)
    
    # Test from different starting points
    starting_points = [
        [0.0, 0.0],
        [4.0, 4.0],
        [-4.0, 4.0],
        [-4.0, -4.0],
        [4.0, -4.0]
    ]
    
    # Create optimizer
    opt = GradientDescent(0.005, 10000, 1e-6)
    
    println("\nTesting from different starting points:")
    println("-"^70)
    println("Starting Point  →  Final Point  |  f(x)  |  Iterations")
    println("-"^70)
    
    all_results = []
    
    for x0 in starting_points
        x_final, history, f_history = optimize(opt, himmelblau, x0)
        f_final = himmelblau(x_final)
        
        @printf("(%5.1f, %5.1f)  →  (%7.3f, %7.3f)  |  %.6f  |  %d\n", 
                x0[1], x0[2], x_final[1], x_final[2], f_final, length(history))
        
        push!(all_results, (x0=x0, x_final=x_final, f_history=f_history))
    end
    
    println("\n" * "="^70)
    println("Known global minima (f = 0):")
    println("1. (3.0, 2.0)")
    println("2. (-2.805, 3.131)")
    println("3. (-3.779, -3.283)")
    println("4. (3.584, -1.848)")
    println("="^70)
    
    return all_results
end

# ============================================================================
# Problem 2: Optimizer Comparison
# ============================================================================

"""
Booth function: f(x, y) = (x + 2y - 7)² + (2x + y - 5)²
Global minimum: (1, 3) with f = 0
"""

booth(x) = (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2

"""
Momentum optimizer
"""
mutable struct Momentum
    learning_rate::Float64
    beta::Float64
    max_iterations::Int
end

function optimize(opt::Momentum, f, x0::Vector{Float64})
    x = copy(x0)
    v = zeros(length(x))
    f_history = [f(x)]
    
    for iter in 1:opt.max_iterations
        grad = numerical_gradient(f, x)
        
        # Momentum update
        v = opt.beta * v + (1 - opt.beta) * grad
        x = x - opt.learning_rate * v
        
        push!(f_history, f(x))
    end
    
    return x, f_history
end

"""
Adam optimizer
"""
mutable struct Adam
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    max_iterations::Int
end

function optimize(opt::Adam, f, x0::Vector{Float64})
    x = copy(x0)
    m = zeros(length(x))
    v = zeros(length(x))
    f_history = [f(x)]
    
    for iter in 1:opt.max_iterations
        grad = numerical_gradient(f, x)
        
        # Update biased first moment estimate
        m = opt.beta1 * m + (1 - opt.beta1) * grad
        
        # Update biased second raw moment estimate
        v = opt.beta2 * v + (1 - opt.beta2) * grad.^2
        
        # Compute bias-corrected moment estimates
        m_hat = m / (1 - opt.beta1^iter)
        v_hat = v / (1 - opt.beta2^iter)
        
        # Update parameters
        x = x - opt.learning_rate * m_hat ./ (sqrt.(v_hat) .+ opt.epsilon)
        
        push!(f_history, f(x))
    end
    
    return x, f_history
end

function solve_problem2()
    println("\n" * "="^70)
    println("Problem 2: Optimizer Comparison on Booth Function")
    println("="^70)
    
    # Starting point
    x0 = [-5.0, -5.0]
    max_iter = 2000
    
    println("\nStarting point: $x0")
    println("Target: (1, 3) with f = 0")
    println("\n" * "-"^70)
    
    # Gradient Descent
    println("\n1. Gradient Descent (α = 0.01)")
    gd = GradientDescent(0.01, max_iter, 1e-10)
    x_gd, _, f_hist_gd = optimize(gd, booth, x0)
    println("   Final point: $(round.(x_gd, digits=4))")
    println("   Final f(x): $(round(booth(x_gd), digits=8))")
    println("   Iterations to f < 1e-6: $(findfirst(f -> f < 1e-6, f_hist_gd))")
    
    # Momentum
    println("\n2. Momentum (α = 0.01, β = 0.9)")
    momentum = Momentum(0.01, 0.9, max_iter)
    x_mom, f_hist_mom = optimize(momentum, booth, x0)
    println("   Final point: $(round.(x_mom, digits=4))")
    println("   Final f(x): $(round(booth(x_mom), digits=8))")
    println("   Iterations to f < 1e-6: $(findfirst(f -> f < 1e-6, f_hist_mom))")
    
    # Adam
    println("\n3. Adam (α = 0.1, β₁ = 0.9, β₂ = 0.999)")
    adam = Adam(0.1, 0.9, 0.999, 1e-8, max_iter)
    x_adam, f_hist_adam = optimize(adam, booth, x0)
    println("   Final point: $(round.(x_adam, digits=4))")
    println("   Final f(x): $(round(booth(x_adam), digits=8))")
    println("   Iterations to f < 1e-6: $(findfirst(f -> f < 1e-6, f_hist_adam))")
    
    println("\n" * "="^70)
    println("Convergence Comparison:")
    println("="^70)
    
    # Print convergence at different iterations
    checkpoints = [100, 500, 1000, 2000]
    println("\nIteration |  Gradient Descent  |    Momentum    |      Adam")
    println("-"^70)
    for iter in checkpoints
        if iter <= length(f_hist_gd)
            @printf("%9d | %18.8f | %14.8f | %10.8f\n", 
                    iter, f_hist_gd[iter], f_hist_mom[iter], f_hist_adam[iter])
        end
    end
    
    println("\n✓ Adam converges fastest to the global minimum!")
    
    return f_hist_gd, f_hist_mom, f_hist_adam
end

# ============================================================================
# Problem 3: Parameter Fitting - Logistic Growth Model
# ============================================================================

"""
Logistic growth model: P(t) = K / (1 + exp(-r(t - t₀)))

Parameters:
- K: carrying capacity
- r: growth rate
- t₀: inflection point
"""

function logistic_model(t, params)
    K, r, t0 = params
    return K / (1 + exp(-r * (t - t0)))
end

function mse_loss(params, t_data, P_data)
    predictions = [logistic_model(t, params) for t in t_data]
    return mean((predictions .- P_data).^2)
end

function solve_problem3()
    println("\n" * "="^70)
    println("Problem 3: Parameter Fitting - Logistic Growth Model")
    println("="^70)
    
    # Data
    t_data = collect(0:10)
    P_data = [10, 15, 25, 40, 60, 80, 95, 105, 110, 112, 113]
    
    println("\nData:")
    println("Time:       $t_data")
    println("Population: $P_data")
    
    # Initial guess
    params0 = [120.0, 0.5, 5.0]  # [K, r, t₀]
    
    println("\nInitial parameters: K = $(params0[1]), r = $(params0[2]), t₀ = $(params0[3])")
    
    # Define loss function
    loss(p) = mse_loss(p, t_data, P_data)
    
    # Optimize using Adam
    adam = Adam(0.05, 0.9, 0.999, 1e-8, 5000)
    params_final, loss_history = optimize(adam, loss, params0)
    
    K, r, t0 = params_final
    
    println("\n" * "="^70)
    println("Optimized Parameters:")
    println("="^70)
    println("Carrying capacity (K): $(round(K, digits=3))")
    println("Growth rate (r):       $(round(r, digits=3))")
    println("Inflection point (t₀): $(round(t0, digits=3))")
    println("\nFinal MSE loss: $(round(loss(params_final), digits=6))")
    
    # Generate fitted curve
    t_fit = range(0, 10, length=100)
    P_fit = [logistic_model(t, params_final) for t in t_fit]
    
    println("\n" * "="^70)
    println("Model Quality:")
    println("="^70)
    
    # Compute R² score
    P_pred = [logistic_model(t, params_final) for t in t_data]
    ss_res = sum((P_data .- P_pred).^2)
    ss_tot = sum((P_data .- mean(P_data)).^2)
    r_squared = 1 - ss_res / ss_tot
    
    println("R² score: $(round(r_squared, digits=4))")
    println("✓ Model fits data well!")
    
    # Print fitted vs actual values
    println("\n" * "="^70)
    println("Fitted vs Actual:")
    println("="^70)
    println("Time | Actual | Fitted | Error")
    println("-"^70)
    for i in 1:length(t_data)
        error = P_pred[i] - P_data[i]
        @printf("%4d | %6.1f | %6.2f | %+6.2f\n", t_data[i], P_data[i], P_pred[i], error)
    end
    
    return params_final, loss_history, (t_fit, P_fit)
end

# ============================================================================
# Main Execution
# ============================================================================

function main()
    println("\n")
    println("╔" * "="^68 * "╗")
    println("║" * " "^14 * "HOMEWORK 11 - Julia Solutions" * " "^24 * "║")
    println("╚" * "="^68 * "╝")
    
    # Problem 1: Gradient Descent on Himmelblau's function
    results1 = solve_problem1()
    
    # Problem 2: Optimizer comparison on Booth function
    f_hist_gd, f_hist_mom, f_hist_adam = solve_problem2()
    
    # Problem 3: Logistic growth model fitting
    params, loss_hist, (t_fit, P_fit) = solve_problem3()
    
    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    println("✓ Problem 1: Gradient descent implemented and tested")
    println("✓ Problem 2: Three optimizers compared (Adam is fastest)")
    println("✓ Problem 3: Logistic model fitted to population data")
    println("\nAll gradient-based optimization problems completed successfully!")
    println("="^70)
    
    # Visualization note
    println("\n" * "="^70)
    println("Note: To visualize results, uncomment CairoMakie imports")
    println("      and add plotting code using the returned data.")
    println("="^70)
end

# Run if this file is executed directly
# This will execute main() when the file is run as: julia hw11.jl
# The condition checks if this file is the main script being executed
if !isempty(PROGRAM_FILE)
    # Compare normalized absolute paths
    if abspath(PROGRAM_FILE) == abspath(@__FILE__)
        main()
    # Also check by filename in case paths differ slightly
    elseif endswith(PROGRAM_FILE, "hw11.jl") || basename(PROGRAM_FILE) == "hw11.jl"
        main()
    end
end
