using Optimisers
using ForwardDiff
using LinearAlgebra
using Plots
using Logging
using Printf

# Suppress the default logging messages for a cleaner output
Logging.disable_logging(Logging.Info)

## 1. Define the Booth Function and its Gradient

"""
Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
Global minimum is at (1, 3) with f(1, 3) = 0.
"""
function booth_function(p::AbstractVector)
    x, y = p
    return (x + 2y - 7)^2 + (2x + y - 5)^2
end

# Automatic differentiation for the gradient using ForwardDiff.jl
function booth_gradient(p::AbstractVector)
    return ForwardDiff.gradient(booth_function, p)
end

## 2. Setup Optimization Parameters

const N_ITERATIONS = 2000
const INITIAL_POINT = [-5.0, -5.0]
const PLOT_FILENAME = "booth_convergence_comparison.png"

# Optimizers with specified hyperparameters
const OPTIMIZERS = Dict(
    "Gradient Descent (α=0.01)" => Optimisers.Descent(0.01),
    "Momentum (α=0.01, β=0.9)" => Optimisers.Momentum(0.01, 0.9),
    "Adam (α=0.1, β1=0.9, β2=0.999)" => Optimisers.Adam(0.1, (0.9, 0.999))
)

## 3. Run Optimization for Each Method

function run_optimization(optimizer, initial_p::AbstractVector, n_iterations::Int)
    # Initialize parameters and optimizer state
    p = copy(initial_p)
    opt_state = Optimisers.setup(optimizer, p)
    
    # Store the objective function values at each iteration
    f_history = Float64[]
    
    for i in 1:n_iterations
        # 1. Compute the current function value and gradient
        f_val = booth_function(p)
        push!(f_history, f_val)
        grad = booth_gradient(p)
        
        # 2. Update the optimizer state and get the parameter update (dp)
        opt_state, dp = Optimisers.update(opt_state, p, grad)
        
        # 3. Apply the parameter update
        p .-= dp 
    end
    
    return f_history, p
end

# Store results
results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

println("Starting optimization runs...")
for (name, optimizer) in OPTIMIZERS
    print("Running $name...")
    f_history, final_p = run_optimization(optimizer, INITIAL_POINT, N_ITERATIONS)
    results[name] = (f_history, final_p)
    f_final = f_history[end]
    x, y = final_p
    println(" Done. Final f(x,y): $(@sprintf("%.2e", f_final)), Final point: ($(@sprintf("%.4f", x)), $(@sprintf("%.4f", y)))")
end

## 4. Plot Convergence Curves and Save Plot

# Create the plot
p = plot(
    title="Convergence of Optimizers on Booth Function",
    xlabel="Iteration",
    ylabel="Objective Function Value f(x,y) (Log Scale)",
    yscale=:log10,
    legend=:topright,
    size=(800, 600)
)

# Plot each method's convergence history
for (name, (f_history, _)) in results
    plot!(p, f_history, label=name, linewidth=2)
end

display(p)

# Save the plot in current directory
savefig(p, PLOT_FILENAME)
println("\nPlot saved to: $PLOT_FILENAME")