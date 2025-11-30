# Problem 2: Optimizer Comparison (Gradient Descent, Momentum, Adam)
# Mingxu Zhang

using ForwardDiff
using CairoMakie

# Booth function: f(x, y) = (x + 2y - 7)² + (2x + y - 5)²
function booth(x)
    return (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2
end

# Gradient Descent
function gradient_descent(f, x0; α=0.01, max_iter=2000)
    x = copy(x0)
    history = Float64[]
    
    for i in 1:max_iter
        fval = f(x)
        push!(history, fval)
        
        grad = ForwardDiff.gradient(f, x)
        x = x - α * grad
    end
    
    return x, history
end

# Momentum optimizer
function momentum_optimizer(f, x0; α=0.01, β=0.9, max_iter=2000)
    x = copy(x0)
    v = zeros(length(x0))  # velocity
    history = Float64[]
    
    for i in 1:max_iter
        fval = f(x)
        push!(history, fval)
        
        grad = ForwardDiff.gradient(f, x)
        v = β * v + α * grad  # update velocity
        x = x - v  # update parameters
    end
    
    return x, history
end

# Adam optimizer
function adam_optimizer(f, x0; α=0.1, β1=0.9, β2=0.999, ε=1e-8, max_iter=2000)
    x = copy(x0)
    m = zeros(length(x0))  # first moment estimate
    v = zeros(length(x0))  # second moment estimate
    history = Float64[]
    
    for t in 1:max_iter
        fval = f(x)
        push!(history, fval)
        
        grad = ForwardDiff.gradient(f, x)
        
        # Update biased first moment estimate
        m = β1 * m + (1 - β1) * grad
        
        # Update biased second raw moment estimate
        v = β2 * v + (1 - β2) * (grad .^ 2)
        
        # Bias-corrected first moment estimate
        m_hat = m / (1 - β1^t)
        
        # Bias-corrected second raw moment estimate
        v_hat = v / (1 - β2^t)
        
        # Update parameters
        x = x - α * m_hat ./ (sqrt.(v_hat) .+ ε)
    end
    
    return x, history
end

# Starting point
x0 = [-5.0, -5.0]
max_iterations = 2000

println("=" ^ 60)
println("Problem 2: Optimizer Comparison on Booth Function")
println("=" ^ 60)
println("Starting point: $x0")
println("Global minimum: (1, 3) with f = 0")
println("=" ^ 60)

# Run all three optimizers
println("\n1. Gradient Descent (α = 0.01)")
x_gd, history_gd = gradient_descent(booth, x0, α=0.01, max_iter=max_iterations)
println("   Final point: [$(round(x_gd[1], digits=6)), $(round(x_gd[2], digits=6))]")
println("   Final value: $(round(booth(x_gd), digits=10))")

println("\n2. Momentum (α = 0.01, β = 0.9)")
x_mom, history_mom = momentum_optimizer(booth, x0, α=0.01, β=0.9, max_iter=max_iterations)
println("   Final point: [$(round(x_mom[1], digits=6)), $(round(x_mom[2], digits=6))]")
println("   Final value: $(round(booth(x_mom), digits=10))")

println("\n3. Adam (α = 0.1, β₁ = 0.9, β₂ = 0.999)")
x_adam, history_adam = adam_optimizer(booth, x0, α=0.1, β1=0.9, β2=0.999, max_iter=max_iterations)
println("   Final point: [$(round(x_adam[1], digits=6)), $(round(x_adam[2], digits=6))]")
println("   Final value: $(round(booth(x_adam), digits=10))")

# Find convergence iteration (when f < threshold)
function find_convergence_iter(history, threshold=1e-6)
    for (i, val) in enumerate(history)
        if val < threshold
            return i
        end
    end
    return length(history)
end

conv_gd = find_convergence_iter(history_gd)
conv_mom = find_convergence_iter(history_mom)
conv_adam = find_convergence_iter(history_adam)

println("\n" * "=" ^ 60)
println("Convergence Analysis (threshold = 1e-6):")
println("=" ^ 60)
println("  Gradient Descent: converged at iteration $conv_gd")
println("  Momentum: converged at iteration $conv_mom")
println("  Adam: converged at iteration $conv_adam")

# Determine fastest
fastest = argmin([conv_gd, conv_mom, conv_adam])
methods = ["Gradient Descent", "Momentum", "Adam"]
println("\n  >> Fastest method: $(methods[fastest])")

# Create comparison plot
fig = Figure(size=(1200, 500))

# Plot 1: Full convergence curves (log scale)
ax1 = Axis(fig[1, 1],
    xlabel="Iteration",
    ylabel="Function Value (log scale)",
    title="Convergence Comparison (Full 2000 iterations)",
    yscale=log10)

# Filter history for log plot
history_gd_filtered = max.(history_gd, 1e-20)
history_mom_filtered = max.(history_mom, 1e-20)
history_adam_filtered = max.(history_adam, 1e-20)

lines!(ax1, 1:length(history_gd_filtered), history_gd_filtered, 
       label="Gradient Descent (α=0.01)", color=:blue, linewidth=2)
lines!(ax1, 1:length(history_mom_filtered), history_mom_filtered, 
       label="Momentum (α=0.01, β=0.9)", color=:red, linewidth=2)
lines!(ax1, 1:length(history_adam_filtered), history_adam_filtered, 
       label="Adam (α=0.1, β₁=0.9, β₂=0.999)", color=:green, linewidth=2)

axislegend(ax1, position=:rt)

# Plot 2: Zoomed in first 500 iterations
ax2 = Axis(fig[1, 2],
    xlabel="Iteration",
    ylabel="Function Value (log scale)",
    title="Convergence Comparison (First 500 iterations)",
    yscale=log10)

n_zoom = min(500, length(history_gd))
lines!(ax2, 1:n_zoom, history_gd_filtered[1:n_zoom], 
       label="Gradient Descent", color=:blue, linewidth=2)
lines!(ax2, 1:n_zoom, history_mom_filtered[1:n_zoom], 
       label="Momentum", color=:red, linewidth=2)
lines!(ax2, 1:n_zoom, history_adam_filtered[1:n_zoom], 
       label="Adam", color=:green, linewidth=2)

axislegend(ax2, position=:rt)

save("problem2_comparison.png", fig)
println("\nComparison plot saved to problem2_comparison.png")

# Summary
println("\n" * "=" ^ 60)
println("Summary:")
println("=" ^ 60)
println("Method           | Final Point          | Final f(x,y)     | Conv. Iter")
println("-" ^ 70)
println("Gradient Descent | ($(round(x_gd[1], digits=4)), $(round(x_gd[2], digits=4))) | $(round(booth(x_gd), digits=10)) | $conv_gd")
println("Momentum         | ($(round(x_mom[1], digits=4)), $(round(x_mom[2], digits=4))) | $(round(booth(x_mom), digits=10)) | $conv_mom")
println("Adam             | ($(round(x_adam[1], digits=4)), $(round(x_adam[2], digits=4))) | $(round(booth(x_adam), digits=10)) | $conv_adam")
