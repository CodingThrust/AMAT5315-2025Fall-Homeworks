using ForwardDiff, CairoMakie, LinearAlgebra

# Define the Himmelblau function - a common test function for optimization with 4 local minima
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

# Gradient descent implementation
function gradient_descent(f, x0; lr=0.01, max_iters=1000)
    x, history = copy(x0), Float64[]  # Initialize position and history array
    for i in 1:max_iters
        grad = ForwardDiff.gradient(f, x)  # Compute gradient using automatic differentiation
        x -= lr * grad  # Update position: move opposite to gradient direction
        push!(history, f(x))  # Store function value at current position
        norm(grad) < 1e-8 && break  # Convergence check: stop if gradient is near zero
    end
    return x, history
end

# Test different starting points - chosen to explore different regions of the search space
starts = [[6.0, 6.0], [-6.0, 6.0], [-6.0, -6.0], [6.0, -6.0], [0.0, 0.0]]
results = [gradient_descent(himmelblau, start) for start in starts]

# Create convergence plot
fig = Figure()
ax = Axis(fig[1, 1], title="Gradient Descent Convergence", 
          xlabel="Iteration", ylabel="Function Value")

# Plot convergence history for each starting point
colors = [:blue, :red, :green, :orange, :purple]
for (i, (point, hist)) in enumerate(results)
    lines!(ax, eachindex(hist), hist, color=colors[i], linewidth=2,
           label="Start $(starts[i])")
end

# Add legend to identify different starting points
Legend(fig[1, 2], ax)
display(fig)
save("convergence_plot.png", fig)
println("saved convergence_plot.png")

# Print optimization results
println("Optimization Results:")
for (i, (point, hist)) in enumerate(results)
    println("Start $(starts[i]) -> Final: $(round.(point, digits=4)), Value: $(round(himmelblau(point), digits=10))")
end