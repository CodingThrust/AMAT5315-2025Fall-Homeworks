# Problem 1: Gradient Descent Implementation on Himmelblau's Function
# Mingxu Zhang

using ForwardDiff
using CairoMakie

# Himmelblau's function: f(x, y) = (x² + y - 11)² + (x + y² - 7)²
function himmelblau(x)
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

# Gradient descent implementation from scratch
function gradient_descent(f, x0; learning_rate=0.01, max_iter=1000, tol=1e-8)
    x = copy(x0)
    history = Float64[]
    x_history = [copy(x)]
    
    for i in 1:max_iter
        # Compute gradient using ForwardDiff
        grad = ForwardDiff.gradient(f, x)
        
        # Store current function value
        fval = f(x)
        push!(history, fval)
        
        # Check convergence
        if norm(grad) < tol
            println("Converged at iteration $i")
            break
        end
        
        # Update parameters
        x = x - learning_rate * grad
        push!(x_history, copy(x))
    end
    
    return x, history, x_history
end

# Helper function for L2 norm
function norm(x)
    return sqrt(sum(x .^ 2))
end

# Test from different starting points
starting_points = [
    [0.0, 0.0],
    [5.0, 5.0],
    [-5.0, 5.0],
    [5.0, -5.0],
    [-5.0, -5.0],
    [1.0, 1.0],
    [-4.0, -4.0],
    [4.0, -2.0]
]

println("=" ^ 60)
println("Problem 1: Gradient Descent on Himmelblau's Function")
println("=" ^ 60)

results = []
all_histories = []

for (i, x0) in enumerate(starting_points)
    x_final, history, x_history = gradient_descent(himmelblau, x0, learning_rate=0.01, max_iter=5000)
    final_value = himmelblau(x_final)
    push!(results, (x0=x0, x_final=x_final, f_final=final_value))
    push!(all_histories, history)
    
    println("\nStarting point $i: $(x0)")
    println("  Final point: [$(round(x_final[1], digits=6)), $(round(x_final[2], digits=6))]")
    println("  Final function value: $(round(final_value, digits=10))")
end

# Known minima of Himmelblau's function
println("\n" * "=" ^ 60)
println("Known global minima (all with f = 0):")
println("  (3, 2)")
println("  (-2.805118, 3.131312)")
println("  (-3.779310, -3.283186)")
println("  (3.584428, -1.848126)")
println("=" ^ 60)

# Create convergence plot
fig = Figure(size=(1000, 800))

# Plot 1: Convergence curves for all starting points
ax1 = Axis(fig[1, 1], 
    xlabel="Iteration", 
    ylabel="Function Value (log scale)",
    title="Convergence of Gradient Descent from Different Starting Points",
    yscale=log10)

colors = [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :brown]
for (i, history) in enumerate(all_histories)
    # Filter out very small values that cause log issues
    filtered_history = max.(history, 1e-15)
    lines!(ax1, 1:length(filtered_history), filtered_history, 
           label="Start: $(starting_points[i])", color=colors[i], linewidth=2)
end
axislegend(ax1, position=:rt)

# Plot 2: Contour plot with optimization paths
ax2 = Axis(fig[2, 1],
    xlabel="x",
    ylabel="y",
    title="Himmelblau's Function Contour with Optimization Paths",
    aspect=1)

# Create contour
x_range = range(-6, 6, length=200)
y_range = range(-6, 6, length=200)
z = [himmelblau([x, y]) for x in x_range, y in y_range]

# Clip z values for better visualization
z_clipped = min.(z, 200)

contourf!(ax2, x_range, y_range, z_clipped, levels=30, colormap=:viridis)

# Mark the known minima
minima = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
for (mx, my) in minima
    scatter!(ax2, [mx], [my], color=:white, markersize=15, marker=:star5)
end

# Mark starting points
for (i, x0) in enumerate(starting_points)
    scatter!(ax2, [x0[1]], [x0[2]], color=colors[i], markersize=10, marker=:circle)
end

# Mark final points
for (i, result) in enumerate(results)
    scatter!(ax2, [result.x_final[1]], [result.x_final[2]], 
             color=colors[i], markersize=12, marker=:diamond)
end

save("problem1_convergence.png", fig)
println("\nConvergence plot saved to problem1_convergence.png")

# Summary table
println("\n" * "=" ^ 60)
println("Summary of Results:")
println("=" ^ 60)
println("Starting Point       | Final Point                    | f(x,y)")
println("-" ^ 60)
for result in results
    x0_str = "($(round(result.x0[1], digits=1)), $(round(result.x0[2], digits=1)))"
    xf_str = "($(round(result.x_final[1], digits=4)), $(round(result.x_final[2], digits=4)))"
    f_str = "$(round(result.f_final, digits=8))"
    println("$(rpad(x0_str, 20)) | $(rpad(xf_str, 30)) | $f_str")
end
