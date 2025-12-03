using Plots

# Booth function and gradient
f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
∇f(x, y) = [10x + 8y - 24, 8x + 10y - 34]

# Optimization methods
function gradient_descent(x0, y0, α, iterations)
    x, y = x0, y0
    history = Float64[]
    for i in 1:iterations
        push!(history, f(x, y))
        g = ∇f(x, y)
        x -= α * g[1]
        y -= α * g[2]
    end
    return history
end

function momentum(x0, y0, α, β, iterations)
    x, y = x0, y0
    vx, vy = 0.0, 0.0
    history = Float64[]
    for i in 1:iterations
        push!(history, f(x, y))
        g = ∇f(x, y)
        vx = β * vx + α * g[1]
        vy = β * vy + α * g[2]
        x -= vx
        y -= vy
    end
    return history
end

function adam(x0, y0, α, β1, β2, ϵ=1e-8, iterations=2000)
    x, y = x0, y0
    m, v = [0.0, 0.0], [0.0, 0.0]
    history = Float64[]
    for t in 1:iterations
        push!(history, f(x, y))
        g = ∇f(x, y)
        m = β1 .* m + (1 - β1) .* g
        v = β2 .* v + (1 - β2) .* g.^2
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)
        update = α .* m̂ ./ (sqrt.(v̂) .+ ϵ)
        x -= update[1]
        y -= update[2]
    end
    return history
end

# Run optimizations
x0, y0 = -5.0, -5.0
iterations = 2000

gd_history = gradient_descent(x0, y0, 0.01, iterations)
mom_history = momentum(x0, y0, 0.01, 0.9, iterations)
adam_history = adam(x0, y0, 0.1, 0.9, 0.999, 1e-8, iterations)

# Plot convergence diagram
plot(1:iterations, [gd_history mom_history adam_history], 
     label=["Gradient Descent" "Momentum" "Adam"], 
     yscale=:log10, linewidth=2,
     xlabel="Iterations", ylabel="f(x,y) (log scale)",
     title="Optimization Methods Convergence",
     legend=:topright)