using ForwardDiff, CairoMakie, LinearAlgebra

# Himmelblau function
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

# Gradient descent implementation
function gradient_descent(f, x0; lr=0.01, max_iters=1000, tol=1e-8)
    x, history = copy(x0), Float64[]
    for i in 1:max_iters
        grad = ForwardDiff.gradient(f, x)
        x_new = x - lr * grad
        f_val = f(x)
        push!(history, f_val)
        norm(x_new - x) < tol && f_val < tol && break
        x = x_new
    end
    return x, history
end

# Test points and known minima
test_points = [[0,0], [4,4], [-4,4], [4,-4], [-4,-4]]
minima = [[3.0,2.0], [-2.805118,3.131312], [-3.779310,-3.283186], [3.584428,-1.848126]]

# Run optimization
results = []
for x0 in test_points
    x_opt, hist = gradient_descent(himmelblau, x0; lr=0.01)
    push!(results, (x0, x_opt, himmelblau(x_opt), hist))
    println("Start: $x0 → Opt: $x_opt, f=$(himmelblau(x_opt))")
end

# Convergence plot
fig = Figure()
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="f(x)", yscale=log10)
for (i,res) in enumerate(results)
    lines!(ax, res[4], label="Start $i")
end
axislegend(ax)
save("D://juliahw//hw11//conv.png", fig)