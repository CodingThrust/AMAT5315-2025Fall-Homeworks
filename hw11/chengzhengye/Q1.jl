using ForwardDiff
using CairoMakie

himmelblau(xy::AbstractVector) = (xy[1]^2 + xy[2] - 11)^2 + (xy[1] + xy[2]^2 - 7)^2

function gradient_descent(
    f,
    x0::AbstractVector;
    η::Float64 = 0.01,
    maxiter::Int = 5000,
    tol::Float64 = 1e-6
)
    x = copy(x0)
    xs = [copy(x0)]
    fvals = [f(x0)]

    for k in 1:maxiter
        g = ForwardDiff.gradient(f, x)
        x -= η .* g
        push!(xs, copy(x))
        push!(fvals, f(x))
        if norm(g) < tol
            println("Converged at iter = $k, ∥∇f∥ = $(norm(g))")
            break
        end
    end

    return xs, fvals
end

starts = [
    [-3.0, 3.0],
    [3.0, 3.0],
    [-3.0, -3.0],
    [3.0, -3.0]
]

η = 0.01
maxiter = 2000

results = Dict{Vector{Float64}, Tuple{Vector{Vector{Float64}}, Vector{Float64}}}()

for s in starts
    xs, fvals = gradient_descent(himmelblau, s; η=η, maxiter=maxiter)
    results[s] = (xs, fvals)
    x_final = xs[end]
    println("Start = $s  →  final x = $x_final, f(x) = ", himmelblau(x_final))
end

f = Figure(resolution = (800, 500))
ax = Axis(f[1, 1],
    xlabel = "Iteration",
    ylabel = "f(x)",
    title = "Gradient Descent on Himmelblau's Function"
)

for (s, (xs, fvals)) in results
    lines!(ax, 0:length(fvals)-1, fvals, label = "start = $(s)")
end

axislegend(ax)
current_figure()
save("himmelblau_gd_convergence_solution1.png", f)