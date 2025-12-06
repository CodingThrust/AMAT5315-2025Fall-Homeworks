using ForwardDiff
using CairoMakie

himmelblau_xy(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
himmelblau(v::AbstractVector) = himmelblau_xy(v[1], v[2])

struct GDConfig
    η0::Float64 
    decay::Float64
    decay_step::Int 
    maxiter::Int
    tol::Float64
end

struct GDResult
    start::Vector{Float64}
    traj::Vector{Vector{Float64}}
    fvals::Vector{Float64}
end

function run_gd(
    f,
    x0::AbstractVector,
    cfg::GDConfig
)::GDResult
    x = copy(x0)
    traj = Vector{Vector{Float64}}(undef, 0)
    fvals = Float64[]

    η = cfg.η0

    for k in 1:cfg.maxiter
        push!(traj, copy(x))
        push!(fvals, f(x))

        g = ForwardDiff.gradient(f, x)

        if norm(g) < cfg.tol
            println("Converged from $x0 at iter $k with ∥∇f∥ = $(norm(g))")
            break
        end

        if k % cfg.decay_step == 0
            η *= cfg.decay
        end

        x -= η .* g
    end

    push!(traj, copy(x))
    push!(fvals, f(x))

    GDResult(collect(x0), traj, fvals)
end

cfg = GDConfig(0.02, 0.7, 200, 3000, 1e-6)

starts = [
    [-4.0, 4.0],
    [4.0, 4.0],
    [-4.0, -2.0],
    [3.5, -3.5]
]

results = GDResult[]

for s in starts
    res = run_gd(himmelblau, s, cfg)
    push!(results, res)
    x_final = res.traj[end]
    println("Start = $(res.start) → final x = $x_final, f(x) = ", himmelblau(x_final))
end

fig = Figure(resolution = (800, 500))
ax = Axis(fig[1, 1],
    xlabel = "Iteration",
    ylabel = "f(x)",
    title = "Convergence of Gradient Descent on Himmelblau (decayed step size)"
)

for res in results
    its = 0:length(res.fvals)-1
    lines!(ax, its, res.fvals, label = "start = $(res.start)")
end

axislegend(ax)
current_figure()
save("himmelblau_gd_convergence_solution2.png", fig)