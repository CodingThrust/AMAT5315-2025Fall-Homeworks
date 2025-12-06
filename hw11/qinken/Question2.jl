using CairoMakie
using Optimisers

booth(x) = (x[1] + 2x[2] - 7)^2 + (2x[1] + x[2] - 5)^2

function grad_booth(x)
    g1 = 2 * (x[1] + 2x[2] - 7) * 1 + 2 * (2x[1] + x[2] - 5) * 2
    g2 = 2 * (x[1] + 2x[2] - 7) * 2 + 2 * (2x[1] + x[2] - 5) * 1
    return Float64[g1, g2]
end

function run_with_optimizer(opt, x0; iters = 2000)
    x = copy(x0)
    st = Optimisers.setup(opt, x)
    fvals = Vector{Float64}(undef, iters)

    for k in 1:iters
        g = grad_booth(x)
        x, st = Optimisers.update(opt, st, x, g)
        fvals[k] = booth(x)
    end
    return x, fvals
end

x0 = [-5.0, -5.0]
iters = 2000

opt_gd   = Descent(0.01)
opt_mom  = Momentum(0.01, 0.9)
opt_adam = Adam(0.1, (0.9, 0.999))

x_gd, f_gd        = run_with_optimizer(opt_gd,   x0; iters = iters)
x_mom, f_momentum = run_with_optimizer(opt_mom,  x0; iters = iters)
x_adam, f_adam    = run_with_optimizer(opt_adam, x0; iters = iters)

println("GD final x    = $x_gd,    f(x) = ", booth(x_gd))
println("Momentum x    = $x_mom,   f(x) = ", booth(x_mom))
println("Adam x        = $x_adam,  f(x) = ", booth(x_adam))

fig = Figure(resolution = (800, 500))
ax = Axis(fig[1, 1],
    xlabel = "Iteration",
    ylabel = "log10(f(x))",
    title = "Booth Function - GD vs Momentum vs Adam (Optimisers.jl)"
)

its = 1:iters
lines!(ax, its, log10.(f_gd .+ 1e-12),        label = "GD (α=0.01)")
lines!(ax, its, log10.(f_momentum .+ 1e-12),  label = "Momentum (α=0.01, β=0.9)")
lines!(ax, its, log10.(f_adam .+ 1e-12),      label = "Adam (α=0.1, β1=0.9, β2=0.999)")

axislegend(ax)
save("booth_optimizers_comparison_sol2.png", fig)
fig