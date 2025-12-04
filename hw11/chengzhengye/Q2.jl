using CairoMakie

booth(x) = (x[1] + 2x[2] - 7)^2 + (2x[1] + x[2] - 5)^2

function grad_booth(x)
    g1 = 2 * (x[1] + 2x[2] - 7) * 1 + 2 * (2x[1] + x[2] - 5) * 2
    g2 = 2 * (x[1] + 2x[2] - 7) * 2 + 2 * (2x[1] + x[2] - 5) * 1
    return SVector(g1, g2)
end

function run_gd(f, ∇f, x0; α = 0.01, iters = 2000)
    x = SVector{2,Float64}(x0...)
    fvals = Vector{Float64}(undef, iters)
    for k in 1:iters
        g = ∇f(x)
        x = x .- α .* g
        fvals[k] = f(x)
    end
    return x, fvals
end

function run_momentum(f, ∇f, x0; α = 0.01, β = 0.9, iters = 2000)
    x = SVector{2,Float64}(x0...)
    v = zeros(SVector{2,Float64})
    fvals = Vector{Float64}(undef, iters)
    for k in 1:iters
        g = ∇f(x)
        v = β .* v .+ g   
        x = x .- α .* v
        fvals[k] = f(x)
    end
    return x, fvals
end

function run_adam(f, ∇f, x0; α = 0.1, β1 = 0.9, β2 = 0.999, ϵ = 1e-8, iters = 2000)
    x = SVector{2,Float64}(x0...)
    m = zeros(SVector{2,Float64})
    v = zeros(SVector{2,Float64})
    fvals = Vector{Float64}(undef, iters)

    for t in 1:iters
        g = ∇f(x)
        m = β1 .* m .+ (1 - β1) .* g
        v = β2 .* v .+ (1 - β2) .* (g .^ 2)
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)
        x = x .- α .* m̂ ./ (sqrt.(v̂) .+ ϵ)
        fvals[t] = f(x)
    end
    return x, fvals
end


x0 = (-5.0, -5.0)
iters = 2000

x_gd, f_gd         = run_gd(booth, grad_booth, x0; α = 0.01, iters = iters)
x_mom, f_momentum  = run_momentum(booth, grad_booth, x0; α = 0.01, β = 0.9, iters = iters)
x_adam, f_adam     = run_adam(booth, grad_booth, x0; α = 0.1, β1 = 0.9, β2 = 0.999, iters = iters)

println("Gradient Descent final x = $x_gd,   f(x) = ", booth(x_gd))
println("Momentum        final x = $x_mom,  f(x) = ", booth(x_mom))
println("Adam            final x = $x_adam, f(x) = ", booth(x_adam))


fig = Figure(resolution = (800, 500))
ax = Axis(fig[1, 1],
    xlabel = "Iteration",
    ylabel = "log10(f(x))",
    title = "Booth Function - Optimizer Comparison"
)

its = 1:iters
lines!(ax, its, log10.(f_gd .+ 1e-12),        label = "GD (α=0.01)")
lines!(ax, its, log10.(f_momentum .+ 1e-12),  label = "Momentum (α=0.01, β=0.9)")
lines!(ax, its, log10.(f_adam .+ 1e-12),      label = "Adam (α=0.1, β1=0.9, β2=0.999)")

axislegend(ax)
save("booth_optimizers_comparison_sol1.png", fig)
fig