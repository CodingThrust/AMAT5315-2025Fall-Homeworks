using ForwardDiff
using CairoMakie

# -----------------------------
# Common utilities
# -----------------------------

grad(f, x) = ForwardDiff.gradient(f, x)

function gradient_descent(f, x0; α=0.01, max_iter=2000)
    x = copy(x0)
    xs = Vector{Vector{Float64}}(undef, max_iter + 1)
    fs = Vector{Float64}(undef, max_iter + 1)
    xs[1] = copy(x)
    fs[1] = f(x)
    for k in 1:max_iter
        x .= x .- α .* grad(f, x)
        xs[k + 1] = copy(x)
        fs[k + 1] = f(x)
    end
    return xs, fs
end

function momentum_descent(f, x0; α=0.01, β=0.9, max_iter=2000)
    x = copy(x0)
    v = zeros(length(x0))
    xs = Vector{Vector{Float64}}(undef, max_iter + 1)
    fs = Vector{Float64}(undef, max_iter + 1)
    xs[1] = copy(x)
    fs[1] = f(x)
    for k in 1:max_iter
        g = grad(f, x)
        v .= β .* v .+ g
        x .= x .- α .* v
        xs[k + 1] = copy(x)
        fs[k + 1] = f(x)
    end
    return xs, fs
end

function adam(f, x0; α=0.1, β1=0.9, β2=0.999, ϵ=1e-8, max_iter=2000, clamp_fn=nothing)
    x = copy(x0)
    m = zeros(length(x0))
    v = zeros(length(x0))
    xs = Vector{Vector{Float64}}(undef, max_iter + 1)
    fs = Vector{Float64}(undef, max_iter + 1)
    xs[1] = copy(x)
    fs[1] = f(x)
    for k in 1:max_iter
        g = grad(f, x)
        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* (g .^ 2)
        mhat = m ./ (1 - β1^k)
        vhat = v ./ (1 - β2^k)
        x .= x .- α .* mhat ./ (sqrt.(vhat) .+ ϵ)
        if clamp_fn !== nothing
            x .= clamp_fn(x)
        end
        xs[k + 1] = copy(x)
        fs[k + 1] = f(x)
    end
    return xs, fs
end

# -----------------------------
# Problem 1: Himmelblau's function
# -----------------------------

himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

starts = [
    [-4.0, 0.0],
    [0.0, 0.0],
    [-4.0, 4.0],
    [4.0, 4.0]
]

α_himmel = 0.01
max_iter_himmel = 5000
himmel_results = []

fig1 = Figure(resolution=(800, 500))
ax1 = Axis(fig1[1, 1], xlabel="Iteration", ylabel="f(x)", yscale=log10)

for (i, x0) in enumerate(starts)
    xs, fs = gradient_descent(himmelblau, x0; α=α_himmel, max_iter=max_iter_himmel)
    push!(himmel_results, (x0=x0, xfinal=xs[end], ffinal=fs[end]))
    lines!(ax1, 0:max_iter_himmel, fs, label="start $(x0)")
end
axislegend(ax1, position=:rb)
fig1
save("hw11/xiweipan/himmelblau_convergence.png", fig1)

# -----------------------------
# Problem 2: Booth function optimizer comparison
# -----------------------------

booth(x) = (x[1] + 2x[2] - 7)^2 + (2x[1] + x[2] - 5)^2

x0_booth = [-5.0, -5.0]
max_iter_booth = 2000

xs_gd, fs_gd = gradient_descent(booth, x0_booth; α=0.01, max_iter=max_iter_booth)
xs_mom, fs_mom = momentum_descent(booth, x0_booth; α=0.01, β=0.9, max_iter=max_iter_booth)
xs_adam, fs_adam = adam(booth, x0_booth; α=0.1, β1=0.9, β2=0.999, max_iter=max_iter_booth)

fig2 = Figure(resolution=(800, 500))
ax2 = Axis(fig2[1, 1], xlabel="Iteration", ylabel="f(x)", yscale=log10)
lines!(ax2, 0:max_iter_booth, fs_gd, label="GD")
lines!(ax2, 0:max_iter_booth, fs_mom, label="Momentum")
lines!(ax2, 0:max_iter_booth, fs_adam, label="Adam")
axislegend(ax2, position=:rb)
fig2
save("hw11/xiweipan/booth_convergence.png", fig2)

# -----------------------------
# Problem 3: Logistic growth fitting with Adam
# -----------------------------

t_data = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
P_data = [10.0, 15, 25, 40, 60, 80, 95, 105, 110, 112, 113]

function logistic(t, params)
    K, r, t0 = params
    return K ./ (1 .+ exp.(-r .* (t .- t0)))
end

function mse_loss(params)
    pred = logistic(t_data, params)
    return sum((P_data .- pred) .^ 2) / length(P_data)
end

clamp_params(x) = [max(x[1], 1e-6), max(x[2], 1e-6), x[3]]

params0 = [120.0, 0.5, 5.0]
max_iter_logistic = 5000

xs_log, fs_log = adam(mse_loss, params0; α=0.05, β1=0.9, β2=0.999,
    max_iter=max_iter_logistic, clamp_fn=clamp_params)

params_fit = xs_log[end]

fig3 = Figure(resolution=(800, 500))
ax3 = Axis(fig3[1, 1], xlabel="t", ylabel="P(t)")
scatter!(ax3, t_data, P_data, color=:black, label="Data")

# Smooth curve for fitted model

t_plot = range(minimum(t_data), maximum(t_data), length=200)
P_fit = logistic(t_plot, params_fit)
lines!(ax3, t_plot, P_fit, color=:blue, label="Fitted logistic")
axislegend(ax3, position=:rb)
fig3
save("hw11/xiweipan/logistic_fit.png", fig3)

# -----------------------------
# Print results for markdown
# -----------------------------

println("Problem 1: Himmelblau results")
for res in himmel_results
    println("start = $(res.x0), final = $(res.xfinal), f = $(res.ffinal)")
end

println("\nProblem 2: Booth results")
println("GD final = $(xs_gd[end]), f = $(fs_gd[end])")
println("Momentum final = $(xs_mom[end]), f = $(fs_mom[end])")
println("Adam final = $(xs_adam[end]), f = $(fs_adam[end])")

println("\nProblem 3: Logistic fit")
println("params = $(params_fit), loss = $(fs_log[end])")
