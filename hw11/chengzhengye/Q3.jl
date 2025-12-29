using ForwardDiff
using CairoMakie
using Statistics

t_data = collect(0.0:10.0)
P_data = [10.0, 15, 25, 40, 60, 80, 95, 105, 110, 112, 113] .|> Float64

logistic(K, r, t0, t) = K ./ (1 .+ exp.(-r .* (t .- t0)))

function loss(θ)
    K  = exp(θ[1])
    r  = exp(θ[2])
    t0 = θ[3]
    P_pred = logistic(K, r, t0, t_data)
    return mean((P_data .- P_pred).^2)
end

function adam_opt(loss_fun, θ0;
    α = 0.05,
    β1 = 0.9,
    β2 = 0.999,
    ϵ = 1e-8,
    iters = 2000
)
    θ = copy(θ0)
    m = zeros(length(θ))
    v = zeros(length(θ))
    losses = Float64[]
    θ_hist = Vector{Vector{Float64}}()

    for t in 1:iters
        g = ForwardDiff.gradient(loss_fun, θ)

        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* (g .^ 2)

        m_hat = m ./ (1 .- β1^t)
        v_hat = v ./ (1 .- β2^t)

        θ .= θ .- α .* m_hat ./ (sqrt.(v_hat) .+ ϵ)

        push!(losses, loss_fun(θ))
        push!(θ_hist, copy(θ))
    end

    return θ, losses, θ_hist
end

θ0 = [log(120.0), log(0.5), 5.0]

θ_opt, loss_hist, θ_hist = adam_opt(loss, θ0; α = 0.05, iters = 2000)

K_opt  = exp(θ_opt[1])
r_opt  = exp(θ_opt[2])
t0_opt = θ_opt[3]

println("方案一结果：")
println("K ≈ $K_opt")
println("r ≈ $r_opt")
println("t0 ≈ $t0_opt")
println("Final MSE ≈ ", loss(θ_opt))

t_dense = range(0, 10, length=200)
P_fit = logistic(K_opt, r_opt, t0_opt, t_dense)

fig = Figure(resolution = (800, 400))

ax1 = Axis(fig[1, 1],
    xlabel = "t",
    ylabel = "P(t)",
    title = "Logistic Fit (Adam from scratch)"
)
scatter!(ax1, t_data, P_data, markersize = 10)
lines!(ax1, t_dense, P_fit)

ax2 = Axis(fig[1, 2],
    xlabel = "Iteration",
    ylabel = "MSE",
    yscale = log10,
    title = "Loss Convergence (Adam)"
)
lines!(ax2, 1:length(loss_hist), loss_hist)

save("logistic_fit_adam_solution1.png", fig)
fig