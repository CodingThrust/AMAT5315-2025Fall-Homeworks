using ForwardDiff
using CairoMakie
using Statistics
using Optimisers

t_data = collect(0.0:10.0)
P_data = [10.0, 15, 25, 40, 60, 80, 95, 105, 110, 112, 113] .|> Float64

logistic(K, r, t0, t) = K ./ (1 .+ exp.(-r .* (t .- t0)))

function loss_params(θ::AbstractVector)
    K  = θ[1]
    r  = θ[2]
    t0 = θ[3]
    P_pred = logistic(K, r, t0, t_data)
    return mean((P_data .- P_pred).^2)
end

function clamp_params!(θ)
    θ[1] = clamp(θ[1], 1e-3, 1e4)   
    θ[2] = clamp(θ[2], 1e-4, 10.0)  
    θ[3] = clamp(θ[3], -10.0, 20.0) 
    return θ
end

function adam_fit(θ0;
    α = 0.05,
    β1 = 0.9,
    β2 = 0.999,
    iters = 2000
)
    opt = Adam(α, (β1, β2))
    θ = copy(θ0)
    state = Optimisers.setup(opt, θ)

    losses = Float64[]
    θ_hist = Vector{Vector{Float64}}()

    for k in 1:iters
        g = ForwardDiff.gradient(loss_params, θ)
        θ, state = Optimisers.update(opt, state, θ, g)
        clamp_params!(θ)
        push!(losses, loss_params(θ))
        push!(θ_hist, copy(θ))
    end

    return θ, losses, θ_hist
end

θ0 = [120.0, 0.5, 5.0]

θ_opt, loss_hist, θ_hist = adam_fit(θ0; α = 0.05, iters = 2000)

K_opt  = θ_opt[1]
r_opt  = θ_opt[2]
t0_opt = θ_opt[3]

println("结果：")
println("K ≈ $K_opt")
println("r ≈ $r_opt")
println("t0 ≈ $t0_opt")
println("Final MSE ≈ ", loss_params(θ_opt))

t_dense = range(0, 10, length=200)
P_fit = logistic(K_opt, r_opt, t0_opt, t_dense)

fig = Figure(resolution = (800, 400))

ax1 = Axis(fig[1, 1],
    xlabel = "t",
    ylabel = "P(t)",
    title = "Logistic Fit (Adam via Optimisers.jl)"
)
scatter!(ax1, t_data, P_data, markersize = 10)
lines!(ax1, t_dense, P_fit)

ax2 = Axis(fig[1, 2],
    xlabel = "Iteration",
    ylabel = "MSE",
    yscale = log10,
    title = "Loss Convergence (Adam, Optimisers.jl)"
)
lines!(ax2, 1:length(loss_hist), loss_hist)

save("logistic_fit_adam_solution2.png", fig)
fig