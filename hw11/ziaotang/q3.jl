using Plots

# Data
t = [0,1,2,3,4,5,6,7,8,9,10]
P = [10,15,25,40,60,80,95,105,110,112,113]
n = length(t)

# Logistic model and loss function
P_model(K, r, t0, t) = K / (1 + exp(-r*(t - t0)))
loss(K, r, t0) = sum((P[i] - P_model(K, r, t0, t[i]))^2 for i in 1:n) / n

# Gradient via automatic differentiation
function ∇loss(K, r, t0, ϵ=1e-6)
    ∂L_∂K = (loss(K+ϵ, r, t0) - loss(K-ϵ, r, t0)) / (2ϵ)
    ∂L_∂r = (loss(K, r+ϵ, t0) - loss(K, r-ϵ, t0)) / (2ϵ)
    ∂L_∂t0 = (loss(K, r, t0+ϵ) - loss(K, r, t0-ϵ)) / (2ϵ)
    return [∂L_∂K, ∂L_∂r, ∂L_∂t0]
end

# Adam optimizer
function adam_optimize(K0, r0, t00; α=0.05, β1=0.9, β2=0.999, ϵ=1e-8, iterations=5000)
    θ = [K0, r0, t00]
    m, v = zeros(3), zeros(3)
    
    for t in 1:iterations
        g = ∇loss(θ[1], θ[2], θ[3])
        m = β1 .* m + (1 - β1) .* g
        v = β2 .* v + (1 - β2) .* g.^2
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)
        θ .-= α .* m̂ ./ (sqrt.(v̂) .+ ϵ)
        
        # Apply bounds
        θ[1] = max(θ[1], 100)  # K ≥ 100
        θ[2] = max(θ[2], 0.1)  # r ≥ 0.1
    end
    return θ
end

# Optimize
K_opt, r_opt, t0_opt = adam_optimize(120, 0.5, 5)

# Results
println("Optimal parameters: K = $(round(K_opt, digits=3)), r = $(round(r_opt, digits=3)), t0 = $(round(t0_opt, digits=3))")
println("Final loss: $(round(loss(K_opt, r_opt, t0_opt), digits=6))")

# Plot
t_plot = 0:0.1:10
P_fit = [P_model(K_opt, r_opt, t0_opt, ti) for ti in t_plot]

plot(t_plot, P_fit, label="Fitted curve", linewidth=2)
scatter!(t, P, label="Data points", markersize=4)
xlabel!("Time")
ylabel!("Population")
title!("Logistic Growth Model Fit")