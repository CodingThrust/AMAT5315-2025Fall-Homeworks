# Problem 3: Parameter Fitting using Logistic Growth Model
# Mingxu Zhang

using ForwardDiff
using CairoMakie

# Logistic growth model: P(t) = K / (1 + exp(-r * (t - t0)))
function logistic_model(t, params)
    K, r, t0 = params
    return K / (1 + exp(-r * (t - t0)))
end

# Mean Squared Error loss function
function mse_loss(params, t_data, P_data)
    n = length(t_data)
    predictions = [logistic_model(t, params) for t in t_data]
    return sum((P_data .- predictions).^2) / n
end

# Adam optimizer for parameter fitting
function adam_fit(loss_fn, params0, t_data, P_data; 
                  α=0.1, β1=0.9, β2=0.999, ε=1e-8, max_iter=5000, tol=1e-10)
    params = copy(params0)
    m = zeros(length(params0))
    v = zeros(length(params0))
    history = Float64[]
    params_history = [copy(params)]
    
    # Wrapper for gradient computation
    loss_wrapper(p) = loss_fn(p, t_data, P_data)
    
    for t in 1:max_iter
        current_loss = loss_wrapper(params)
        push!(history, current_loss)
        
        # Compute gradient
        grad = ForwardDiff.gradient(loss_wrapper, params)
        
        # Check for convergence
        if sqrt(sum(grad .^ 2)) < tol
            println("Converged at iteration $t")
            break
        end
        
        # Adam update
        m = β1 * m + (1 - β1) * grad
        v = β2 * v + (1 - β2) * (grad .^ 2)
        
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        
        params = params - α * m_hat ./ (sqrt.(v_hat) .+ ε)
        
        # Ensure parameters stay positive and reasonable
        params[1] = max(params[1], 1.0)    # K > 0
        params[2] = max(params[2], 0.01)   # r > 0
        # t0 can be any value
        
        push!(params_history, copy(params))
    end
    
    return params, history, params_history
end

# Given data
t_data = Float64[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
P_data = Float64[10, 15, 25, 40, 60, 80, 95, 105, 110, 112, 113]

# Initial parameter guesses
K0 = 120.0
r0 = 0.5
t0_init = 5.0
params0 = [K0, r0, t0_init]

println("=" ^ 60)
println("Problem 3: Logistic Growth Model Parameter Fitting")
println("=" ^ 60)
println("Model: P(t) = K / (1 + exp(-r * (t - t₀)))")
println("Loss function: MSE = (1/n) * Σ(Pᵢ - P(tᵢ))²")
println("=" ^ 60)

println("\nData points:")
println("t = $t_data")
println("P = $P_data")

println("\nInitial parameters:")
println("  K = $K0 (carrying capacity)")
println("  r = $r0 (growth rate)")
println("  t₀ = $t0_init (inflection point)")
println("  Initial MSE: $(round(mse_loss(params0, t_data, P_data), digits=4))")

# Run Adam optimizer
println("\nRunning Adam optimizer (α=0.1, β₁=0.9, β₂=0.999)...")
optimal_params, loss_history, params_history = adam_fit(
    mse_loss, params0, t_data, P_data,
    α=0.1, β1=0.9, β2=0.999, max_iter=10000
)

K_opt, r_opt, t0_opt = optimal_params

println("\n" * "=" ^ 60)
println("Optimization Results:")
println("=" ^ 60)
println("Optimal parameters:")
println("  K (carrying capacity) = $(round(K_opt, digits=4))")
println("  r (growth rate) = $(round(r_opt, digits=4))")
println("  t₀ (inflection point) = $(round(t0_opt, digits=4))")
println("  Final MSE: $(round(mse_loss(optimal_params, t_data, P_data), digits=6))")

# Calculate R² (coefficient of determination)
P_mean = sum(P_data) / length(P_data)
P_pred = [logistic_model(t, optimal_params) for t in t_data]
SS_res = sum((P_data .- P_pred).^2)
SS_tot = sum((P_data .- P_mean).^2)
R_squared = 1 - SS_res / SS_tot

println("  R² = $(round(R_squared, digits=6))")

# Print predictions vs actual
println("\n" * "=" ^ 60)
println("Predictions vs Actual:")
println("=" ^ 60)
println("  t  |  Actual  |  Predicted  |  Error")
println("-" ^ 45)
for (i, t) in enumerate(t_data)
    pred = logistic_model(t, optimal_params)
    error = P_data[i] - pred
    println("  $(Int(t))  |   $(round(P_data[i], digits=1))    |    $(round(pred, digits=2))    |  $(round(error, digits=2))")
end

# Create visualization
fig = Figure(size=(1200, 800))

# Plot 1: Data points and fitted curve
ax1 = Axis(fig[1, 1],
    xlabel="Time (t)",
    ylabel="Population (P)",
    title="Logistic Growth Model Fitting")

# Plot original data points
scatter!(ax1, t_data, P_data, color=:red, markersize=15, label="Data points")

# Plot fitted curve
t_smooth = collect(range(0, 10, length=100))
P_smooth = [logistic_model(t, optimal_params) for t in t_smooth]
lines!(ax1, t_smooth, P_smooth, color=:blue, linewidth=3, label="Fitted curve")

# Plot initial guess curve
P_initial = [logistic_model(t, params0) for t in t_smooth]
lines!(ax1, t_smooth, P_initial, color=:gray, linewidth=2, linestyle=:dash, label="Initial guess")

# Add horizontal line at K (carrying capacity) - simplified
lines!(ax1, [0.0, 10.0], [K_opt, K_opt], color=:green, linewidth=1, linestyle=:dot, label="K=$(round(K_opt, digits=1))")

axislegend(ax1, position=:rb)

# Plot 2: Loss convergence
ax2 = Axis(fig[1, 2],
    xlabel="Iteration",
    ylabel="MSE Loss (log scale)",
    title="Optimization Convergence",
    yscale=log10)

loss_filtered = max.(loss_history, 1e-20)
lines!(ax2, collect(1:length(loss_filtered)), loss_filtered, color=:purple, linewidth=2)

# Plot 3: Parameter evolution
ax3 = Axis(fig[2, 1],
    xlabel="Iteration",
    ylabel="Parameter Value",
    title="Parameter Evolution During Optimization")

K_history = [p[1] for p in params_history]
r_history = [p[2] for p in params_history]
t0_history = [p[3] for p in params_history]

n_plot = min(1000, length(K_history))
iters = collect(1:n_plot)
lines!(ax3, iters, K_history[1:n_plot], color=:blue, linewidth=2, label="K")
lines!(ax3, iters, r_history[1:n_plot] .* 100, color=:red, linewidth=2, label="r × 100")
lines!(ax3, iters, t0_history[1:n_plot] .* 10, color=:green, linewidth=2, label="t₀ × 10")

axislegend(ax3, position=:rt)

# Plot 4: Residuals
ax4 = Axis(fig[2, 2],
    xlabel="Time (t)",
    ylabel="Residual (Actual - Predicted)",
    title="Residual Plot")

residuals = P_data .- P_pred
scatter!(ax4, t_data, residuals, color=:orange, markersize=12)
# Add zero line
lines!(ax4, [0.0, 10.0], [0.0, 0.0], color=:black, linewidth=1, linestyle=:dash)

save("problem3_fitting.png", fig)
println("\nFitting visualization saved to problem3_fitting.png")

# Final summary
println("\n" * "=" ^ 60)
println("FINAL SUMMARY")
println("=" ^ 60)
println("Fitted Logistic Model: P(t) = $(round(K_opt, digits=2)) / (1 + exp(-$(round(r_opt, digits=4)) × (t - $(round(t0_opt, digits=2)))))")
println("R² = $(round(R_squared, digits=4)) ($(round(R_squared*100, digits=2))% of variance explained)")
println("Final MSE = $(round(mse_loss(optimal_params, t_data, P_data), digits=6))")
