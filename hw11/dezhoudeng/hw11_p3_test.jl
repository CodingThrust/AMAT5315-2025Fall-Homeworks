using Optimisers
using ForwardDiff
using Random
using Plots

# Data
t_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
P_data = [10, 15, 25, 40, 60, 80, 95, 105, 110, 112, 113]

# Improved model - Added numerical stability protection
function model(t, K, r, t0)
    # Prevent numerical overflow
    exponent = -r * (t - t0)
    # Limit exponent range to prevent exp from producing infinities
    exponent = clamp(exponent, -50.0, 50.0)
    return K / (1.0 + exp(exponent))
end

# Loss function
function loss(p)
    K, r, t0 = p
    total = 0.0
    for i in 1:length(t_data)
        pred = model(t_data[i], K, r, t0)
        total += (P_data[i] - pred)^2
    end
    return total / length(t_data)
end

# Single-start Adam optimization - Removed all clamp constraints
function adam_optimize(p_init; learning_rate=0.02, iterations=50000)  # Reduced learning rate
    p = copy(p_init)
    opt = Optimisers.setup(Optimisers.Adam(learning_rate), p)
    
    best_loss = Inf
    best_p = copy(p)
    
    println("Starting Adam optimization (no parameter constraints)...")
    
    for i in 1:iterations
        grad = ForwardDiff.gradient(loss, p)
        opt, p = Optimisers.update(opt, p, grad)
        
        current_loss = loss(p)
        
        # Save best result
        if current_loss < best_loss
            best_loss = current_loss
            best_p = copy(p)
        end
        
        # Print progress
        if i % 10000 == 0
            println("  Iteration $i: Current loss=$(round(current_loss, digits=8))")
            println("          Parameters: K=$(round(p[1], digits=4)), r=$(round(p[2], digits=6)), t0=$(round(p[3], digits=4))")
        end
        
        # Early stopping mechanism
        if i > 20000 && abs(current_loss - best_loss) < 1e-10
            println("  Early stopping: Loss change is minimal")
            break
        end
    end
    
    return best_p
end

# Multi-start random search - Removed all clamp constraints
function multi_start_optimization(n_starts=6; learning_rate=0.02, iterations=50000)
    best_loss = Inf
    best_params = nothing
    all_results = []  # Ensure this variable is defined
    
    Random.seed!(123)
    for i in 1:n_starts
        # Reasonable random initial value ranges
        K_init = rand() * 50 + 100   # Between 100-150
        r_init = rand() * 1.5 + 0.1  # Between 0.1-1.6  
        t0_init = rand() * 6 + 2     # Between 2-8
        
        p_init = [K_init, r_init, t0_init]
        
        println("\n--- Start $i ---")
        println("Random initial values: K=$(round(K_init, digits=2)), r=$(round(r_init, digits=4)), t0=$(round(t0_init, digits=2))")
        
        # Use unconstrained Adam optimization
        p_opt = adam_optimize(p_init, learning_rate=learning_rate, iterations=iterations)
        current_loss = loss(p_opt)
        
        push!(all_results, (params=p_opt, loss=current_loss, init=p_init))
        
        println("Optimized loss: $(round(current_loss, digits=10))")
        println("Optimized parameters: K=$(round(p_opt[1], digits=4)), r=$(round(p_opt[2], digits=6)), t0=$(round(p_opt[3], digits=4))")
        
        if current_loss < best_loss
            best_loss = current_loss
            best_params = p_opt
        end
    end
    
    # Sort by loss - Add safety check
    if !isempty(all_results)
        sort!(all_results, by=x->x.loss)
        
        println("\n" * "="^60)
        println("Multi-start search results (sorted by loss):")
        
        # Safely handle result display
        num_to_show = min(5, length(all_results))
        for i in 1:num_to_show
            result = all_results[i]
            println("$i. Loss: $(round(result.loss, digits=10)), K=$(round(result.params[1], digits=4)), r=$(round(result.params[2], digits=6)), t0=$(round(result.params[3], digits=4))")
        end
    else
        println("\nWarning: No valid results found")
    end
    
    return best_params, all_results
end

# Calculate fitting errors
function calculate_errors(p)
    K, r, t0 = p
    err_sum = 0.0
    errors = Float64[]
    
    for i in 1:length(t_data)
        pred = model(t_data[i], K, r, t0)
        err = abs(P_data[i] - pred)
        push!(errors, err)
        err_sum += err
    end
    
    avg_err = err_sum / length(t_data)
    return errors, err_sum, avg_err
end

# Main program
println("Logistic Growth Model Auto-Fitting (No Parameter Constraints)")
println("-"^50)

# Use multi-start random search
println("Starting multi-start random search (unconstrained optimization)...")
best_params, all_results = multi_start_optimization(4, learning_rate=0.015, iterations=30000)

# Check if results are valid
if best_params === nothing
    error("Optimization failed, no valid parameters found")
end

K, r, t0 = best_params
final_loss = loss(best_params)
println("\n" * "="^60)
println("Final Best Result:")
println("K=$(round(K, digits=4)), r=$(round(r, digits=8)), t0=$(round(t0, digits=4))")
println("Best loss: $(round(final_loss, digits=12))")

# Calculate detailed errors
errors, err_sum, avg_err = calculate_errors(best_params)

println("\nFitting Result Details:")
for i in 1:length(t_data)
    pred = model(t_data[i], K, r, t0)
    rel_err = abs(P_data[i] - pred) / P_data[i] * 100
    println("t=$(t_data[i]): Actual=$(P_data[i]), Predicted=$(round(pred, digits=6)), Absolute Error=$(round(errors[i], digits=6)), Relative Error=$(round(rel_err, digits=4))%")
end
println("Mean Absolute Error: $(round(avg_err, digits=6))")

# Plotting - Detailed title version
t_fit = 0:0.1:15
P_fit = [model(t, K, r, t0) for t in t_fit]

detailed_title = "Logistic Growth Model Fit (Unconstrained Optimization)\nK=$(round(K, digits=2)), r=$(round(r, digits=4)), t0=$(round(t0, digits=2))\nLoss: $(round(final_loss, digits=6))"

p1 = plot(t_data, P_data, 
     seriestype=:scatter, 
     label="Data", 
     markersize=6,
     xlabel="Time t", 
     ylabel="Population P",
     title=detailed_title,
     titlefontsize=10,
     legend=:bottomright)

plot!(p1, t_fit, P_fit, 
      label="Fitted Curve", 
      linewidth=2,
      color=:red)

savefig(p1, "logistic_fit_no_constraints.png")
println("\nUnconstrained optimization plot saved as logistic_fit_no_constraints.png")