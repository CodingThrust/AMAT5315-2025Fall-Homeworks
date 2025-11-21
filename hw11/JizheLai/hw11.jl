# hw11.jl

using ForwardDiff
using CairoMakie
using Optimisers
using LinearAlgebra   # for norm
using Statistics      # for mean

println("Loaded packages, starting script...")

############################################################
# Utility: gradient descent from scratch (Part 1)          #
############################################################

function gradient_descent(f, x0; η=0.01, maxiter=10_000, tol=1e-8)
    x = copy(x0)
    values = Float64[]
    push!(values, f(x))

    for k in 1:maxiter
        g = ForwardDiff.gradient(f, x)
        x_new = x .- η .* g
        fx_new = f(x_new)

        push!(values, fx_new)

        if norm(g) < tol || abs(values[end] - values[end-1]) < tol
            x = x_new
            break
        end

        x = x_new
    end

    return x, values
end

# Himmelblau function
himmelblau(v::AbstractVector) = (v[1]^2 + v[2] - 11)^2 + (v[1] + v[2]^2 - 7)^2

function run_part1()
    println("=== Part 1: Himmelblau ===")

    starts = [
        [-4.0,  0.0],
        [ 0.0,  0.0],
        [ 4.0,  4.0],
        [-4.0,  4.0],
        [ 4.0, -4.0],
    ]

    η = 0.01
    maxiter = 20_000

    fig = Figure(size = (800, 500))   # use size instead of resolution
    ax = Axis(fig[1, 1];
              xlabel = "Iteration",
              ylabel = "f(x)",
              yscale = log10,
              title = "Himmelblau – Gradient Descent (η = $η)")

    colors = [:blue, :red, :green, :orange, :purple]

    for (i, x0) in enumerate(starts)
        println("  Starting GD from $x0 ...")
        x_final, vals = gradient_descent(himmelblau, x0;
                                         η = η, maxiter = maxiter)
        println("    Final point: ", x_final)
        println("    Final f: ", himmelblau(x_final))

        iters = 0:(length(vals)-1)
        # 给每条线设置 label，legend 会自动读取
        lines!(ax, iters, vals;
               color = colors[i],
               label = "start = $(x0)")
    end

    # 让 Makie 自己从带 label 的 plots 中生成 legend
    axislegend(ax; position = :rt)

    save("part1_himmelblau_convergence.png", fig)
    println("  Saved plot: part1_himmelblau_convergence.png")
end

########################################################
# Utility: optimiser runner using Optimisers.jl        #
########################################################

function run_optimiser(optim, f, x0; maxiter=2_000)
    x = copy(x0)
    st = Optimisers.setup(optim, x)

    values = Float64[]
    push!(values, f(x))

    for k in 1:maxiter
        g = ForwardDiff.gradient(f, x)
        st, x = Optimisers.update(st, x, g)
        push!(values, f(x))
    end

    return x, values
end

# Booth function
booth(v::AbstractVector) = (v[1] + 2v[2] - 7)^2 + (2v[1] + v[2] - 5)^2

function run_part2()
    println("=== Part 2: Booth optimizer comparison ===")

    x0 = [-5.0, -5.0]
    maxiter = 2_000

    opt_gd       = Descent(0.01)
    opt_momentum = Momentum(0.01, 0.9)
    opt_adam     = Adam(0.1, (0.9, 0.999))

    optimisers = Dict(
        "Gradient Descent" => opt_gd,
        "Momentum"         => opt_momentum,
        "Adam"             => opt_adam
    )

    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1];
              xlabel = "Iteration",
              ylabel = "f(x)",
              yscale = log10,
              title  = "Booth – Optimizer Comparison")

    colors = Dict(
        "Gradient Descent" => :blue,
        "Momentum"         => :red,
        "Adam"             => :green
    )

    for (name, opt) in optimisers
        println("  Running $name ...")
        x_final, vals = run_optimiser(opt, booth, x0; maxiter = maxiter)
        println("    Final point: ", x_final)
        println("    Final f: ", booth(x_final))

        iters = 0:(length(vals)-1)
        lines!(ax, iters, vals;
               color = colors[name],
               label = name)
    end

    axislegend(ax; position = :rt)
    save("part2_booth_optimizer_comparison.png", fig)
    println("  Saved plot: part2_booth_optimizer_comparison.png")
end

########################################################
# Part 3 – Logistic model fitting                      #
########################################################

function logistic(t, θ::AbstractVector)
    K, r, t0 = θ
    K ./ (1 .+ exp.(-r .* (t .- t0)))
end

function logistic_loss(θ, t, Pdata)
    P̂ = logistic(t, θ)
    mean((Pdata .- P̂).^2)
end

function run_part3()
    println("=== Part 3: Logistic fit ===")

    t_data = collect(0:10)
    P_data = [10.0, 15.0, 25.0, 40.0, 60.0, 80.0, 95.0, 105.0, 110.0, 112.0, 113.0]

    θ0 = [120.0, 0.5, 5.0]   # [K, r, t0]
    opt_adam = Adam(0.05, (0.9, 0.999))
    maxiter = 5_000

    fθ(θ) = logistic_loss(θ, t_data, P_data)

    θ = copy(θ0)
    st = Optimisers.setup(opt_adam, θ)

    losses = Float64[]
    push!(losses, fθ(θ))

    for k in 1:maxiter
        g = ForwardDiff.gradient(fθ, θ)
        st, θ = Optimisers.update(st, θ, g)
        push!(losses, fθ(θ))
    end

    K̂, r̂, t0̂ = θ
    println("  Estimated K  ≈ $K̂")
    println("  Estimated r  ≈ $r̂")
    println("  Estimated t0 ≈ $t0̂")
    println("  Final loss   ≈ ", losses[end])

    # loss curve (no legend)
    fig1 = Figure(size = (800, 500))
    ax1 = Axis(fig1[1, 1];
               xlabel = "Iteration",
               ylabel = "Loss (MSE)",
               yscale = log10,
               title  = "Logistic – Adam Loss")
    iters = 0:(length(losses)-1)
    lines!(ax1, iters, losses, color = :purple)
    save("part3_logistic_loss_curve.png", fig1)
    println("  Saved plot: part3_logistic_loss_curve.png")

    # data vs fit (with legend)
    t_fine = range(0, 10, length = 400)
    P_fit = logistic(t_fine, θ)

    fig2 = Figure(size = (800, 500))
    ax2 = Axis(fig2[1, 1];
               xlabel = "t",
               ylabel = "Population",
               title  = "Logistic fit to data")
    scatter!(ax2, t_data, P_data;
             color = :black, markersize = 10, label = "data")
    lines!(ax2, t_fine, P_fit;
           color = :orange, linewidth = 3, label = "fit")
    axislegend(ax2)  # 或 axislegend(ax2; position = :rb)
    save("part3_logistic_fit.png", fig2)
    println("  Saved plot: part3_logistic_fit.png")
end

#########################
# Main entry point      #
#########################

function main()
    println("Entering main() ...")
    CairoMakie.activate!()
    run_part1()
    run_part2()
    run_part3()
    println("All parts finished.")
end

main()