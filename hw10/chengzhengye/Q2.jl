using JuMP
using SCIP
using Printf


function build_csp_model!(model::Model, instance_data)

    @variable(model, x[1:10], Bin)
    @constraint(model, sum(x) >= 3)
    @objective(model, Min, sum(x))
end

function configure_scip!(model::Model; tuned::Bool)
    set_optimizer_attribute(model, "display/verblevel", 0)

    if !tuned
        return
    end

    set_optimizer_attribute(model, "presolving/emphasis", "aggressive")

    set_optimizer_attribute(model, "heuristics/emphasis", "aggressive")

    set_optimizer_attribute(model, "separating/emphasis", "fast")

    set_optimizer_attribute(model, "separating/maxroundsroot", 5)
    set_optimizer_attribute(model, "separating/maxcutsroot", 200)

    set_optimizer_attribute(model, "branching/preferbinary", true)
    set_optimizer_attribute(model, "timing/statistictiming", false)
end

function solve_instance(instance_data; tuned::Bool=false)
    model = Model(SCIP.Optimizer)
    configure_scip!(model; tuned=tuned)
    build_csp_model!(model, instance_data)

    t = @elapsed optimize!(model)

    status = termination_status(model)
    obj    = has_values(model) ? objective_value(model) : NaN

    return t, obj, status
end

function run_benchmark(instances)
    baseline_times = Float64[]
    tuned_times    = Float64[]

    println("Running baseline (default SCIP settings)...")
    for inst in instances
        t, obj, status = solve_instance(inst; tuned=false)
        push!(baseline_times, t)
        @printf("  baseline: instance=%s  time=%.3f s  status=%s\n",
                string(inst), t, string(status))
    end

    println("\nRunning tuned settings (方案 A)...")
    for inst in instances
        t, obj, status = solve_instance(inst; tuned=true)
        push!(tuned_times, t)
        @printf("  tuned:    instance=%s  time=%.3f s  status=%s\n",
                string(inst), t, string(status))
    end

    avg_base = mean(baseline_times)
    avg_tune = mean(tuned_times)
    speedup  = avg_base / avg_tune

    println("\nSummary:")
    @printf("  average baseline time = %.3f s\n", avg_base)
    @printf("  average tuned time    = %.3f s\n", avg_tune)
    @printf("  speedup (baseline / tuned) = %.2fx\n", speedup)
end

instances = ["instance1", "instance2", "instance3", "instance4"]

run_benchmark(instances)
