using JuMP
using SCIP

function petersen_edges()
    return [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), # outer cycle
        (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), # spokes
        (6, 8), (8, 10), (10, 7), (7, 9), (9, 6) # inner star
    ]
end

function solve_maximum_independent_set_petersen(; optimizer=SCIP.Optimizer)
    edges = petersen_edges()
    model = Model(optimizer)
    @variable(model, x[1:10], Bin)
    @objective(model, Max, sum(x))
    for (i, j) in edges
        @constraint(model, x[i] + x[j] <= 1)
    end
    optimize!(model)
    chosen = [i for i in 1:10 if value(x[i]) > 0.5]
    return objective_value(model), chosen
end

function try_set_param!(model::Model, name::String, value)
    try
        set_optimizer_attribute(model, name, value)
    catch err
        @warn "SCIP parameter not available in this build" name value err
    end
    return nothing
end

function apply_scip_tuning!(model::Model)
    params = Dict(
        "display/verblevel" => 0,
        "limits/gap" => 0.0,
        "presolving/maxrounds" => 20,
        "presolving/maxroundsroot" => 15,
        "separating/maxrounds" => 5,
        "separating/maxroundsroot" => 10,
        "separating/maxcuts" => 200,
        "separating/maxcutsroot" => 500,
        "heuristics/feaspump/freq" => 1,
        "heuristics/rins/freq" => 10,
        "heuristics/rounding/freq" => 1,
        "heuristics/pscostdiving/freq" => 5
    )
    for (name, value) in params
        try_set_param!(model, name, value)
    end
    return nothing
end

function scip_tuning_options()
    return Dict(
        "display/verblevel" => 0,
        "limits/gap" => 0.0,
        "presolving/maxrounds" => 20,
        "presolving/maxroundsroot" => 15,
        "separating/maxrounds" => 5,
        "separating/maxroundsroot" => 10,
        "separating/maxcuts" => 200,
        "separating/maxcutsroot" => 500,
        "heuristics/feaspump/freq" => 1,
        "heuristics/rins/freq" => 10,
        "heuristics/rounding/freq" => 1,
        "heuristics/pscostdiving/freq" => 5
    )
end

function filter_supported_scip_options(options::Dict{String, T}) where {T}
    model = Model(SCIP.Optimizer)
    supported = Dict{String, T}()
    for (name, value) in options
        try
            set_optimizer_attribute(model, name, value)
            supported[name] = value
        catch
            # Ignore unsupported parameters to avoid runtime errors.
        end
    end
    return supported
end

function build_model_with_optimizer(build_model, optimizer)
    try
        return build_model(optimizer)
    catch err
        if err isa MethodError
            model = build_model()
            set_optimizer(model, optimizer)
            return model
        end
        rethrow()
    end
end

function solve_crystal_model(build_model; tuned::Bool=false)
    model = build_model_with_optimizer(build_model, SCIP.Optimizer)
    if tuned
        apply_scip_tuning!(model)
    end
    optimize!(model)
    return objective_value(model), termination_status(model)
end

function benchmark_scip_tuning(build_model; repeats::Int=3)
    default_times = Float64[]
    tuned_times = Float64[]
    for _ in 1:repeats
        model = build_model_with_optimizer(build_model, SCIP.Optimizer)
        t = @elapsed optimize!(model)
        push!(default_times, t)
    end
    for _ in 1:repeats
        model = build_model_with_optimizer(build_model, SCIP.Optimizer)
        apply_scip_tuning!(model)
        t = @elapsed optimize!(model)
        push!(tuned_times, t)
    end
    default_mean = sum(default_times) / length(default_times)
    tuned_mean = sum(tuned_times) / length(tuned_times)
    speedup = default_mean / tuned_mean
    return default_mean, tuned_mean, speedup
end

function benchmark_SrTiO3(; repeats::Int=3, use_quadratic_problem::Bool=false)
    default_times = Float64[]
    tuned_times = Float64[]
    for _ in 1:repeats
        t = @elapsed run_SrTiO3_prediction(;
            use_quadratic_problem,
            optimizer=SCIP.Optimizer,
            optimizer_options=Dict(),
            save_figure=false
        )
        push!(default_times, t)
    end
    tuned_opts = scip_tuning_options()
    tuned_opts = filter_supported_scip_options(tuned_opts)
    for _ in 1:repeats
        t = @elapsed run_SrTiO3_prediction(;
            use_quadratic_problem,
            optimizer=SCIP.Optimizer,
            optimizer_options=tuned_opts,
            save_figure=false
        )
        push!(tuned_times, t)
    end
    default_mean = sum(default_times) / length(default_times)
    tuned_mean = sum(tuned_times) / length(tuned_times)
    speedup = default_mean / tuned_mean
    return default_mean, tuned_mean, speedup
end

objective_value, chosen = solve_maximum_independent_set_petersen()
println("Objective value: $objective_value")
println("Chosen vertices: $chosen")