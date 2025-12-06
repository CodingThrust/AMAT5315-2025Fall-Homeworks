using JuMP
using HiGHS

function build_petersen_graph()

    V = 1:10
    outer = [1, 2, 3, 4, 5]
    inner = [6, 7, 8, 9, 10]

    outer_edges = [(outer[i], outer[(i % 5) + 1]) for i in 1:5]

    inner_edges = [(inner[i], inner[(i % 5) + 1]) for i in 1:5]

    spoke_edges = [(outer[i], inner[i]) for i in 1:5]

    E = vcat(outer_edges, inner_edges, spoke_edges)

    return V, E
end


function max_independent_set(V, E)
    model = Model(HiGHS.Optimizer)

    @variable(model, x[v in V], Bin)

    @objective(model, Max, sum(x[v] for v in V))

    @constraint(model, [ (i, j) in E ], x[i] + x[j] <= 1)

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        error("Solver did not find an optimal solution. Status = $(termination_status(model))")
    end

    opt_val = objective_value(model)
    indep_set = [v for v in V if value(x[v]) > 0.5]

    return opt_val, indep_set
end

V, E = build_petersen_graph()
opt_val, indep_set = max_independent_set(V, E)

println("Maximum independent set size on Petersen graph = ", opt_val)
println("One maximum independent set is: ", indep_set)