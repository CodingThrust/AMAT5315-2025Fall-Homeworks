using JuMP
using GLPK


vertices = 1:10

edges = [

    (1, 2), (2, 3), (3, 4), (4, 5), (5, 1),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 6),

    (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
]

model = Model(GLPK.Optimizer)

@variable(model, x[v in vertices], Bin)

@objective(model, Max, sum(x[v] for v in vertices))

for (i, j) in edges
    @constraint(model, x[i] + x[j] <= 1)
end

optimize!(model)

term_status = termination_status(model)
primal_status = primal_status(model)

println("Termination status: ", term_status)
println("Primal status: ", primal_status)

if term_status == MOI.OPTIMAL
    max_size = objective_value(model)
    println("Maximum independent set size = ", max_size)

    chosen_vertices = [v for v in vertices if value(x[v]) > 0.5]
    println("One maximum independent set is: ", chosen_vertices)
else
    println("Solver did not find an optimal solution.")
end