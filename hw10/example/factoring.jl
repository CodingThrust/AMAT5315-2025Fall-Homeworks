# This script is contributed by Zhongyi Ni
using ProblemReductions
using JuMP
using Test

function findmin(problem::AbstractProblem, optimizer, tag::Bool, verbose::Bool)
    cons = constraints(problem)
    nsc = ProblemReductions.num_variables(problem)
    maxN = maximum([length(c.variables) for c in cons])
    combs = [ProblemReductions.combinations(2,i) for i in 1:maxN]

    objs = objectives(problem)

    model = JuMP.Model(optimizer)
    verbose || set_silent(model)
    set_string_names_on_creation(model, false)

    JuMP.@variable(model, 0 <= x[i = 1:nsc] <= 1, Int)
    
    for con in cons
        f_vec = findall(!,con.specification)
        num_vars = length(con.variables)
        for f in f_vec
            JuMP.@constraint(model, sum(j-> iszero(combs[num_vars][f][j]) ? (1 - x[con.variables[j]]) : x[con.variables[j]], 1:num_vars) <= num_vars -1)
        end
    end
    if isempty(objs)
        JuMP.@objective(model,  Min, 0)
    else
        obj_sum = sum(objs) do obj
            (1-x[obj.variables[1]])*obj.specification[1] + x[obj.variables[1]]*obj.specification[2]
        end
        tag ? JuMP.@objective(model,  Min, obj_sum) : JuMP.@objective(model,  Max, obj_sum)
    end

    JuMP.optimize!(model)
    @assert JuMP.is_solved_and_feasible(model) "The problem is infeasible"
    return round.(Int, JuMP.value.(x))
end

"""
    factoring(m, n, N, optimizer; verbose::Bool=false)

Factorize the number N using the method of Integer Programming.

# Arguments
- `m`: the number of bit length of the first prime
- `n`: the number of bit length of the second prime
- `N`: the semiprime number to factor
- `optimizer`: the optimizer to use

# Keyword Arguments
- `verbose`: whether to print the verbose output of the IP solver

# Available Optimizers <: MathOptInterface.AbstractOptimizer
- SCIP.Optimizer (Open Source)
- HiGHS.Optimizer (Open Source)
- Gurobi.Optimizer (Need License)
- CPLEX.Optimizer (Need License)
"""
function factoring(m, n, N, optimizer; verbose::Bool=false)
    fact3 = Factoring(m, n, N)
    res3 = reduceto(CircuitSAT, fact3)
    problem = CircuitSAT(res3.circuit.circuit; use_constraints=true)
    vals = findmin(problem, optimizer, true, verbose)
    a, b = ProblemReductions.read_solution(fact3, [vals[res3.p]...,vals[res3.q]...])

    if BigInt(a) * BigInt(b) == N
        @info "✓ Factorization successful! $N = $a * $b"
        return true, (a, b)
    else
        @error "✗ Factorization failed: a*b = $(BigInt(a)*BigInt(b)) != N = $N"
        return false, (a, b)
    end
end
