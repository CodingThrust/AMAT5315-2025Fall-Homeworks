# This script is contributed by Zhongyi Ni
using ProblemReductions
using JuMP
using SCIP
using Test

function findmin(problem::AbstractProblem,optimizer,tag::Bool)
    cons = constraints(problem)
    nsc = ProblemReductions.num_variables(problem)
    maxN = maximum([length(c.variables) for c in cons])
    combs = [ProblemReductions.combinations(2,i) for i in 1:maxN]

    objs = objectives(problem)

    # IP by JuMP
    model = JuMP.Model(optimizer)

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

function factoring(m,n,N)
    fact3 = Factoring(m, n, N)
    res3 = reduceto(CircuitSAT, fact3)
    problem = CircuitSAT(res3.circuit.circuit; use_constraints=true)
    vals = findmin(problem, SCIP.Optimizer,true)
    return ProblemReductions.read_solution(fact3, [vals[res3.p]...,vals[res3.q]...])
end
@info "Factoring: 1267650600228168602901733704409"
a,b = factoring(50,50,1267650600228168602901733704409)
@info "The factors are $a and $b"
@test BigInt(a)*BigInt(b) == 1267650600228168602901733704409

@info "Factoring: 1146749307995035755805410447651043470398282494584134934736730302757809653488752086493475676497348575363759 (this is expected to stuck!)"
a,b = factoring(175,175,1146749307995035755805410447651043470398282494584134934736730302757809653488752086493475676497348575363759)
@test a*b == 1146749307995035755805410447651043470398282494584134934736730302757809653488752086493475676497348575363759