using JuMP
using SCIP
using Statistics

const V = Set(1:10)
const E = Set([
    1=>2, 2=>3, 3=>4, 4=>5, 5=>1,
    6=>7, 7=>8, 8=>9, 9=>10, 10=>6,
    1=>6, 2=>7, 3=>8, 4=>9, 5=>10
])

struct ScipProfile
    heurs::Int
    cutroot::Int
    branchprio::Int
    seed::Int
end

function solve_graph(profile::ScipProfile)
    m = Model(SCIP.Optimizer)
    @variable(m, x[V], Bin)
    @objective(m, Max, sum(x))
    @constraint(m, [e in E], x[first(e)] + x[last(e)] <= 1)


    set_optimizer_attribute(m, "heuristics/alns/freq", profile.heurs)
    set_optimizer_attribute(m, "separating/maxcutsroot", profile.cutroot)
    set_optimizer_attribute(m, "nodeselection/conflict/stdpriority", profile.branchprio)
    set_optimizer_attribute(m, "randomization/randomseedshift", profile.seed + 100)
    set_optimizer_attribute(m, "limits/time", 300.0)

    optimize!(m)
    return objective_value(m), [v for v in V if value(x[v]) > 0.5]
end

baseline = ScipProfile(3, 30, 100, 1)
tuned    = ScipProfile(20, 180, 25000, 9999)

t_default = @elapsed obj1, set1 = solve_graph(baseline)
t_tuned   = @elapsed obj2, set2 = solve_graph(tuned)

println("\nBaseline: value=$obj1, set=$set1, time=$t_default")
println("Tuned:   value=$obj2, set=$set2, time=$t_tuned")
println("Speedup ≈ ", t_default/t_tuned, " x")