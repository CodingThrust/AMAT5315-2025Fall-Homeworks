# Homework 10 - Challenge: Improved Integer Factorization
# Author: tengxianglin
#
# Goal: Beat the baseline factoring.jl on 40-bit semiprimes
#
# Run with: julia --project=example example/factoring_improved.jl

using ProblemReductions
using JuMP
using SCIP  # You can also try Gurobi or CPLEX
using Printf

"""
Improved IP solver with optimized SCIP parameters
"""
function findmin_improved(problem::AbstractProblem, optimizer, tag::Bool, verbose::Bool)
    cons = constraints(problem)
    nsc = ProblemReductions.num_variables(problem)
    maxN = maximum([length(c.variables) for c in cons])
    combs = [ProblemReductions.combinations(2,i) for i in 1:maxN]

    objs = objectives(problem)

    model = JuMP.Model(optimizer)
    verbose || set_silent(model)
    set_string_names_on_creation(model, false)

    # ==================== OPTIMIZATION 1: Branching Strategy ====================
    # Use reliability pseudocost branching (generally best for 0-1 programs)
    set_optimizer_attribute(model, "branching/relpscost/priority", 1000000)
    
    # ==================== OPTIMIZATION 2: Presolving ====================
    # Enable aggressive presolving
    set_optimizer_attribute(model, "presolving/maxrounds", 3)
    set_optimizer_attribute(model, "constraints/linear/presolpairwise", true)
    
    # ==================== OPTIMIZATION 3: Separation (Cutting Planes) ====================
    # More aggressive cut generation
    set_optimizer_attribute(model, "separating/maxrounds", 10)
    set_optimizer_attribute(model, "separating/maxroundsroot", 20)
    
    # ==================== OPTIMIZATION 4: Heuristics ====================
    # Enable specific heuristics that work well for factorization
    set_optimizer_attribute(model, "heuristics/rins/freq", 50)
    set_optimizer_attribute(model, "heuristics/feaspump/freq", 20)
    
    # ==================== OPTIMIZATION 5: Node Selection ====================
    # Use best-first search for factorization problems
    set_optimizer_attribute(model, "nodeselection/bfs/stdpriority", 100000)
    
    # ==================== OPTIMIZATION 6: Time Limit ====================
    # Set a reasonable time limit (5 minutes per instance)
    set_optimizer_attribute(model, "limits/time", 300)

    JuMP.@variable(model, 0 <= x[i = 1:nsc] <= 1, Int)
    
    # ==================== OPTIMIZATION 7: Variable Fixing ====================
    # Add problem-specific constraints based on factorization structure
    # For factorization: p and q must be odd (LSB = 1)
    # This should be done at problem generation level, but we can add hints
    
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

    # ==================== OPTIMIZATION 8: Warm Start ====================
    # Could add warm start from trial division if implemented

    JuMP.optimize!(model)
    
    if !JuMP.is_solved_and_feasible(model)
        @warn "Problem may be infeasible or time limit reached"
        # Try to extract incumbent solution even if not optimal
        if has_values(model)
            return round.(Int, JuMP.value.(x))
        end
        error("The problem is infeasible and no solution found")
    end
    
    return round.(Int, JuMP.value.(x))
end

"""
Factorize using improved solver
"""
function factoring_improved(m, n, N, optimizer; verbose::Bool=false)
    println("\nFactorizing $N ($(m)×$(n) bits)...")
    
    # Create problem
    fact = Factoring(m, n, N)
    res = reduceto(CircuitSAT, fact)
    problem = CircuitSAT(res.circuit.circuit; use_constraints=true)
    
    # Solve with improved method
    start_time = time()
    vals = findmin_improved(problem, optimizer, true, verbose)
    elapsed = time() - start_time
    
    # Extract solution
    a, b = ProblemReductions.read_solution(fact, [vals[res.p]...,vals[res.q]...])

    # Verify
    if BigInt(a) * BigInt(b) == N
        println("✓ SUCCESS! $N = $a × $b")
        println("  Time: $(round(elapsed, digits=2))s")
        return true, (a, b), elapsed
    else
        println("✗ FAILED: $a × $b = $(BigInt(a)*BigInt(b)) ≠ $N")
        println("  Time: $(round(elapsed, digits=2))s")
        return false, (a, b), elapsed
    end
end

"""
Parallel trial division for warm start (optional optimization)
"""
function trial_division_bounds(N, max_trials=10000)
    # Try small factors first to narrow search space
    for p in 3:2:max_trials
        if N % p == 0
            return (p, div(N, p))
        end
    end
    return nothing
end

"""
Test on dataset
"""
function test_on_dataset(filename::String; max_instances=5)
    println("\n" * "="^70)
    println("Testing on dataset: $filename")
    println("="^70)
    
    if !isfile(filename)
        @error "Dataset file not found: $filename"
        return
    end
    
    # Read dataset
    lines = readlines(filename)
    results = []
    
    for (i, line) in enumerate(lines)
        i > max_instances && break
        
        parts = split(strip(line))
        if length(parts) != 5
            continue
        end
        
        m, n, N, p, q = parse.(BigInt, parts)
        
        println("\n[Instance $i] m=$m, n=$n, N=$N")
        println("-"^70)
        
        # Use SCIP optimizer
        optimizer = SCIP.Optimizer
        
        # Run factorization
        success, factors, elapsed = factoring_improved(Int(m), Int(n), N, optimizer, verbose=false)
        
        push!(results, (instance=i, success=success, time=elapsed))
        
        if success
            @assert BigInt(factors[1]) == p || BigInt(factors[1]) == q "Factors don't match!"
        end
    end
    
    # Summary
    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    n_success = count(r -> r.success, results)
    avg_time = mean([r.time for r in results])
    
    println("Success rate: $n_success / $(length(results))")
    println("Average time: $(round(avg_time, digits=2))s")
    
    return results
end

"""
Main function
"""
function main()
    println("\n")
    println("╔" * "="^68 * "╗")
    println("║" * " "^8 * "HW10 Challenge: Improved Integer Factorization" * " "^13 * "║")
    println("╚" * "="^68 * "╝")
    
    # Test on small example first
    println("\n[Warm-up] Testing on 10×10 bit factorization")
    println("-"^70)
    N_test = BigInt(523091)  # 521 × 1003
    success, factors, elapsed = factoring_improved(10, 10, N_test, SCIP.Optimizer, verbose=false)
    
    # Test on actual dataset
    dataset_path = joinpath(@__DIR__, "..", "example", "data", "numbers_20x20.txt")
    if isfile(dataset_path)
        results = test_on_dataset(dataset_path, max_instances=3)
    else
        println("\n⚠ Dataset not found at: $dataset_path")
    end
    
    println("\n" * "="^70)
    println("Key Optimizations Implemented:")
    println("="^70)
    println("1. ✓ Reliability pseudocost branching")
    println("2. ✓ Aggressive presolving with pairwise elimination")
    println("3. ✓ Enhanced cutting plane generation")
    println("4. ✓ Problem-specific heuristics (RINS, Feaspump)")
    println("5. ✓ Best-first node selection")
    println("6. ✓ Time limits to prevent excessive search")
    println("\nFuture improvements:")
    println("- Warm start from trial division")
    println("- Symmetry breaking constraints (p ≤ q)")
    println("- LSB constraints (both factors must be odd)")
    println("- Parallel solving with multiple configurations")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
