#!/usr/bin/env julia

using JuMP
using HiGHS
import MathOptInterface as MOI
using Random
using Printf
using Dates

# ---------------- Q1: MIS on Petersen graph via IP ----------------
# Petersen graph: 10 vertices; standard adjacency
function petersen_adj()
    # Label outer 0..4, inner 5..9; edges: outer cycle (i,i+1), inner star (i,i+2), spokes (i, i+5)
    adj = [Int[] for _ in 1:10]
    # outer cycle 0..4 -> vertices 1..5
    for i in 0:4
        u = i + 1
        v = ((i + 1) % 5) + 1
        push!(adj[u], v); push!(adj[v], u)
    end
    # spokes
    for i in 0:4
        u = i + 1
        v = 5 + i + 1  # 6..10
        push!(adj[u], v); push!(adj[v], u)
    end
    # inner star edges (i -> i+2 mod 5) between 5..9 -> vertices 6..10
    for i in 0:4
        u = 5 + i + 1
        v = 5 + ((i + 2) % 5) + 1
        push!(adj[u], v); push!(adj[v], u)
    end
    # dedup
    for v in 1:10
        adj[v] = sort(unique(adj[v]))
    end
    return adj
end

function mis_ip(adj; optimizer=HiGHS.Optimizer, verbose=false)
    n = length(adj)
    model = Model(optimizer)
    verbose || set_silent(model)
    try
        MOI.set(model, MOI.TimeLimitSec(), 30.0)
    catch err
        @warn "Failed to set time limit: $err"
    end
    @variable(model, x[1:n], Bin)
    for v in 1:n
        for u in adj[v]
            if u > v
                @constraint(model, x[v] + x[u] <= 1)
            end
        end
    end
    @objective(model, Max, sum(x))
    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL
    xsol = value.(x)
    S = [i for i in 1:n if round(Int, xsol[i]) == 1]
    return objective_value(model), S
end

function run_q1()
    println("--- Q1: MIS on Petersen graph via IP ---")
    adj = petersen_adj()
    z, S = mis_ip(adj; optimizer=HiGHS.Optimizer)
    println(@sprintf("Optimal MIS size: %.0f", z))
    println("Vertices in an MIS: ", S)
end

# ---------------- Q2: (Proxy) Tuning harness for IP solver ----------------
# We provide a simple tuning comparison using HiGHS by toggling presolve, as SCIP may not be available.
# If SCIP is installed, users can switch optimizer and set SCIP parameters similarly via MOI.RawParameter.

function with_params(optimizer_ctor; params=Dict{String,Any}())
    # Returns a zero-argument constructor suitable for JuMP.Model
    return () -> begin
        o = optimizer_ctor()
        # Best-effort raw parameter setting if supported by the solver
        for (k, v) in params
            try
                MOI.set(o, MOI.RawOptimizerAttribute(k), v)
            catch err
                @warn "Failed to set parameter $k=$v: $err"
            end
        end
        return o
    end
end

function timed_solve(f; repeats=1)
    t = Inf
    for _ in 1:repeats
        GC.gc()
        t = min(t, @elapsed f())
    end
    return t
end

function tuning_workload()
    # Use a moderately sized random MIS instance as proxy workload
    n = 150; d = 3
    function random_regular_graph(n, d)
        @assert n*d % 2 == 0
        for _ in 1:10_000
            stubs = repeat(collect(1:n), inner=d)
            shuffle!(stubs)
            adj = [Int[] for _ in 1:n]
            valid = true
            for i in 1:2:length(stubs)
                u, v = stubs[i], stubs[i+1]
                if u == v || v in adj[u]
                    valid = false; break
                end
                push!(adj[u], v); push!(adj[v], u)
            end
            if valid && all(length(adj[v]) == d for v in 1:n)
                return adj
            end
        end
        error("failed to generate 3-regular graph")
    end
    return random_regular_graph(n, d)
end

function run_q2()
    println("\n--- Q2: Solver tuning (proxy with HiGHS) ---")
    adj = tuning_workload()

    baseline_opt = with_params(HiGHS.Optimizer; params=Dict("presolve" => "off", "output_flag" => false))
    tuned_opt    = with_params(HiGHS.Optimizer; params=Dict("presolve" => "on",  "output_flag" => false))

    # Build one model instance for timing abstraction: re-create inside closure to apply parameters
    function run_with(opt_ctor)
        # warm-up small model
        mis_ip(petersen_adj(); optimizer=opt_ctor)
        # time actual workload
        t = timed_solve(() -> mis_ip(adj; optimizer=opt_ctor), repeats=1)
        return t
    end

    t_base = run_with(baseline_opt)
    t_tuned = run_with(tuned_opt)

    speedup = t_base / max(t_tuned, 1e-9)
    println(@sprintf("Baseline: %.3fs (presolve=off)", t_base))
    println(@sprintf("Tuned:    %.3fs (presolve=on)",  t_tuned))
    println(@sprintf("Speedup:  %.2fx", speedup))

    open("tuning_results.csv", "w") do io
        println(io, "config,time_seconds")
        println(io, @sprintf("baseline,%.6f", t_base))
        println(io, @sprintf("tuned,%.6f", t_tuned))
        println(io, @sprintf("speedup,%.6f", speedup))
    end
    println("Saved tuning results to tuning_results.csv")
end

# ---------------- Q3: 0-1 ILP factorization (20x20) ----------------
# Long multiplication with binary partial products and integer carries.

function factor_ilp(m::Int, n::Int, N::BigInt; optimizer=HiGHS.Optimizer, verbose=false)
    # Bits are indexed from 0 (LSB). Extract N bits up to m+n.
    total_bits = m + n
    N_bits = [Int((N >> k) & 1) for k in 0:total_bits]

    model = Model(optimizer)
    verbose || set_silent(model)

    @variable(model, p[0:m-1], Bin)
    @variable(model, q[0:n-1], Bin)
    @variable(model, y[0:m-1, 0:n-1], Bin)

    # McCormick linearization for y_ij = p_i * q_j (since binary, these are tight)
    for i in 0:m-1, j in 0:n-1
        @constraint(model, y[i,j] <= p[i])
        @constraint(model, y[i,j] <= q[j])
        @constraint(model, y[i,j] >= p[i] + q[j] - 1)
    end

    # Carry variables
    # Upper bounds for carries: at position k, sum of partials <= min(k+1, m, n)
    ub_c = [min(k+1, m, n) for k in 0:total_bits]
    @variable(model, c[0:total_bits] >= 0, Int)
    for k in 0:total_bits
        set_upper_bound(c[k], ub_c[k+1])
    end

    # c[-1] interpreted as 0; we use c[0] as incoming carry to bit 0.
    @constraint(model, c[0] == 0)

    # Bit-balance constraints: sum_{i+j=k} y[i,j] + c[k] = N_k + 2*c[k+1]
    for k in 0:total_bits-1
        @constraint(model, sum(y[i, k - i] for i in max(0, k-(n-1)) : min(m-1, k)) + c[k] == N_bits[k+1] + 2*c[k+1])
    end

    # Final carry must match MSB beyond total_bits-1
    @constraint(model, c[total_bits] == 0)

    # No objective: feasibility problem
    @objective(model, Min, 0)
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("ILP did not solve to optimality: $status")
    p_bits = [round(Int, value(p[i])) for i in 0:m-1]
    q_bits = [round(Int, value(q[j])) for j in 0:n-1]

    # Convert bits (LSB-first) to integers
    function bits_to_int(bits)
        x = BigInt(0)
        for (k, b) in enumerate(bits)
            x += BigInt(b) << (k-1)
        end
        return x
    end
    a = bits_to_int(p_bits)
    b = bits_to_int(q_bits)
    return a, b
end

function run_q3(; max_lines::Int=3)
    println("\n--- Q3: 0-1 ILP factorization on numbers_12x12.txt (demo) ---")
    data_path = joinpath(@__DIR__, "..", "example", "data", "numbers_12x12.txt")
    open(data_path, "r") do io
        println(@sprintf("%-6s %-6s %-14s %-14s", "m", "n", "N", "check"))
        for (line_num, line) in enumerate(eachline(io))
            line = strip(line); isempty(line) && continue
            parts = split(line)
            m = parse(Int, parts[1])
            n = parse(Int, parts[2])
            N = parse(BigInt, parts[3])
            t0 = time()
            a, b = factor_ilp(m, n, N; optimizer=HiGHS.Optimizer)
            dt = time() - t0
            ok = (a*b == N)
            println(@sprintf("%-6d %-6d %-14s %s (%.2fs)", m, n, string(N), ok ? "OK" : "FAIL", dt))
            if line_num >= max_lines
                break
            end
        end
    end
end

function main()
    run_q1()
    run_q2()
    run_q3()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
