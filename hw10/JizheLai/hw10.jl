# hw10.jl
#
# 放在 hw10/JizheLai/hw10.jl
#
# 运行方式（从 hw10 目录）：
#   cd hw10
#   julia --project=example JizheLai/hw10.jl

using JuMP
using SCIP
using Printf
using Test

include(joinpath(@__DIR__, "..", "example", "dataset.jl"))

# ================================================================
# 1. Petersen 最大独立集
# ================================================================

function build_petersen_graph()
    n = 10
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 1),
        (6, 8), (8,10), (10, 7), (7, 9), (9, 6),
        (1, 6), (2, 7), (3, 8), (4, 9), (5,10),
    ]
    return n, edges
end

function solve_max_independent_set_petersen(; optimizer_factory = SCIP.Optimizer,
                                            verbose::Bool = true)
    n, edges = build_petersen_graph()
    model = Model(optimizer_factory)
    verbose || set_silent(model)

    @variable(model, x[1:n], Bin)
    for (u, v) in edges
        @constraint(model, x[u] + x[v] <= 1)
    end
    @objective(model, Max, sum(x))

    optimize!(model)
    status = termination_status(model)
    status == JuMP.MOI.OPTIMAL || error("MIS not optimal, status = $status")

    mis_size = Int(round(objective_value(model)))
    mis_vertices = [i for i in 1:n if value(x[i]) > 0.5]
    return mis_size, mis_vertices, model
end

function demo_petersen_mis()
    mis_size, mis_vertices, _ = solve_max_independent_set_petersen()
    @info "Petersen MIS size = $mis_size, vertices = $mis_vertices"
    @test mis_size == 4
end

# ================================================================
# 2. 0-1 IP 因式分解（用默认 SCIP，无调参逻辑）
# ================================================================

"""
    factor_semiprime_ip(m, n, N; timelimit=30.0, verbose=false)

用 0-1 乘法模型分解 N = p*q。m,n 是 p,q 的 bit 长度。

timelimit 是 SCIP 的总时间限制（秒），可设到 90s。
"""
function factor_semiprime_ip(m::Int, n::Int, N::BigInt;
                             timelimit::Float64 = 30.0,
                             verbose::Bool = false)

    N_int = Int(N)  # 对 24×24 (~48bit) 足够安全

    model = Model(SCIP.Optimizer)
    verbose || set_silent(model)
    set_string_names_on_creation(model, false)

    # 给 SCIP 设 time limit（通过 JuMP 的属性）
    set_optimizer_attribute(model, "limits/time", timelimit)

    @variable(model, p_bits[0:m-1], Bin)
    @variable(model, q_bits[0:n-1], Bin)

    @constraint(model, p_bits[m-1] == 1)
    @constraint(model, q_bits[n-1] == 1)

    # 对称破坏：p ≤ q
    p_expr = @expression(model, sum((1 << i) * p_bits[i] for i in 0:m-1))
    q_expr = @expression(model, sum((1 << j) * q_bits[j] for j in 0:n-1))
    @constraint(model, p_expr <= q_expr)

    # 线性化 z[i,j] = p[i]*q[j]
    @variable(model, z[0:m-1, 0:n-1], Bin)
    for i in 0:m-1, j in 0:n-1
        @constraint(model, z[i,j] <= p_bits[i])
        @constraint(model, z[i,j] <= q_bits[j])
        @constraint(model, z[i,j] >= p_bits[i] + q_bits[j] - 1)
    end

    # 乘积约束：∑ 2^(i+j) z[i,j] = N
    @constraint(model,
        sum((1 << (i + j)) * z[i,j] for i in 0:m-1, j in 0:n-1) == N_int
    )

    @objective(model, Min, 0)

    optimize!(model)
    status = termination_status(model)

    if status != JuMP.MOI.OPTIMAL && status != JuMP.MOI.LOCALLY_SOLVED
        error("0-1 IP factoring did not finish within timelimit=$timelimit s, status=$status")
    end

    p_val = 0
    q_val = 0
    for i in 0:m-1
        p_val += (1 << i) * Int(round(value(p_bits[i])))
    end
    for j in 0:n-1
        q_val += (1 << j) * Int(round(value(q_bits[j])))
    end

    if verbose
        @info "0-1 IP factoring" N p_val q_val prod = (BigInt(p_val) * BigInt(q_val))
    end

    return p_val, q_val
end

"""
    benchmark_factoring_file(filepath; n_instances=3, line_offset=0, timelimit=30.0)

在某个 dataset 文件上对我们的 0-1 IP 因式分解做简单时间测试。
"""
function benchmark_factoring_file(filepath::String;
        n_instances::Int = 3, line_offset::Int = 0, timelimit::Float64 = 30.0)

    lines = open(filepath, "r") do io
        collect(eachline(io))
    end
    nlines = length(lines)
    if line_offset >= nlines
        error("line_offset=$line_offset ≥ nlines=$nlines in $filepath")
    end

    t_new = Float64[]

    n_run = min(n_instances, nlines - line_offset)
    n_run <= 0 && error("No instances to run")

    for k in 1:n_run
        line = strip(lines[line_offset + k])
        isempty(line) && continue
        parts = split(line)
        length(parts) < 3 && continue

        m     = parse(Int,    parts[1])
        nbits = parse(Int,    parts[2])
        N     = parse(BigInt, parts[3])

        @info "Benchmarking line $(line_offset + k): m=$m n=$nbits N=$N"

        tn = @elapsed begin
            p, q = factor_semiprime_ip(m, nbits, N;
                                       timelimit=timelimit,
                                       verbose=false)
            prod = BigInt(p) * BigInt(q)
            if prod == N
                @info "✓ factorization ok: p=$p, q=$q"
            else
                @warn "✗ factorization mismatch: p=$p, q=$q, p*q=$prod, N=$N"
            end
        end
        push!(t_new, tn)
        @info @sprintf("  0-1 IP factoring time = %.3f s", tn)
    end

    avg_new  = sum(t_new) / length(t_new)
    @info "0-1 IP factoring summary" avg_new
    return (avg_new = avg_new,)
end

# ================================================================
# Main：一次性跑 MIS + 12×12~24×24 所有数据的前几行
# ================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running hw10.jl demo under project=example ..."

    # 1. MIS
    demo_petersen_mis()

    # 2. 各 bit 长度数据的 0-1 IP 因式分解
    data_dir = joinpath(@__DIR__, "..", "example", "data")
    fnames = [
        "numbers_12x12.txt",
        "numbers_14x14.txt",
        "numbers_16x16.txt",
        "numbers_18x18.txt",
        "numbers_20x20.txt",
        "numbers_22x22.txt",
        "numbers_24x24.txt",
    ]

    for fname in fnames
        fpath = joinpath(data_dir, fname)
        if !isfile(fpath)
            @warn "Data file $fname not found, skipping"
            continue
        end

        # 对每个文件跑前 2 个实例，24×24 给更高 timelimit
        tl = occursin("24x24", fname) ? 90.0 : 30.0

        try
            @info "=== Benchmarking $fname with timelimit=$tl s ==="
            bench = benchmark_factoring_file(fpath; n_instances=2, timelimit=tl)
            @info "Summary for $fname" bench
        catch err
            @warn "Benchmark on $fname failed" err
        end
    end
end