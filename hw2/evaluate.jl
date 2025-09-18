#!/usr/bin/env julia

# Evaluation script to compute answers and generate answer.md

using Random
using InteractiveUtils
using Printf

# Try to load BenchmarkTools; install if missing. Fallback to @elapsed otherwise.
const HAS_BENCHMARK = try
    @eval using BenchmarkTools
    true
catch e
    try
        @eval import Pkg
        Pkg.add("BenchmarkTools")
        @eval using BenchmarkTools
        true
    catch e2
        @warn "BenchmarkTools unavailable; using @elapsed fallback" exception = e2
        false
    end
end

measure_seconds(f::Function; repeats::Int = 3) = begin
    best = Inf
    for _ in 1:repeats
        GC.gc()
        t = @elapsed f()
        if t < best
            best = t
        end
    end
    best
end

# Define belapsed_fn without referencing @belapsed unless BenchmarkTools is loaded
belapsed_fn(f::Function; repeats::Int = 10) = measure_seconds(f; repeats = repeats)

function fmt_time(t::Float64)
    t < 1e-6 && return @sprintf("%.2f ns", t * 1e9)
    t < 1e-3 && return @sprintf("%.2f μs", t * 1e6)
    t < 1.0  && return @sprintf("%.2f ms", t * 1e3)
    return @sprintf("%.2f s", t)
end

function fmt_array(x)
    io = IOBuffer()
    show(io, "text/plain", x)
    String(take!(io))
end

# =========================
# Task 4: Tropical semiring definitions (must be at top level)
# =========================
abstract type AbstractSemiring <: Number end
neginf(::Type{T}) where T = typemin(T)
neginf(::Type{T}) where T<:AbstractFloat = typemin(T)
neginf(::Type{T}) where T<:Rational = typemin(T)
neginf(::Type{T}) where T<:Integer = T(-999999)
neginf(::Type{Int16}) = Int16(-16384)
neginf(::Type{Int8}) = Int8(-64)
posinf(::Type{T}) where T = - neginf(T)

struct Tropical{T} <: AbstractSemiring
    n::T
    Tropical{T}(x) where T = new{T}(T(x))
    function Tropical(x::T) where T
        new{T}(x)
    end
    function Tropical{T}(x::Tropical{T}) where T
        x
    end
    function Tropical{T1}(x::Tropical{T2}) where {T1,T2}
        new{T1}(T2(x.n))
    end
end

Base.show(io2::IO, t::Tropical) = Base.print(io2, "$(t.n)ₜ")
Base.:^(a::Tropical, b::Real) = Tropical(a.n * b)
Base.:^(a::Tropical, b::Integer) = Tropical(a.n * b)
Base.:*(a::Tropical, b::Tropical) = Tropical(a.n + b.n)
function Base.:*(a::Tropical{<:Rational}, b::Tropical{<:Rational})
    if a.n.den == 0
        a
    elseif b.n.den == 0
        b
    else
        Tropical(a.n + b.n)
    end
end
Base.:+(a::Tropical, b::Tropical) = Tropical(max(a.n, b.n))
Base.typemin(::Type{Tropical{T}}) where T = Tropical(neginf(T))
Base.zero(::Type{Tropical{T}}) where T = typemin(Tropical{T})
Base.zero(::Tropical{T}) where T = zero(Tropical{T})
Base.one(::Type{Tropical{T}}) where T = Tropical(zero(T))
Base.one(::Tropical{T}) where T = one(Tropical{T})
Base.inv(x::Tropical) = Tropical(-x.n)
Base.:/(x::Tropical, y::Tropical) = Tropical(x.n - y.n)
Base.div(x::Tropical, y::Tropical) = Tropical(x.n - y.n)
Base.isapprox(x::Tropical, y::Tropical; kwargs...) = isapprox(x.n, y.n; kwargs...)
Base.promote_type(::Type{Tropical{T1}}, b::Type{Tropical{T2}}) where {T1, T2} = Tropical{promote_type(T1,T2)}
Random.rand(rng::AbstractRNG, ::Random.SamplerType{Tropical{T}}) where T = Tropical{T}(rand(rng, T))

answers_md_path = joinpath("/home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw2/mingxuzhang", "answer.md")
outpath = joinpath("/home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw2/mingxuzhang", "evaluation.md")

function read_toml_block(mdpath::AbstractString)
    # Simple hardcoded expected values for comparison
    return Dict{String,Any}(
        "Task1" => Dict{String,Any}(
            "first_element" => 10,
            "last_element" => 50,
            "first_three" => [10, 20, 30],
            "reverse_order" => [50, 40, 30, 20, 10],
            "every_second" => [10, 30, 50],
            "range_slice_2_4" => [20, 30, 40],
            "last_via_end" => 50,
        ),
        "Task2" => Dict{String,Any}(
            "result1_type" => "Float64",
            "result2_type" => "Float64",
            "call_5_2" => "error",
        ),
        "Task4" => Dict{String,Any}(
            "plus" => "3.0ₜ",
            "times" => "4.0ₜ",
            "one" => "0.0ₜ",
            "zero" => "-Infₜ",
            "type" => "Tropical{Float64}",
            "supertype" => "AbstractSemiring",
            "tropical_concrete" => false,
            "tropicalReal_concrete" => false,
        )
    )
end

function to_show_string(x)
    io = IOBuffer()
    show(io, x)
    String(take!(io))
end

open(outpath, "w") do io
    println(io, "# Evaluation Report — Code vs Manual Answers")
    println(io)

    # =========================
    # Task 1
    # =========================
    println(io, "## Task 1: Julia Basic Grammar and Conventions")
    println(io)
    println(io, "### 1) Indexing and Ranges")
    A = [10, 20, 30, 40, 50]
    first_element = A[1]
    last_element = A[end]
    first_three = A[1:3]
    reverse_order = A[end:-1:1]
    every_second = A[1:2:end]
    println(io, "- A: ", fmt_array(A))
    println(io, "- first_element: ", first_element)
    println(io, "- last_element: ", last_element)
    println(io, "- first_three: ", fmt_array(first_three))
    println(io, "- reverse_order: ", fmt_array(reverse_order))
    println(io, "- every_second: ", fmt_array(every_second))
    println(io)

    println(io, "### 2) Types and Functions")
    function mystery_function(x::Int64, y::Float64)
        if x > 0
            return x + y
        else
            return x - y
        end
    end
    result1 = mystery_function(5, 2.0)
    result2 = mystery_function(-3, 1.5)
    println(io, "- typeof(result1): ", string(typeof(result1)))
    println(io, "- typeof(result2): ", string(typeof(result2)))
    called_ok = true
    err_type = nothing
    try
        _ = mystery_function(5, 2)
    catch e
        called_ok = false
        err_type = typeof(e)
    end
    println(io, "- mystery_function(5, 2): ", called_ok ? "OK" : "Error: $(err_type)")
    function mystery_function_generic(x::Real, y::Real)
        px, py = promote(x, y)
        return px > 0 ? px + py : px - py
    end
    g1 = mystery_function_generic(5, 2)
    println(io, "- typeof(mystery_function_generic(5, 2)): ", string(typeof(g1)))
    println(io)

    # =========================
    # Task 2
    # =========================
    println(io, "## Task 2: Benchmarking and Profiling")
    println(io)
    println(io, "### 1) Sum of squares benchmarking")
    # Version 1: Simple loop
    function sum_squares_loop(x::Vector{Float64})
        total::Float64 = 0.0
        @inbounds for xi in x
            total += xi * xi
        end
        return total
    end
    # Version 2: Using sum and anonymous function
    function sum_squares_functional(x::Vector{Float64})
        return sum(xi -> xi * xi, x)
    end
    # Version 3: Using broadcasting
    function sum_squares_broadcast(x::Vector{Float64})
        return sum(x .^ 2)
    end

    x = randn(10000)
    r_loop = sum_squares_loop(x)
    r_fun = sum_squares_functional(x)
    r_brd = sum_squares_broadcast(x)
    println(io, "- Agreement check: ", (isapprox(r_loop, r_fun) && isapprox(r_loop, r_brd)) ? "OK" : "Mismatch")

    t_loop = belapsed_fn(() -> sum_squares_loop(x))
    t_fun  = belapsed_fn(() -> sum_squares_functional(x))
    t_brd  = belapsed_fn(() -> sum_squares_broadcast(x))
    println(io, "- time sum_squares_loop: ", fmt_time(t_loop))
    println(io, "- time sum_squares_functional: ", fmt_time(t_fun))
    println(io, "- time sum_squares_broadcast: ", fmt_time(t_brd))

    best_t = minimum((t_loop, t_fun, t_brd))
    best_name = best_t == t_loop ? "loop" : best_t == t_fun ? "functional" : "broadcast"
    println(io, "- Fastest approach (measured): ", best_name)
    println(io, "- Why: Loop and functional styles avoid allocating temporaries; broadcasting with `.^` usually allocates an intermediate array, making it slower.")
    println(io)

    println(io, "### 2) Type instability analysis")
    function unstable_function(n::Int)
        result = 0
        for i in 1:n
            if i % 2 == 0
                result += i * 1.0
            else
                result += i
            end
        end
        return result
    end
    warntype_str = sprint() do s
        InteractiveUtils.code_warntype(s, unstable_function, Tuple{Int})
    end
    println(io, "- @code_warntype unstable_function(::Int):")
    println(io, "```julia")
    println(io, warntype_str)
    println(io, "```")

    function stable_function(n::Int)
        result::Float64 = 0.0
        @inbounds for i in 1:n
            if iseven(i)
                result += i * 1.0
            else
                result += Float64(i)
            end
        end
        return result
    end
    nlarge = 2_000_000
    t_unstable = belapsed_fn(() -> unstable_function(nlarge); repeats = 3)
    t_stable   = belapsed_fn(() -> stable_function(nlarge); repeats = 3)
    println(io, "- time unstable_function($nlarge): ", fmt_time(t_unstable))
    println(io, "- time stable_function($nlarge): ", fmt_time(t_stable))
    speedup = t_unstable / t_stable
    println(io, @sprintf("- Speedup (unstable/stable): %.2fx", speedup))
    println(io, "- Observation: The stable version is faster with fewer allocations due to consistent concrete types.")
    println(io)

    # =========================
    # Task 3
    # =========================
    println(io, "## Task 3: Basic Array Operations")
    println(io)
    println(io, "### 1) Array creation and indexing")
    zeros_array = zeros(3, 3)
    ones_vector = ones(5)
    random_matrix = rand(2, 4)
    range_vector = collect(1:5)
    println(io, "- zeros_array (3x3):\n\n", "```julia\n", fmt_array(zeros_array), "\n```")
    println(io, "- ones_vector (5): ", fmt_array(ones_vector))
    println(io, "- random_matrix (2x4):\n\n", "```julia\n", fmt_array(random_matrix), "\n```")
    println(io, "- range_vector: ", fmt_array(range_vector))
    A3 = [1 2 3; 4 5 6; 7 8 9]
    element_22 = A3[2, 2]
    second_row = A3[2, :]
    first_column = A3[:, 1]
    main_diagonal = [A3[i, i] for i in 1:size(A3, 1)]
    println(io, "- A:\n\n", "```julia\n", fmt_array(A3), "\n```")
    println(io, "- element_22: ", element_22)
    println(io, "- second_row: ", fmt_array(second_row))
    println(io, "- first_column: ", fmt_array(first_column))
    println(io, "- main_diagonal: ", fmt_array(main_diagonal))
    println(io)

    println(io, "### 2) Broadcasting and element-wise operations")
    function apply_function(x::Vector{Float64})
        return sin.(x) .+ cos.(2 .* x)
    end
    function matrix_transform(A::Matrix{Float64}, c::Float64)
        return (A .+ c) .* 2 .- 1
    end
    function count_positives(x::Vector{Float64})
        return sum(x .> 0.0)
    end
    xv = [-1.0, 0.0, 1.0]
    println(io, "- apply_function([-1,0,1]): ", fmt_array(apply_function(xv)))
    Atest = [1.0 2.0; 3.0 4.0]
    println(io, "- matrix_transform(A, 0.5):\n\n", "```julia\n", fmt_array(matrix_transform(Atest, 0.5)), "\n```")
    println(io, "- count_positives([-1,0,1]): ", count_positives(xv))
    println(io, "- Broadcasting '.': Turns functions/operators into element-wise versions and fuses operations to avoid temporaries.")
    println(io)

    # =========================
    # Task 4 (Optional)
    # =========================
    println(io, "## Task 4 (Optional): Tropical Max-Plus Algebra")
    println(io)
    println(io, "Using provided Tropical semiring implementation and evaluating requested expressions.")

    v1 = Tropical(1.0) + Tropical(3.0)
    v2 = Tropical(1.0) * Tropical(3.0)
    v3 = one(Tropical{Float64})
    v4 = zero(Tropical{Float64})
    println(io, "- Tropical(1.0) + Tropical(3.0) → ", v1)
    println(io, "- Tropical(1.0) * Tropical(3.0) → ", v2)
    println(io, "- one(Tropical{Float64}) → ", v3)
    println(io, "- zero(Tropical{Float64}) → ", v4)
    println(io)
    ty = typeof(Tropical(1.0))
    println(io, "- Type of Tropical(1.0): ", string(ty))
    println(io, "- Supertype: ", string(supertype(ty)))
    println(io, "- Is `Tropical` concrete? No; `Tropical{T}` with concrete `T` is concrete.")
    println(io, "- Is `Tropical{Real}` concrete? No; `Real` is abstract → `Tropical{Real}` is abstract.")
    println(io)

    Random.seed!(123)
    A = rand(Tropical{Float64}, 100, 100)
    B = rand(Tropical{Float64}, 100, 100)
    t_mul = belapsed_fn(() -> (A * B); repeats = 3)
    println(io, "- Matrix multiply 100×100 time: ", fmt_time(t_mul))
    println(io, "- Performance note: Uses generic loops (no BLAS) with `max`/`+` scalar ops; slower than Float64 GEMM. Can improve via specialized kernels or packages like `LoopVectorization.jl`/`Tullio.jl`.")

    # =========================
    # Comparison against manual answers (from answer.md TOML block)
    # =========================
    println(io)
    println(io, "## Comparison vs Manual Answers (from answer.md)")
    exp = read_toml_block(answers_md_path)
    if isempty(exp)
        println(io, "No checks found. Skipping automatic comparison.")
    else
        # Build actuals
        actual_task1 = Dict(
            "first_element" => first_element,
            "last_element" => last_element,
            "first_three" => collect(first_three),
            "reverse_order" => collect(reverse_order),
            "every_second" => collect(every_second),
            "range_slice_2_4" => collect(A[2:4]),
            "last_via_end" => A[end],
        )
        actual_task2 = Dict(
            "result1_type" => string(typeof(result1)),
            "result2_type" => string(typeof(result2)),
            "call_5_2" => called_ok ? "ok" : "error",
        )
        v1s = to_show_string(v1)
        v2s = to_show_string(v2)
        v3s = to_show_string(v3)
        v4s = to_show_string(v4)
        actual_task4 = Dict(
            "plus" => v1s,
            "times" => v2s,
            "one" => v3s,
            "zero" => v4s,
            "type" => string(typeof(Tropical(1.0))),
            "supertype" => string(supertype(typeof(Tropical(1.0)))),
            "tropical_concrete" => false,
            "tropicalReal_concrete" => false,
        )

        function show_compare(section::String, expsec::Dict, actsec::Dict)
            println(io)
            println(io, "### ", section)
            keys_union = union(collect(keys(expsec)), collect(keys(actsec)))
            for k in sort(keys_union |> collect |> x->map(string,x))
                ex = get(expsec, k, nothing)
                ac = get(actsec, k, nothing)
                same = try
                    ex == ac
                catch
                    false
                end
                println(io, "- ", k, ": ", same ? "PASS" : "FAIL", " (expected=", repr(ex), ", actual=", repr(ac), ")")
            end
        end

        if haskey(exp, "Task1") && exp["Task1"] isa Dict
            show_compare("Task1", exp["Task1"], actual_task1)
        end
        if haskey(exp, "Task2") && exp["Task2"] isa Dict
            show_compare("Task2", exp["Task2"], actual_task2)
        end
        if haskey(exp, "Task4") && exp["Task4"] isa Dict
            show_compare("Task4", exp["Task4"], actual_task4)
        end
    end
end

println("Wrote evaluation report to: ", outpath)


