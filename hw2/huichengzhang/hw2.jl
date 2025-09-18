## Task 1: Julia Basic Grammar and Conventions
# Given array
A = [10, 20, 30, 40, 50]

# Fill in the correct indices/expressions:
first_element = A[1]        # Get first element
last_element = A[end]       # Get last element  
first_three = A[1:3]          # Get first three elements
reverse_order = A[end:-1:1]        # Get all elements in reverse order
every_second = A[1:2:end]         # Get every second element (10, 30, 50)

# Answer: 
# i. 1
# ii. A[2:4]
# iii. A[end]

function mystery_function(x::Int64, y::Float64)
    if x > 0
        return x + y
    else
        return x - y
    end
end

result1 = mystery_function(5, 2.0)
result2 = mystery_function(-3, 1.5)

# Answer:
# i. Float64, Float64
# ii. ERROR: MethodError: no method matching mystery_function(::Int64, ::Int64)
# The function `mystery_function` exists, but no method is defined for this combination of argument types.

# Closest candidates are:
#   mystery_function(::Int64, ::Float64)
#    @ Main REPL[13]:1

# Stacktrace:
#  [1] top-level scope
#    @ REPL[16]:1

# iii. 推荐的通用数值版本（不覆盖上面的签名，避免与答案矛盾）
function mystery_function_any(x::Number, y::Number)
    x1, y1 = promote(x, y)
    return x1 > zero(x1) ? x1 + y1 : x1 - y1
end

## Task 2: Benchmarking and Profiling
using BenchmarkTools
using InteractiveUtils

# Version 1: Simple loop
function sum_squares_loop(x::Vector{Float64})
    total = 0.0
    for i in 1:length(x)
        total += x[i]^2
    end
    return total
end

# Version 2: Using sum and anonymous function
function sum_squares_functional(x::Vector{Float64})
    return sum(y -> y^2, x)
end

# Version 3: Using broadcasting
function sum_squares_broadcast(x::Vector{Float64})
    return sum(x .^ 2)
end

# Create test vector
x = randn(10000)

# Benchmark each function
println("Loop version:")
@btime sum_squares_loop($x)
# 8.514 μs (0 allocations: 0 bytes)
# 9896.848205481949

println("Functional version:")
@btime sum_squares_functional($x)
# 1.104 μs (0 allocations: 0 bytes)
# 9896.848205481956

println("Broadcasting version:")
@btime sum_squares_broadcast($x)
# 2.432 μs (3 allocations: 96.06 KiB)
# 9896.848205481958

# Answer:
# iV. functional version (sum(y -> y^2, x)) is the fastest.
# because sum with a generator-like function is highly optimized in Julia. It applies the squaring operation and performs summation in a single pass without creating any intermediate data structures, resulting in zero memory allocation.

## BenchmarkTools 已在上方导入

# Original unstable function
function unstable_function(n::Int)
    result = 0    # starts as Int
    for i in 1:n
        if i % 2 == 0
            result += i * 1.0    # becomes Float64
        else
            result += i          # stays Int
        end
    end
    return result
end

# 1. Use @code_warntype to see type instability
println("Type instability analysis:")
@code_warntype unstable_function(10)
# MethodInstance for unstable_function(::Int64)
#   from unstable_function(n::Int64) @ Main ~/Desktop/Desktop/HKUST/scientific-computing/AMAT5315-2025Fall-Homeworks/hw2/laijizhe/hw2.jl:105
# Arguments
#   #self#::Core.Const(Main.unstable_function)
#   n::Int64
# Locals
#   @_3::Union{Nothing, Tuple{Int64, Int64}}
#   result::Union{Float64, Int64}
#   i::Int64
# Body::Union{Float64, Int64}
# 1 ─       (result = 0)
# │   %2  = (1:n)::Core.PartialStruct(UnitRange{Int64}, Any[Core.Const(1), Int64])
# │         (@_3 = Base.iterate(%2))
# │   %4  = @_3::Union{Nothing, Tuple{Int64, Int64}}
# │   %5  = (%4 === nothing)::Bool
# │   %6  = Base.not_int(%5)::Bool
# └──       goto #7 if not %6
# 2 ┄ %8  = @_3::Tuple{Int64, Int64}
# │         (i = Core.getfield(%8, 1))
# │   %10 = Core.getfield(%8, 2)::Int64
# │   %11 = Main.:(==)::Core.Const(==)
# │   %12 = Main.:%::Core.Const(rem)
# │   %13 = i::Int64
# │   %14 = (%12)(%13, 2)::Int64
# │   %15 = (%11)(%14, 0)::Bool
# └──       goto #4 if not %15
# 3 ─ %17 = Main.:+::Core.Const(+)
# │   %18 = result::Union{Float64, Int64}
# │   %19 = Main.:*::Core.Const(*)
# │   %20 = i::Int64
# │   %21 = (%19)(%20, 1.0)::Float64
# │         (result = (%17)(%18, %21))
# └──       goto #5
# 4 ─ %24 = result::Union{Float64, Int64}
# │   %25 = i::Int64
# └──       (result = %24 + %25)
# 5 ┄       (@_3 = Base.iterate(%2, %10))
# │   %28 = @_3::Union{Nothing, Tuple{Int64, Int64}}
# │   %29 = (%28 === nothing)::Bool
# │   %30 = Base.not_int(%29)::Bool
# └──       goto #7 if not %30
# 6 ─       goto #2
# 7 ┄ %33 = result::Union{Float64, Int64}
# └──       return %33

# 2. Rewrite the function to be type-stable
function stable_function(n::Int)
    result = 0.0  # Start with Float64
    for i in 1:n
        if i % 2 == 0
            result += i * 1.0
        else
            result += i           # Adding Int to Float64 promotes to Float64
        end
    end
    return result
end

# 3. Benchmark both versions with large n
n = 1000000

println("\nBenchmarking unstable version:")
@btime unstable_function($n)
# 5.199 ms (0 allocations: 0 bytes)
# 5.000005e11

println("\nBenchmarking stable version:")
@btime stable_function($n)
# 856.083 μs (0 allocations: 0 bytes)
# 5.000005e11

# the stable function has a concrete return type (Body::Float64) 
# and its local variable result is consistently typed as Float64. 
# This allows the Julia compiler to generate highly optimized, 
# specialized machine code that performs direct floating-point arithmetic without any runtime type checks.

## Task 3: Basic Array Operations

# Create arrays
zeros_array = zeros(3, 3)              # Create 3x3 matrix of zeros
ones_vector = ones(5)                  # Create vector of 5 ones
random_matrix = rand(2, 4)             # Create 2x4 matrix of random numbers (uniformly distributed between 0 and 1)
range_vector = collect(1:5)            # Create vector [1, 2, 3, 4, 5]

# Matrix operations
A = [1 2 3; 4 5 6; 7 8 9]
B = [1 0 1; 0 1 0; 1 0 1]

# Fill in operations:
element_22 = A[2, 2]                   # Get element at row 2, column 2 (returns 5)
second_row = A[2, :]                   # Get entire second row (returns [4, 5, 6])
first_column = A[:, 1]                 # Get entire first column (returns [1, 4, 7])

using LinearAlgebra  # Import the LinearAlgebra module to use `diag`
main_diagonal = diag(A)                # Get main diagonal elements [1, 5, 9]



function apply_function(x::Vector{Float64})
    return sin.(x) .+ cos.(2 .* x)
end

function matrix_transform(A::Matrix{Float64}, c::Float64)
    return (A .+ c) .* 2 .- 1
end

function count_positives(x::Vector{Float64})
    return sum(x .> 0)
end

# Test data
x = [0.0, π/2, π, 3π/2, 2π]  # ≈ [0, 1.57, 3.14, 4.71, 6.28]
A = [1.0 2.0; 3.0 4.0]
c = 1.0

# Test Function 1
println("apply_function(x):")
println(apply_function(x))

# Test Function 2
println("\nmatrix_transform(A, c):")
println(matrix_transform(A, c))

# Test Function 3
println("\nNumber of positives in x:")
println(count_positives(x))

# apply_function(x):

# [1.0, 0.0, 1.0000000000000002, -2.0, 0.9999999999999998]


# matrix_transform(A, c):

# [3.0 5.0; 7.0 9.0]


# Number of positives in x:

# 4

# f.(x) applies function f to each element of x.
# A .+ B adds corresponding elements of A and B, even if their shapes differ (with broadcasting rules).
# Scalars are automatically "expanded" when used with . operators: x .+ 1 adds 1 to every element.
# Dots fuse in loops (via loop fusion), meaning sin.(x) .+ cos.(x) is efficient and runs in one loop.
#
# Sample run (local):
# Loop version:
#   8.167 μs (0 allocations: 0 bytes)
# Functional version:
#   1.065 μs (0 allocations: 0 bytes)
# Broadcasting version:
#   164.463 μs (2 allocations: 78.17 KiB)
# Type instability analysis:
#   Body::Union{Float64, Int64}
#
# Benchmarking unstable version:
#   2.504 ms (0 allocations: 0 bytes)
#
# Benchmarking stable version:
#   841.295 μs (0 allocations: 0 bytes)
# apply_function(x):
# [1.0, 0.0, 1.0000000000000002, -2.0, 0.9999999999999998]
#
# matrix_transform(A, c):
# [3.0 5.0; 7.0 9.0]
#
# Number of positives in x:
# 4
