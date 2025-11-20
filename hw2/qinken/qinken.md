# hw2
## Task 1
### Q1
A = [10, 20, 30, 40, 50]

first_element = A[1]        # Get first element
last_element = A[end]         # Get last element  
first_three = A[1:3]          # Get first three elements
reverse_order = A[end:-1:1]        # Get all elements in reverse order
every_second = A[1:2:end]         # Get every second element (10, 30, 50)

q1.1 1
q1.2 A[2:4]
q1.3 A[end]

### Q2
q2.1 float 64 and float 64
q2.2 no method matching mystery_function(::Int64, ::Int64)
q2.3 function mystery_function(x::Number,, y::Number,)
    if x > 0
        return x + y
    else
        return x - y
    end
end


## Task 2
### Q1
q1.1, q1.2 and q1.3
using BenchmarkTools  

function sum_squares_loop(x::Vector{Float64})
    result = 0.0
    for val in x
        result += val^2
    end
    return result
end

function sum_squares_functional(x::Vector{Float64})
    sum(y -> y^2, x)
end

function sum_squares_broadcast(x::Vector{Float64})
    sum(x .^ 2)
end

x = randn(10000)  

@btime sum_squares_loop($x)
@btime sum_squares_functional($x)
@btime sum_squares_broadcast($x)

The result

Precompiling project...
  8 dependencies successfully precompiled in 9 seconds. 26 already precompiled.

  4.923 μs (0 allocations: 0 bytes)
  808.517 ns (0 allocations: 0 bytes)
  2.046 μs (3 allocations: 96.06 KiB)
10043.73146813045

q1.4 
Although the sum with an anonymous function appears fastest here, broadcasting is actually faster according to research/information found.

### Q2
q2.1， q2.2 and q2.3
#1. Analyze type instability
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

using InteractiveUtils
println("Type instability in unstable_function:")
@code_warntype unstable_function(10)

#2. Rewrite as type-stable
function stable_function(n::Int)
    result = 0.0
    for i in 1:n
        if i % 2 == 0
            result += i * 1.0
        else
            result += i
        end
    end
    return result
end

println("\nType stability in stable_function (no red warnings):")
@code_warntype stable_function(10)

#3. Benchmark
import Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools

n_large = 1_000_000
println("\nBenchmark with n = 1,000,000:")
println("Unstable version:")
@btime unstable_function($n_large)
println("\nStable version:")
@btime stable_function($n_large)

The result
Benchmark with n = 1,000,000:
Unstable version:
  4.302 ms (0 allocations: 0 bytes)

Stable version:
  515.375 μs (0 allocations: 0 bytes)
5.000005e11

## Task3
q1.1 
#Import required library for diagonal function
using LinearAlgebra

#Create arrays
zeros_array = zeros(3, 3)       # Create 3x3 matrix of zeros
ones_vector = ones(5)           # Create vector of 5 ones
random_matrix = rand(2, 4)      # Create 2x4 matrix of random numbers
range_vector = collect(1:5)              # Create vector [1, 2, 3, 4, 5]

#Matrix operations
A = [1 2 3; 4 5 6; 7 8 9]
B = [1 0 1; 0 1 0; 1 0 1]

#Indexing operations
element_22 = A[2, 2]            # Get element at row 2, column 2
second_row = A[2, :]            # Get entire second row
first_column = A[:, 1]          # Get entire first column
main_diagonal = diag(A)         # Get main diagonal elements [1, 5, 9]


The result
3-element Vector{Int64}:
 1
 5
 9

3-element Vector{Int64}:
 1
 5
 9

3-element Vector{Int64}:
 1
 5
 9

3-element Vector{Int64}:
 1
 5
 9

q1.2 and q1.3
#Function 1: Apply operation to each element
function apply_function(x::Vector{Float64})
    # Return: a vector whose ith entry is sin(x_i) + cos(2*x_i)
    sin.(x) .+ cos.(2 .* x)
end

#Function 2: Matrix-scalar operations
function matrix_transform(A::Matrix{Float64}, c::Float64)
    # Return: a matrix whose (i,j)-entry is (A_ij + c) * 2 - 1
    (A .+ c) .* 2 .- 1
end

#Function 3: Element-wise comparison
function count_positives(x::Vector{Float64})
    # Count how many elements are positive
    sum(x .> 0)
end

#Test apply_function
x_test = [0.0, π/2, π]::Vector{Float64}
println("apply_function test result: ", apply_function(x_test))
#Expected: [1.0, 0.0, 1.0]

#Test matrix_transform
A_test = [1.0 2.0; 3.0 4.0]::Matrix{Float64}
c_test = 1.0
println("matrix_transform test result: \n", matrix_transform(A_test, c_test))
#Expected: [3.0 5.0; 7.0 9.0]

#Test count_positives
x_pos_test = [1.5, -2.0, 3.0, 0.0, 4.5]::Vector{Float64}
println("count_positives test result: ", count_positives(x_pos_test))
#Expected: 3

The result
apply_function test result: [1.0, 0.0, 1.0000000000000002]
matrix_transform test result: 
[3.0 5.0; 7.0 9.0]
count_positives test result: 3

q1.4
In Julia, the . (dot) operator enables broadcasting, which applies an operation element-wise to arrays without requiring explicit loops. It automatically handles operations between arrays of different shapes by expanding dimensions to make them compatible, allowing element-by-element computations. This makes the code more concise while maintaining high performance.
For example, A .+ c broadcasts the scalar c to match the shape of matrix A before performing element-wise addition, and sin.(x) applies the sine function to each element of x.