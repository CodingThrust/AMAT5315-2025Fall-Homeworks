# Homework2
## Taks1:
1.(Indexing and Ranges) Complete the following code snippets and answer the questions:

```julia
# Given array
A = [10, 20, 30, 40, 50]

# Fill in the correct indices/expressions:
first_element = A[1]        # Get first element
last_element = A[end]         # Get last element  
first_three = A[1:3]          # Get first three elements
reverse_order = A[end:-1:1]        # Get all elements in reverse order
every_second = A[1:2:end]         # Get every second element (10, 30, 50)

```

### Questions:

<ol type="i">
<li>What index does Julia use for the first element of an array? (0 or 1)</li>
<li>Write the expression to get elements from index 2 to 4 (inclusive)
</li>
<li>How do you get the last element without knowing the array length?</li>
</ol>
    

### Answers:

i.  Julia use 1 for the first element

ii. 
```julia
A[2:4]
```

iii.
```julia
A[end]
```

## Taks 2:Benchmarking and Profiling

1. (Basic Benchmarking) Implement and benchmark different approaches to compute the sum of squares:
```julia
using BenchmarkTools

# Version 1: Simple loop
function sum_squares_loop(x::Vector{Float64})
    s = 0
    for i in eachindex(x)
        s+=x[i]^2
    end
    return s
end

# Version 2: Using sum and anonymous function
function sum_squares_functional(x::Vector{Float64})
    return sum(x -> x^2, x)
end

# Version 3: Using broadcasting
function sum_squares_broadcast(x::Vector{Float64})
    return sum(x.^2)
end

```

### result:
After finishing the code and creating test vector, using the benchmark tools, I get the following results:

```julia-repl

julia> @btime sum_squares_loop(x)
  16.749 μs (0 allocations: 0 bytes)
9980.978394126201

julia> @btime sum_squares_functional(x)
  1.415 μs (0 allocations: 0 bytes)
9980.978394126234

julia> @btime sum_squares_broadcast(x)
  6.828 μs (3 allocations: 78.19 KiB)
9980.978394126234
```

Using sum and anonymous function is the fastest way because it allows the Julia compiler to perform "loop fusion", which eliminates the creation of temporary intermediate arrays. The anonymous function is inlined into the "sum" function.

2. (Performance Analysis) Analyze this type-unstable function:
```julia

function unstable_function(n::Int)
    result = 0    # starts as Int
    for i in 1:n
        if i % 2 == 0
            result += i * 1.0    # becomes Float64
        else
            result += i
        end
    end
    return result
end
```
### Tasks:

i. Use `@code_warntype unstable_function(10)` to see type instability

ii.Rewrite the function to be type-stable

iii.Benchmark both versions with large n and compare performance

### Answers:
i. The output is:
```julia-repl
julia> @code_warntype unstable_function(10)
MethodInstance for unstable_function(::Int64)
  from unstable_function(n::Int64) @ Main REPL[28]:1
Arguments
  #self#::Core.Const(Main.unstable_function)
  n::Int64
Locals
  @_3::Union{Nothing, Tuple{Int64, Int64}}
  result::Union{Float64, Int64}
  i::Int64
Body::Union{Float64, Int64}
1 ─       (result = 0)
│   %2  = (1:n)::Core.PartialStruct(UnitRange{Int64}, Any[Core.Const(1), Int64])
│         (@_3 = Base.iterate(%2))
│   %4  = @_3::Union{Nothing, Tuple{Int64, Int64}}
│   %5  = (%4 === nothing)::Bool
│   %6  = Base.not_int(%5)::Bool
└──       goto #7 if not %6
2 ┄ %8  = @_3::Tuple{Int64, Int64}
│         (i = Core.getfield(%8, 1))
│   %10 = Core.getfield(%8, 2)::Int64
│   %11 = Main.:(==)::Core.Const(==)
│   %12 = Main.:%::Core.Const(rem)
│   %13 = i::Int64
│   %14 = (%12)(%13, 2)::Int64
│   %15 = (%11)(%14, 0)::Bool
└──       goto #4 if not %15
3 ─ %17 = Main.:+::Core.Const(+)
│   %18 = result::Union{Float64, Int64}
│   %19 = Main.:*::Core.Const(*)
│   %20 = i::Int64
│   %21 = (%19)(%20, 1.0)::Float64
│         (result = (%17)(%18, %21))
└──       goto #5
4 ─ %24 = result::Union{Float64, Int64}
│   %25 = i::Int64
└──       (result = %24 + %25)
5 ┄       (@_3 = Base.iterate(%2, %10))
│   %28 = @_3::Union{Nothing, Tuple{Int64, Int64}}
│   %29 = (%28 === nothing)::Bool
│   %30 = Base.not_int(%29)::Bool
└──       goto #7 if not %30
6 ─       goto #2
7 ┄ %33 = result::Union{Float64, Int64}
└──       return %33
```

ii. The rewrited version is:
```julia
function stable_function(n::Int)
    result = 0    
    for i in 1:n
        if i % 2 == 0
            result += i * 1
        else
            result += i
        end
    end
    return result
end
```

iii. Using the benchmark, I get:
```julia-repl
julia> @btime unstable_function(100000)
  483.531 μs (0 allocations: 0 bytes)
5.00005e9

julia> @btime stable_function(100000)
  3.376 ns (0 allocations: 0 bytes)
5000050000
```
The stable version is much much faster than the stable version because the integer addition is faster than the float number addition.

## Task 3: Basic Array Operations
1. (Array Creation and Indexing) Complete the following array operations:
```julia
# Create arrays
zeros_array = zeros(Int64,3,3)              # Create 3x3 matrix of zeros
ones_vector = ones(Int64,5)              # Create vector of 5 ones
random_matrix = rand(2,4)          # Create 2x4 matrix of random numbers
range_vector = collect(1:1:5)             # Create vector [1, 2, 3, 4, 5]

# Matrix operations
A = [1 2 3; 4 5 6; 7 8 9]
B = [1 0 1; 0 1 0; 1 0 1]

# Fill in operations:
element_22 = A[2,2]               # Get element at row 2, column 2
second_row = A[2,:]               # Get entire second row
first_column = A[:,1]             # Get entire first column
main_diagonal = diag(A)            # Get main diagonal elements [1, 5, 9]
```
2. (Broadcasting and Element-wise Operations) Implement functions using Julia's broadcasting:
```julia
# Function 1: Apply operation to each element
function apply_function(x::Vector{Float64})
    # Return: a vector whose ith entry is sin(x_i) + cos(2*x_i)
    # Use broadcasting (dot notation)
    return sin.(x) .+ cos.(2 .* x)
end

# Function 2: Matrix-scalar operations
function matrix_transform(A::Matrix{Float64}, c::Float64)
    # Return: a matrix whose (i,j)-entry is (A_ij + c) * 2 - 1
    # Apply this transformation element-wise
    return (A .+ c).*2 .- 1
end

# Function 3: Element-wise comparison
function count_positives(x::Vector{Float64})
    # Count how many elements are positive
    # Hint: use broadcasting and sum
    return sum(x.>0)
end
```

Here is my test code:
```julia

@testset "apply_function" begin
    n = 100
    x = rand(n)
    y = apply_function(x)
    for i in 1:n
        @test y[i] == (sin(x[i]) + cos(2*x[i]))
    end

end

@testset "matrix_transform" begin
    n = 4
    c = rand()
    A = rand(n,n)
    B = matrix_transform(A,c)
    for i in 1:n
        for j in 1:n
            @test B[i,j] == (A[i,j]+c)*2 - 1
        end
    end
end

@testset "count_positives" begin
    n = 100
    x = rand(n) .- 0.5
    count = 0
    for i in 1:n
        if x[i]>0
            count += 1
        end
    end
    @test count_positives(x) == count
end
```
and the result is:
```julia-repl
Test Summary:  | Pass  Total  Time
apply_function |  100    100  0.0s
Test.DefaultTestSet("apply_function", Any[], 100, false, false, true, 1.758509078926641e9, 1.758509078926698e9, false, "/Users/hongkuanhisa/Test/test/hw2.jl")

Test Summary:    | Pass  Total  Time
matrix_transform |   16     16  0.0s
Test.DefaultTestSet("matrix_transform", Any[], 16, false, false, true, 1.758509079690804e9, 1.758509079690827e9, false, "/Users/hongkuanhisa/Test/test/hw2.jl")

Test Summary:   | Pass  Total  Time
count_positives |    1      1  0.0s
Test.DefaultTestSet("count_positives", Any[], 1, false, false, true, 1.758509080096513e9, 1.75850908009653e9, false, "/Users/hongkuanhisa/Test/test/hw2.jl")

```
.(dot) applies operations element-by-element to arrays of different sizes

## Task 4: (Optional) Design a new algebraic system in Julia

1. What are the outputs of the following expressions?
```julia-repl
julia> Tropical(1.0) + Tropical(3.0)
julia> Tropical(1.0) * Tropical(3.0)
julia> one(Tropical{Float64})
julia> zero(Tropical{Float64})
```
2. What is the type and supertype of `Tropical(1.0)`?
3. Is `Tropical` a concrete type or an abstract type?
4. Is `Tropical{Real}` a concrete type or an abstract type?
5. Benchmark and profile the performance of Tropical matrix multiplication:
```julia
A = rand(Tropical{Float64}, 100, 100)
B = rand(Tropical{Float64}, 100, 100)
C = A * B   # please measure the time taken
```
write a brief report on the performance of the tropical matrix multiplication.
## Answers:
1. The output is:
```julia-repl
julia> Tropical(1.0) + Tropical(3.0)
3.0ₜ
julia> Tropical(1.0) * Tropical(3.0)
4.0ₜ
julia> one(Tropical{Float64})
0.0ₜ
julia> zero(Tropical{Float64})
-Infₜ
```
2. The type of `Tropical(1.0)` is `Tropical{Float64}`, and the supertype of it is `AbstractSemiring`

3. `Tropical` is neither concrete type nor abstract type:
```julia-repl
julia> isconcretetype(Tropical)
false
julia> isabstracttype(Tropical)
false
```

4. `Tropical{Real}` is a concrete type:
```julia-repl
julia> isconcretetype(Tropical{Real})
true
```
5. Using the `@btime`, I get the following result for the multiplication of Tropcial numbers:
```julia-repl
julia> @btime C = A * B
  1.399 ms (3 allocations: 78.20 KiB)
```
Comparing to the usual `Matrix{Float64}` multiplication:
```julia
A = rand(Float64, 100, 100)
B = rand(Float64, 100, 100)
C = A * B   
```
I get the following benchmark result:
```julia-repl
julia> @btime C = A * B
  91.763 μs (3 allocations: 78.20 KiB)
```
it is not surpurised that the allocations are the same, but with more time cost in the Tropical matrix multiplication. Since the multiplication of the Tropical number is just the `min()` function and the `+` for `Float64` numbers, I do the following benchmark:
```julia
a = rand()
b = rand()
@btime min(a,b)
@btime a + b
```
The result is:
```julia-repl
julia> @btime min(a,b)
  18.629 ns (1 allocation: 16 bytes)
0.3552214072852681

julia> @btime a + b
  18.888 ns (1 allocation: 16 bytes)
0.8861627529397925
```
With little computation, if ignoring the time cost of calling `Tropical()`, the time cost for basic `min()` and `+` is about 10000*37.5ns = 37.5μs, which is much less than 1.399ms, so the time cost for initialized function `Tropical()` is very high:
```julia-repl
julia> @btime Tropical(a)
  184.482 ns (1 allocation: 16 bytes)
0.5309413456545244ₜ
```
