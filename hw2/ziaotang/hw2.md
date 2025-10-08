## Task 1

### 1.

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

i. 1

ii. `A[2:4]`

iii. `A[end]`

### 2.

i. Both are `Float64`

ii. `MethodError`

iii. 

```julia
function mystery_function(x::Number, y::Number)
    if x > 0
        return x + y
    else
        return x - y
    end
end
```



## Task 2

### 1.

```julia
using BenchmarkTools

function sum_squares_loop(x::Vector{Float64})
    total = 0.0
    for i in eachindex(x)
        total += x[i] * x[i]
    end
    return total
end


function sum_squares_functional(x::Vector{Float64})
    return sum(val -> val * val, x)
end

function sum_squares_broadcast(x::Vector{Float64})
    return sum(x .^ 2)
end
```

```julia
x = randn(10000);
@btime sum_squares_loop($x)
  23.599 μs (0 allocations: 0 bytes)
9934.187931529203

@btime sum_squares_functional($x)
  1.286 μs (0 allocations: 0 bytes)
9934.187931529263

@btime sum_squares_broadcast($x)
  8.540 μs (2 allocations: 78.20 KiB)
9934.187931529263
```

Functional method is faster sicne Julia's compiler is designed to optimize functions like `sum`, and avoids creating a temporary array.

### 2.

i.

```julia
 @code_warntype unstable_function(10)

Variables
  #self#::Core.Compiler.Const(unstable_function, false)
  n::Int64
  result::Union{Float64, Int64}
  @_4::Union{Nothing, Tuple{Int64,Int64}}
  i::Int64

Body::Union{Float64, Int64}
1 ─       (result = 0)
│   %2  = (1:n)::Core.Compiler.PartialStruct(UnitRange{Int64}, Any[Core.Compiler.Const(1, false), Int64])
│         (@_4 = Base.iterate(%2))
│   %4  = (@_4 === nothing)::Bool
│   %5  = Base.not_int(%4)::Bool
└──       goto #7 if not %5
2 ┄ %7  = @_4::Tuple{Int64,Int64}::Tuple{Int64,Int64}
│         (i = Core.getfield(%7, 1))
│   %9  = Core.getfield(%7, 2)::Int64
│   %10 = (i % 2)::Int64
│   %11 = (%10 == 0)::Bool
└──       goto #4 if not %11
3 ─ %13 = result::Union{Float64, Int64}
│   %14 = (i * 1.0)::Float64
│         (result = %13 + %14)
└──       goto #5
4 ─       (result = result + i)
5 ┄       (@_4 = Base.iterate(%2, %9))
│   %19 = (@_4 === nothing)::Bool
│   %20 = Base.not_int(%19)::Bool
└──       goto #7 if not %20
6 ─       goto #2
7 ┄       return result
```

ii.

```julia
function stable_function(n::Int)
    result = 0.0 
    for i in 1:n
        if i % 2 == 0
            result += i * 1.0
        else
            result += Float64(i) 
        end
    end
    return result
end
```



```julia
@code_warntype stable_function(10)

Variables
  #self#::Core.Compiler.Const(stable_function, false)
  n::Int64
  result::Float64
  @_4::Union{Nothing, Tuple{Int64,Int64}}
  i::Int64

Body::Float64
1 ─       (result = 0.0)
│   %2  = (1:n)::Core.Compiler.PartialStruct(UnitRange{Int64}, Any[Core.Compiler.Const(1, false), Int64])
│         (@_4 = Base.iterate(%2))
│   %4  = (@_4 === nothing)::Bool
│   %5  = Base.not_int(%4)::Bool
└──       goto #7 if not %5
2 ┄ %7  = @_4::Tuple{Int64,Int64}::Tuple{Int64,Int64}
│         (i = Core.getfield(%7, 1))
│   %9  = Core.getfield(%7, 2)::Int64
│   %10 = (i % 2)::Int64
│   %11 = (%10 == 0)::Bool
└──       goto #4 if not %11
3 ─ %13 = result::Float64
│   %14 = (i * 1.0)::Float64
│         (result = %13 + %14)
└──       goto #5
4 ─ %17 = result::Float64
│   %18 = Main.Float64(i)::Float64
└──       (result = %17 + %18)
5 ┄       (@_4 = Base.iterate(%2, %9))
│   %21 = (@_4 === nothing)::Bool
│   %22 = Base.not_int(%21)::Bool
└──       goto #7 if not %22
6 ─       goto #2
7 ┄       return result

```

iii.

```julia
n = 1000000;

@btime unstable_function($n)
  2.768 ms (0 allocations: 0 bytes)
5.000005e11

@btime stable_function($n)
  1.542 ms (0 allocations: 0 bytes)
5.000005e11
```



stable_function is better





## Task 3

### 1.

```julia
zeros_array = zeros(3, 3)
3×3 Array{Float64,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

ones_vector = ones(5)
5-element Array{Float64,1}:
 1.0
 1.0
 1.0
 1.0
 1.0

random_matrix = rand(2, 4)
2×4 Array{Float64,2}:
 0.922778  0.980256  0.126836   0.531031
 0.547728  0.860628  0.0168512  0.25902

range_vector = collect(1:5)
5-element Array{Int64,1}:
 1
 2
 3
 4
 5


element_22 = A[2, 2]
5



second_row = A[2, :]
3-element Array{Int64,1}:
 4
 5
 6


first_column = A[:, 1]
3-element Array{Int64,1}:
 1
 4
 7

using LinearAlgebra
main_diagonal = diag(A)
3-element Array{Int64,1}:
 1
 5
 9
```

### 2.

```julia
# Function 1: Apply operation to each element
function apply_function(x::Vector{Float64})
    # Return: a vector whose ith entry is sin(x_i) + cos(2*x_i)
    # Use broadcasting (dot notation)
    return sin.(x) + cos.(2 .* x)
end

# Function 2: Matrix-scalar operations
function matrix_transform(A::Matrix{Float64}, c::Float64)
    # Return: a matrix whose (i,j)-entry is (A_ij + c) * 2 - 1
    # Apply this transformation element-wise
    return (A .+ c) .* 2 .- 1
end

# Function 3: Element-wise comparison
function count_positives(x::Vector{Float64})
    # Count how many elements are positive
    # Hint: use broadcasting and sum
    return sum(x .> 0)
end
```



```julia
x_vector = [1.0,2.0,3.0,4.0,5.0];
sample_matrix = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0];
c = 1.5;

apply_function(x_vector)
5-element Array{Float64,1}:
  0.4253241482607541
  0.25565380596206977
  1.1012902947102332
 -0.9023025291165417
 -1.797995803739591

matrix_transform(sample_matrix, c)
3×3 Array{Float64,2}:
  4.0   6.0   8.0
 10.0  12.0  14.0
 16.0  18.0  20.0

count_positives(x_vector)
5
```



iv. dot operator is an element-wise operation to arrays
