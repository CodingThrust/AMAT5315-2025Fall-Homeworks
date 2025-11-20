# Task 1: Julia Basic Grammar and Conventions

## 1. (Indexing and Ranges) Complete the following code snippets and answer the questions:
    # Given array
    A = [10, 20, 30, 40, 50]

    # Fill in the correct indices/expressions:
    first_element = A[1]        # Get first element
    last_element = A[5]         # Get last element  
    first_three = A[1:3]          # Get first three elements
    reverse_order = reverse(A)        # Get all elements in reverse order
    every_second = A[2:2:end]         # Get every second element (10, 30, 50)

### Answers of the Questions:

i. The index Julia use for the first element of an array is 1.

ii. If there is an array A, the expression of elements from index 2 to 4 is: A[2:4]

iii. Suppose we have an unknown array A, to get the last element, type: A[end]

## 2.(Types and Functions) Analyze this Julia code

    julia> function mystery_function(x::Int64, y::Float64)
           if x > 0
               return x + y
           else
               return x - y
           end
       end
    mystery_function (generic function with 1 method)

    julia> result1 = mystery_function(5, 2.0)
    7.0

    julia> result2 = mystery_function(-3, 1.5)
    -4.5

### Answers of the Questions:

i.

    julia> typeof(result1)
    Float64

    julia> typeof(result2)
    Float64

Clearly, both result1 and result2 are type Float64.

ii.

    julia> mystery_function(5, 2)
    ERROR: MethodError: no method matching mystery_function(::Int64, ::Int64)
    The function `mystery_function` exists, but no method is defined for this combination of argument types.

    Closest candidates are:
    mystery_function(::Int64, ::Float64)
    @ Main REPL[12]:1

    Stacktrace:
    [1] top-level scope
    @ REPL[19]:1
iii.

    function mystery_function(x::Number, y::Number)
        if x > 0
            return x + y
        else
            return x - y
        end
    end

# Task 2: Benchmarking and Profiling

## 1.(Basic Benchmarking) Implement and benchmark different approaches to compute the sum of squares:

i.
```
julia> using BenchmarkTools

julia> function sum_squares_loop(x::Vector{Float64})
           # TODO
       end
sum_squares_loop (generic function with 1 method)

julia> function sum_squares_loop(x::Vector{Float64})
       total = 0.0
           for i in eachindex(x)
               total += x[i] * x[i]
           end
           return total
       end
sum_squares_loop (generic function with 1 method)

julia> function sum_squares_functional(x::Vector{Float64})
           return sum(xi -> xi * xi, x)
       end
sum_squares_functional (generic function with 1 method)

julia> function sum_squares_broadcast(x::Vector{Float64})
           return sum(x .^ 2)
       end
sum_squares_broadcast (generic function with 1 method)
    ```

ii.
    ```
    julia> x = randn(10000)
    10000-element Vector{Float64}:
    -0.24425382254854885
    0.5886920777147305
    0.033688970431918035
    -1.3298838139050604
    -0.877668058294783
    -0.4280237198693283
    -0.41801219347880375
    1.933705033847068
    -1.0076954727856362
    -1.0948377368457047
    0.0588582108391274
    -1.3007635657625267
    -1.0086576805001162
    ⋮
    0.2614829681789775
    0.5731062751057356
    -0.05349467469210498
    0.01885036975527749
    1.430142616164312
    1.6196324846373575
    -0.42919246309591264
    0.34105542777106307
    1.8229813194526585
    1.5411319022909586
    -1.6434321077065424
    -0.2580234839205887
        ```

iii.

First Try:
    julia> @btime sum_squares_functional(x)
    1.779 ns (0 allocations: 0 bytes)

    julia> @btime sum_squares_loop(x)
    1.780 ns (0 allocations: 0 bytes)

    julia> @btime sum_squares_broadcast(x)
    1.993 ns (0 allocations: 0 bytes)
Another Try:
    julia> @btime sum_squares_loop(x)
    1.780 ns (0 allocations: 0 bytes)

    julia> @btime sum_squares_functional(x)
    1.781 ns (0 allocations: 0 bytes)

    julia> @btime sum_squares_broadcast(x)
    1.992 ns (0 allocations: 0 bytes)
iv.

From above, using sum and anonymous function is just as fast as simple loop, and the broadcasting function is the lowest.The reason is that the broadcasting function does copy inputs and create a virtual array, while other two functions don't. It makes the broadcasting function less efficienct.
2025/10/09
Edited:
```
julia> function sum_squares_loop(x::Vector{Float64})
       total = 0.0
           for i in eachindex(x)
               total += x[i] * x[i]
           end
           return total
       end
sum_squares_loop (generic function with 1 method)

julia> function sum_squares_functional(x::Vector{Float64})
           return sum(xi -> xi * xi, x)
       end
sum_squares_functional (generic function with 1 method)

julia> function sum_squares_broadcast(x::Vector{Float64})
           return sum(x .^ 2)
       end
sum_squares_broadcast (generic function with 1 method)

julia> x = randn(10000)
10000-element Vector{Float64}:
 -1.0107491423578914
 -0.23678470423960365
 -0.2416896757169892
 -1.1765384319300072
  1.8722700117129343
 -1.4775312226695783
  0.6130922100493651
  0.24507544276644444
  0.6796189843410887
 -0.2170355206313096
 -0.6387465842068056
  0.47529425294986727
  0.4773075805808943
  ⋮
 -0.6208663422546729
 -0.6896869080304372
  0.515450617980146
  1.1624721024882336
 -3.482011643476688
  0.7016582011346962
  1.2363614602166404
  0.12923769041946356
  0.601917545292705
  0.05992993571385328
 -0.626682974826411
  0.7025691559684631

julia> @btime sum_squares_functional(x)
  639.659 ns (0 allocations: 0 bytes)
9740.811634271548

julia> @btime sum_squares_loop(x)
  3.711 μs (0 allocations: 0 bytes)
9740.811634271535

julia> @btime sum_squares_broadcast(x)
  2.271 μs (3 allocations: 78.20 KiB)
9740.811634271548

```

## 2.(Performance Analysis) Analyze this type-unstable function:

### Answers of the Tasks:

i. 

``` julia> @code_warntype unstable_function(10)
MethodInstance for unstable_function(::Int64)
  from unstable_function(n::Int64) @ Main REPL[27]:1
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

It has Union{Float64, Int64}, which is clearly instable.

ii. Rewrite the function:

    ```
    function stable_function(n::Int)
        result = 0    # starts as Int
            for i in 1:n
                if i % 2 == 0
                    result += i * 1    # still Int
                else
                    result += i
                end
            end
            return result
        end
    ```

iii. Let large number n=1000000.

        julia> @btime stable_function1(1000000)
        1.386 ns (0 allocations: 0 bytes)
        500000500000

        julia> @btime unstable_function(1000000)
        1.405 ms (0 allocations: 0 bytes)
        5.000005e11
The stable function is much faster than the unstable function.

# Task 3: Basic Array Operations

### Answers of the Tasks:
i. 

        # Create arrays
        zeros_array = zeros(3, 3)              # Create 3x3 matrix of zeros
        ones_vector = ones(5)              # Create vector of 5 ones
        random_matrix = rand(2, 4)            # Create 2x4 matrix of random numbers
        range_vector = [1, 2, 3, 4, 5]             # Create vector [1, 2, 3, 4, 5]

        # Matrix operations
        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 0 1; 0 1 0; 1 0 1]

        # Fill in operations:
        element_22 = A[2, 2]               # Get element at row 2, column 2
        second_row = A[2, :]               # Get entire second row
        first_column = A[:, 1]             # Get entire first column
        main_diagonal = diag(A)            # Get main diagonal elements [1, 5, 9]

ii.

    ```
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

iii. Test your functions with sample data

    ```
            # Function #1 Test:
                julia> apply_function([-π/2,0,π/2])
                3-element Vector{Float64}:
                -2.0
                1.0
                0.0
            # Function #2 Test:
                julia> A = [1 2 3; 4 5 6; 7 8 9]
                3×3 Matrix{Int64}:
                1  2  3
                4  5  6
                7  8  9
                julia> matrix_transform(Float64.(A), 2.1)
                3×3 Matrix{Float64}:
                5.2   7.2   9.2
                11.2  13.2  15.2
                17.2  19.2  21.2
            # Function #3 Test:
                julia> count_positives(Float64.(A[:, 1]))
                3
    ```

iv. Explain what the . (dot) operator does in broadcasting
The . (dot) operator enables element-wise operations. For example, in sum(Float64.(A[:, 1]) .> 0), dot means to take each element of A[:, 1], and compare them to zero individually.
