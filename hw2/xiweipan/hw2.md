## Task 1: Julia Basic Grammar and Conventions

1. (Indexing and Ranges)

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

    **Questions:**
    1. What index does Julia use for the first element of an array? (0 or 1) \
    A: 1
    2. Write the expression to get elements from index 2 to 4 (inclusive) \
    A: A[2:4]
    3. How do you get the last element without knowing the array length? \
    A: A[end]

2. (Types and Functions) Analyze this Julia code:

    ```julia
    function mystery_function(x::Int64, y::Float64)
        if x > 0
            return x + y
        else
            return x - y
        end
    end

    # Test calls:
    result1 = mystery_function(5, 2.0)
    result2 = mystery_function(-3, 1.5)
    ```

    **Questions:**
    1. What will be the type of `result1` and `result2`? \
    A: Both Float64.
    2. What happens if you call `mystery_function(5, 2)` (integer as second argument)? \
    A: It will throw a MethodError: no method matching mystery_function(::Int64, ::Int64).
    3. Rewrite the function to accept any numeric types for both parameters \
    A:
    ```julia
    function mystery_function(x::Number, y::Number)
        if x > 0
            return x + y
        else
            return x - y
        end
    end
    ```
## Task 2: Benchmarking and Profiling

Learn to measure and analyze performance in Julia.

1. (Basic Benchmarking) Implement and benchmark different approaches to compute the sum of squares:

    ```julia
    using BenchmarkTools

    # Version 1: Simple loop
    function sum_squares_loop(x::Vector{Float64})
        result = 0.0
        @inbounds @simd for i in eachindex(x)
            y = x[i]
            result += y * y
        end
        return result
    end

    # Version 2: Using sum and anonymous function
    function sum_squares_functional(x::Vector{Float64})
        return sum(x -> x * x, x)
    end

    # Version 3: Using broadcasting
    function sum_squares_broadcast(x::Vector{Float64})
        return sum(x .* x)
    end
    ```

    **Tasks:**
    1. Implement all three functions
    2. Create test vector: `x = randn(10000)`
    3. Benchmark each function using `@btime`
    4. Which approach is fastest? Explain why (2-3 sentences)

    **Answer:**
    The version 2 using sum with an anonymous function is the fastest because it fuses the mapping and reduction into a single pass, so no temporary array is allocated for intermediate results. Moreover, it leverages Julia’s highly optimized `mapreduce` implementation.

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

    **Tasks:**
    1. Use `@code_warntype unstable_function(10)` to see type instability
    2. Rewrite the function to be type-stable
    3. Benchmark both versions with large n and compare performance
   
    **Answer:**
    The second version is type-stable because the type of `result` is always `Float64` after the first iteration.
    ```julia
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
    ```
    Benchmark results:
    ```julia
    @btime unstable_function(100000)
    @btime stable_function(100000)
    ```
    The average time of the second version is 66.958 $\mu$s, while the average time of the first version is 468.292 $\mu$s.

## Task 3: Basic Array Operations
Practice fundamental array operations that are commonly used in Julia.

1. (Array Creation and Indexing) Complete the following array operations:

    ```julia
    # Create arrays
    zeros_array = zeros(3, 3)   # Create 3x3 matrix of zeros
    ones_vector = ones(5)       # Create vector of 5 ones
    random_matrix = rand(2, 4)  # Create 2x4 matrix of random numbers
    range_vector = 1:5          # Create vector [1, 2, 3, 4, 5]

    # Matrix operations
    A = [1 2 3; 4 5 6; 7 8 9]
    B = [1 0 1; 0 1 0; 1 0 1]

    # Fill in operations:
    element_22 = A[2,2]               # Get element at row 2, column 2
    second_row = A[2,:]               # Get entire second row
    first_column = A[:,1]             # Get entire first column
    using LinearAlgebra
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
        return (A .+ c) .* 2 .- 1
    end

    # Function 3: Element-wise comparison
    function count_positives(x::Vector{Float64})
        # Count how many elements are positive
        # Hint: use broadcasting and sum
        return sum(x .> 0)
    end
    ```

    **Tasks:**
    1. Complete all the array creation and indexing operations
    2. Implement the three broadcasting functions
    3. Test your functions with sample data
    4. Explain what the `.` (dot) operator does in broadcasting
   
    **Answer:**
    The `.` (dot) operator does element-wise operations.

## Task 4: (Optional) Design a new algebraic system in Julia

Tropical Max-Plus Algebra is a Semiring algebra. It maps
* `+` to `max` in regular algebra,
* `*` to `+` in regular algebra,
* `1` to `0` in regular algebra,
* `0` to `-Inf` in regular algebra (for integer content types, this is chosen as a small integer).

Consider the following implementation of the Tropical Max-Plus Algebra:

```julia
using Random
abstract type AbstractSemiring <: Number end

# define the -inf
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
        # new is the default constructor
        new{T1}(T2(x.n))
    end
end

# customize the print
Base.show(io::IO, t::Tropical) = Base.print(io, "$(t.n)ₜ")

# power is mapped to multiplication
Base.:^(a::Tropical, b::Real) = Tropical(a.n * b)
Base.:^(a::Tropical, b::Integer) = Tropical(a.n * b)

# multiplication is mapped to addition
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

# addition is mapped to max
Base.:+(a::Tropical, b::Tropical) = Tropical(max(a.n, b.n))

# minimum value of the semiring
Base.typemin(::Type{Tropical{T}}) where T = Tropical(neginf(T))

# additive identity (zero element) - defined on types
Base.zero(::Type{Tropical{T}}) where T = typemin(Tropical{T})
# additive identity (zero element)
Base.zero(::Tropical{T}) where T = zero(Tropical{T})

# multiplicative identity (one element)
Base.one(::Type{Tropical{T}}) where T = Tropical(zero(T))
Base.one(::Tropical{T}) where T = one(Tropical{T})

# inverse is mapped to negative
Base.inv(x::Tropical) = Tropical(-x.n)

# division is mapped to subtraction
Base.:/(x::Tropical, y::Tropical) = Tropical(x.n - y.n)
# `div` is similar to `/`, the only difference is that `div(::Int, ::Int) -> Int`, but `/(::Int, ::Int) -> Float64`
Base.div(x::Tropical, y::Tropical) = Tropical(x.n - y.n)

# two numbers are approximately equal. For floating point numbers, this is often preferred to `==` due to the rounding error.
Base.isapprox(x::Tropical, y::Tropical; kwargs...) = isapprox(x.n, y.n; kwargs...)

# promotion rules
Base.promote_type(::Type{Tropical{T1}}, b::Type{Tropical{T2}}) where {T1, T2} = Tropical{promote_type(T1,T2)}

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Tropical{T}}) where T
    Tropical{T}(rand(rng, T))
end
```

Please open a Julia REPL, run the code above, and answer the following questions:
1. What are the outputs of the following expressions?
    ```julia
    julia> Tropical(1.0) + Tropical(3.0)
    3.0ₜ
    julia> Tropical(1.0) * Tropical(3.0)
    4.0ₜ
    julia> one(Tropical{Float64})
    0.0ₜ
    julia> zero(Tropical{Float64})
    -Infₜ
    ```

2. What is the type and supertype of `Tropical(1.0)`? \
    A: The type of `Tropical(1.0)` is `Tropical{Float64}`, and the supertype is `AbstractSemiring`.
3. Is `Tropical` a concrete type or an abstract type? \
    A: `Tropical` is NOT either concrete or abstract type.
4. Is `Tropical{Real}` a concrete type or an abstract type? \
    A: `Tropical{Real}` is a concrete type.
5. Benchmark and profile the performance of Tropical matrix multiplication:
   ```julia
   A = rand(Tropical{Float64}, 100, 100)
   B = rand(Tropical{Float64}, 100, 100)
   C = A * B   # please measure the time taken
   ```
   write a brief report on the performance of the tropical matrix multiplication. \
    **Answer:**
    - Result: `@btime C = A*B;`  # 538.834 μs (3 allocations: 80.08 KiB)
	- Profiler trace: falls into `LinearAlgebra.generic_matmatmul!` → SIMDed inner loop → `getindex/axes/size`. That’s the generic triple loop for non-BLAS element types (expected for `Tropical{Float64}`).
