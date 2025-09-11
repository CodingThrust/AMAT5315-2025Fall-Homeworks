# Homework 2

Note: Submission should be in markdown format.

## Task 1: Julia Basic Grammar and Conventions

Test your knowledge of Julia's syntax and basic conventions.

1. (Indexing and Ranges) Complete the following code snippets and answer the questions:

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
    1. What index does Julia use for the first element of an array? (0 or 1)
    2. Write the expression to get elements from index 2 to 4 (inclusive)
    3. How do you get the last element without knowing the array length?
   
    Ans: `1`; `A[2:4]`; `A[end]`

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
    1. What will be the type of `result1` and `result2`?
    2. What happens if you call `mystery_function(5, 2)` (integer as second argument)?
    3. Rewrite the function to accept any numeric types for both parameters
   
    Ans: 
    1. `typeof(result1) = Float64`, `typeof(result2) = Float64`.

    2. Get a ERROR:
   ```julia
   ERROR: MethodError: no method matching mystery_function(::Int64, ::Int64)
   The function `mystery_function` exists, but no method is defined for this combination of argument types.

   Closest candidates are:
     mystery_function(::Int64, ::Float64)
      @ Main REPL[7]:1

   Stacktrace:
    [1] top-level scope
      @ REPL[11]:1
   ```
    3. rewritten function:
   ```julia
   function mystery_function(x::Tx, y::Ty) where {Tx<:Number, Ty<:Number}
       r = real(x)
       if r > 0
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
        result = zero(Float64)
        for xi in x
            result += xi ^ 2
        end
        return result
    end

    # Version 2: Using sum and anonymous function
    function sum_squares_functional(x::Vector{Float64})
        return sum(xi -> xi ^ 2, x)
    end

    # Version 3: Using broadcasting
    function sum_squares_broadcast(x::Vector{Float64})
        return sum(x .^ 2)
    end
    ```

    **Tasks:**
    1. Implement all three functions
    2. Create test vector: `x = randn(10000)`
    3. Benchmark each function using `@btime`
    4. Which approach is fastest? Explain why (2-3 sentences)

    Ans: The benchmark results are shown below
    ```julia
   julia> x = randn(10000);

   julia> @btime sum_squares_loop($x)
     6.525 μs (0 allocations: 0 bytes)
   9740.017762248772

   julia> @btime sum_squares_functional($x)
     936.000 ns (0 allocations: 0 bytes)
   9740.017762248755

   julia> @btime sum_squares_broadcast($x)
     2.094 μs (3 allocations: 96.06 KiB)
   9740.017762248755
    ```

    Explanation:
    Julia's higher-order functions like sum with anonymous functions are highly optimized and can leverage efficient reduction algorithms without creating intermediate arrays.
    The broadcast version creates a temporary array for the squared elements before summing, which adds memory overhead.
    The simple loop version may not benefit from Julia's same level of compiler optimizations as the functional approach for this specific operation.

1. (Performance Analysis) Analyze this type-unstable function:

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
    
    Ans: The `@code_warntype unstable_function(10)` results:
    ```julia
   julia> @code_warntype unstable_function(10)
   MethodInstance for unstable_function(::Int64)
     from unstable_function(n::Int64) @ Main REPL[1]:1
   Arguments
     #self#::Core.Const(Main.unstable_function)
     n::Int64
   Locals
     @_3::Union{Nothing, Tuple{Int64, Int64}}
     result::Union{Float64, Int64}
     i::Int64
   Body::Union{Float64, Int64}
   1 ─       (result = 0)
   │   %2  = Main.:(:)::Core.Const(Colon())
   │   %3  = (%2)(1, n)::Core.PartialStruct(UnitRange{Int64}, Any[Core.Const(1), Int64])
   │         (@_3 = Base.iterate(%3))
   │   %5  = @_3::Union{Nothing, Tuple{Int64, Int64}}
   │   %6  = (%5 === nothing)::Bool
   │   %7  = Base.not_int(%6)::Bool
   └──       goto #7 if not %7
   2 ┄ %9  = @_3::Tuple{Int64, Int64}
   │         (i = Core.getfield(%9, 1))
   │   %11 = Core.getfield(%9, 2)::Int64
   │   %12 = Main.:(==)::Core.Const(==)
   │   %13 = Main.:%::Core.Const(rem)
   │   %14 = i::Int64
   │   %15 = (%13)(%14, 2)::Int64
   │   %16 = (%12)(%15, 0)::Bool
   └──       goto #4 if not %16
   3 ─ %18 = Main.:+::Core.Const(+)
   │   %19 = result::Union{Float64, Int64}
   │   %20 = Main.:*::Core.Const(*)
   │   %21 = i::Int64
   │   %22 = (%20)(%21, 1.0)::Float64
   │         (result = (%18)(%19, %22))
   └──       goto #5
   4 ─ %25 = Main.:+::Core.Const(+)
   │   %26 = result::Union{Float64, Int64}
   │   %27 = i::Int64
   └──       (result = (%25)(%26, %27))
   5 ┄       (@_3 = Base.iterate(%3, %11))
   │   %30 = @_3::Union{Nothing, Tuple{Int64, Int64}}
   │   %31 = (%30 === nothing)::Bool
   │   %32 = Base.not_int(%31)::Bool
   └──       goto #7 if not %32
   6 ─       goto #2
   7 ┄ %35 = result::Union{Float64, Int64}
   └──       return %35
    ```

    There are two ways of rewriting:
    ```julia
   function int_stable_function(n::Int)
       result = 0
       for i in 1:n
           if i % 2 == 0
               result += i
           else
               result += i
           end
       end
       return result
   end

   function float_stable_function(n::Int)
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

    The benchmark results:
    ```julia
   julia> n = 2 ^ 16
   65536

   julia> @btime unstable_function($n)
     306.916 μs (0 allocations: 0 bytes)
   2.147516416e9

   julia> @btime int_stable_function($n)
     1.666 ns (0 allocations: 0 bytes)
   2147516416

   julia> @btime float_stable_function($n)
     44.875 μs (0 allocations: 0 bytes)
   2.147516416e9
    ```

## Task 3: Basic Array Operations
Practice fundamental array operations that are commonly used in Julia.

1. (Array Creation and Indexing) Complete the following array operations:

    ```julia
    # Create arrays
    zeros_array = zeros(3, 3)              # Create 3x3 matrix of zeros
    ones_vector = ones(5)              # Create vector of 5 ones
    random_matrix = rand(2, 4)            # Create 2x4 matrix of random numbers
    range_vector = collect(1:5)             # Create vector [1, 2, 3, 4, 5]

    # Matrix operations
    A = [1 2 3; 4 5 6; 7 8 9]
    B = [1 0 1; 0 1 0; 1 0 1]

    # Fill in operations:
    element_22 = A[2, 2]               # Get element at row 2, column 2
    second_row = A[2, :]               # Get entire second row
    first_column = A[:, 1]             # Get entire first column
    main_diagonal = [A[k, k] for k in min(size(A)...)]            # Get main diagonal elements [1, 5, 9]
    ```

2. (Broadcasting and Element-wise Operations) Implement functions using Julia's broadcasting:

    ```julia
    # Function 1: Apply operation to each element
    function apply_function(x::Vector{Float64})
        return sin.(x) + cos.(2 * x)
    end

    # Function 2: Matrix-scalar operations
    function matrix_transform(A::Matrix{Float64}, c::Float64)
        return 2 * (A .+ c) .- 1
    end

    # Function 3: Element-wise comparison
    function count_positives(x::Vector{Float64})
        return sum(x .> 0)
    end
    ```

    **Tasks:**
    1. Complete all the array creation and indexing operations
    2. Implement the three broadcasting functions
    3. Test your functions with sample data
    4. Explain what the `.` (dot) operator does in broadcasting
    
    Ans: test results:
    ```julia
   julia> x1 = π * ones(5);

   julia> apply_function(x1)
   5-element Vector{Float64}:
    1.0000000000000002
    1.0000000000000002
    1.0000000000000002
    1.0000000000000002
    1.0000000000000002

   julia> A, c = ones(2, 2), 1.0;

   julia> matrix_transform(A, c)
   2×2 Matrix{Float64}:
    3.0  3.0
    3.0  3.0

   julia> x2 = ones(5);

   julia> count_positives(x2)
   5
    ```
    dot operator apply the function elementwise to the array.

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

    Ans:

    1. Results are
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
    2. results:
    ```julia
   julia> typeof(Tropical(1.0))
   Tropical{Float64}

   julia> supertype(typeof(Tropical(1.0)))
   AbstractSemiring
    ```

    3. it seems like a struct type:
    ```julia
   julia> isabstracttype(Tropical)
   false

   julia> isconcretetype(Tropical)
   false

   julia> isstructtype(Tropical)
   true
    ```

    4. it is a concretetype
    ```julia
   julia> isabstracttype(Tropical{Real})
   false

   julia> isconcretetype(Tropical{Real})
   true
    ```

    5. benchmark result:
    ```julia
   julia> @btime $A * $B;
     554.875 μs (3 allocations: 96.08 KiB)
    ```
    and the profiling result
    ```julia
   julia> Profile.clear()

   julia> @profile C = A * B;

   julia> Profile.print()
   Overhead ╎ [+additional indent] Count File:Line; Function
   =========================================================
   ┌ Warning: There were no samples collected.
   │ Run your program longer (perhaps by running it multiple times),
   │ or adjust the delay between samples with `Profile.init()`.
   └ @ Profile ~/.julia/juliaup/julia-1.11.6+0.aarch64.apple.darwin14/share/julia/stdlib/v1.11/Profile/src/Profile.jl:1246

   julia> A = rand(Tropical{Float64}, 1000, 1000);

   julia> B = rand(Tropical{Float64}, 1000, 1000);

   julia> Profile.clear()

   julia> @profile C = A * B;

   julia> Profile.print()
   Overhead ╎ [+additional indent] Count File:Line; Function
   =========================================================
      ╎470 @Base/client.jl:541; _start()
      ╎ 470 @Base/client.jl:567; repl_main
      ╎  470 @Base/client.jl:430; run_main_repl(interactive::Bool, quiet::Bool, banner::Symbol, history_file::Bool, color_set::Bool)
      ╎   470 @Base/essentials.jl:1052; invokelatest
      ╎    470 @Base/essentials.jl:1055; #invokelatest#2
      ╎     470 @Base/client.jl:446; (::Base.var"#1150#1152"{Bool, Symbol, Bool})(REPL::Module)
      ╎    ╎ 470 @REPL/src/REPL.jl:486; run_repl(repl::REPL.AbstractREPL, consumer::Any)
      ╎    ╎  470 @REPL/src/REPL.jl:500; run_repl(repl::REPL.AbstractREPL, consumer::Any; backend_on_current_task::Bool, backend::Any)
      ╎    ╎   470 @REPL/src/REPL.jl:340; kwcall(::NamedTuple, ::typeof(REPL.start_repl_backend), backend::REPL.REPLBackend, consumer::Any)
      ╎    ╎    470 @REPL/src/REPL.jl:343; start_repl_backend(backend::REPL.REPLBackend, consumer::Any; get_module::Function)
      ╎    ╎     470 @REPL/src/REPL.jl:368; repl_backend_loop(backend::REPL.REPLBackend, get_module::Function)
      ╎    ╎    ╎ 470 @REPL/src/REPL.jl:261; eval_user_input(ast::Any, backend::REPL.REPLBackend, mod::Module)
      ╎    ╎    ╎  470 @Base/boot.jl:430; eval
      ╎    ╎    ╎   470 @LinearAlgebra/src/matmul.jl:114; *(A::Matrix{Tropical{Float64}}, B::Matrix{Tropical{Float64}})
      ╎    ╎    ╎    470 @LinearAlgebra/src/matmul.jl:253; mul!
      ╎    ╎    ╎     470 @LinearAlgebra/src/matmul.jl:285; mul!
      ╎    ╎    ╎    ╎ 470 @LinearAlgebra/src/matmul.jl:287; _mul!
      ╎    ╎    ╎    ╎  470 @LinearAlgebra/src/matmul.jl:868; generic_matmatmul!
      ╎    ╎    ╎    ╎   1   @LinearAlgebra/src/matmul.jl:890; _generic_matmatmul!(C::Matrix{Tropical{Float64}}, A::Matrix{Tropical{Float64}}, B::Matrix{Tropical…
      ╎    ╎    ╎    ╎    1   @LinearAlgebra/src/generic.jl:103; _rmul_or_fill!
      ╎    ╎    ╎    ╎     1   @Base/array.jl:329; fill!
      ╎    ╎    ╎    ╎    ╎ 1   @Base/array.jl:987; setindex!
      ╎    ╎    ╎    ╎   469 @LinearAlgebra/src/matmul.jl:896; _generic_matmatmul!(C::Matrix{Tropical{Float64}}, A::Matrix{Tropical{Float64}}, B::Matrix{Tropical…
     5╎    ╎    ╎    ╎    5   @Base/simdloop.jl:75; macro expansion
      ╎    ╎    ╎    ╎    464 @Base/simdloop.jl:77; macro expansion
      ╎    ╎    ╎    ╎     464 @LinearAlgebra/src/matmul.jl:897; macro expansion
      ╎    ╎    ╎    ╎    ╎ 8   @Base/array.jl:930; getindex
      ╎    ╎    ╎    ╎    ╎  5   @Base/abstractarray.jl:1347; _to_linear_index
      ╎    ╎    ╎    ╎    ╎   5   @Base/abstractarray.jl:3048; _sub2ind
      ╎    ╎    ╎    ╎    ╎    5   @Base/abstractarray.jl:98; axes
     5╎    ╎    ╎    ╎    ╎     5   @Base/array.jl:194; size
     3╎    ╎    ╎    ╎    ╎  3   @Base/essentials.jl:917; getindex
   456╎    ╎    ╎    ╎    ╎ 456 @Base/array.jl:994; setindex!
   Total snapshots: 470. Utilization: 100% across all threads and tasks. Use the `groupby` kwarg to break down by thread and/or task.

    ```

    Let us take a look at regular matrix multiplication:
    ```julia
   julia> regular_A = rand(100, 100);

   julia> regular_B = rand(100, 100);

   julia> @btime $regular_A * $regular_B;
     31.584 μs (3 allocations: 96.08 KiB)
    ```
    It is tens times faster than tropical matrix multiplication.
