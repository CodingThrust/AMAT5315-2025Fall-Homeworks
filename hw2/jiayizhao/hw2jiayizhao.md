# Homework 2 - Solution

## Task 1: Julia Basic Grammar and Conventions

### 1. Indexing and Ranges



**(Indexing and Ranges) Complete the following code snippets and answer the questions:**
```julia
# Given array
A = [10, 20, 30, 40, 50]

# Fill in the correct indices/expressions:
first_element = A[1]            # Get first element
last_element = A[end]           # Get last element  
first_three = A[1:3]            # Get first three elements
reverse_order = A[end:-1:1]     # Get all elements in reverse order
every_second = A[1:2:end]       # Get every second element (10, 30, 50)
```


**Output**
```julia
julia> A = [10, 20, 30, 40, 50]
5-element Vector{Int64}:
 10
 20
 30
 40
 50

julia> first_element = A[1]
10

julia> last_element = A[end]
50

julia> first_three = A[1:3]
3-element Vector{Int64}:
 10
 20
 30

julia> reverse_order = A[end:-1:1]
5-element Vector{Int64}:
 50
 40
 30
 20
 10

julia> every_second = A[1:2:end]
3-element Vector{Int64}:
 10
 30
 50
```



**Questions:**
1. **What index does Julia use for the first element of an array?** 
   ```julia
   1
   ```

2. **Write the expression to get elements from index 2 to 4 (inclusive)**
   ```julia
   A[2:4]
   ```

3. **How do you get the last element without knowing the array length?**
   ```julia
   A[end]
   ```

### 2. (Types and Functions) Analyze this Julia code:

```julia
function mystery_function(x::Int64, y::Float64)
    if x > 0
        return x + y
    else
        return x - y
    end
end

# Test calls:
result1 = mystery_function(5, 2.0)    # result1 = 7.0
result2 = mystery_function(-3, 1.5)   # result2 = -4.5
```

**Questions:**

1. **What will be the type of `result1` and `result2`?**
   - Both `result1` and `result2` will be of type `Float64` because Julia promotes the result to the more general type when mixing Int64 and Float64.
    ```julia
    julia> result1 = mystery_function(5, 2.0)
    7.0

    julia> typeof(result1)
    Float64

    julia> result2 = mystery_function(-3, 1.5)
    -4.5

    julia> typeof(result2)
    Float64
    ```
1. **What happens if you call `mystery_function(5, 2)`?**
   - This will throw a `MethodError` because the function expects the second argument to be `Float64`, but `2` is an `Int64`.

    ```julia
    julia> mystery_function(5, 2)
    ERROR: MethodError: no method matching mystery_function(::Int64, ::Int64)
    The function `mystery_function` exists, but no method is defined for this combination of argument types.
    ```

2. **Rewrite the function to accept any numeric types:**
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

### 1. Basic Benchmarking

```julia
using BenchmarkTools

# Version 1: Simple loop
function sum_squares_loop(x::Vector{Float64})
    result = 0.0
    for i in 1:length(x)
        result += x[i]^2
    end
    return result
end

# Version 2: Using sum and anonymous function
function sum_squares_functional(x::Vector{Float64})
    return sum(xi -> xi^2, x)
end

# Version 3: Using broadcasting
function sum_squares_broadcast(x::Vector{Float64})
    return sum(x.^2)
end
```

```julia
julia> x = randn(10000)
10000-element Vector{Float64}:
 -0.27599233675464413
 -0.7105769165599556
  1.1655139745107623
  2.357357600160997
 -0.05645272411836845
 -1.0570765789890266
 -0.24343070607186743
  1.0398215274749025
 -1.2132715367393352
 -0.21914332122417415
 -1.1171522060514005
 -0.5241546153346109
  ⋮
  1.0885812697771686
 -0.9082468219336222
 -1.797230463405286
 -0.1736097269417227
  0.8585898997395109
  1.0131820480903595
 -1.54525219745237
 -0.5815923409062306
 -0.7702618030084764
  1.898355478955154
 -0.6071954025416891

julia> using BenchmarkTools

julia> @btime sum_squares_loop($x)
  12.052 μs (0 allocations: 0 bytes)
10147.901071012582

julia> @btime sum_squares_functional($x)
  1.288 μs (0 allocations: 0 bytes)
10147.901071012571

julia> @btime sum_squares_broadcast($x)
  4.274 μs (3 allocations: 78.20 KiB)
10147.90107101257
```

**Which approach is fastest?**
The functional version is the fastest. It is because the functional version does not need to create another list to store each x^2 like broadcast. 
Mybe the reason why simple loop is slow is because it needs do the check of the condition of the for loop each time.
### 2. Performance Analysis

**Original type-unstable function:**
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



**Testing type stability:**
```julia
# Check type instability
julia> @code_warntype unstable_function(10)
MethodInstance for unstable_function(::Int64)
  from unstable_function(n::Int64) @ Main REPL[13]:1
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

**Type-stable rewrite:**
```julia
function stable_int_function(n::Int)
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
```



```julia
julia> @code_warntype stable_int_function(10)
MethodInstance for stable_int_function(::Int64)
  from stable_int_function(n::Int64) @ Main REPL[19]:1
Arguments
  #self#::Core.Const(Main.stable_int_function)
  n::Int64
Locals
  @_3::Union{Nothing, Tuple{Int64, Int64}}
  result::Int64
  i::Int64
Body::Int64
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
│   %12 = i::Int64
│   %13 = (%12 % 2)::Int64
│   %14 = (%11)(%13, 0)::Bool
└──       goto #4 if not %14
3 ─ %16 = result::Int64
│   %17 = i::Int64
│         (result = %16 + %17)
└──       goto #5
4 ─ %20 = result::Int64
│   %21 = i::Int64
└──       (result = %20 + %21)
5 ┄       (@_3 = Base.iterate(%2, %10))
│   %24 = @_3::Union{Nothing, Tuple{Int64, Int64}}
│   %25 = (%24 === nothing)::Bool
│   %26 = Base.not_int(%25)::Bool
└──       goto #7 if not %26
6 ─       goto #2
7 ┄ %29 = result::Int64
└──       return %29
```

```julia
# Benchmark comparison
julia> @btime unstable_function(10000)
  7.835 μs (0 allocations: 0 bytes)
5.0005e7

julia> @btime stable_int_function(10000)
  1.823 ns (0 allocations: 0 bytes)
50005000
```
The stable version is significantly faster.

## Task 3: Basic Array Operations

Practice fundamental array operations that are commonly used in Julia.

### 1. Array Creation and Indexing

```julia
# Create arrays
zeros_array = zeros(3, 3)              # Create 3x3 matrix of zeros
ones_vector = ones(5)                  # Create vector of 5 ones  
random_matrix = rand(2, 4)             # Create 2x4 matrix of random numbers
range_vector = collect(1:5)            # Create vector [1, 2, 3, 4, 5]

# Matrix operations
A = [1 2 3; 4 5 6; 7 8 9]
B = [1 0 1; 0 1 0; 1 0 1]

# Fill in operations:
element_22 = A[2, 2]                   # Get element at row 2, column 2
second_row = A[2, :]                   # Get entire second row
first_column = A[:, 1]                 # Get entire first column
main_diagonal = [A[1,1], A[2,2], A[3,3]]  # Get main diagonal
```

```julia
julia> zeros_array = zeros(3,3)
3×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> ones_vector = ones(5)
5-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0

julia> random_matrix = randn(2,4)
2×4 Matrix{Float64}:
 -0.439427  -0.447998  -1.59355   0.444236
 -0.314669   0.188532   0.757983  1.09389

julia> range_vector = 1:5
1:5

julia> range_vector = collect(1:5)
5-element Vector{Int64}:
 1
 2
 3
 4
 5

julia> range_vector = [i for i in 1:5]
5-element Vector{Int64}:
 1
 2
 3
 4
 5

julia> A = [1 2 3; 4 5 6; 7 8 9]
3×3 Matrix{Int64}:
 1  2  3
 4  5  6
 7  8  9

julia> element_22 = A[2, 2]
5

julia> second_row = A[2, :]
3-element Vector{Int64}:
 4
 5
 6

julia> first_column = A[:, 1]
3-element Vector{Int64}:
 1
 4
 7

julia> A = [1 2 3; 4 5 6; 7 8 9]
3×3 Matrix{Int64}:
 1  2  3
 4  5  6
 7  8  9

julia> main_diagonal = [A[1,1], A[2,2], A[3,3]]
3-element Vector{Int64}:
 1
 5
 9
```



### 2. Broadcasting and Element-wise Operations

```julia
# Function 1: Apply operation to each element
function apply_function(x::Vector{Float64})
    # Return: a vector whose ith entry is sin(x_i) + cos(2*x_i)
    return sin.(x) .+ cos.(2 .* x)
end

# Function 2: Matrix-scalar operations
function matrix_transform(A::Matrix{Float64}, c::Float64)
    # Return: a matrix whose (i,j)-entry is (A_ij + c) * 2 - 1
    return (A .+ c) .* 2 .- 1
end

# Function 3: Element-wise comparison
function count_positives(x::Vector{Float64})
    # Count how many elements are positive
    return sum(x .> 0)
end
```

```julia
julia> test_vector = [1.0, -2.0, 3.0, -4.0, 5.0]
5-element Vector{Float64}:
  1.0
 -2.0
  3.0
 -4.0
  5.0

julia> test_matrix = [1.0 2.0; 3.0 4.0]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> apply_function(test_vector)
5-element Vector{Float64}:
  0.4253241482607541
 -1.5629410476892938
  1.1012902947102332
  0.6113024614993147
 -1.797995803739591

julia> matrix_transform(test_matrix, 1.0)
2×2 Matrix{Float64}:
 3.0  5.0
 7.0  9.0

julia> count_positives(test_vector)
3
```

**What does the `.` (dot) operator do?**
The dot (.) operator in Julia enables broadcasting, which applies operations element-wise to arrays. For example:
- `sin.(x)` applies `sin` to each element of `x`
- `A .+ B` adds corresponding elements of `A` and `B`
- `x .> 0` compares each element of `x` to 0

## Task 4: Tropical Max-Plus Algebra (Optional)

After running the provided code in Julia REPL, here are the answers:

### 1. Expression Outputs

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

### 2. Type Analysis

```julia
julia> typeof(Tropical(1.0))
Tropical{Float64}

julia> supertype(Tropical{Float64})
AbstractSemiring
```

### 3. Type Classification

```julia
julia> isabstracttype(Tropical)
false

julia> isconcretetype(Tropical)
false
```
So, Tropical is neither of them.
### 4. Parameterized Type

```julia
julia> isconcretetype(Tropical{Real})
true
```
So, Tropical{Real} is a concrete type.
### 5. Performance Benchmarking

```julia
using BenchmarkTools

# Create test matrices
A = rand(Tropical{Float64}, 100, 100)
B = rand(Tropical{Float64}, 100, 100)

# Benchmark multiplication
println("Tropical matrix multiplication benchmark:")
@btime C = $A * $B

# Profile the operation
using Profile
@profile begin
    for i in 1:100
        C = A * B
    end
end
Profile.print()
```

```julia
julia> A = rand(Tropical{Float64}, 100, 100)
100×100 Matrix{Tropical{Float64}}:
  0.005049416241310611ₜ   0.03444497240076505ₜ  …     0.551150799200599ₜ
  0.057328376110501345ₜ    0.7819160157631034ₜ       0.2090977035731798ₜ
    0.2028013404480573ₜ    0.2949713300054354ₜ       0.8648280035170831ₜ
    0.4622915933859333ₜ    0.5668851375981335ₜ       0.2540268466792014ₜ
    0.8293175116832691ₜ  0.045052357856946834ₜ       0.7565670286163342ₜ
   0.48381009393707486ₜ   0.22623930322729457ₜ  …    0.5868237849137815ₜ
   0.05811580618977019ₜ    0.7935904308631011ₜ       0.8691500393106593ₜ
    0.2816624476723809ₜ   0.17464694926091862ₜ      0.08610112568725348ₜ
    0.5771464624331033ₜ    0.7498331534365196ₜ       0.8404753777163005ₜ
    0.3369354187851794ₜ    0.5404098487964114ₜ     0.024234510159483746ₜ
    0.9708823728851894ₜ   0.09664060641457695ₜ  …    0.5999970411940907ₜ
 0.0031282227861000322ₜ   0.09261947129699788ₜ       0.6085086349845931ₜ
                      ⋮                         ⋱
    0.8234521357912734ₜ    0.4774276207291409ₜ       0.6562900197225049ₜ
    0.6513935842077443ₜ   0.07957125268194543ₜ  …    0.5490808292925097ₜ
    0.7179098737756212ₜ    0.6383286276776126ₜ       0.3322065655412709ₜ
   0.20942232547425543ₜ    0.1937160728604319ₜ       0.4865141913919677ₜ
    0.4304634287530552ₜ    0.4242374282795005ₜ       0.4641267109177777ₜ
    0.8562267311692592ₜ   0.24983903448727718ₜ        0.710791705201829ₜ
   0.15126215354900518ₜ    0.8373754667260255ₜ  …    0.6558534358186668ₜ
  0.016866951164385635ₜ    0.7325934442352621ₜ       0.6443227353931047ₜ
  0.047446653489900625ₜ    0.6869947820263187ₜ      0.32153404201149294ₜ
   0.08705742622786217ₜ   0.28322191937768304ₜ      0.02382419396224622ₜ
    0.9819586060098326ₜ    0.0371605769685579ₜ       0.8709433183275397ₜ


julia> B = rand(Tropical{Float64}, 100, 100)
100×100 Matrix{Tropical{Float64}}:
  0.14525422599279847ₜ   0.1786242020077694ₜ  …   0.21578889068393092ₜ
   0.4617028741237561ₜ   0.2704427643758526ₜ       0.6513960232846893ₜ
   0.7712730172547292ₜ  0.35666845902309074ₜ      0.12065534938308353ₜ
 0.045813592821369986ₜ  0.24029658714109758ₜ         0.19564271996991ₜ
   0.6539851575414104ₜ   0.5104045610363207ₜ       0.8152048598006565ₜ
   0.6882790123497988ₜ    0.914551355606518ₜ  …    0.7789301492714972ₜ
   0.7129069603541169ₜ   0.6314942876380011ₜ     0.010000921322575906ₜ
 0.010225553231009599ₜ   0.8943115955244633ₜ       0.0178712042133361ₜ
   0.0839511858768297ₜ  0.49088623290679934ₜ      0.31510010715939896ₜ
   0.8234731272488173ₜ   0.4774947958750664ₜ       0.8829186821975268ₜ
   0.3988871461534722ₜ  0.22431083705495192ₜ  …    0.8326914077658414ₜ
  0.06652447992167909ₜ   0.8699168162224282ₜ       0.7060747845618459ₜ
                     ⋮                        ⋱
  0.49737724117292514ₜ   0.7077155260648421ₜ       0.4469267541469527ₜ
   0.9216356432161624ₜ  0.30153461126796666ₜ  …   0.12517377587192557ₜ
  0.21963227013115294ₜ   0.9972496216510485ₜ       0.9088353082665708ₜ
   0.7258121264921992ₜ   0.2570989796940576ₜ        0.965186354690642ₜ
   0.5627566146505113ₜ   0.1350229750964631ₜ       0.5201833079011009ₜ
   0.9024271719673966ₜ   0.1343717068224186ₜ      0.16204030124156787ₜ
  0.10433610521335213ₜ    0.404158659557446ₜ  …    0.3702331021816335ₜ
   0.6025921182152609ₜ   0.8607514529931929ₜ       0.4953395493902363ₜ
   0.4151826178788024ₜ   0.8375740125082538ₜ      0.12400383297077866ₜ
  0.23534710028482464ₜ   0.7246364691627496ₜ       0.2135627989657698ₜ
  0.12934785500146906ₜ  0.18459347480632193ₜ      0.33198100307837664ₜ

julia> using BenchmarkTools

julia> @btime C = $A * $B
  712.208 μs (3 allocations: 78.21 KiB)
100×100 Matrix{Tropical{Float64}}:
 1.9163393758044505ₜ  1.9451825953804107ₜ  …  1.8689248389297632ₜ
  1.934490715976358ₜ    1.93836936723214ₜ     1.8403450534511698ₜ
 1.8695109678164576ₜ  1.9037586235576538ₜ     1.9246868933135977ₜ
  1.882578809008824ₜ  1.8367614963514454ₜ     1.9358172406837992ₜ
 1.9140412090495396ₜ   1.853803455399398ₜ     1.8666266721748523ₜ
 1.8836077514152325ₜ  1.8919501879876883ₜ  …  1.8361932145405455ₜ
 1.7996230515110572ₜ  1.8105587030906993ₜ     1.8590686064597666ₜ
 1.9172149204141347ₜ  1.8055306751498392ₜ      1.994650307520935ₜ
 1.9232956445016438ₜ  1.8569188522141045ₜ     1.7685045388296268ₜ
 1.9230869489698885ₜ  1.8410714247762825ₜ      1.919577615341331ₜ
 1.8570864098141817ₜ  1.7497222129536834ₜ  …   1.815849553368928ₜ
  1.895342918412254ₜ  1.8624290838883346ₜ      1.914365986076554ₜ
                   ⋮                       ⋱
   1.93988546682753ₜ  1.8530641902197833ₜ       1.83462719516771ₜ
 1.8796396223558436ₜ  1.7965186165474445ₜ  …  1.8036693955431269ₜ
 1.8636442038723264ₜ  1.9057259958654345ₜ     1.7143090085951957ₜ
 1.7964729789716536ₜ  1.8604370311336451ₜ      1.877405136828299ₜ
  1.879511042748732ₜ  1.8331137778483342ₜ     1.8458583815439735ₜ
 1.9277332519581891ₜ  1.9368988333731167ₜ     1.6956296015411731ₜ
 1.8650364215879995ₜ   1.745793978168579ₜ  …  1.7871125133796213ₜ
 1.8041790676372287ₜ  1.9315942663996721ₜ     1.8805308990946639ₜ
   1.80500074466368ₜ  1.7636518036698114ₜ     1.8582391763386552ₜ
  1.898477225694205ₜ  1.8092998954652577ₜ      1.812341974810733ₜ
  1.886690701039682ₜ   1.844783907799763ₜ     1.8392761641649948ₜ

julia> using Profile

julia> @profile begin
           for i in 1:100
               C = A * B
           end
       end

julia> Profile.print()
Overhead ╎ [+additional indent] Count File:Line; Function
=========================================================
  ╎88 @Base/client.jl:541; _start()
  ╎ 88 @Base/client.jl:567; repl_main
  ╎  88 @Base/client.jl:430; run_main_repl(interactive::Bool, quiet::Bool, banner…
  ╎   88 @Base/essentials.jl:1052; invokelatest
  ╎    88 @Base/essentials.jl:1055; #invokelatest#2
  ╎     88 @Base/client.jl:446; (::Base.var"#1150#1152"{Bool, Symbol, Bool})(REPL…
  ╎    ╎ 88 @REPL/src/REPL.jl:486; run_repl(repl::REPL.AbstractREPL, consumer::An…
  ╎    ╎  88 @REPL/src/REPL.jl:500; run_repl(repl::REPL.AbstractREPL, consumer::A…
  ╎    ╎   88 @REPL/src/REPL.jl:340; kwcall(::NamedTuple, ::typeof(REPL.start_rep…
  ╎    ╎    88 @REPL/src/REPL.jl:343; start_repl_backend(backend::REPL.REPLBacken…
  ╎    ╎     88 @REPL/src/REPL.jl:368; repl_backend_loop(backend::REPL.REPLBacken…
  ╎    ╎    ╎ 88 @REPL/src/REPL.jl:261; eval_user_input(ast::Any, backend::REPL.R…
  ╎    ╎    ╎  88 @Base/boot.jl:430; eval
  ╎    ╎    ╎   88 REPL[64]:1; top-level scope
  ╎    ╎    ╎    88 …file/src/Profile.jl:58; macro expansion
 4╎    ╎    ╎     88 REPL[64]:3; macro expansion
  ╎    ╎    ╎    ╎ 83 …bra/src/matmul.jl:114; *(A::Matrix{Tropical{Float64}}, B::…
  ╎    ╎    ╎    ╎  83 …bra/src/matmul.jl:253; mul!
  ╎    ╎    ╎    ╎   83 …bra/src/matmul.jl:285; mul!
  ╎    ╎    ╎    ╎    83 …ra/src/matmul.jl:287; _mul!
  ╎    ╎    ╎    ╎     83 …ra/src/matmul.jl:868; generic_matmatmul!
  ╎    ╎    ╎    ╎    ╎ 9  …a/src/matmul.jl:890; _generic_matmatmul!(C::Matrix{Tr…
  ╎    ╎    ╎    ╎    ╎  9  …/src/generic.jl:103; _rmul_or_fill!
  ╎    ╎    ╎    ╎    ╎   9  @Base/array.jl:329; fill!
 9╎    ╎    ╎    ╎    ╎    9  @Base/array.jl:987; setindex!
  ╎    ╎    ╎    ╎    ╎ 73 …a/src/matmul.jl:896; _generic_matmatmul!(C::Matrix{Tr…
 5╎    ╎    ╎    ╎    ╎  5  …ase/simdloop.jl:75; macro expansion
  ╎    ╎    ╎    ╎    ╎  68 …ase/simdloop.jl:77; macro expansion
  ╎    ╎    ╎    ╎    ╎   68 …a/src/matmul.jl:897; macro expansion
  ╎    ╎    ╎    ╎    ╎    15 @Base/array.jl:930; getindex
15╎    ╎    ╎    ╎    ╎     15 …/essentials.jl:917; getindex
19╎    ╎    ╎    ╎    ╎    19 @Base/array.jl:994; setindex!
  ╎    ╎    ╎    ╎    ╎    34 …e/promotion.jl:633; muladd
  ╎    ╎    ╎    ╎    ╎     7  REPL[28]:1; *
 7╎    ╎    ╎    ╎    ╎    ╎ 7  @Base/float.jl:491; +
  ╎    ╎    ╎    ╎    ╎     27 REPL[30]:1; +
  ╎    ╎    ╎    ╎    ╎    ╎ 3  @Base/math.jl:838; max
 3╎    ╎    ╎    ╎    ╎    ╎  3  @Base/float.jl:492; -
  ╎    ╎    ╎    ╎    ╎    ╎ 15 @Base/math.jl:839; max
13╎    ╎    ╎    ╎    ╎    ╎  13 …essentials.jl:796; ifelse
 2╎    ╎    ╎    ╎    ╎    ╎  2  …floatfuncs.jl:15; signbit
  ╎    ╎    ╎    ╎    ╎    ╎ 9  @Base/math.jl:841; max
 9╎    ╎    ╎    ╎    ╎    ╎  9  …essentials.jl:796; ifelse
  ╎    ╎    ╎    ╎    ╎ 1  …a/src/matmul.jl:899; _generic_matmatmul!(C::Matrix{Tr…
 1╎    ╎    ╎    ╎    ╎  1  @Base/range.jl:908; iterate
Total snapshots: 90. Utilization: 100% across all threads and tasks. Use the `groupby` kwarg to break down by thread and/or task.

```



**Performance Report:**

**Performance Metrics:**
- 100% CPU utilization  
- 83/90 samples in matrix multiplication function
- 68/90 samples in core computation loop

**Time Breakdown:**
- Memory operations: 34/90 (getindex 15/90, setindex 19/90)
- Tropical arithmetic: 34/90 (max 27/90, add 7/90)
- Loop overhead: 5/90

**Key Findings:** Excellent CPU utilization, most time spent in actual multiplication logic, minimal loop overhead.