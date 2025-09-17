# Homework 2 — Manual Answers (Q&A Format)

## Task 1: Julia Basic Grammar and Conventions

### 1) Indexing and Ranges

**Complete the code snippets:**
```julia
first_element = A[1]        # Get first element
last_element = A[end]       # Get last element  
first_three = A[1:3]        # Get first three elements
reverse_order = A[end:-1:1] # Get all elements in reverse order
every_second = A[1:2:end]   # Get every second element (10, 30, 50)
```

**Questions and Answers:**
1. **Q:** What index does Julia use for the first element of an array? (0 or 1)
   **A:** 1

2. **Q:** Write the expression to get elements from index 2 to 4 (inclusive)
   **A:** `A[2:4]`

3. **Q:** How do you get the last element without knowing the array length?
   **A:** `A[end]`

### 2) Types and Functions

**Questions and Answers:**
1. **Q:** What will be the type of `result1` and `result2`?
   **A:** Both `Float64`

2. **Q:** What happens if you call `mystery_function(5, 2)` (integer as second argument)?
   **A:** `MethodError` (second argument must be `Float64`)

3. **Q:** Rewrite the function to accept any numeric types for both parameters
   **A:** 
   ```julia
   function mystery_function(x::Real, y::Real)
       px, py = promote(x, y)
       px > 0 ? px + py : px - py
   end
   ```

## Task 2: Benchmarking and Profiling

### 1) Basic Benchmarking

**Q:** Which approach is fastest? Explain why (2-3 sentences)
**A:** Loop/functional approaches are typically fastest (no temporaries); broadcasting may allocate intermediate arrays.

### 2) Performance Analysis

**Q:** Benchmark both versions with large n and compare performance
**A:** Type-stable version keeps `result::Float64`; faster with fewer allocations due to consistent concrete types.

## Task 3: Basic Array Operations

### 1) Array Creation and Indexing

**Complete the operations:**
```julia
zeros_array = zeros(3, 3)              # Create 3x3 matrix of zeros
ones_vector = ones(5)                  # Create vector of 5 ones
random_matrix = rand(2, 4)             # Create 2x4 matrix of random numbers
range_vector = collect(1:5)            # Create vector [1, 2, 3, 4, 5]

element_22 = A[2, 2]                   # Get element at row 2, column 2
second_row = A[2, :]                   # Get entire second row
first_column = A[:, 1]                 # Get entire first column
main_diagonal = [A[i, i] for i in 1:size(A, 1)]  # Get main diagonal elements [1, 5, 9]
```

### 2) Broadcasting and Element-wise Operations

**Implement the functions:**
```julia
function apply_function(x::Vector{Float64})
    return sin.(x) .+ cos.(2 .* x)
end

function matrix_transform(A::Matrix{Float64}, c::Float64)
    return (A .+ c) .* 2 .- 1
end

function count_positives(x::Vector{Float64})
    return sum(x .> 0.0)
end
```

**Q:** Explain what the `.` (dot) operator does in broadcasting
**A:** Element-wise operations + fusion, avoids temporaries.

## Task 4 (Optional): Tropical Max-Plus Algebra

### Questions and Answers:

1. **Q:** What are the outputs of the following expressions?
   **A:** 
   - `Tropical(1.0) + Tropical(3.0)` → `3.0ₜ`
   - `Tropical(1.0) * Tropical(3.0)` → `4.0ₜ`
   - `one(Tropical{Float64})` → `0.0ₜ`
   - `zero(Tropical{Float64})` → `-Infₜ`

2. **Q:** What is the type and supertype of `Tropical(1.0)`?
   **A:** Type: `Tropical{Float64}`; Supertype: `AbstractSemiring`

3. **Q:** Is `Tropical` a concrete type or an abstract type?
   **A:** `Tropical` is not concrete (needs parametrization)

4. **Q:** Is `Tropical{Real}` a concrete type or an abstract type?
   **A:** Abstract (because `Real` is abstract)

5. **Q:** Write a brief report on the performance of the tropical matrix multiplication.
   **A:** 100×100 tropical matrix multiplication is slower than BLAS-optimized Float64 operations; can be improved with specialized kernels or vectorization packages.

```toml
[Task1]
first_element = 10
last_element = 50
first_three = [10, 20, 30]
reverse_order = [50, 40, 30, 20, 10]
every_second = [10, 30, 50]
range_slice_2_4 = [20, 30, 40]
last_via_end = 50

[Task2]
result1_type = "Float64"
result2_type = "Float64"
call_5_2 = "error"

[Task4]
plus = "3.0ₜ"
times = "4.0ₜ"
one = "0.0ₜ"
zero = "-Infₜ"
type = "Tropical{Float64}"
supertype = "AbstractSemiring"
tropical_concrete = false
tropicalReal_concrete = false
```
