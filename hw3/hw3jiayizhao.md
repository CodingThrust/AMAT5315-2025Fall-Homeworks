# Homework 3

## Problem 1
**GitHub Repository**: https://github.com/jyzhau/MyFirstPackage.jl

## Problem 2: Time Complexity Analysis of Fibonacci Algorithms

### Part 1: Recursive Fibonacci Implementation

#### Theoretical Analysis

First, we calculate theoretically and then verify our theory through experiments.

#### Tree Diagram Analysis

Let's visualize the computation process of the recursive Fibonacci function using a tree structure:

```
fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
├── fib(n - 1) = n - 1 <= 2 ? 1 : fib(n - 2) + fib(n - 3)
└── fib(n - 2) = n - 2 <= 2 ? 1 : fib(n - 3) + fib(n - 4)
    ...
    └── fib(n - (n-3)) = n - (n-3) <= 2 ? 1 : fib(n - n + 2) + fib(n - n + 1)
```

#### Pattern Recognition

By counting the nodes in the tree, we can identify the following pattern:

| Function Call | Number of Occurrences |
|--------------|----------------------|
| `fib(n)`     | 1                    |
| `fib(n-1)`   | 1                    |
| `fib(n-2)`   | 2                    |
| `fib(n-3)`   | 3                    |
| `fib(n-4)`   | 5                    |
| `fib(n-k)`   | `fib(k)`             |

These counts represent nodes in the tree diagram. Since computing each node requires a constant amount of operations from its child nodes, the time complexity is **O(fib(n))**.

According to research, for large values of n, this approximates to **O(1.618^n)** (where 1.618 is the golden ratio).

### Experimental Verification

Let's verify our hypothesis through empirical measurement:

#### Implementation

```julia
function fib_recursive(n)
    n <= 2 ? 1 : fib_recursive(n - 1) + fib_recursive(n - 2)
end

function measure_with_time()
    println("Run Time Measurements")
    for n in 30:47
        print("fib($n) = ")
        result = @time fib_recursive(n)
        println("result: $result")
    end
end
```

#### Experimental Results

```julia
julia> measure_with_time()
Run Time Measurements
fib(30) =   0.004504 seconds
result: 832040
fib(31) =   0.006488 seconds
result: 1346269
fib(32) =   0.011033 seconds
result: 2178309
fib(33) =   0.019864 seconds
result: 3524578
fib(34) =   0.028244 seconds
result: 5702887
fib(35) =   0.045799 seconds
result: 9227465
fib(36) =   0.073851 seconds
result: 14930352
fib(37) =   0.123557 seconds
result: 24157817
fib(38) =   0.202646 seconds
result: 39088169
fib(39) =   0.389508 seconds
result: 63245986
fib(40) =   0.536954 seconds
result: 102334155
fib(41) =   0.904852 seconds
result: 165580141
fib(42) =   1.421690 seconds
result: 267914296
fib(43) =   2.223354 seconds
result: 433494437
fib(44) =   3.688630 seconds
result: 701408733
fib(45) =   6.251350 seconds
result: 1134903170
fib(46) =   9.517873 seconds
result: 1836311903
fib(47) =  15.431084 seconds
result: 2971215073
```

### Analysis of Results

Examining the growth rate between consecutive measurements:

| Transition | Time Ratio |
|------------|------------|
| fib(31)/fib(30) | 1.44 |
| fib(32)/fib(31) | 1.70 |
| fib(33)/fib(32) | 1.80 |
| fib(40)/fib(39) | 1.38 |
| fib(45)/fib(44) | 1.69 |
| fib(47)/fib(46) | 1.62 |

The observed growth rate approximately matches our theoretical prediction of **≈1.618** (the golden ratio), confirming that the time complexity of the recursive Fibonacci implementation is indeed **O(φ^n)** where φ ≈ 1.618.

### Part 2: Iterative Fibonacci Implementation

#### Algorithm Implementation

```julia
function fib_while(n)
    a, b = 1, 1
    for i in 3:n
        a, b = b, a + b
    end
    return b
end
```

#### Complexity Analysis

In this iterative approach:
- Each iteration performs 2 assignments and 1 addition operation
- The loop runs from 3 to n, which is (n - 2) iterations
- Total operations: O(n) steps

Therefore, the time complexity of the iterative Fibonacci algorithm is **O(n)**.