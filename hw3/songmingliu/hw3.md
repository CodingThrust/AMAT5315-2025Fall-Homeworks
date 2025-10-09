# Homework 3

## 1. Package Creation
https://github.com/Zzero76/MyFirstPackage.jl.git 

## 2. Big-O Analysis
2.1 Recursive Fibonacci function:
fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)

julia> for n in [10, 20, 25, 30, 35]
           time_taken = @belapsed fib_recursive($n)
               println("n=$n, time=$(time_taken) seconds")
               end
n=10, time=2.0718214285714286e-7 seconds
n=20, time=2.5811e-5 seconds
n=25, time=0.000286435 seconds
n=30, time=0.003199794 seconds
n=35, time=0.035541253 seconds

Time Complexity: Θ(((1+√5)/2)^n)


2.2 Iterate Fibonacci function:
function fib_while(n)
    a, b = 1, 1
    for i in 3:n
        a, b = b, a + b
    end
    return b
end

julia> for n in [10, 100, 1000, 10000]
           time_taken = @belapsed fib_iterative($n)
               println("n=$n, time=$(time_taken) seconds")
               end
n=10, time=7.427427427427427e-9 seconds
n=100, time=6.646319018404908e-8 seconds
n=1000, time=6.523374233128834e-7 seconds
n=10000, time=6.46e-6 seconds

time complexity: O(n)
