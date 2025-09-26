# Homework 3
1.

    https://github.com/HsiaHongKuan/MyFirstPackage.jl

2. The time complexity of the recursive Fibonacci function:
```julia
fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
```
is $O((\frac{\sqrt{5}+1}{2})^n)$

The time complexity of the second method:
```julia
function fib_while(n)
    a, b = 1, 1
    for i in 3:n
        a, b = b, a + b
    end
    return b
end
```
is $O(n)$.
