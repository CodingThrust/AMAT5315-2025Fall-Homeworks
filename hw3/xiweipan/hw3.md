# Homework 3

1. (Package Creation) https://github.com/isPANN/MyFirstPackage.jl

2. (Big-O Analysis) Analyze the time complexity of the following Fibonacci implementations.

    Consider this recursive Fibonacci function:
    ```julia
    fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
    ```
    What is the time complexity of this function in Big-O notation?

    Answer: $O(2^n)$.

    $T(n)=T(n-1)+T(n-2)+O(1)$

    $T(n)\le 2T(n-1)+O(1)\Rightarrow T(n)=O(2^n)$

    Now consider this alternative iterative implementation:
    ```julia
    function fib_while(n)
        a, b = 1, 1
        for i in 3:n
            a, b = b, a + b
        end
        return b
    end
    ```
    What is the time complexity of this function in Big-O notation?

    Answer: $O(n)$.