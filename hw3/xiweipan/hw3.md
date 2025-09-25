# Homework 3

1. (Package Creation) https://github.com/isPANN/MyFirstPackage.jl

2. (Big-O Analysis) Analyze the time complexity of the following Fibonacci implementations.

    Consider this recursive Fibonacci function:
    ```julia
    fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
    ```
    What is the time complexity of this function in Big-O notation?

    Answer: a rough estimate is $O(2^n)$.

    $T(n)=T(n-1)+T(n-2)+O(1)$

    $T(n)\le 2T(n-1)+O(1)\Rightarrow T(n)=O(2^n)$

    To be more accurate, let the time complexity be `T(n)` and the total number of calls be `C(n)`.

    Recurrence:
    - `C(1) = C(2) = 1`
    - `C(n) = C(n-1) + C(n-2) + 1`

    By induction:
    $C(n) = 2F_n - 1$
    where $F_n$ is the $n$-th Fibonacci number.

    From Binet’s formula:
    $$F_n = \frac{\varphi^n - \psi^n}{\sqrt{5}}, \quad 
    \varphi = \tfrac{1+\sqrt{5}}{2}, \ |\psi| < 1$$

    Thus:
    $$F_n = \Theta(\varphi^n), \quad \varphi = \tfrac{1+\sqrt{5}}{2}$$

    Final result:
    $$T(n) = \Theta(F_n) = \Theta(\varphi^n), \quad \varphi = \tfrac{1+\sqrt{5}}{2}$$

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
