# Homework 3
Dezhou Deng
2025/09/25

## Problem 1 

My GitHub repository link:
https://github.com/Honfoglalas/MyFirstPackage.jl

## Problem 2

### 1)
    fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
### Answer

    O(2^n), for it has fib(n - 1) + fib(n - 2), which are two recursion parts.
### Edited on 2025/10/09:
    Assume the time cost of calculating fib(n), which has the exponential form k^n with unknown k.
    Each time we have fib(n) = fib(n - 1) + fib(n - 2). 

$$
\begin{gather}
k^{n} = k^{n-1} + k^{n-2}. 
\end{gather}
$$

$$
\begin{gather}
k = \dfrac{1+\sqrt{5}}{2}.
\end{gather}
$$

    Hence, the answer is
$$
\begin{gather}
O((\dfrac{1+\sqrt{5}}{2})^{n})
\end{gather}
$$

### 2)

    julia
    function fib_while(n)
        a, b = 1, 1
        for i in 3:n
            a, b = b, a + b
        end
        return b
    end

### Answer

    O(n), for it has only one iteration.

$$
\begin{gather}
 T(n) = O(1) + O(n) \times O(1) 
\end{gather}
$$

The first O(1) is for initialization of a,b. The term O(n) is approximated from O(n-2). Because from i=3, there are n-2 iterations. The last O(1) is for the a,b = b, and a+b.


