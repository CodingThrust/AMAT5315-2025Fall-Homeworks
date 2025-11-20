# Homework 3

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. (Package Creation) Following the provided [guide](https://scfp.jinguo-group.science/chap1-julia/julia-release.html), create a package called `MyFirstPackage.jl` and upload it to your GitHub account. Requirements:
    - Set up CI/CD properly with all tests passing and test coverage above 80%
    - Submit the GitHub repository link, not the package files themselves
    - **Warning:** Do not complete the final step in the guide that registers the package to the General registry

2. (Big-O Analysis) Analyze the time complexity of the following Fibonacci implementations.

    Consider this recursive Fibonacci function:
    ```julia
    fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
    ```
    What is the time complexity of this function in Big-O notation?

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