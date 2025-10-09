# Homework 3 Solutions

**Task 1:** I have successfully created a Julia package called `MyFirstPackage.jl` following the provided guide and uploaded it to GitHub with proper CI/CD setup, all tests passing, and test coverage above 80%. The GitHub repository link is: https://github.com/mingxuZhang2/MyFirstPackage.jl

**Task 2:** The recursive Fibonacci implementation `fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)` has time complexity **O($(\frac{1+\sqrt{5}}{2})^n$)** because each function call spawns two recursive calls, creating an exponential growth in the number of operations. The iterative implementation has time complexity **O(n)** because it uses a single loop that runs exactly (n-2) times, performing constant work in each iteration, making it significantly more efficient than the exponential recursive approach.
