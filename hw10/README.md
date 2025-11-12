# Homework 10

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. **(Integer Programming)** Use integer programming to solve the maximum independent set problem. The maximum independent set problem is a well-known NP-complete problem that asks for the largest subset of vertices in a graph such that no two vertices in the subset are connected by an edge. Use the Petersen graph to construct a test case (its maximum independent set is 4). 
   
   **Hint:** Use a boolean variable $x_i$ to indicate whether vertex $i$ is in the independent set.

2. **(Integer Programming - AI Allowed)** Improve the performance of crystal structure prediction by tuning the integer programming solver SCIP. It is highly recommended to read the thesis [^Achterberg2009] to better understand the [parameters in SCIP](https://scip.zib.de/doc/html/PARAMETERS.php). Try to achieve a performance improvement of at least 2x. Submit your code and a report of your tuning process.

3. **(Challenge: 0-1 Programming)** Factorize a 350-bit number using integer programming.
   
   **Setup:**
   ```bash
   cd example
   julia --project=. -e 'using Pkg; Pkg.instantiate();'
   julia --project=. factoring.jl
   ```
   
   **Hints:** 
   - This problem is also known as 0-1 programming
   - There are optimization tricks for 0-1 programming described in the thesis [^Achterberg2009]
   - Consider combining branching with state-of-the-art integer programming solvers like CPLEX and Gurobi
   - You can use `Primes.jl` to generate prime numbers:
     ```julia
     julia> using Primes; prevprime(60)
     59
     ```

[^Achterberg2009]: Achterberg, T., 2009. Constraint Integer Programming.