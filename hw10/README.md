# Homework 10

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. **(Integer Programming)** Use integer programming to solve the maximum independent set problem. The maximum independent set problem is a well-known NP-complete problem that asks for the largest subset of vertices in a graph such that no two vertices in the subset are connected by an edge. Use the Petersen graph to construct a test case (its maximum independent set is 4). 
   
   **Hint:** Use a boolean variable $x_i$ to indicate whether vertex $i$ is in the independent set.

2. **(Integer Programming - AI Allowed)** Improve the performance of crystal structure prediction by tuning the integer programming solver SCIP. It is highly recommended to read the thesis [^Achterberg2009] to better understand the [parameters in SCIP](https://scip.zib.de/doc/html/PARAMETERS.php). Try to achieve a performance improvement of at least 2x. Submit your code and a report of your tuning process.

3. **(Challenge: 0-1 Programming)** Factorize ~40-bit semiprime (the product of two 20-bit primes) in the dataset file `data/numbers_20x20.txt` using integer programming, beating the `factoring.jl` baseline on the same setup.
   
   **Setup:**
   ```bash
   cd example
   julia --project=. -e 'using Pkg; Pkg.instantiate();'
   julia --project=. example.jl
   ```
   
   **Hints:** 
   - This problem is also known as 0-1 programming
   - There are optimization tricks for 0-1 programming described in the thesis [^Achterberg2009]
   - Consider combining branching with state-of-the-art integer programming solvers like CPLEX and Gurobi.
   - The dataset is provided as a plain text file (.txt) in the `data` folder, where each line contains one instance in the following format: `m n N p q` separated by spaces. 
   Here m and n are the bit lengths of the factors, N is the semiprime to be factorized, and p, q are its prime factors.
   Additional instances of various bit-lengths are included in the data folder to test and benchmark your implementation across different problem scales.
   - The `example.jl` file contains the code to (1) read the dataset file and (2) generate a random semiprime number given the bit length of the two primes.
   - The `factoring.jl` file contains the example code to factorize a number using integer programming need to be optimized.
   
   Note: by solving this problem, you can get an A+.

[^Achterberg2009]: Achterberg, T., 2009. Constraint Integer Programming.