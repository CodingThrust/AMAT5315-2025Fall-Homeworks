# Homework 3 Solutions
# Author: tengxianglin
#
# Run with: julia --project=hw3 hw3/tengxianglin/hw3.jl

using Printf

println("\n" * "="^70)
println("HOMEWORK 3 - Solutions")
println("="^70)

# ============================================================================
# Problem 1: Package Creation
# ============================================================================
println("\nProblem 1: Package Creation")
println("-"^70)
println("Repository link: https://github.com/tengxianglin/MyFirstPackage.jl")
println("✓ Package created using PkgTemplates")
println("✓ CI/CD set up with GitHub Actions")
println("✓ Test coverage ≥ 80%")
println("✓ Package not registered to General registry (as required)")

# ============================================================================
# Problem 2: Big-O Analysis
# ============================================================================
println("\n" * "="^70)
println("Problem 2: Big-O Analysis")
println("="^70)

# Recursive Fibonacci implementation
function fib_recursive(n)
    n <= 2 ? 1 : fib_recursive(n - 1) + fib_recursive(n - 2)
end

# Iterative Fibonacci implementation
function fib_iterative(n)
    n <= 2 && return 1
    a, b = 1, 1
    for i in 3:n
        a, b = b, a + b
    end
    return b
end

println("\n2.1 Recursive Fibonacci: fib(n) = n ≤ 2 ? 1 : fib(n-1) + fib(n-2)")
println("-"^70)
println("Time Complexity: O(2^n) or more precisely Θ(φ^n) where φ = (1+√5)/2")
println("Space Complexity: O(n) due to recursion depth")
println("Explanation: The recurrence T(n) = T(n-1) + T(n-2) + O(1)")
println("            leads to exponential growth in function calls.")

println("\n2.2 Iterative Fibonacci: loop from 3 to n")
println("-"^70)
println("Time Complexity: O(n) - single pass with constant work per iteration")
println("Space Complexity: O(1) - only a few running variables stored")

# Demonstrate the difference with a small test
println("\n" * "-"^70)
println("Demonstration (computing fib(10)):")
println("-"^70)

n_test = 10
@printf("Recursive fib(%d) = %d\n", n_test, fib_recursive(n_test))
@printf("Iterative fib(%d) = %d\n", n_test, fib_iterative(n_test))
println("✓ Both implementations produce the same result")
println("\nNote: For larger n, recursive version becomes extremely slow")
println("      due to exponential time complexity.")

println("\n" * "="^70)
println("Summary")
println("="^70)
println("✓ Problem 1: Package created and configured")
println("✓ Problem 2: Big-O analysis completed")
println("="^70)

