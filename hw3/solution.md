# Homework 3 Submission

1) Package Creation: MyFirstPackage.jl
- Repository link: https://github.com/YourUsername/MyFirstPackage.jl
- Status checklist:
  - Generated via PkgTemplates (License, Git, GitHubActions, Codecov, Documenter)
  - Pushed to GitHub; Actions run and tests pass on default branch
  - Coverage ≥ 80% and uploaded to Codecov
  - Do NOT register to General

Minimal reproduce steps (Julia REPL):
```julia
using Pkg; Pkg.add("PkgTemplates"); using PkgTemplates
tpl = Template(; user="YourUsername", authors="Your Name", julia=v"1",
    plugins=[License(; name="MIT"), Git(; ssh=true), GitHubActions(; x86=false),
             Codecov(), Documenter{GitHubActions}()])
tpl("MyFirstPackage")
```
Then:
- cd ~/.julia/dev/MyFirstPackage; git init (if needed); add remote; push
- Add tests to test/ to exercise exported APIs; run locally:
```julia
(@v1.10) pkg> test --coverage
```
- Enable Codecov for the repo; set CODECOV_TOKEN if required

2) Big-O Analysis
- Recursive Fibonacci (fib(n) = n <= 2 ? 1 : fib(n-1) + fib(n-2))
  - Time: O(φ^n) ≈ O(2^n)
- Iterative Fibonacci (loop from 3:n)
  - Time: O(n)
- Local test and coverage:
```julia
(@v1.10) pkg> test --coverage
```
Tip: Add tests under test/ to cover public APIs; aim for small, focused tests.

Reminder: Do not register to General.

2) Big-O Analysis
- Recursive Fibonacci
  - Code: fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
  - Time: Θ(φ^n) (exponential), where φ ≈ 1.618; Big-O: O(2^n) is acceptable
  - Space: O(n) due to recursion depth
- Iterative Fibonacci
  - Code: fib_while(n) with a loop from 3:n
  - Time: O(n)
  - Space: O(1)
