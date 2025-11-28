# Homework 3 — tengxianglin

## 1. Package Creation
Repository link: https://github.com/tengxianglin/MyFirstPackage.jl

What I did:
- Used PkgTemplates to bootstrap the package (MIT license, Git, GitHub Actions, Codecov, Documenter).
- Pushed to GitHub; CI on the default branch is passing.
- Wrote tests for the exported API and ran them locally with coverage enabled.
- Turned on Codecov for the repo; coverage ≥ 80% is reported.
- Did not submit to General (kept unregistered).

Quick steps I followed (high level):
- Create template with PkgTemplates → generate MyFirstPackage.
- Initialize repo (if needed), set remote, and push to GitHub.
- Add tests under test/, run pkg> test --coverage locally.
- Connect Codecov to the repo; upload coverage from CI.

## 2. Big-O Analysis

- Recursive Fibonacci (fib(n) = n ≤ 2 ? 1 : fib(n−1) + fib(n−2))
  - Time: Θ(φ^n) (exponential), φ = (1+√5)/2; equivalently O(2^n) as a loose bound.
  - Space: O(n) due to recursion depth.
  - Why: the recurrence T(n) = T(n−1) + T(n−2) + O(1) leads to a number of calls on the order of F_n = Θ(φ^n).

- Iterative Fibonacci (loop from 3 to n)
  - Time: O(n) — single pass with constant work per iteration.
  - Space: O(1) — only a few running variables are stored.
