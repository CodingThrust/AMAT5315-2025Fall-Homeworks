# Homework 10 - Solutions

**Author:** tengxianglin

## Contents

- `hw10.jl` - Complete Julia implementation

## Problems Completed

### ✅ Problem 1: Integer Programming for Maximum Independent Set
- Formulated MIS as integer program
- Variables: xᵢ ∈ {0, 1} for each vertex
- Constraints: xᵢ + xⱼ ≤ 1 for all edges (i,j)
- Tested on Petersen graph (expected size = 4)
- **Result:** Successfully found MIS of size 4 ✓

### ⚠️ Problem 2: SCIP Parameter Tuning
- **Status:** Template and strategy provided
- Detailed parameter tuning guidelines included
- Key parameters identified:
  - Branching strategy
  - Separation settings
  - Presolving options
  - Heuristics configuration
  - Node selection
- **Note:** Requires SCIP solver and crystal structure prediction benchmark

### ⚠️ Problem 3: Challenge - Semi-prime Factorization
- **Status:** Approach discussed, not fully implemented
- This is an A+ challenge problem
- Requires:
  - Binary encoding of multiplication
  - Sophisticated 0-1 programming techniques
  - Potentially hybrid IP-SAT approach
- Factorizing 40-bit semiprimes is extremely challenging

## How to Run

```bash
cd hw10/tengxianglin
julia hw10.jl
```

## Dependencies

Required Julia packages:
```julia
using JuMP
using GLPK  # or SCIP, Gurobi, CPLEX for advanced problems
using Graphs
using Printf
```

Install with:
```julia
using Pkg
Pkg.add(["JuMP", "GLPK", "Graphs"])
```

## Key Results

### Problem 1: Petersen Graph MIS
- Successfully solved using integer programming
- Found maximum independent set of size 4 (matches expected result)
- Verified solution correctness

### Problem 2: SCIP Tuning
- Comprehensive tuning strategy provided
- Parameter recommendations for different problem types
- Focus on branching, cuts, and heuristics

### Problem 3: Factorization Challenge
- Detailed approach and formulation discussed
- Requires advanced implementation for A+ credit
- Binary multiplication constraints are the key challenge

## Notes

- Problem 1 fully implemented and tested
- Problems 2 and 3 are advanced challenges requiring specialized tools
- Code includes templates and extensive documentation
- For A+ credit, problems 2 and 3 require significant additional work
