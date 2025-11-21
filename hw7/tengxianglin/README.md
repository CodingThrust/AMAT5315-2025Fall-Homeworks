# Homework 7 - Solutions

**Author:** tengxianglin

## Contents

- `hw7.jl` - Complete Julia implementation

## Problems Completed

### ✅ Problem 1: Ground State Energy (Anti-ferromagnetic Ising Model)
- Implemented simulated annealing algorithm
- Constructed Fullerene graph with 60 vertices
- Found ground state energy through optimization

### ✅ Problem 2: Spectral Gap Analysis
- **Part a:** Analyzed spectral gap vs temperature (T = 0.1 to 2.0)
- **Part b:** Analyzed spectral gap vs system size (n = 4 to 12)
- Used eigenvalue decomposition for small systems

### ⚠️ Problem 3: Challenge (Parallel Tempering)
- **Status:** Template provided, full implementation not completed
- **Note:** This is an A+ challenge problem requiring advanced algorithms

## How to Run

```bash
cd hw7/tengxianglin
julia hw7.jl
```

## Dependencies

Required Julia packages:
```julia
using Graphs
using ProblemReductions
using LinearAlgebra
using SparseArrays
using Random
```

Install with:
```julia
using Pkg
Pkg.add(["Graphs", "ProblemReductions", "LinearAlgebra", "SparseArrays", "Random"])
```

## Notes

- Simulated annealing runs multiple trials to ensure convergence to ground state
- Spectral gap calculations limited to small graphs due to exponential complexity
- Challenge problem (parallel tempering) requires significant additional work for A+ credit
