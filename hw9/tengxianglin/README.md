# Homework 9 - Solutions

**Author:** tengxianglin

## Contents

- `hw9.jl` - Complete Julia implementation

## Problems Completed

### ✅ Problem 1: Circuit SAT - Half Adder to Spin Glass
- Reduced half adder circuit to spin glass problem
- Created spin glass representation with couplings and local fields
- Verified truth table for all input combinations
- Variables: A (input 1), B (input 2), S (sum), C (carry)

### ✅ Problem 2: Spin Dynamics Simulation
- Implemented spin dynamics using gradient descent
- Fixed outputs: S = 0, C = 1
- Successfully recovered inputs: A = 1, B = 1
- Used thermal noise and multiple trials for robustness

### ✅ Problem 3: Greedy Algorithm for Maximum Independent Set
- Implemented greedy MIS algorithm (minimum degree heuristic)
- Generated random 3-regular graphs at various sizes (n = 10 to 50)
- Analyzed approximation ratio scaling
- Computed exact MIS for small graphs for comparison

## How to Run

```bash
cd hw9/tengxianglin
julia hw9.jl
```

## Dependencies

Required Julia packages:
```julia
using Graphs
using LinearAlgebra
using Random
using Statistics
using Printf
```

## Key Results

### Problem 1: Half Adder Encoding
- Successfully encoded XOR and AND gates as spin glass constraints
- Energy minimization corresponds to satisfying circuit logic

### Problem 2: Input Recovery
- From outputs S=0, C=1, correctly recovered inputs A=1, B=1
- Spin dynamics converges to valid solution

### Problem 3: MIS Analysis
- Greedy algorithm provides ~1/4 approximation for 3-regular graphs
- Approximation ratio does NOT remain constant as graph size increases
- Theoretical bound: α ≥ n/4 for 3-regular graphs

## Notes

- All three main problems completed
- Code includes detailed comments and explanations
- Results verified against known circuit behavior and graph theory
