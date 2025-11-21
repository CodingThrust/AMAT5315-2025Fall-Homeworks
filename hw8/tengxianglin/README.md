# Homework 8 - Solutions

**Author:** tengxianglin

## Contents

- `hw8.md` - Complete solutions in Markdown format with embedded Julia code

## Problems Completed

### ✅ Problem 1: Einsum Notation
All einsum notations provided:
1. Matrix multiplication with transpose: `einsum"ij,kj->ik"`
2. Summing over all elements: `einsum"ij->"`
3. Element-wise multiplication: `einsum"ij,ij,ij->ij"`
4. Kronecker product: `einsum"ij,kl,mn->ijklmn"`

### ✅ Problem 2: Contraction Order
- General strategy and algorithm provided
- Complexity analysis included
- Optimal order principles explained

### ✅ Problem 3: Partition Function
- Implemented partition function calculation for AFM Ising model
- Scanned inverse temperature β from 0.1 to 2.0
- Applied to Fullerene graph (60 vertices, 90 edges)

### ⚠️ Problem 4: Challenge (Better Contraction Order Algorithm)
- **Status:** Not attempted
- **Note:** This is an A+ challenge requiring novel algorithm development

## Dependencies

Required Julia packages:
```julia
using Graphs
using ProblemReductions
using OMEinsum
using LinearAlgebra
```

## Notes

- Problem 1: Theoretical einsum notation definitions
- Problem 2: General approach (specific diagram not provided)
- Problem 3: Includes working Julia code for partition function
- Challenge problem requires developing state-of-the-art contraction order algorithms
