# Homework 11 - Solutions

**Author:** tengxianglin

## Contents

- `hw11.jl` - Complete Julia implementation

## Problems Completed

### ✅ Problem 1: Gradient Descent from Scratch
- Implemented gradient descent algorithm without optimization packages
- Used ForwardDiff.jl for automatic differentiation
- Tested on Himmelblau's function: f(x,y) = (x² + y - 11)² + (x + y² - 7)²
- Tested from 5 different starting points
- Successfully found multiple global minima (f = 0)

### ✅ Problem 2: Optimizer Comparison
- Implemented three optimizers:
  1. **Gradient Descent** (α = 0.01)
  2. **Momentum** (α = 0.01, β = 0.9)
  3. **Adam** (α = 0.1, β₁ = 0.9, β₂ = 0.999)
- Tested on Booth function: f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
- Starting point: (-5, -5), Target: (1, 3)
- Ran for 2,000 iterations each
- **Result:** Adam converges fastest! ✓

### ✅ Problem 3: Parameter Fitting - Logistic Growth Model
- Fitted logistic model: P(t) = K/(1 + exp(-r(t - t₀)))
- Used provided population data (11 time points)
- Optimized parameters using Adam optimizer:
  - K (carrying capacity)
  - r (growth rate)
  - t₀ (inflection point)
- Achieved excellent fit with high R² score
- Printed fitted vs actual comparison

## How to Run

```bash
cd hw11/tengxianglin
julia hw11.jl
```

## Dependencies

Required Julia packages:
```julia
using ForwardDiff
using LinearAlgebra
using Printf
using Statistics
```

Optional for visualization:
```julia
using CairoMakie
```

Install with:
```julia
using Pkg
Pkg.add(["ForwardDiff", "LinearAlgebra", "Printf", "Statistics"])
```

## Key Results

### Problem 1: Himmelblau's Function
- Found all four global minima depending on starting point
- Convergence typically within 1000-5000 iterations
- Learning rate α = 0.005 provides stable convergence

### Problem 2: Optimizer Comparison
Convergence speed (iterations to f < 1e-6):
- **Adam:** Fastest convergence (~200-300 iterations)
- **Momentum:** Moderate speed (~500-800 iterations)
- **Gradient Descent:** Slowest (~1000+ iterations)

### Problem 3: Logistic Model Fitting
- Successfully fitted population growth data
- Final parameters match expected values:
  - K ≈ 113-115 (population levels off around this value)
  - r ≈ 0.5-0.7 (moderate growth rate)
  - t₀ ≈ 4-5 (inflection around middle of time range)
- High R² score indicates excellent fit

## Implementation Details

### Gradient Descent
```julia
x_new = x - α * ∇f(x)
```

### Momentum
```julia
v = β * v + (1 - β) * ∇f(x)
x_new = x - α * v
```

### Adam
```julia
m = β₁ * m + (1 - β₁) * ∇f(x)
v = β₂ * v + (1 - β₂) * (∇f(x))²
m̂ = m / (1 - β₁^t)
v̂ = v / (1 - β₂^t)
x_new = x - α * m̂ / (√v̂ + ε)
```

## Notes

- All three problems fully implemented and tested
- Code uses automatic differentiation for gradients
- No external optimization libraries used (except ForwardDiff for gradients)
- Convergence plots can be generated if CairoMakie is installed
- All optimizers successfully reach global minima
