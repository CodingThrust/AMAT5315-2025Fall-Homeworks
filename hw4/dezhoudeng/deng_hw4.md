# Homework 4 2025/10/13 Dezhou Deng

## Problem 1

### (a)

```

julia> A = [10.0^10 0.0; 0.0 10.0^-10]
2×2 Matrix{Float64}:
 1.0e10  0.0
 0.0     1.0e-10

julia> cond(A)
1.0e20

```

ill-conditioned matrix

### (b)

```
julia> B = [10.0^10 0.0; 0.0 10.0^10]
2×2 Matrix{Float64}:
 1.0e10  0.0
 0.0     1.0e10

julia> cond(B)
1.0
```

well-conditioned matrix

### (c)

```
julia> C = [10.0^-10 0.0; 0.0 10.0^-10]
2×2 Matrix{Float64}:
 1.0e-10  0.0
 0.0      1.0e-10

julia> cond(C)
1.0
```

well-conditioned matrix

### (d)

```
julia> D = [1 2; 2 4]
2×2 Matrix{Int64}:
 1  2
 2  4

julia> cond(D)
2.517588727560788e16
```
ill-conditioned matrix

## Problem 2

```
julia> A = [2.0  1.0 -1.0  0.0  1.0;
            1.0  3.0  1.0 -1.0  0.0;
            0.0  1.0  4.0  1.0 -1.0;
           -1.0  0.0  1.0  3.0  1.0;
            1.0 -1.0  0.0  1.0  2.0]
5×5 Matrix{Float64}:
  2.0   1.0  -1.0   0.0   1.0
  1.0   3.0   1.0  -1.0   0.0
  0.0   1.0   4.0   1.0  -1.0
 -1.0   0.0   1.0   3.0   1.0
  1.0  -1.0   0.0   1.0   2.0

julia> b = [4.0, 6.0, 2.0, 5.0, 3.0]
5-element Vector{Float64}:
 4.0
 6.0
 2.0
 5.0
 3.0

julia> x = A \ b
5-element Vector{Float64}:
 -0.04651162790697683
  2.186046511627907
  0.30232558139534904
  0.8139534883720929
  2.2093023255813957
```

$x_{1}$ = -0.04651162790697683, $x_{2}$ =  2.186046511627907, $x_{3}$ = 0.30232558139534904, $x_{4}$ =  0.8139534883720929, x_${5}$ =  2.2093023255813957

## Problem 3

problem3.jl script:

```
using Makie, CairoMakie
using Polynomials

# Population data
years = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
population = [2374, 2250, 2113, 2120, 2098, 2052, 2057, 2028, 1934, 1827, 1765, 1696, 1641, 1594, 1588, 1612, 1581, 1591, 1604, 1587, 1588, 1600, 1800, 1640, 1687, 1655, 1786, 1723, 1523, 1465, 1200, 1062]

# Shift years for numerical stability
time = years .- 1990
y = population

# Fit third-degree polynomial
poly_fit = fit(time, y, 3)

# Get coefficients
coefficients = coeffs(poly_fit)
println("Polynomial coefficients:")
println("a0 = ", coefficients[1])
println("a1 = ", coefficients[2])
println("a2 = ", coefficients[3])
println("a3 = ", coefficients[4])

# Create figure and plot data points
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Year", ylabel="Population")
scatter!(ax, years, y, color=:blue, marker=:circle, markersize=10, label="Data")

# Plot fitted curve
poly = Polynomial([coefficients[1], coefficients[2], coefficients[3],coefficients[4]])
fitted_values = poly.(time)
lines!(ax, years, fitted_values, color=:red, label="Fitted Curve")

# Add legend and display
axislegend(; position=:lt)
fig
save("population_fit.png", fig)

# Predict 2024
predict_2024 = poly_fit(2024 - 1990)
println("2024 prediction: ", predict_2024)
```

By runing julia script "problem3.jl":

```
shell> julia problem3.jl
Polynomial coefficients:
a0 = 2451.5802139037437
a1 = -131.88430413372748
a2 = 7.607889485048743
a3 = -0.14811528059499654
2024 prediction: 940.7111295676117
```

The fitting plot is population_fit.png under the folder "dezhoudeng".
