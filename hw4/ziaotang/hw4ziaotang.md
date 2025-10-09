## 1

``` julia
julia> A = [10^10 0;0 10^(-10)];

julia> cond(A)
1.0e20

julia> B = [10^10 0;0 10^10];

julia> cond(B)
1.0

julia> C = [10^(-10) 0;0 10^(-10)];

julia> cond(C)
1.0

julia> D = [1 2;2 4];

julia> cond(D)
2.517588727560788e16
```

A and D are ill-conditioned, B and C are well-conditioned.



## 2

```julia
julia> A = [
            2   1  -1   0   1;
            1   3   1  -1   0;
            0   1   4   1  -1;
           -1   0   1   3   1;
            1  -1   0   1   2
       ]

julia> b = [4, 6, 2, 5, 3];

julia> x = A\b
5-element Vector{Float64}:
 -0.04651162790697683
  2.186046511627907
  0.30232558139534904
  0.8139534883720929
  2.2093023255813957
```

## 3

```julia
using CairoMakie, LinearAlgebra
using Makie
using Polynomials
years = collect(1990:2021);
population = [2374, 2250, 2113, 2120, 2098, 2052, 2057, 2028, 1934, 1827, 
       1765, 1696, 1641, 1594, 1588, 1612, 1581, 1591, 1604, 1587, 
       1588, 1600, 1800, 1640, 1687, 1655, 1786, 1723, 1523, 1465, 
       1200, 1062];

x = Float64.(years .- 1990);

X = hcat(ones(length(x)), x, x.^2, x.^3);
y = Float64.(population);

Q, R = qr(X);
coef = R \ (Matrix(Q)' * y);
a0, a1, a2, a3 = coef;
t = 2024 - 1990;
pred = a0 + a1*t + a2*t^2 + a3*t^3;

println("a0, a1, a2, a3 = ", coef);
println("Prediction value = ", pred);

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time", ylabel="Population")
scatter!(ax, years, population, color=:blue, marker=:circle, markersize=20, label="Data")
 
 # Example polynomial fit (replace with your fitted coefficients)
poly = Polynomial([a0, a1, a2, a3])  # Replace with your coefficients
fitted_values = poly.(x)
lines!(ax, years, fitted_values, color=:red, label="Fitted Curve")
 
# Add legend and display
axislegend(; position=:lt)
fig  # Display figure
save("D:\\juliahw\\population_fit.png", fig)  # Save plot
```

Output:

```
a0, a1, a2, a3 = [2451.5802139037473, -131.88430413372816, 7.607889485048778, -0.14811528059499718]
Prediction value = 940.7111295676077
```

