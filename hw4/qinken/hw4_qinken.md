# HW4
## Question1
```
using LinearAlgebra

A = [1e10 0; 0 1e-10]
B = [1e10 0; 0 1e10]
C = [1e-10 0; 0 1e-10]
D = [1 2; 2 4]

cond_A = cond(A)
cond_B = cond(B)
cond_C = cond(C)
cond_D = cond(D)

println("cond(A) = ", cond_A)
println("cond(B) = ", cond_B)
println("cond(C) = ", cond_C)
println("cond(D) = ", cond_D)

```
a is ill-conditioned
b is well-conditioned
c is well-conditioned
d is ill-conditioned

## Question2
```
using LinearAlgebra

A = [
    2  1 -1  0  1;
    1  3  1 -1  0;
    0  1  4  1 -1;
   -1  0  1  3  1;
    1 -1  0  1  2
]

b = [4, 6, 2, 5, 3]


x_float = A \ b
println("x (float) = ", x_float)


A_rat = Rational.(A)  
b_rat = Rational.(b)
x_rat = A_rat \ b_rat
println("x (rational) = ", x_rat)

```
result:
```
  x₁ = −0.04651163
  x₂ = 2.18604651
  x₃ = 0.30232558
  x₄ = 0.81395349
  x₅ = 2.20930233
```

## Task3
```using LinearAlgebra, Polynomials
using CairoMakie


years = 1990:2021
population = [
    2374,2250,2113,2120,2098,2052,2057,2028,1934,1827,1765,1696,1641,1594,1588,1612,
    1581,1591,1604,1587,1588,1600,1800,1640,1687,1655,1786,1723,1523,1465,1200,1062
]


x = years .- 1990


p = fit(x, population, 3)
println("Fitted polynomial: ", p)


x_pred = 2024 - 1990
y_pred = p(x_pred)
println("Predicted newborn population in 2024 ≈ ", y_pred)


fig = Figure()
ax = Axis(fig[1, 1], xlabel="Year", ylabel="Population (×10⁴)")
scatter!(ax, years, population, color=:blue, markersize=8, label="Data")

x_fit = range(0, stop=years[end]-1990, length=300)
y_fit = p.(x_fit)
lines!(ax, x_fit .+ 1990, y_fit, color=:red, linewidth=2, label="Fitted Curve")

axislegend(; position=:rb)
fig
save("china_newborn_fit.png", fig)
```
Fitted polynomial: 2451.58 - 131.884*x + 7.60789*x^2 - 0.148115*x^3
Predicted newborn population in 2024 ≈ 940.7111295676123