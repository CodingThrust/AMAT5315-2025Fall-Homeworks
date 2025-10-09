using LinearAlgebra
using Plots

# --- Data ---
years = [
    1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,
    2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
    2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
    2020,2021
]
y = [
    2374,2250,2113,2120,2098,2052,2057,2028,1934,1827,
    1765,1696,1641,1594,1588,1612,1581,1591,1604,1587,
    1588,1600,1800,1640,1687,1655,1786,1723,1523,1465,
    1200,1062
]

x = years .- 1990

A = hcat(ones(length(x)), x, x.^2, x.^3)

# Solve least squares A * a ≈ y
a = A \ y
println("Cubic model with x = year - 1990:")
println("y = a0 + a1*x + a2*x^2 + a3*x^3")
println("a = ", a)

# Predict for 2024
x_pred = 2024 - 1990
y_pred = a[1] + a[2]*x_pred + a[3]*x_pred^2 + a[4]*x_pred^3
println("Prediction for 2024: ", round(y_pred; digits=1), " ×10^4 people (≈ ",
        round(y_pred/10; digits=2), " million)")

# Plot
xmin, xmax = minimum(x), maximum(x)
xs = range(xmin, stop = xmax + 5, length = 300)
ys = a[1] .+ a[2].*xs .+ a[3].*xs.^2 .+ a[4].*xs.^3

plt = scatter(years, y; label="Data", markerstrokewidth=0, legend=:bottomleft)
plot!(plt, xs .+ 1990, ys; label="Fitted Curve", linewidth=2)
vline!(plt, [2024]; l=(:dash, 0.8), label="")
scatter!(plt, [2024], [y_pred]; label="2024 pred: $(round(y_pred; digits=1))", marker=:xcross, ms=6)

xlabel!("Year")
ylabel!("Newborn population (×10^4)")
title!("HW4 by Xiwei Pan: China Newborn Population")

savefig(plt, "china_newborn_fit.png")