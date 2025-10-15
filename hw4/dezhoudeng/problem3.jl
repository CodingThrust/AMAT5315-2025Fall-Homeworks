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