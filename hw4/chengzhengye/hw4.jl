using Polynomials, CairoMakie


# 1. Prepare data
years = 1990:2021
population = [2374, 2250, 2113, 2120, 2098, 2052, 2057, 2028, 1934, 1827, 1765, 
              1696, 1641, 1594, 1588, 1612, 1581, 1591, 1604, 1587, 1588, 1600, 
              1800, 1640, 1687, 1655, 1786, 1723, 1523, 1465, 1200, 1062]

# Create shifted x-values (years since 1990)
x = collect(years .- 1990)
y = population

# 2. Perform 3rd-degree polynomial fitting
poly_fit = fit(x, y, 3)
println("Fitted polynomial coefficients (a0 + a1*x + a2*x² + a3*x³):")
# Access coefficients directly through the 'coeffs' field (compatible with most versions)
println(poly_fit.coeffs)

# 3. Visualization
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1], 
          xlabel = "Years since 1990", 
          ylabel = "Newborn Population (10,000)",
          title = "China's Newborn Population (1990-2021) with 3rd-Degree Fit")

scatter!(ax, x, y, color = :blue, markersize = 8, label = "Actual Data")
x_fit = range(minimum(x), maximum(x), length=200)
y_fit = poly_fit.(x_fit)
lines!(ax, x_fit, y_fit, color = :red, linewidth = 2, label = "3rd-Degree Polynomial")

axislegend(ax, position = :lt)
save("population_fit.png", fig)
display(fig)

# 4. Predict 2024 population
x_2024 = 2024 - 1990  # Years since 1990
prediction_2024 = poly_fit(x_2024)
println("\nPredicted 2024 newborn population: ", round(prediction_2024, digits=1), " (10,000 people)")