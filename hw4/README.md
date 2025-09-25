# Homework 4

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. **(Condition Number Analysis)** Classify each of the following matrices as well-conditioned or ill-conditioned by computing their condition numbers:

```math
\text{(a)} \quad \left(\begin{matrix}10^{10} & 0\\ 0 & 10^{-10}\end{matrix}\right)
```

```math
\text{(b)} \quad \left(\begin{matrix}10^{10} & 0\\ 0 & 10^{10}\end{matrix}\right)
```

```math
\text{(c)} \quad \left(\begin{matrix}10^{-10} & 0\\ 0 & 10^{-10}\end{matrix}\right)
```

```math
\text{(d)} \quad \left(\begin{matrix}1 & 2\\ 2 & 4\end{matrix}\right)
```

2. **(Solving Linear Equations)** Solve the following system of linear equations using Julia. Let $x_1, x_2, \ldots, x_5$ be real numbers:

```math
   \begin{align*}
   2x_1 + x_2 - x_3 + 0x_4 + x_5 &= 4, \\
   x_1 + 3x_2 + x_3 - x_4 + 0x_5 &= 6, \\
   0x_1 + x_2 + 4x_3 + x_4 - x_5 &= 2, \\
   -x_1 + 0x_2 + x_3 + 3x_4 + x_5 &= 5, \\
   x_1 - x_2 + 0x_3 + x_4 + 2x_5 &= 3.
   \end{align*}
```

3. **(Polynomial Data Fitting)** Analyze the newborn population data in China and perform polynomial regression.

   **Dataset:** China's newborn population (1990-2021)
   - First column: Year
   - Second column: Population in 万 (×10⁴ people)

   ```text
   Year    Population (万)
   1990    2374
   1991    2250
   1992    2113
   1993    2120
   1994    2098
   1995    2052
   1996    2057
   1997    2028
   1998    1934
   1999    1827
   2000    1765
   2001    1696
   2002    1641
   2003    1594
   2004    1588
   2005    1612
   2006    1581
   2007    1591
   2008    1604
   2009    1587
   2010    1588
   2011    1600
   2012    1800
   2013    1640
   2014    1687
   2015    1655
   2016    1786
   2017    1723
   2018    1523
   2019    1465
   2020    1200
   2021    1062
   ```

   **Tasks:**
   - Fit the data using a third-degree polynomial: $y = a_0 + a_1 x + a_2 x^2 + a_3 x^3$
   - Create a plot showing both the original data points and the fitted curve
   - Use your model to predict the newborn population for 2024

   **Hint:** Consider shifting the x-axis (e.g., using years relative to 1990) to improve numerical stability and fitting quality.

   **Sample Code for Visualization:**

   ```julia
    using Makie, CairoMakie
    using Polynomials

    # Example data (replace with your actual population data)
    time = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    y = [2.9, 2.7, 4.8, 5.3, 7.1, 7.6, 7.7, 7.6, 9.4, 9.0]

    # Create figure and plot data points
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="Population")
    scatter!(ax, time, y, color=:blue, marker=:circle, markersize=20, label="Data")
    
    # Example polynomial fit (replace with your fitted coefficients)
    poly = Polynomial([1.0, 2.0, 3.0])  # Replace with your coefficients
    fitted_values = poly.(time)
    lines!(ax, time, fitted_values, color=:red, label="Fitted Curve")
    
    # Add legend and display
    axislegend(; position=:lt)
    fig  # Display figure
    save("population_fit.png", fig)  # Save plot
    ```

4. **(Extra points: Eigen-decomposition)** Solve the following problem:

    Consider a dual species spring chain, with the number of sites as large as possible. The mass is 1 on even sites and 2 on odd sites, and stiffness constant is $C = 1$. The boundary condition is periodic.
    - Show the density of states at different energy with binned bar plot. The $x$-axis is the energy, the $y$-axis is the population.
    - Compare with the result of the single species spring chain.

    Ref: the `hist` function in CairoMakie: https://docs.makie.org/dev/reference/plots/hist
