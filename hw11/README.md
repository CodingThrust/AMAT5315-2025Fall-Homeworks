# Homework 11

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. **(Gradiented Based Optimization Implementation)** Implement gradient descent from scratch and test it on Himmelblau's function $f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$. This function has four global minima and is commonly used to test optimization algorithms.

   Implement a gradient based optimization algorithm without using any optimization packages. Use `ForwardDiff.jl` to compute gradients automatically. Test your implementation starting from different starting points and report the final point and function value. Create a convergence plot showing the function value vs iteration number.

   **Hints:**
   - All minima have function value $f = 0$
   - If the algorithm oscillates, try reducing the learning rate
   - Use `using CairoMakie` to create visualizations

2. **(Optimizer Comparison)** Compare the performance of three optimization methods: Gradient Descent, Momentum, and Adam on the Booth function $f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2$.

   For each method, run from the same starting point $(-5, -5)$ for 2,000 iterations. Use the following hyperparameters: Gradient Descent with $\alpha = 0.01$, Momentum with $\alpha = 0.01$ and $\beta = 0.9$, Adam with $\alpha = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$. Create a single plot comparing the convergence curves and report which method converges fastest.

   **Hints:**
   - The global minimum is at $(1, 3)$ with function value $f = 0$
   - You can use `Optimisers.jl` package for advanced optimizers
   - Use log scale for the y-axis to better visualize convergence
   - Make sure all methods start from exactly the same initial point

3. **(Parameter Fitting)** Use gradient-based optimization to fit a logistic growth model to real population data. The logistic model is $P(t) = \frac{K}{1 + e^{-r(t-t_0)}}$, where $K$ is the carrying capacity, $r$ is the growth rate, and $t_0$ is the inflection point.

   Generate synthetic population data or use the following dataset: $t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]$ and $P = [10, 15, 25, 40, 60, 80, 95, 105, 110, 112, 113]$. Define the loss function as mean squared error: $L(K, r, t_0) = \frac{1}{n}\sum_{i=1}^n (P_i - P(t_i))^2$. Use Adam optimizer to find the optimal parameters that best fit the data.

   **Hints:**
   - Initialize parameters with reasonable guesses: $K \approx 120$, $r \approx 0.5$, $t_0 \approx 5$
   - Use learning rate around $0.01-0.1$ for Adam
   - Plot both the original data points and fitted curve to visualize the fit quality
   - This type of model fitting is common in biology, epidemiology, and technology adoption studies
   - Consider using parameter bounds to ensure physically meaningful solutions
