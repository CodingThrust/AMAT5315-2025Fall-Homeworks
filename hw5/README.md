# Homework 5

**Note:** Submit your solutions in either `.md` (Markdown) or `.jl` (Julia) format.

1. **(Singular Value Decomposition for Image Compression)** Use singular value decomposition to compress the MNIST dataset. 

   **Dataset:** Use the provided `download_mnist.jl` function to download the MNIST dataset.

   **Tasks:**
   - Download the MNIST dataset using the provided function (see `download_mnist.jl`).
   - Vectorize the images (flatten each 28×28 image into a 784-dimensional vector).
   - Construct a data matrix where each column (or row) represents a flattened image.
   - Apply SVD to the data matrix.
   - Compress the dataset by retaining only the top $k$ singular values (experiment with different values of $k$, e.g., $k = 10, 50, 100, 200$).
   - Reconstruct the images from the compressed representation.
   - Visualize and compare the original vs. reconstructed images.
   - Plot the compression ratio and reconstruction error (e.g., mean squared error or Frobenius norm) as a function of $k$.

2. **(Image Processing with Fourier Transform)** Based on the ["cat" image](cat.png) that we used in class, perform the following analysis:
   - Compress the image with Fourier transformation in HSV channel (check [this youtube video](https://www.youtube.com/watch?app=desktop&v=DGojI9xcCfg)). Keep only the most significant 1% components (by magnitude) in the Fourier space, and then reconstruct the image.
   - Repeat the same process in the RGB channel, and compare the two reconstructed images. How much are they different? explain the observation.