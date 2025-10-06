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
   - Report the compression ratio and reconstruction error (e.g., mean squared error or Frobenius norm).

   **Discussion:**
   - How does the reconstruction quality vary with the number of retained singular values?
   - What is the trade-off between compression ratio and image quality?

2. **(Image Processing with Fourier Transform)** Based on the ["cat" image](cat.png) that we used in class, perform the following analysis:

   **Part A: Edge-Based Frequency Mask**
   - Apply edge detection (check [this youtube video](https://www.youtube.com/watch?app=desktop&v=DGojI9xcCfg)) to the cat image.
   - Apply Fourier transformation on the edge-detected image.
   - Keep the most significant 1% components (by magnitude) and create a binary mask $M$.
   - Apply the mask $M$ to the Fourier transformation of the original image.
   - Recover the compressed image with the inverse Fourier transformation.
   - Compare the results with direct compression (keeping the top 1000 components from the original image's Fourier transform directly, without using edge detection).

   **Part B: SVD-Based Compression**
   - Repeat the same process using SVD instead of Fourier transform for compression, please make sure the compression ratio is comparable to the edge-guided Fourier compression.
   - Compare all three methods: edge-guided Fourier, direct Fourier, and edge-guided SVD.

   **Discussion:**
   - Does edge detection help identify important frequency/singular components?
   - How do the three compression methods compare in terms of visual quality and compression efficiency?
   - What are the advantages and disadvantages of each approach?