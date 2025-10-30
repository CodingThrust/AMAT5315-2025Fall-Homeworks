# Homework 5 Solution - Mingxu Zhang-50046133

## Overview

This directory contains the complete solution for AMAT 5315 Homework 5, which covers:
1. **SVD for Image Compression** on the MNIST dataset
2. **Fourier Transform Image Processing** on a cat image

## Files

- `homework5_solution.jl` - Main solution file with all implementations (core file for submission)
- `install_packages.jl` - Helper script to automatically install all required packages
- `run_solution.jl` - Wrapper script that checks dependencies and runs the solution
- `README.md` - This documentation file


## Requirements

The following Julia packages are required:
- MLDatasets
- LinearAlgebra
- Images
- Plots
- Statistics
- FileIO
- ColorTypes
- JLD2
- FFTW

## Installation

### Option 1: Use the automatic installer (Recommended)

Simply run the provided installer script:

```bash
cd /home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw5/mingxuzhang
julia install_packages.jl
```

This script will automatically install all required packages and verify the installation.

### Option 2: Manual installation

Install packages manually in Julia REPL:

```julia
using Pkg
Pkg.add(["MLDatasets", "Images", "Plots", "FileIO", "ColorTypes", "JLD2", "FFTW"])
```

### Option 3: Use parent project environment

```julia
cd("/home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw5")
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

### Option 1: Use the runner script (Easiest)

The runner script checks all dependencies before running:

```bash
cd /home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw5/mingxuzhang
julia run_solution.jl
```

This script will:
- Check if all required packages are installed
- Verify that the cat.png file exists
- Run the complete homework solution
- Display any errors with helpful messages

### Option 2: Run directly from command line

```bash
cd /home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw5/mingxuzhang
julia homework5_solution.jl
```

### Option 3: Run from Julia REPL

```julia
cd("/home/data/mingxu/PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw5/mingxuzhang")
include("homework5_solution.jl")
main()
```

### Run individual problems:

**Problem 1 only (SVD):**
```julia
include("homework5_solution.jl")
U, S, V, k_vals, comp_ratios, recon_errors = svd_mnist_compression()
```

**Problem 2 only (Fourier Transform):**
```julia
include("homework5_solution.jl")
img_orig, img_hsv, img_rgb = fourier_image_compression()
```

## Output

The solution generates two output directories:

### `output_problem1/` - SVD Compression Results
- `compression_ratio.png` - Plot of compression ratio vs k
- `reconstruction_error.png` - Plot of reconstruction error vs k
- `combined_plot.png` - Combined plot showing both metrics
- `singular_values.png` - Spectrum of singular values
- `reconstruction_k10.png` - Original vs reconstructed images for k=10
- `reconstruction_k50.png` - Original vs reconstructed images for k=50
- `reconstruction_k100.png` - Original vs reconstructed images for k=100
- `reconstruction_k200.png` - Original vs reconstructed images for k=200

### `output_problem2/` - Fourier Transform Results
- `original.png` - Original cat image
- `reconstructed_hsv.png` - Image reconstructed from HSV compression
- `reconstructed_rgb.png` - Image reconstructed from RGB compression
- `comparison.png` - Side-by-side comparison with error maps
- `analysis.txt` - Detailed analysis and explanation

## Problem 1: SVD Image Compression

### Implementation Details

1. **Data Loading**: Downloads 60,000 MNIST training images ($28 \times 28$ pixels)
   
2. **Vectorization**: Reshapes each image into a 784-dimensional vector
   $$\text{vec}(I) \in \mathbb{R}^{784}, \quad I \in \mathbb{R}^{28 \times 28}$$

3. **Data Matrix**: Creates a $784 \times 60000$ matrix where each column is a flattened image
   $$X = [x_1, x_2, \ldots, x_{60000}] \in \mathbb{R}^{784 \times 60000}$$

4. **SVD Decomposition**: Computes the full SVD
   $$X = U \Sigma V^T$$
   
5. **Compression**: Retains only top $k$ singular values ($k = 10, 50, 100, 200$)
   $$\hat{X}_k = U_k \Sigma_k V_k^T, \quad \text{where } U_k \in \mathbb{R}^{784 \times k}, \Sigma_k \in \mathbb{R}^{k \times k}, V_k \in \mathbb{R}^{60000 \times k}$$

6. **Reconstruction**: Reconstructs images from compressed representation
   $$\hat{x}_i = U_k \Sigma_k v_{k,i}, \quad \text{reshape to } 28 \times 28$$

7. **Evaluation**: Calculates compression ratio and reconstruction error
   - Compression ratio: $\frac{784 \times 60000}{(784 \times k) + k + (60000 \times k)}$
   - Relative error: $\frac{\|X - \hat{X}_k\|_F}{\|X\|_F}$
   - MSE: $\frac{1}{784 \times 60000}\|X - \hat{X}_k\|_F^2$

### Key Metrics

- **Compression Ratio**: 
  $$\text{Compression Ratio} = \frac{\text{Original Size}}{\text{Compressed Size}} = \frac{m \times n}{(m \times k) + k + (k \times n)}$$
  
  Where:
  - Original: $m \times n = 784 \times 60000$ elements
  - Compressed: $(m \times k) + k + (k \times n) = (784 \times k) + k + (k \times 60000)$ elements
  - $m = 784$ (image dimension), $n = 60000$ (number of images), $k$ = number of singular values retained

- **Reconstruction Error**: Relative Frobenius norm
  $$\text{Error} = \frac{\|X - \hat{X}\|_F}{\|X\|_F}$$
  
  Where $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ is the Frobenius norm

### Experimental Results

The following results were obtained from running the SVD compression on the MNIST dataset:

| k Value | Compression Ratio | Relative Frobenius Error | Mean Squared Error | Quality |
|---------|-------------------|--------------------------|-------------------|---------|
| 10      | 77.39×           | 0.555331                | 0.034541         | Low - significant information loss |
| 50      | 15.48×           | 0.324635                | 0.011804         | Medium - recognizable digits |
| 100     | 7.74×            | 0.226452                | 0.005744         | Good - clear digits |
| 200     | 3.87×            | 0.141944                | 0.002257         | Excellent - high fidelity |

**Key Observations:**
- As k increases, compression ratio decreases but reconstruction quality improves
- With k=10, we achieve 77× compression but lose significant detail
- With k=200, we still achieve nearly 4× compression with excellent quality
- The reconstruction error decreases exponentially as k increases
- Around k=100 provides a good balance between compression and quality

## Problem 2: Fourier Transform Image Processing

### Implementation Details

1. **Image Loading**: Loads the cat.png image $I \in \mathbb{R}^{H \times W \times 3}$

2. **HSV Compression**:
   - **Convert** RGB → HSV color space: $I_{RGB} \rightarrow I_{HSV} = [H, S, V]$
   - **Apply 2D FFT** to each channel independently:
     $$F_H(u,v) = \text{FFT2D}(H), \quad F_S(u,v) = \text{FFT2D}(S), \quad F_V(u,v) = \text{FFT2D}(V)$$
   - **Threshold**: Keep only top 1% of coefficients by magnitude
     $$\tilde{F}_c(u,v) = \begin{cases} F_c(u,v) & \text{if } |F_c(u,v)| \geq \tau_c \\ 0 & \text{otherwise} \end{cases}$$
     where $\tau_c$ is chosen such that 99% of coefficients are zeroed
   - **Inverse FFT** and convert back to RGB:
     $$\tilde{H} = \text{IFFT2D}(\tilde{F}_H), \quad \tilde{I}_{HSV} = [\tilde{H}, \tilde{S}, \tilde{V}] \rightarrow \tilde{I}_{RGB}$$

3. **RGB Compression**:
   - **Apply 2D FFT** to each channel (R, G, B):
     $$F_R(u,v) = \text{FFT2D}(R), \quad F_G(u,v) = \text{FFT2D}(G), \quad F_B(u,v) = \text{FFT2D}(B)$$
   - **Threshold**: Keep only top 1% of coefficients by magnitude (same strategy as HSV)
   - **Inverse FFT** to reconstruct:
     $$\tilde{R} = \text{IFFT2D}(\tilde{F}_R), \quad \tilde{I}_{RGB} = [\tilde{R}, \tilde{G}, \tilde{B}]$$

4. **Comparison**: Calculates quality metrics
   - MSE: $\text{MSE} = \frac{1}{HWC}\sum_{i,j,c}(I_{ijc} - \tilde{I}_{ijc})^2$
   - PSNR: $\text{PSNR} = 10\log_{10}(1/\text{MSE})$ dB
   - Difference maps: $\Delta_{HSV-RGB} = |\tilde{I}_{HSV} - \tilde{I}_{RGB}|$

### Key Insights

**Why HSV typically performs better than RGB:**

1. **Perceptual Separation**: HSV separates luminance (V) from chrominance (H, S)
2. **Human Visual System**: More sensitive to luminance than color
3. **Channel Decorrelation**: RGB channels are highly correlated; HSV channels are more independent
4. **Efficient Compression**: V channel contains structural information and needs less compression; H and S can be heavily compressed
5. **Similar to JPEG**: Professional image compression (JPEG) uses YCbCr for similar reasons

### Experimental Results

The following results were obtained from Fourier transform compression (keeping 1% of coefficients):

| Method | Mean Squared Error | PSNR (dB) | Quality Assessment |
|--------|-------------------|-----------|-------------------|
| **HSV Compression** | 0.028664 | 15.43 | Acceptable - preserves color well |
| **RGB Compression** | 0.027882 | 15.55 | Acceptable - similar to HSV |

**Comparison between HSV and RGB:**
- Mean absolute difference between reconstructions: 0.0484
- Maximum absolute difference: 0.3289
- Both methods perform similarly for this image
- Difference is relatively small, indicating both approaches work well

**Key Observations:**
1. **Similar Performance**: For this particular cat image, HSV and RGB compression perform nearly identically (PSNR difference: 0.12 dB)
2. **Low PSNR Values**: Both achieve ~15.5 dB PSNR because we're keeping only 1% of coefficients (99% compression)
3. **Visual Quality**: Despite low PSNR, the images are still recognizable due to preservation of low-frequency components
4. **HSV Advantages**: 
   - Better separation of luminance and chrominance
   - More aligned with human perception
   - Typically better for color preservation in natural images
5. **Why Similar Results**: The cat image likely has similar spectral distributions across color spaces for this compression level

### Quality Metrics

- **MSE (Mean Squared Error)**: Average squared pixel differences (lower is better)
  $$\text{MSE} = \frac{1}{HWC} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{C} (I_{ij}^c - \hat{I}_{ij}^c)^2$$
  
  Where $H$ = height, $W$ = width, $C$ = channels (3 for RGB)

- **PSNR (Peak Signal-to-Noise Ratio)**: Quality measure in dB (higher is better)
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right) = 10 \cdot \log_{10}\left(\frac{1}{\text{MSE}}\right) \text{ (when MAX=1)}$$

- Typical PSNR values:
  - > 40 dB: Excellent quality
  - 30-40 dB: Good quality
  - 20-30 dB: Acceptable quality
  - < 20 dB: Poor quality
  - Our results (~15.5 dB): Reflects aggressive 99% compression

## Mathematical Background

### SVD Decomposition

**Full SVD:**
$$X = U \Sigma V^T$$

Where:
- $X \in \mathbb{R}^{m \times n}$: Data matrix ($784 \times 60000$ for MNIST)
- $U \in \mathbb{R}^{m \times m}$: Left singular vectors (eigenimages) - orthogonal matrix
- $\Sigma \in \mathbb{R}^{m \times n}$: Diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- $V \in \mathbb{R}^{n \times n}$: Right singular vectors (image weights) - orthogonal matrix

**Rank-k Approximation (Compressed form):**
$$\hat{X}_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

Where:
- $U_k \in \mathbb{R}^{m \times k}$: First $k$ columns of $U$
- $\Sigma_k \in \mathbb{R}^{k \times k}$: Top $k$ singular values
- $V_k \in \mathbb{R}^{n \times k}$: First $k$ columns of $V$

**Eckart-Young-Mirsky Theorem:**
$$\hat{X}_k = \arg\min_{\text{rank}(A) \leq k} \|X - A\|_F$$

This means the rank-k SVD approximation is optimal in terms of Frobenius norm.

### 2D Discrete Fourier Transform

**Forward Transform:**
$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-2\pi i \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

**Inverse Transform:**
$$f(x,y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u,v) \cdot e^{2\pi i \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

Where:
- $f(x,y)$: Spatial domain image ($M \times N$ pixels)
- $F(u,v)$: Frequency domain representation
- $(u,v)$: Frequency coordinates
- $i = \sqrt{-1}$

**Magnitude and Phase:**
$$|F(u,v)| = \sqrt{\text{Re}(F(u,v))^2 + \text{Im}(F(u,v))^2}$$
$$\phi(u,v) = \arctan\left(\frac{\text{Im}(F(u,v))}{\text{Re}(F(u,v))}\right)$$

**Properties:**
- **Low frequencies** $(u,v) \approx (0,0)$: Overall structure, smooth regions, most image energy
- **High frequencies** $(u,v)$ far from origin: Details, edges, noise, texture
- **Energy concentration**: $\sim99\%$ of image energy typically in $<10\%$ of coefficients

**Compression Strategy:**
Keep only coefficients with $|F(u,v)| \geq \tau$ where $\tau$ is a threshold chosen to retain top 1% by magnitude.

## Notes

- First run will download MNIST dataset (~11 MB) and cache it
- SVD computation may take a few minutes for 60,000 images
- All computations are done in Float64 for numerical stability
- Visualizations are saved automatically to output directories

## File Structure Explanation

This submission includes three Julia files:

### 1. `homework5_solution.jl` (Core file - 586 lines)
**Purpose**: Contains all the homework implementation code
- Problem 1: SVD image compression on MNIST
- Problem 2: Fourier transform image processing
- All function definitions and main logic
- **This is the main submission file required by the assignment**

### 2. `install_packages.jl` (Helper script - 58 lines)
**Purpose**: Automated package installation
- Automatically installs all required Julia packages
- Provides installation verification
- Shows clear success/error messages
- **Optional**: For convenience, not required for grading

**When to use**: Run this first if packages are not yet installed

### 3. `run_solution.jl` (Helper script - 88 lines)
**Purpose**: Pre-flight checks and execution wrapper
- Checks if all required packages are installed
- Verifies cat.png exists
- Runs homework5_solution.jl with error handling
- Provides user-friendly status messages
- **Optional**: For convenience, not required for grading

**When to use**: Use this if you want automatic dependency checking before running

### Quick Start Guide

**For first-time setup:**
```bash
# Step 1: Install packages
julia install_packages.jl

# Step 2: Run solution
julia run_solution.jl
```

**For subsequent runs:**
```bash
# Directly run the solution
julia homework5_solution.jl
```
