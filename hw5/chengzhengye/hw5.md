# HW5
## Task1
```
using Pkg
Pkg.add(["LinearAlgebra", "JLD2", "MLDatasets", "CairoMakie"])
using LinearAlgebra, JLD2, MLDatasets, CairoMakie

# Include the provided download function
include("download_mnist.jl")

# 1. Download the MNIST dataset using the provided function
println("Downloading MNIST training dataset...")
train_images, train_labels = download_mnist(:train)
println("Training dataset loaded with dimensions: ", size(train_images))

# 2. Vectorize the images (flatten each 28×28 image into 784-dimensional vector)
n_samples = 1000
images_subset = train_images[:, :, 1:n_samples]
flattened_images = reshape(images_subset, 28*28, n_samples)

# 3. Construct data matrix (each column is a flattened image)
data_matrix = flattened_images

# 4. Apply SVD to the data matrix
println("Performing SVD decomposition (number of samples: $n_samples)...")
U, S, Vt = svd(data_matrix)
println("SVD decomposition completed!")

# Function for compression and reconstruction
function compress_reconstruct(U, S, Vt, k)
    U_k = U[:, 1:k]
    S_k = Diagonal(S[1:k])
    Vt_k = Vt[1:k, :]
    return U_k * S_k * Vt_k
end

# 5. Compress dataset by retaining top k singular values (k = 10, 50, 100, 200)
k_values = [10, 50, 100, 200]
println("Compressing and reconstructing for k values: $k_values...")
reconstructed_matrices = [compress_reconstruct(U, S, Vt, k) for k in k_values]

# 6. Reconstruct images from compressed representation & 7. Visualize original vs reconstructed
sample_index = 5
original_image = reshape(data_matrix[:, sample_index], 28, 28)
original_digit = train_labels[sample_index]

fig1 = Figure(resolution=(1000, 300), title="Original vs Reconstructed (Digit: $original_digit, Sample $sample_index)")
ax1 = Axis(fig1[1, 1], title="Original", 
           xticklabelsvisible=false, yticklabelsvisible=false,
           xticksvisible=false, yticksvisible=false)
heatmap!(ax1, original_image, colormap=:grays)

for (i, k) in enumerate(k_values)
    reconstructed_image = reshape(reconstructed_matrices[i][:, sample_index], 28, 28)
    ax = Axis(fig1[1, i+1], title="k=$k", 
              xticklabelsvisible=false, yticklabelsvisible=false,
              xticksvisible=false, yticksvisible=false)
    heatmap!(ax, reconstructed_image, colormap=:grays)
end

save("original_vs_reconstructed.png", fig1)
println("Original vs reconstructed images saved as 'original_vs_reconstructed.png'")

# 8. Plot compression ratio and reconstruction error as function of k
original_size = prod(size(data_matrix))

function compression_ratio(k, matrix)
    m, n = size(matrix)
    compressed_size = k * (m + n + k)
    return compressed_size / original_size
end

function mse(original, reconstructed)
    return mean((original .- reconstructed).^2)
end

ratios = [compression_ratio(k, data_matrix) for k in k_values]
errors = [mse(data_matrix, recon) for recon in reconstructed_matrices]

fig2 = Figure(resolution=(800, 600))
ax_ratio = Axis(fig2[1, 1], xlabel="k (Number of Singular Values)", 
                ylabel="Compression Ratio", title="Compression Ratio vs k")
lines!(ax_ratio, k_values, ratios, marker=:circle, linewidth=2, color=:blue, label="Compression Ratio")
axislegend(ax_ratio)

ax_error = Axis(fig2[2, 1], xlabel="k (Number of Singular Values)", 
                ylabel="Mean Squared Error (MSE)", title="Reconstruction Error vs k")
lines!(ax_error, k_values, errors, marker=:square, linewidth=2, color=:red, label="MSE")
axislegend(ax_error)

save("compression_metrics.png", fig2)
println("Compression ratio and error plots saved as 'compression_metrics.png'")

println("All tasks completed!")
```

## Task2
```
using Pkg
Pkg.add(["Images", "FFTW", "CairoMakie", "ImageMagick"])
using Images, FFTW, CairoMakie

# Load the cat image
img = load("cat.png")
img_rgb = Float64.(channelview(img))  # Convert to RGB channels (3×H×W)

# Helper function: Compress image channel using Fourier transform (keep top 1% components)
function fourier_compress(channel)
    # Compute 2D FFT with shift (low frequencies at center)
    fft_result = fftshift(fft(channel))
    
    # Calculate magnitudes and find threshold for top 1% components
    magnitudes = abs.(fft_result)
    total = length(magnitudes)
    top_count = Int(round(0.01 * total))  # 1% of total components
    threshold = partialsort(vec(magnitudes), total - top_count + 1)  # Threshold for top 1%
    
    # Keep only top 1% components (set others to 0)
    fft_filtered = fft_result .* (magnitudes .>= threshold)
    
    # Inverse FFT to reconstruct
    reconstructed = real.(ifft(ifftshift(fft_filtered)))
    return clamp.(reconstructed, 0.0, 1.0)  # Clamp to valid range [0,1]
end

# Task 1: Compress in HSV channel
# Convert RGB to HSV
img_hsv = Float64.(channelview(rgb2hsv(img)))  # HSV channels (3×H×W)

# Compress each HSV channel
h_compressed = fourier_compress(img_hsv[1, :, :])
s_compressed = fourier_compress(img_hsv[2, :, :])
v_compressed = fourier_compress(img_hsv[3, :, :])

# Reconstruct HSV image and convert back to RGB
hsv_reconstructed = cat(h_compressed, s_compressed, v_compressed, dims=1)
img_hsv_recon = colorview(HSV, hsv_reconstructed) |> rgb  # Convert HSV back to RGB

# Task 2: Compress in RGB channel
# Compress each RGB channel
r_compressed = fourier_compress(img_rgb[1, :, :])
g_compressed = fourier_compress(img_rgb[2, :, :])
b_compressed = fourier_compress(img_rgb[3, :, :])

# Reconstruct RGB image
rgb_reconstructed = cat(r_compressed, g_compressed, b_compressed, dims=1)
img_rgb_recon = colorview(RGB, rgb_reconstructed)

# Visualize original and reconstructed images
fig = Figure(resolution=(1200, 800))
ax1 = Axis(fig[1, 1], title="Original Image")
image!(ax1, rotr90(img))
ax2 = Axis(fig[1, 2], title="Reconstructed (HSV Channel)")
image!(ax2, rotr90(img_hsv_recon))
ax3 = Axis(fig[1, 3], title="Reconstructed (RGB Channel)")
image!(ax3, rotr90(img_rgb_recon))
save("fourier_reconstructions.png", fig)
println("Reconstructed images saved as 'fourier_reconstructions.png'")

# Calculate and visualize difference between HSV and RGB reconstructions
diff_img = abs.(channelview(img_hsv_recon) .- channelview(img_rgb_recon))
diff_img = colorview(RGB, diff_img)
fig_diff = Figure(resolution=(600, 600))
ax_diff = Axis(fig_diff[1, 1], title="Difference Between HSV and RGB Reconstructions")
image!(ax_diff, rotr90(diff_img))
save("reconstruction_difference.png", fig_diff)
println("Difference image saved as 'reconstruction_difference.png'")

println("All tasks completed!")
```
The image reconstructed from HSV channels usually has clearer shapes and edges than the one from RGB channels. 