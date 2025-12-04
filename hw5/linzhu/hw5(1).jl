include("../download_mnist.jl")

using LinearAlgebra
using Images, ImageView
using Plots

# ------------------------------------------------------------
# 1. Load MNIST test set
# ------------------------------------------------------------
test_images, test_labels = download_mnist(:test)
println("Test set: ", size(test_images))          # 28×28×10000

img_h, img_w, N = size(test_images)               # N = number of images

# ------------------------------------------------------------
# 2. Build data matrix and compute SVD
#    Each column of X is one flattened image (size 784)
# ------------------------------------------------------------
X = Float64.(reshape(test_images, img_h * img_w, N))   # 784×N

U, S, V = svd(X; full = false)                         # X ≈ U * Diagonal(S) * V'

# Small helper to clamp values into [0, 1]
clamp01(A) = clamp.(A, 0.0, 1.0)

# ------------------------------------------------------------
# 3. Save an example original vs compressed image (for k_vis)
# ------------------------------------------------------------
k_vis = 50
S_vis = S[1:k_vis]
Xk_vis = U[:, 1:k_vis] * Diagonal(S_vis) * (V[:, 1:k_vis])'
compressed_images_vis = reshape(Xk_vis, img_h, img_w, N)

# Original image is already Gray in [0,1], but we clamp for safety
orig_img = clamp01(Float64.(test_images[:, :, 1]))
comp_img = clamp01(compressed_images_vis[:, :, 1])

gray_img_ini = Gray.(orig_img)
gray_img_compressed = Gray.(comp_img)

save("hw5/linzhu/original_image.png", gray_img_ini)
save("hw5/linzhu/compressed_image_k$(k_vis).png", gray_img_compressed)

# ------------------------------------------------------------
# 4. Compression for different k, error + compression ratio
# ------------------------------------------------------------
maxDim = [10, 50, 100, 200]         # k values
err = zeros(Float64, length(maxDim))
compressed_ratio = zeros(Float64, length(maxDim))

m, n = size(X)                      # m = 784, n = N

for (i, dim) in pairs(maxDim)
    # Rank-dim approximation of X
    S_k = S[1:dim]
    Xk = U[:, 1:dim] * Diagonal(S_k) * (V[:, 1:dim])'

    # Compression ratio = original_size / compressed_size
    # Original: m * n numbers
    # Compressed: U_k (m×dim) + S_k (dim) + V_k (n×dim) -> dim*(m + n + 1)
    original_size = m * n
    compressed_size = dim * (m + n + 1)
    compressed_ratio[i] = original_size / compressed_size

    # Reconstruction error: mean squared error per pixel
    diff = X - Xk
    mse = sum(abs2, diff) / (m * n)   # Frobenius^2 / #entries
    err[i] = mse
end

# ------------------------------------------------------------
# 5. Plots
# ------------------------------------------------------------

# Compression ratio vs k
plot(maxDim, compressed_ratio,
     xlabel = "k",
     ylabel = "compression ratio (original / compressed)",
     label = false,
     title = "compressed ratio")
savefig("hw5/linzhu/compressed_ratio.png")

# Reconstruction error vs k
plot(maxDim, err,
     xlabel = "k",
     ylabel = "MSE per pixel",
     label = false,
     title = "reconstructed error")
savefig("hw5/linzhu/reconstructed_error.png")
