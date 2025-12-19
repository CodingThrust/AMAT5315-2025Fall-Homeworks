using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("download_mnist.jl")
include("main.jl")

using LinearAlgebra
using Images
using CairoMakie
using Statistics

# Task 1
train_images, train_labels = download_mnist(:train)
@show size(train_images)

X = reshape(train_images, 28*28, :)
U, S, V = svd(X)
@show S

k_values = [10, 50, 100, 200]

function calculate_mse(original, reconstructed)
    return mean((original .- reconstructed).^2)
end

function calculate_frobenius_error(original, reconstructed)
    return norm(original - reconstructed)
end

function compression_ratio(k, m, n)
    return (m * n) / (k * (m + n + 1))
end

crs = Float64[]
mse_errors = Float64[]
plot_images = Matrix{Float64}[]

for k in k_values
    X_compressed = U[:, 1:k] * diagm(S[1:k]) * V[:, 1:k]'
    images = reshape(X_compressed, 28, 28, :)
    push!(plot_images, images[:, :, 1])
    @assert size(images) == (28, 28, 60000)
    push!(crs, compression_ratio(k, 28*28, 60000))
    push!(mse_errors, calculate_mse(X, X_compressed))
end

push!(plot_images, train_images[:, :, 1])

fig = Figure()
CairoMakie.Axis(fig[1, 1], xlabel = "k", ylabel = "compression ratio")
scatterlines!(k_values, crs, color = :red)
CairoMakie.Axis(fig[1, 2], xlabel = "k", ylabel = "mse error")
scatterlines!(k_values, mse_errors, color = :blue)
#plot the first image in the plot_images array
fig
save("svd_compression_ratio.png", fig)

fig = Figure()
for i in 1:5
    if i <= 4
        ax = CairoMakie.Axis(fig[1, i], aspect=DataAspect(), title="k = $(k_values[i])")
    else
        ax = CairoMakie.Axis(fig[1, i], aspect=DataAspect(), title="original")
    end
    image!(fig[1, i], plot_images[i], colorrange = (0, 1))
    hidespines!(ax)
    hidedecorations!(ax)
end

fig
save("svd_images.png", fig)

# Task 2: Image Processing with Fourier Transform

# Load the cat image
img = demo_image("cat.png")
original_size = size(img)

# Calculate target dimensions for 1% compression
ratio = 0.01
scale = sqrt(ratio)
target_nx = round(Int, original_size[1] * scale)
target_ny = round(Int, original_size[2] * scale)

# ===== RGB Channel Compression =====
img_k_rgb = fft_compress(img, target_nx, target_ny)
actual_ratio_rgb = ImageProcessing.compression_ratio(img_k_rgb)
img_reconstructed_rgb = toimage(RGBA{N0f8}, img_k_rgb)

# ===== HSV Channel Compression =====
img_hsv = Images.HSV{Float32}.(img)
img_k_hsv = fft_compress(img_hsv, target_nx, target_ny)
actual_ratio_hsv = ImageProcessing.compression_ratio(img_k_hsv)
img_reconstructed_hsv = toimage(HSV{Float32}, img_k_hsv)

# Convert HSV back to RGB for comparison
img_reconstructed_hsv_rgb = RGBA{N0f8}.(img_reconstructed_hsv)

# ===== Comparison =====
diff_image = abs.(Float64.(channelview(RGB.(img_reconstructed_rgb))) - 
                   Float64.(channelview(RGB.(img_reconstructed_hsv_rgb))))
mse = mean((Float64.(channelview(RGB.(img_reconstructed_rgb))) - 
            Float64.(channelview(RGB.(img_reconstructed_hsv_rgb)))).^2)

# Save images
Images.save("cat_original.png", img)
Images.save("cat_reconstructed_rgb.png", img_reconstructed_rgb)
Images.save("cat_reconstructed_hsv.png", img_reconstructed_hsv_rgb)

# Create comparison figure
fig = Figure(resolution=(1200, 400))

ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Original")
image!(ax1, rotr90(img))
hidespines!(ax1)
hidedecorations!(ax1)

ax2 = Axis(fig[1, 2], aspect=DataAspect(), 
           title="RGB Reconstruction\n(ratio=$(round(actual_ratio_rgb, digits=4)))")
image!(ax2, rotr90(img_reconstructed_rgb))
hidespines!(ax2)
hidedecorations!(ax2)

ax3 = Axis(fig[1, 3], aspect=DataAspect(), 
           title="HSV Reconstruction\n(ratio=$(round(actual_ratio_hsv, digits=4)))")
image!(ax3, rotr90(img_reconstructed_hsv_rgb))
hidespines!(ax3)
hidedecorations!(ax3)

ax4 = Axis(fig[1, 4], aspect=DataAspect(), 
           title="Absolute Difference\n(MSE=$(round(mse, digits=6)))")
heatmap!(ax4, rotr90(mean(diff_image, dims=1)[1, :, :]), colormap=:hot)
hidespines!(ax4)
hidedecorations!(ax4)

fig
save("fft_comparison_hsv_vs_rgb.png", fig)

