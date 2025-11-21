#!/usr/bin/env julia

############################################################
# AMAT5315 — HW5
# Student: Jizhe Lai
#
# Q1: Singular Value Decomposition for Image Compression (MNIST)
# Q2: Image Processing with Fourier Transform (cat image)
############################################################

using LinearAlgebra
using Statistics
using Random
using Printf
using Plots
using ColorTypes, ColorVectorSpace, Colors
using ImageIO
using FFTW
using ImageCore          # for channelview

# ----------------------------------------------------------
# Common utilities
# ----------------------------------------------------------

function ensure_dir(dir::AbstractString)
    isdir(dir) || mkpath(dir)
    return dir
end

results_dir = ensure_dir(joinpath(@__DIR__, "results"))

# ----------------------------------------------------------
# Q1. MNIST: SVD for Image Compression
# ----------------------------------------------------------

include("download_mnist.jl")

println("=== Q1: MNIST SVD Compression ===")

############################
# 1. Load MNIST data
############################

train_images, train_labels = download_mnist(:train)
println("Train images size: ", size(train_images))   # (28, 28, 60000)

############################
# 2. Build data matrix
############################

N_total   = size(train_images, 3)
N_samples = 2000              # number of images used for SVD

Random.seed!(5315)
idxs = sort(randperm(N_total)[1:N_samples])

sub_images = train_images[:, :, idxs]           # 28×28×N_samples
X          = reshape(sub_images, :, N_samples)  # 784×N_samples
println("Using ", N_samples, " images for SVD: X size = ", size(X))

############################
# 3. Mean-centering
############################

μ  = mean(X; dims = 2)
Xc = X .- μ

############################
# 4. SVD
############################

println("\nComputing SVD on centered data...")
@time U, S, Vt = svd(Xc; full = false)
println("U size: ", size(U))        # 784×784
println("S length: ", length(S))    # 784
println("Vt size: ", size(Vt))      # 2000×784  (rows = samples)

############################
# 5. Compression & reconstruction for various k
############################

ks = [10, 50, 100, 200]
errors = Float64[]
ratios = Float64[]

n_vis = 5
vis_indices = collect(1:n_vis)

# Save a few original images
for i in vis_indices
    img_vec = X[:, i]
    img     = reshape(img_vec, 28, 28)
    plt = heatmap(permutedims(img), aspect_ratio=:equal, c=:grays,
                  axis=nothing, colorbar=false, title="Original (idx = $i)")
    png(joinpath(results_dir, @sprintf("mnist_original_%02d.png", i)))
end

for k in ks
    println("\n---- MNIST: k = $k ----")

    Uk = U[:, 1:k]           # 784×k
    Sk = Diagonal(S[1:k])    # k×k
    Vk = Vt[:, 1:k]          # 2000×k

    # 784×k  * k×k *  k×2000  = 784×2000
    Xc_approx = Uk * Sk * Vk'
    X_approx  = Xc_approx .+ μ

    stored_parameters   = length(Uk) + length(Sk) + length(Vk)
    original_parameters = length(X)
    ratio = stored_parameters / original_parameters
    push!(ratios, ratio)

    mse = sum((X .- X_approx).^2) / length(X)
    push!(errors, mse)
    println("Compression ratio (stored / original) ≈ ", ratio)
    println("Reconstruction MSE ≈ ", mse)

    for i in vis_indices
        orig_vec  = X[:, i]
        recon_vec = X_approx[:, i]

        orig_img  = reshape(orig_vec, 28, 28)
        recon_img = reshape(recon_vec, 28, 28)

        p1 = heatmap(permutedims(orig_img), aspect_ratio=:equal, c=:grays,
                     axis=nothing, colorbar=false, title="Original")
        p2 = heatmap(permutedims(recon_img), aspect_ratio=:equal, c=:grays,
                     axis=nothing, colorbar=false, title="Reconstructed (k=$k)")

        plt = plot(p1, p2, layout=(1,2), size=(600,300))
        png(joinpath(results_dir, @sprintf("mnist_recon_idx%02d_k%03d.png", i, k)))
    end
end

############################
# 6. Plot compression ratio & error vs k
############################

plt1 = plot(ks, ratios, marker=:o, xlabel="k",
            ylabel="Compression ratio (stored/original)",
            title="MNIST SVD Compression Ratio vs k")
png(joinpath(results_dir, "mnist_compression_ratio_vs_k.png"))

plt2 = plot(ks, errors, marker=:o, xlabel="k",
            ylabel="Reconstruction MSE",
            title="MNIST Reconstruction Error vs k")
png(joinpath(results_dir, "mnist_reconstruction_error_vs_k.png"))

println("\nMNIST figures saved under: ", results_dir)

# ----------------------------------------------------------
# Q2. Cat image: Fourier compression in HSV vs RGB
# ----------------------------------------------------------

println("\n=== Q2: Fourier Compression on Cat Image ===")

cat_path = joinpath(@__DIR__, "cat.png")
if !isfile(cat_path)
    error("cat.png not found in current directory: $cat_path")
end

img_rgb = load(cat_path)

# keep top p_fraction coefficients (by magnitude) in Fourier space
function compress_fft_channel(ch::AbstractMatrix{<:Real}, p_fraction::Float64)
    F = fft(ch)
    mags = abs.(F)
    vec_mags = vec(mags)
    N = length(vec_mags)
    k = max(1, round(Int, p_fraction * N))
    thresh = partialsort(vec_mags, N-k+1)
    Fmask = F .* (mags .>= thresh)
    recon = real(ifft(Fmask))
    return recon, Fmask
end

p_fraction = 0.01   # keep 1% largest coefficients

############################
# 1. Compression in HSV
############################

println("Fourier compression in HSV space...")

img_hsv = HSV.(img_rgb)          # H×W image, each pixel is HSV

cv_hsv = channelview(img_hsv)    # 3×H×W, first dim = H,S,V
H = Float64.(cv_hsv[1, :, :])
S = Float64.(cv_hsv[2, :, :])
V = Float64.(cv_hsv[3, :, :])

Hc, _ = compress_fft_channel(H, p_fraction)
Sc, _ = compress_fft_channel(S, p_fraction)
Vc, _ = compress_fft_channel(V, p_fraction)

Hc = mod.(Hc, 1)          # hue in [0,1), wrap-around
Sc = clamp.(Sc, 0, 1)
Vc = clamp.(Vc, 0, 1)

Hh, Hw = size(Hc)
hsv_recon = Array{HSV{Float64}}(undef, Hh, Hw)
@inbounds for j in 1:Hw, i in 1:Hh
    hsv_recon[i, j] = HSV(Hc[i, j], Sc[i, j], Vc[i, j])
end

rgb_hsvrec = RGB.(hsv_recon)

############################
# 2. Compression in RGB
############################

println("Fourier compression in RGB space...")

cv_rgb = channelview(img_rgb)           # 3×H×W: (R,G,B)
R = Float64.(cv_rgb[1, :, :])
G = Float64.(cv_rgb[2, :, :])
B = Float64.(cv_rgb[3, :, :])

Rc, _ = compress_fft_channel(R, p_fraction)
Gc, _ = compress_fft_channel(G, p_fraction)
Bc, _ = compress_fft_channel(B, p_fraction)

Rc = clamp.(Rc, 0, 1)
Gc = clamp.(Gc, 0, 1)
Bc = clamp.(Bc, 0, 1)

Rh, Rw = size(Rc)
rgb_recon = Array{RGB{Float64}}(undef, Rh, Rw)
@inbounds for j in 1:Rw, i in 1:Rh
    rgb_recon[i, j] = RGB(Rc[i, j], Gc[i, j], Bc[i, j])
end

############################
# 3. Save comparison images
############################

save(joinpath(results_dir, "cat_original.png"), img_rgb)
save(joinpath(results_dir, "cat_fft_hsv.png"), rgb_hsvrec)
save(joinpath(results_dir, "cat_fft_rgb.png"), rgb_recon)

p_orig = plot(img_rgb, axis=nothing, title="Original cat")
p_hsv  = plot(rgb_hsvrec, axis=nothing, title="FFT 1% (HSV)")
p_rgb  = plot(rgb_recon, axis=nothing, title="FFT 1% (RGB)")

plt_cat1 = plot(p_orig, p_hsv, p_rgb, layout=(1,3), size=(900,300))
png(joinpath(results_dir, "cat_fft_comparison_row.png"))

plt_cat2 = plot(p_orig, p_hsv, p_rgb, layout=(3,1), size=(400,900))
png(joinpath(results_dir, "cat_fft_comparison_col.png"))

println("Cat images saved under: ", results_dir)
println("\n=== HW5 script finished ===")

# === Q1: MNIST SVD Compression ===
# Loading MNIST train data from cache: /Users/hery/Desktop/Desktop/HKUST/scientific-computing/AMAT5315-2025Fall-Homeworks/hw5/JizheLai/mnist_data/mnist_train.jld2
# Train images size: (28, 28, 60000)
# Using 2000 images for SVD: X size = (784, 2000)

# Computing SVD on centered data...
#   0.255067 seconds (208.89 k allocations: 34.716 MiB, 34.95% compilation time)
# U size: (784, 784)
# S length: 784
# Vt size: (2000, 784)

# ---- MNIST: k = 10 ----
# Compression ratio (stored / original) ≈ 0.017818877551020407
# Reconstruction MSE ≈ 0.03404519

# ---- MNIST: k = 50 ----
# Compression ratio (stored / original) ≈ 0.09036989795918367
# Reconstruction MSE ≈ 0.011286998

# ---- MNIST: k = 100 ----
# Compression ratio (stored / original) ≈ 0.18392857142857144
# Reconstruction MSE ≈ 0.0053018937

# ---- MNIST: k = 200 ----
# Compression ratio (stored / original) ≈ 0.3806122448979592
# Reconstruction MSE ≈ 0.0019106633

# MNIST figures saved under: /Users/hery/Desktop/Desktop/HKUST/scientific-computing/AMAT5315-2025Fall-Homeworks/hw5/JizheLai/results

# === Q2: Fourier Compression on Cat Image ===
# Fourier compression in HSV space...
# Fourier compression in RGB space...
# Cat images saved under: /Users/hery/Desktop/Desktop/HKUST/scientific-computing/AMAT5315-2025Fall-Homeworks/hw5/JizheLai/results

# === HW5 script finished ===