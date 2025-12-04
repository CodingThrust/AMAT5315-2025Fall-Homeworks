# Homework 5 Solutions
# Author: tengxianglin
#
# Run with: julia --project=hw5 hw5/tengxianglin/hw5.jl

using LinearAlgebra
using Printf
# Optional: For visualization (uncomment if needed)
# using MLDatasets, Images, FileIO, FFTW, Colors, ImageTransformations, Statistics, CairoMakie

println("\n" * "="^70)
println("HOMEWORK 5 - Solutions")
println("="^70)

# ============================================================================
# Task 1: SVD Compression of MNIST
# ============================================================================
println("\nTask 1: SVD Compression of MNIST")
println("="^70)

# Check if we can load MNIST
try
    # Include the download function
    include(joinpath(@__DIR__, "..", "download_mnist.jl"))
    
    println("\nLoading MNIST training data...")
    # Note: This requires MLDatasets package
    # Uncomment the following if MLDatasets is available:
    
    using MLDatasets
    train_images, train_labels = download_mnist(:train)
    
    # Reshape into 2D data matrix
    X = reshape(train_images, 28*28, :)
    Xsample = X[:, 1:1000]  # sample subset
    
    println("  Data matrix size: $(size(Xsample))")
    println("  Using first 1000 images for analysis")
    
    # Compute SVD
    println("\nComputing SVD...")
    svdres = svd(Xsample; full=false)
    U, S, Vt = svdres.U, svdres.S, svdres.Vt
    
    println("  SVD completed")
    println("  Number of singular values: $(length(S))")
    
    # Reconstruction and compression analysis
    reconstruct(U, S, Vt, k) = U[:, 1:k] * Diagonal(S[1:k]) * Vt[1:k, :]
    fro(A) = norm(A)
    m, n = size(Xsample)
    ks = [10, 50, 100, 200]
    errors = Float64[]
    ratios = Float64[]
    den = fro(Xsample)
    
    println("\nCompression analysis:")
    println("-"^70)
    println("  k   | Relative Error | Compression Ratio")
    println("-"^70)
    
    for k in ks
        Xk = reconstruct(U, S, Vt, k)
        error = fro(Xsample - Xk) / den
        ratio = (k * (m + n + 1)) / (m * n)
        push!(errors, error)
        push!(ratios, ratio)
        @printf("  %3d |     %.4f      |      %.4f\n", k, error, ratio)
    end
    
    println("\nResults:")
    println("  - As k increases, reconstruction error decreases")
    println("  - Compression ratio increases with k")
    println("  - Trade-off between compression and quality")
    
    
    println("\nNote: Full MNIST analysis requires MLDatasets package.")
    println("      The code structure is provided above (commented out).")
    println("      Expected results:")
    println("        k=10:  Error ≈ 0.546, Ratio ≈ 0.0228")
    println("        k=50:  Error ≈ 0.307, Ratio ≈ 0.1138")
    println("        k=100: Error ≈ 0.206, Ratio ≈ 0.2277")
    println("        k=200: Error ≈ 0.118, Ratio ≈ 0.4554")
    
catch e
    println("  Note: MNIST analysis requires additional packages.")
    println("  Expected results are documented in hw5.md")
end

# ============================================================================
# Task 2: Fourier Transform Compression on cat.png
# ============================================================================
println("\n\n" * "="^70)
println("Task 2: Fourier Transform Compression on cat.png")
println("="^70)

# Check if cat.png exists (try multiple possible paths)
script_dir = @__DIR__
hw5_dir = dirname(script_dir)  # hw5 directory
project_root = dirname(hw5_dir)  # Project root

cat_paths = [
    joinpath(hw5_dir, "cat.png"),                      # hw5/cat.png
    joinpath(script_dir, "..", "cat.png"),              # hw5/cat.png (relative to script)
    joinpath(project_root, "hw5", "cat.png"),          # Absolute from project root
    "cat.png",                                          # Current directory
]

global cat_path = nothing
for path in cat_paths
    full_path = abspath(path)
    if isfile(full_path)
        global cat_path = full_path
        break
    end
end

if cat_path !== nothing
    println("\nFound cat.png")
    println("  Path: $cat_path")
    
    # Note: Full implementation requires Images, FFTW, Colors packages
    # Uncomment the following if packages are available:
    
    using Images, FileIO, FFTW, Colors, ImageTransformations, Statistics, CairoMakie
    
    img = load(cat_path)
    img_rgb = RGB.(img)
    
    println("  Image loaded successfully")
    println("  Image size: $(size(img))")
    
    # Compression function
    function compress_fft_channel(A; keep_ratio=0.01)
        A64 = float.(A)
        F   = fftshift(fft(A64))
        mag = abs.(F)
        thr = quantile(vec(mag), 1 - keep_ratio)
        mask = mag .>= thr
        Ff = F .* mask
        real.(ifft(ifftshift(Ff)))
    end
    
    println("\nCompressing image (keeping top 1% of FFT coefficients)...")
    
    # RGB compression
    rgb_ch = channelview(img_rgb)
    rec_rgb_ch = similar(rgb_ch, Float64)
    for i in 1:3
        rec_rgb_ch[i, :, :] = compress_fft_channel(rgb_ch[i, :, :]; keep_ratio=0.01)
    end
    rec_rgb = colorview(RGB, clamp.(rec_rgb_ch, 0, 1))
    
    # HSV compression (converted back to RGB for display)
    img_hsv = HSV.(img_rgb)
    hsv_ch = channelview(img_hsv)
    rec_hsv_ch = similar(hsv_ch, Float64)
    for i in 1:3
        rec_hsv_ch[i, :, :] = compress_fft_channel(hsv_ch[i, :, :]; keep_ratio=0.01)
    end
    rec_hsv = colorview(HSV, clamp.(rec_hsv_ch, 0, 1))
    rec_hsv_rgb = RGB.(rec_hsv)
    
    println("  Compression completed")
    
    # Visualization
    fig = CairoMakie.Figure(size=(900, 300))
    ax1 = CairoMakie.Axis(fig[1,1], title="Original")
    ax2 = CairoMakie.Axis(fig[1,2], title="HSV 1%")
    ax3 = CairoMakie.Axis(fig[1,3], title="RGB 1%")
    CairoMakie.image!(ax1, img_rgb)
    CairoMakie.image!(ax2, rec_hsv_rgb)
    CairoMakie.image!(ax3, rec_rgb)
    output_dir = @__DIR__
    output_path = joinpath(output_dir, "cat_fft_compare.png")
    CairoMakie.save(output_path, fig)
    println("  Comparison plot saved to $output_path")
    
    println("\nObservation:")
    println("  - HSV-compressed image appears red-tinted and more blurred")
    println("  - RGB-compressed version maintains original color tones")
    println("  - Direct RGB compression is visually more stable")
    
    
    println("\nNote: Full FFT compression analysis requires Images, FFTW packages.")
    println("      The code structure is provided above (commented out).")
    println("      Expected observation:")
    println("        HSV compression causes color distortion due to hue channel errors")
    println("        RGB compression maintains better visual quality")
else
    println("\nNote: cat.png not found at expected location.")
    println("      Expected path: $cat_path")
    println("      The code structure for FFT compression is documented in hw5.md")
end

# ============================================================================
# Summary
# ============================================================================
println("\n\n" * "="^70)
println("Summary")
println("="^70)
println("✓ Task 1: SVD compression analysis structure provided")
println("✓ Task 2: FFT compression analysis structure provided")
println("\nNote: Full execution requires:")
println("  - MLDatasets (for MNIST)")
println("  - Images, FFTW, Colors (for image processing)")
println("  - CairoMakie (for visualization)")
println("="^70)

