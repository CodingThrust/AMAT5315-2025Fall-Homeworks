"""
Homework 5 Solution
AMAT 5315 - Fall 2025
Author: mingxuzhang

This file contains solutions for:
1. SVD for Image Compression on MNIST dataset
2. Fourier Transform Image Processing on cat image
"""

using MLDatasets
using LinearAlgebra
using Images
using Plots
using Statistics
using FileIO
using ColorTypes
using JLD2
using FFTW

# Include the download_mnist function
include("../download_mnist.jl")

#=============================================================================
Problem 1: Singular Value Decomposition for Image Compression
=============================================================================#

"""
    svd_mnist_compression()

Perform SVD-based compression on MNIST dataset.
"""
function svd_mnist_compression()
    println("=" ^ 80)
    println("Problem 1: SVD for Image Compression on MNIST")
    println("=" ^ 80)
    
    # Step 1: Download MNIST dataset
    println("\n[1] Downloading MNIST training data...")
    train_images, train_labels = download_mnist(:train)
    println("Training set dimensions: ", size(train_images))
    
    # Step 2: Vectorize the images
    # Each column represents a flattened image
    println("\n[2] Vectorizing images...")
    X = reshape(train_images, 28*28, :)  # 784×60000 matrix
    println("Data matrix dimensions: ", size(X))
    
    # Convert to Float64 for better numerical stability
    X = Float64.(X)
    
    # Step 3: Apply SVD
    println("\n[3] Computing SVD...")
    println("This may take a few moments...")
    U, S, V = svd(X)
    println("SVD complete!")
    println("  U dimensions: ", size(U))
    println("  S length: ", length(S))
    println("  V dimensions: ", size(V))
    
    # Step 4: Compress with different values of k
    k_values = [10, 50, 100, 200]
    println("\n[4] Testing compression with k = ", k_values)
    
    # Store results
    compression_ratios = Float64[]
    reconstruction_errors = Float64[]
    
    # Create output directory for visualizations
    output_dir = "output_problem1"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    for k in k_values
        println("\n  Testing k = $k...")
        
        # Compress: keep only top k singular values
        U_k = U[:, 1:k]
        S_k = S[1:k]
        V_k = V[:, 1:k]
        
        # Reconstruct
        X_reconstructed = U_k * Diagonal(S_k) * V_k'
        
        # Calculate compression ratio
        # Original: 784 × 60000 elements
        # Compressed: (784×k) + k + (k×60000) elements
        original_size = size(X, 1) * size(X, 2)
        compressed_size = size(U_k, 1) * k + k + k * size(V_k, 1)
        compression_ratio = original_size / compressed_size
        
        # Calculate reconstruction error (Frobenius norm)
        error_frobenius = norm(X - X_reconstructed) / norm(X)
        
        # Calculate mean squared error
        mse = mean((X - X_reconstructed).^2)
        
        push!(compression_ratios, compression_ratio)
        push!(reconstruction_errors, error_frobenius)
        
        println("    Compression ratio: ", round(compression_ratio, digits=2))
        println("    Relative Frobenius error: ", round(error_frobenius, digits=6))
        println("    Mean squared error: ", round(mse, digits=6))
        
        # Visualize some examples
        visualize_reconstruction(X, X_reconstructed, k, output_dir)
    end
    
    # Step 5: Plot compression ratio vs reconstruction error
    println("\n[5] Creating plots...")
    
    # Plot 1: Compression ratio vs k
    p1 = plot(k_values, compression_ratios, 
              marker=:circle, markersize=8,
              linewidth=2,
              xlabel="k (number of singular values)",
              ylabel="Compression Ratio",
              title="Compression Ratio vs k",
              legend=false,
              grid=true)
    savefig(p1, joinpath(output_dir, "compression_ratio.png"))
    
    # Plot 2: Reconstruction error vs k
    p2 = plot(k_values, reconstruction_errors,
              marker=:circle, markersize=8,
              linewidth=2,
              xlabel="k (number of singular values)",
              ylabel="Relative Frobenius Error",
              title="Reconstruction Error vs k",
              legend=false,
              grid=true,
              color=:red)
    savefig(p2, joinpath(output_dir, "reconstruction_error.png"))
    
    # Plot 3: Combined plot
    p3 = plot(k_values, compression_ratios,
              marker=:circle, markersize=8,
              linewidth=2,
              xlabel="k (number of singular values)",
              ylabel="Compression Ratio",
              label="Compression Ratio",
              legend=:right,
              grid=true)
    plot!(twinx(), k_values, reconstruction_errors,
          marker=:square, markersize=8,
          linewidth=2,
          ylabel="Relative Frobenius Error",
          label="Reconstruction Error",
          color=:red,
          legend=:topright)
    savefig(p3, joinpath(output_dir, "combined_plot.png"))
    
    # Plot 4: Singular value spectrum
    p4 = plot(1:min(500, length(S)), S[1:min(500, length(S))],
              xlabel="Singular Value Index",
              ylabel="Singular Value",
              title="Singular Value Spectrum",
              legend=false,
              yscale=:log10,
              linewidth=2,
              grid=true)
    vline!(k_values, label="k values", linestyle=:dash, color=:red, linewidth=1.5)
    savefig(p4, joinpath(output_dir, "singular_values.png"))
    
    println("\n✓ Problem 1 complete!")
    println("  Results saved in: $output_dir/")
    
    return U, S, V, k_values, compression_ratios, reconstruction_errors
end

"""
    visualize_reconstruction(X_orig, X_recon, k, output_dir; n_examples=10)

Visualize original vs reconstructed images.
"""
function visualize_reconstruction(X_orig, X_recon, k, output_dir; n_examples=10)
    # Select random examples
    n_images = size(X_orig, 2)
    indices = rand(1:n_images, n_examples)
    
    # Create a large canvas for all images
    # Each row: original and reconstructed side by side
    # Size: 2 columns × n_examples rows, each image is 28×28
    img_height = 28
    img_width = 28
    padding = 2
    
    canvas_width = 2 * img_width + 3 * padding
    canvas_height = n_examples * img_height + (n_examples + 1) * padding
    
    # Create white canvas
    canvas = ones(Float64, canvas_height, canvas_width)
    
    for (idx, i) in enumerate(indices)
        # Original image
        img_orig = reshape(X_orig[:, i], 28, 28)
        # Clamp values to [0, 1] range
        img_orig_display = clamp.(img_orig, 0, 1)
        
        # Reconstructed image
        img_recon = reshape(X_recon[:, i], 28, 28)
        img_recon_display = clamp.(img_recon, 0, 1)
        
        # Calculate position
        row_start = (idx - 1) * (img_height + padding) + padding + 1
        row_end = row_start + img_height - 1
        
        # Place original image (left column)
        col_start_orig = padding + 1
        col_end_orig = col_start_orig + img_width - 1
        canvas[row_start:row_end, col_start_orig:col_end_orig] = img_orig_display
        
        # Place reconstructed image (right column)
        col_start_recon = 2 * padding + img_width + 1
        col_end_recon = col_start_recon + img_width - 1
        canvas[row_start:row_end, col_start_recon:col_end_recon] = img_recon_display
    end
    
    # Convert to grayscale image and save
    img_gray = Gray.(canvas)
    save(joinpath(output_dir, "reconstruction_k$(k).png"), img_gray)
end


#=============================================================================
Problem 2: Image Processing with Fourier Transform
=============================================================================#

"""
    fourier_image_compression()

Perform Fourier transform-based compression on cat image.
"""
function fourier_image_compression()
    println("\n" * "=" ^ 80)
    println("Problem 2: Image Processing with Fourier Transform")
    println("=" ^ 80)
    
    # Load cat image
    println("\n[1] Loading cat image...")
    img_path = "../cat.png"
    img = load(img_path)
    println("Image dimensions: ", size(img))
    
    # Create output directory
    output_dir = "output_problem2"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Save original image for reference
    save(joinpath(output_dir, "original.png"), img)
    
    # Part A: Compress in HSV channel
    println("\n[2] Compressing in HSV color space...")
    img_hsv_compressed, compression_info_hsv = compress_image_hsv(img)
    save(joinpath(output_dir, "reconstructed_hsv.png"), img_hsv_compressed)
    println("  HSV compression complete!")
    
    # Part B: Compress in RGB channel
    println("\n[3] Compressing in RGB color space...")
    img_rgb_compressed, compression_info_rgb = compress_image_rgb(img)
    save(joinpath(output_dir, "reconstructed_rgb.png"), img_rgb_compressed)
    println("  RGB compression complete!")
    
    # Part C: Compare the results
    println("\n[4] Comparing results...")
    compare_reconstructions(img, img_hsv_compressed, img_rgb_compressed, output_dir)
    
    println("\n✓ Problem 2 complete!")
    println("  Results saved in: $output_dir/")
    
    return img, img_hsv_compressed, img_rgb_compressed
end

"""
    compress_image_hsv(img; keep_ratio=0.01)

Compress image using Fourier transform in HSV color space.
Keep only the top `keep_ratio` (1%) of coefficients by magnitude.
"""
function compress_image_hsv(img; keep_ratio=0.01)
    # Convert to HSV
    img_hsv = HSV.(img)
    
    # Extract channels
    h_channel = [pix.h for pix in img_hsv]
    s_channel = [pix.s for pix in img_hsv]
    v_channel = [pix.v for pix in img_hsv]
    
    # Apply FFT to each channel
    H_fft = fft(h_channel)
    S_fft = fft(s_channel)
    V_fft = fft(v_channel)
    
    # Compress each channel
    H_compressed = compress_fft_channel(H_fft, keep_ratio)
    S_compressed = compress_fft_channel(S_fft, keep_ratio)
    V_compressed = compress_fft_channel(V_fft, keep_ratio)
    
    # Inverse FFT
    h_recon = real.(ifft(H_compressed))
    s_recon = real.(ifft(S_compressed))
    v_recon = real.(ifft(V_compressed))
    
    # Clamp values to valid ranges
    h_recon = clamp.(h_recon, 0, 360)
    s_recon = clamp.(s_recon, 0, 1)
    v_recon = clamp.(v_recon, 0, 1)
    
    # Reconstruct HSV image
    img_hsv_recon = [HSV(h_recon[i,j], s_recon[i,j], v_recon[i,j]) 
                     for i in 1:size(img,1), j in 1:size(img,2)]
    
    # Convert back to RGB
    img_rgb_recon = RGB.(img_hsv_recon)
    
    # Calculate statistics
    total_coeffs = length(H_fft)
    kept_coeffs = sum(abs.(H_compressed) .> 0)
    actual_ratio = kept_coeffs / total_coeffs
    
    compression_info = (
        total_coefficients = total_coeffs,
        kept_coefficients = kept_coeffs,
        actual_keep_ratio = actual_ratio,
        target_keep_ratio = keep_ratio
    )
    
    return img_rgb_recon, compression_info
end

"""
    compress_image_rgb(img; keep_ratio=0.01)

Compress image using Fourier transform in RGB color space.
Keep only the top `keep_ratio` (1%) of coefficients by magnitude.
"""
function compress_image_rgb(img; keep_ratio=0.01)
    # Convert to RGB if not already
    img_rgb = RGB.(img)
    
    # Extract channels
    r_channel = [Float64(pix.r) for pix in img_rgb]
    g_channel = [Float64(pix.g) for pix in img_rgb]
    b_channel = [Float64(pix.b) for pix in img_rgb]
    
    # Apply FFT to each channel
    R_fft = fft(r_channel)
    G_fft = fft(g_channel)
    B_fft = fft(b_channel)
    
    # Compress each channel
    R_compressed = compress_fft_channel(R_fft, keep_ratio)
    G_compressed = compress_fft_channel(G_fft, keep_ratio)
    B_compressed = compress_fft_channel(B_fft, keep_ratio)
    
    # Inverse FFT
    r_recon = real.(ifft(R_compressed))
    g_recon = real.(ifft(G_compressed))
    b_recon = real.(ifft(B_compressed))
    
    # Clamp values to [0, 1]
    r_recon = clamp.(r_recon, 0, 1)
    g_recon = clamp.(g_recon, 0, 1)
    b_recon = clamp.(b_recon, 0, 1)
    
    # Reconstruct RGB image
    img_rgb_recon = [RGB(r_recon[i,j], g_recon[i,j], b_recon[i,j])
                     for i in 1:size(img,1), j in 1:size(img,2)]
    
    # Calculate statistics
    total_coeffs = length(R_fft)
    kept_coeffs = sum(abs.(R_compressed) .> 0)
    actual_ratio = kept_coeffs / total_coeffs
    
    compression_info = (
        total_coefficients = total_coeffs,
        kept_coefficients = kept_coeffs,
        actual_keep_ratio = actual_ratio,
        target_keep_ratio = keep_ratio
    )
    
    return img_rgb_recon, compression_info
end

"""
    compress_fft_channel(fft_data, keep_ratio)

Keep only the top `keep_ratio` of Fourier coefficients by magnitude.
"""
function compress_fft_channel(fft_data, keep_ratio)
    # Calculate magnitudes
    magnitudes = abs.(fft_data)
    
    # Find threshold: keep only top keep_ratio% by magnitude
    n_keep = ceil(Int, length(fft_data) * keep_ratio)
    sorted_mags = sort(vec(magnitudes), rev=true)
    threshold = sorted_mags[min(n_keep, length(sorted_mags))]
    
    # Create mask: keep coefficients above threshold
    mask = magnitudes .>= threshold
    
    # Apply mask
    compressed = copy(fft_data)
    compressed[.!mask] .= 0
    
    return compressed
end

"""
    compare_reconstructions(img_orig, img_hsv, img_rgb, output_dir)

Compare original and reconstructed images from HSV and RGB compression.
"""
function compare_reconstructions(img_orig, img_hsv, img_rgb, output_dir)
    # Convert to RGB for fair comparison
    img_orig_rgb = RGB.(img_orig)
    
    # Calculate errors
    # Convert images to matrices for computation
    orig_r = [Float64(pix.r) for pix in img_orig_rgb]
    orig_g = [Float64(pix.g) for pix in img_orig_rgb]
    orig_b = [Float64(pix.b) for pix in img_orig_rgb]
    
    hsv_r = [Float64(pix.r) for pix in img_hsv]
    hsv_g = [Float64(pix.g) for pix in img_hsv]
    hsv_b = [Float64(pix.b) for pix in img_hsv]
    
    rgb_r = [Float64(pix.r) for pix in img_rgb]
    rgb_g = [Float64(pix.g) for pix in img_rgb]
    rgb_b = [Float64(pix.b) for pix in img_rgb]
    
    # MSE for HSV reconstruction
    mse_hsv = mean((orig_r .- hsv_r).^2 .+ (orig_g .- hsv_g).^2 .+ (orig_b .- hsv_b).^2)
    
    # MSE for RGB reconstruction
    mse_rgb = mean((orig_r .- rgb_r).^2 .+ (orig_g .- rgb_g).^2 .+ (orig_b .- rgb_b).^2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr_hsv = 10 * log10(1.0 / mse_hsv)
    psnr_rgb = 10 * log10(1.0 / mse_rgb)
    
    # Difference images
    diff_hsv = abs.(orig_r .- hsv_r) .+ abs.(orig_g .- hsv_g) .+ abs.(orig_b .- hsv_b)
    diff_rgb = abs.(orig_r .- rgb_r) .+ abs.(orig_g .- rgb_g) .+ abs.(orig_b .- rgb_b)
    
    # Difference between HSV and RGB reconstructions
    diff_hsv_rgb = abs.(hsv_r .- rgb_r) .+ abs.(hsv_g .- rgb_g) .+ abs.(hsv_b .- rgb_b)
    
    println("\n  Reconstruction Quality Metrics:")
    println("  " * "-" ^ 60)
    println("  HSV Compression:")
    println("    - Mean Squared Error: ", round(mse_hsv, digits=6))
    println("    - PSNR: ", round(psnr_hsv, digits=2), " dB")
    println()
    println("  RGB Compression:")
    println("    - Mean Squared Error: ", round(mse_rgb, digits=6))
    println("    - PSNR: ", round(psnr_rgb, digits=2), " dB")
    println()
    println("  Difference between HSV and RGB reconstructions:")
    println("    - Mean absolute difference: ", round(mean(diff_hsv_rgb), digits=6))
    println("    - Max absolute difference: ", round(maximum(diff_hsv_rgb), digits=6))
    
    # Create comparison visualization
    p1 = plot(img_orig, title="Original", axis=false, showaxis=false)
    p2 = plot(img_hsv, title="HSV Compressed\n(MSE=$(round(mse_hsv, digits=6)))", 
              axis=false, showaxis=false)
    p3 = plot(img_rgb, title="RGB Compressed\n(MSE=$(round(mse_rgb, digits=6)))", 
              axis=false, showaxis=false)
    p4 = heatmap(diff_hsv, title="Error: Original vs HSV", 
                 color=:hot, axis=false, showaxis=false)
    p5 = heatmap(diff_rgb, title="Error: Original vs RGB", 
                 color=:hot, axis=false, showaxis=false)
    p6 = heatmap(diff_hsv_rgb, title="Error: HSV vs RGB", 
                 color=:hot, axis=false, showaxis=false)
    
    p_combined = plot(p1, p2, p3, p4, p5, p6, 
                     layout=(2, 3), 
                     size=(1200, 800))
    savefig(p_combined, joinpath(output_dir, "comparison.png"))
    
    # Write analysis to text file
    open(joinpath(output_dir, "analysis.txt"), "w") do f
        write(f, """
        Fourier Transform Image Compression Analysis
        ============================================
        
        Compression Settings:
        - Keep ratio: 1% (top 1% of Fourier coefficients by magnitude)
        
        HSV Compression Results:
        - Mean Squared Error: $(mse_hsv)
        - PSNR: $(psnr_hsv) dB
        
        RGB Compression Results:
        - Mean Squared Error: $(mse_rgb)
        - PSNR: $(psnr_rgb) dB
        
        Comparison:
        - Mean absolute difference between HSV and RGB: $(mean(diff_hsv_rgb))
        - Maximum absolute difference: $(maximum(diff_hsv_rgb))
        
        Observations and Explanation:
        
        1. Performance Comparison:
           - HSV compression typically performs $(mse_hsv < mse_rgb ? "better" : "worse") than RGB
             (lower MSE = better reconstruction)
           - The PSNR for HSV is $(round(psnr_hsv, digits=2)) dB vs $(round(psnr_rgb, digits=2)) dB for RGB
           - Higher PSNR indicates better quality
        
        2. Why HSV Often Performs Better:
           - HSV separates luminance (V channel) from chrominance (H, S channels)
           - Human visual system is more sensitive to luminance than color
           - The V (Value) channel contains most of the important structural information
           - H (Hue) and S (Saturation) can be more heavily compressed without noticeable artifacts
           - RGB channels are correlated, so compressing them independently is less efficient
        
        3. Fourier Transform Characteristics:
           - Both methods keep only 1% of coefficients (highest magnitudes)
           - Low frequency components (near DC) contain most image energy
           - These are preserved in both methods
           - High frequency components (details, edges) are mostly discarded
        
        4. Visual Quality:
           - HSV compression tends to preserve color appearance better
           - RGB compression may show more color distortion
           - Both methods may show some ringing artifacts around edges (Gibbs phenomenon)
           - Overall structure is well-preserved in both cases with 1% retention
        
        5. Mathematical Insight:
           - The HSV color space is more aligned with perceptual color organization
           - Compressing in a decorrelated color space (like HSV) is more efficient
           - This is similar to how JPEG uses YCbCr instead of RGB
        """)
    end
end


#=============================================================================
Main Execution
=============================================================================#

"""
    main()

Run both problems.
"""
function main()
    println("\n" * "█" ^ 80)
    println("HOMEWORK 5 - AMAT 5315")
    println("█" ^ 80)
    
    # Problem 1: SVD Image Compression
    try
        U, S, V, k_vals, comp_ratios, recon_errors = svd_mnist_compression()
    catch e
        println("\n❌ Error in Problem 1:")
        println(e)
        rethrow(e)
    end
    
    # Problem 2: Fourier Transform Image Processing
    try
        img_orig, img_hsv, img_rgb = fourier_image_compression()
    catch e
        println("\n❌ Error in Problem 2:")
        println(e)
        rethrow(e)
    end
    
    println("\n" * "█" ^ 80)
    println("ALL PROBLEMS COMPLETED SUCCESSFULLY!")
    println("█" ^ 80)
    println("\nOutput directories:")
    println("  - Problem 1: output_problem1/")
    println("  - Problem 2: output_problem2/")
    println("\nCheck these directories for visualizations and detailed results.")
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

