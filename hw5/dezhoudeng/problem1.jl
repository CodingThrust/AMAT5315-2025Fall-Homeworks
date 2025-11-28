using LinearAlgebra
using MLDatasets
using JLD2
using Plots

include("download_mnist.jl")

train_images, train_labels = download_mnist(:train)
test_images, test_labels = download_mnist(:test)

train_images_vectorized = reshape(train_images, 784, 60000)
test_images_vectorized = reshape(test_images, 784, 10000)

println("="^50)
println("Starting SVD decomposition on training set")
println("="^50)

# 1. Apply SVD decomposition to training set
train_data_float = Float64.(train_images_vectorized)
U, S, V = svd(train_data_float)

println("SVD decomposition completed!")

# 2. Test different k values
k_values = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]
compression_ratios = Float64[]
reconstruction_errors = Float64[]
energy_retained = Float64[]

println("\n" * "="^50)
println("Analyzing compression effects for different k values")
println("="^50)

for k in k_values
    println("Processing k = $k")
    
    # Keep first k singular values
    U_k = U[:, 1:k]
    S_k = S[1:k]
    V_k = V[:, 1:k]
    
    # Reconstruct compressed data
    train_compressed = U_k * Diagonal(S_k) * V_k'
    
    # Calculate compression ratio
    original_size = 784 * 60000
    compressed_size = 784 * k + k + 60000 * k
    compression_ratio = original_size / compressed_size
    push!(compression_ratios, compression_ratio)
    
    # Calculate reconstruction error (Frobenius norm)
    reconstruction_error = norm(train_data_float - train_compressed) / norm(train_data_float)
    push!(reconstruction_errors, reconstruction_error)
    
    # Calculate percentage of energy retained
    energy = sum(S_k.^2) / sum(S.^2) * 100
    push!(energy_retained, energy)
    
    println("  k=$k: Compression ratio=$(round(compression_ratio, digits=2))x, Reconstruction error=$(round(reconstruction_error*100, digits=4))%, Energy retained=$(round(energy, digits=2))%")
end

# 3. Plot compression ratio vs k
println("\n" * "="^50)
println("Generating compression ratio plot")
println("="^50)

# Create compression ratio plot
compression_plot = plot(k_values, compression_ratios, 
          label="Compression Ratio", 
          linewidth=3, 
          marker=:circle,
          markersize=6,
          xlabel="Number of Singular Values k",
          ylabel="Compression Ratio",
          title="Compression Ratio vs Number of Singular Values",
          legend=:topright,
          color=:blue,
          grid=true,
          size=(800, 500))

display(compression_plot)
savefig(compression_plot, "svd_compression_ratio.png")

# 4. Plot reconstruction error vs k
println("\nGenerating reconstruction error plot")
println("="^50)

# Create reconstruction error plot
error_plot = plot(k_values, reconstruction_errors .* 100, 
          label="Reconstruction Error", 
          linewidth=3, 
          marker=:square,
          markersize=6,
          xlabel="Number of Singular Values k",
          ylabel="Reconstruction Error (%)",
          title="Reconstruction Error vs Number of Singular Values",
          legend=:topright,
          color=:red,
          grid=true,
          size=(800, 500))

display(error_plot)
savefig(error_plot, "svd_reconstruction_error.png")

# Output detailed data table
println("\nDetailed Data Table:")
println("k\tCompression Ratio\tReconstruction Error(%)\tEnergy Retained(%)")
println("-"^60)
for (i, k) in enumerate(k_values)
    println("$k\t$(round(compression_ratios[i], digits=2))x\t\t\t$(round(reconstruction_errors[i]*100, digits=3))\t\t\t$(round(energy_retained[i], digits=2))")
end

# Visualization function for reconstruction comparison
function quick_compare_multiple(train_data, U, S, V, image_indices, k_values)
    n_samples = length(image_indices)
    n_k = length(k_values)
    
    # Create subplot array
    plots_array = []
    
    for (idx, img_idx) in enumerate(image_indices)
        # Original image - first column
        original = reshape(train_data[:, img_idx], 28, 28)
        p_original = heatmap(original', color=:grays, 
                           title=(idx == 1 ? "Original" : ""),
                           titlefontsize=10,
                           aspect_ratio=:equal, 
                           showaxis=false, 
                           colorbar=false)
        push!(plots_array, p_original)
        
        # Different k reconstructions - subsequent columns
        for k in k_values
            reconstructed = U[:, 1:k] * Diagonal(S[1:k]) * V[:, 1:k]'
            img = reshape(reconstructed[:, img_idx], 28, 28)
            p = heatmap(img', color=:grays, 
                       title=(idx == 1 ? "k=$k" : ""),
                       titlefontsize=10,
                       aspect_ratio=:equal, 
                       showaxis=false, 
                       colorbar=false)
            push!(plots_array, p)
        end
    end
    
    # Layout: rows = number of samples, columns = number of k values + 1 (original)
    n_cols = n_k + 1
    n_rows = n_samples
    
    # Create combined plot
    plot(plots_array..., 
         layout=(n_rows, n_cols),
         size=(400 * n_cols, 300 * n_rows),
         plot_title="SVD Compression Comparison - Multiple Samples")
end

# Example usage: display samples 1 to 9
println("\nGenerating visualization for samples 1-9...")
image_indices = 1:9
result_plot = quick_compare_multiple(train_data_float, U, S, V, image_indices, [10, 50, 100, 200])
display(result_plot)
savefig(result_plot, "svd_compression_multiple_samples.png")