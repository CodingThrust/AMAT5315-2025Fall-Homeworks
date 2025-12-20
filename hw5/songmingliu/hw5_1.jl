using LinearAlgebra
using Plots
using MLDatasets
using JLD2

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

function download_mnist(split::Symbol=:train)
    cache_dir = "mnist_data"
    if !isdir(cache_dir)
        mkdir(cache_dir)
    end
    
    cache_file = joinpath(cache_dir, "mnist_$(split).jld2")
    
    if isfile(cache_file)
        println("Loading MNIST $(split) data from cache: $cache_file")
        data = load(cache_file)
        return data["images"], data["labels"]
    else
        println("Downloading MNIST $(split) data...")
        data = MNIST(split=split)
        images = data.features
        labels = data.targets
        
        println("Saving to cache: $cache_file")
        save(cache_file, Dict("images" => images, "labels" => labels))
        
        println("Download complete! Data cached for future use.")
        return images, labels
    end
end

function svd_image_compression_analysis()
    println("=== MNIST SVD Compression Analysis ===")
    
    println("Downloading MNIST training data...")
    train_images, train_labels = download_mnist(:train)
    
    n_samples = 1000  # Reduced for faster computation
    subset_indices = 1:n_samples
    train_images_subset = train_images[:, :, subset_indices]
    train_labels_subset = train_labels[subset_indices]
    
    println("Vectorizing images...")
    data_matrix = reshape(Float64.(train_images_subset), 28*28, n_samples)
    
    println("Data matrix dimensions: ", size(data_matrix))
    
    println("Computing SVD...")
    U, S, Vt = svd(data_matrix)
    
    println("Singular values computed. Total: ", length(S))
    
    function perform_compression(num_components::Int)
        # Ensure we don't exceed the number of available singular values
        k = min(num_components, length(S))
        
        # Extract top k components
        U_k = U[:, 1:k]
        S_k = S[1:k]
        Vt_k = Vt[1:k, :]
        
        # Reconstruct using k singular values
        reconstructed_data = U_k * Diagonal(S_k) * Vt_k
        
        # Ensure reconstructed_data has same dimensions as original
        # If k is less than the number of columns, we need to pad with zeros
        if size(reconstructed_data) != size(data_matrix)
            # Pad with zeros to match original dimensions
            reconstructed_full = zeros(Float64, size(data_matrix)...)
            reconstructed_full[:, 1:size(reconstructed_data, 2)] = reconstructed_data
            reconstructed_data = reconstructed_full
        end
        
        # Calculate metrics
        original_norm = norm(data_matrix)
        reconstruction_error = norm(data_matrix - reconstructed_data) / original_norm
        original_size = size(data_matrix, 1) * size(data_matrix, 2)
        compressed_size = size(U_k, 1) * size(U_k, 2) + length(S_k) + size(Vt_k, 1) * size(Vt_k, 2)
        compression_ratio = compressed_size / original_size
        
        return reconstructed_data, reconstruction_error, compression_ratio
    end
    
    k_values = [5, 15, 30, 50, 100, 150, 200]
    errors = Float64[]
    ratios = Float64[]
    
    println("Testing different compression levels...")
    for k in k_values
        if k <= length(S)
            _, error, ratio = perform_compression(k)
            push!(errors, error)
            push!(ratios, ratio)
            println("k = $(k): Error = $(round(error, digits=4)), Ratio = $(round(ratio, digits=4))")
        end
    end
    
    println("Creating visualization of compression quality...")
    
    sample_indices = [1, 100, 500]
    compression_levels = [10, 50, 150]
    
    for img_idx in sample_indices
        original = reshape(data_matrix[:, img_idx], 28, 28)
        
        plots = []
        push!(plots, heatmap(original, aspect_ratio=1, color=:grays, title="Original", 
                           showaxis=false, colorbar=false))
        
        for k in compression_levels
            if k <= length(S)
                reconstructed_full, _, _ = perform_compression(k)
                reconstructed_img = reshape(reconstructed_full[:, img_idx], 28, 28)
                push!(plots, heatmap(reconstructed_img, aspect_ratio=1, color=:grays, 
                                   title="k=$(k)", showaxis=false, colorbar=false))
            end
        end
        
        comparison_plot = plot(plots..., layout=(1, length(plots)), size=(600, 120))
        println("Sample image $(img_idx) reconstruction comparison:")
        display(comparison_plot)
    end
    
    singular_plot = plot(1:min(200, length(S)), S[1:min(200, length(S))], 
                        xlabel="Singular Value Index", ylabel="Singular Value", 
                        title="Singular Values (Top 200)", 
                        marker=:circle, markersize=3, legend=false)
    
    ratio_plot = plot(k_values, ratios, 
                     xlabel="Number of Components (k)", 
                     ylabel="Compression Ratio", 
                     title="Compression Ratio vs k", 
                     marker=:circle, markersize=5, legend=false)
    
    error_plot = plot(k_values, errors, 
                     xlabel="Number of Components (k)", 
                     ylabel="Relative Reconstruction Error", 
                     title="Reconstruction Error vs k", 
                     marker=:circle, markersize=5, legend=false, 
                     yscale=:log10)
    
    metrics_plot = plot(singular_plot, ratio_plot, error_plot, 
                       layout=(3,1), size=(800, 900))
    println("Compression metrics visualization:")
    display(metrics_plot)
    
    cumulative_energy = cumsum(S.^2) ./ sum(S.^2)
    energy_plot = plot(1:length(cumulative_energy), cumulative_energy, 
                      xlabel="Number of Components", 
                      ylabel="Cumulative Energy Preserved", 
                      title="Cumulative Energy Preservation", 
                      legend=false, linewidth=2)
    println("Cumulative energy preservation:")
    display(energy_plot)
    
    results_dict = Dict(
        "k_values" => k_values,
        "reconstruction_errors" => errors,
        "compression_ratios" => ratios,
        "singular_values" => S,
        "cumulative_energy" => cumulative_energy,
        "data_matrix_shape" => size(data_matrix)
    )
    
    save("mnist_svd_analysis.jld2", results_dict)
    println("Results saved to mnist_svd_analysis.jld2")
    
    println("\n=== SVD Compression Summary ===")
    println("Original data size: $(size(data_matrix, 1)) × $(size(data_matrix, 2))")
    println("Total singular values: $(length(S))")
    println("Energy preserved by top 50 components: $(round(cumulative_energy[50]*100, digits=2))%")
    println("Energy preserved by top 100 components: $(round(cumulative_energy[100]*100, digits=2))%")
    
    return results_dict
end

results = svd_image_compression_analysis()

println("\nSVD compression analysis completed successfully!")



