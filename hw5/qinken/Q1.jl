using LinearAlgebra
using Plots
using MLDatasets
using JLD2

# Set to automatically accept dataset downloads
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

include("download_mnist.jl")

function main()
    # Task 1: Download MNIST dataset using provided function
    println("Downloading MNIST dataset...")
    train_images, train_labels = download_mnist(:train)
    test_images, test_labels = download_mnist(:test)
    
    # Task 2: Vectorize images (flatten each 28×28 image into 784-dimensional vector)
    println("Vectorizing images...")
    X_train = reshape(train_images, 28*28, :)
    X_test = reshape(test_images, 28*28, :)
    
    # Use all training data for analysis
    X = Float64.(X_train)
    
    # Task 3: Apply SVD to the data matrix
    println("Performing SVD...")
    U, S, Vt = svd(X)
    
    # Task 4: Compress dataset by retaining top k singular values
    function compress_and_reconstruct(k::Int, U, S, Vt, X_original)
        U_k = U[:, 1:k]
        S_k = Diagonal(S[1:k])
        Vt_k = Vt[1:k, :]
        
        # Reconstruct using k singular values
        temp = S_k * Vt_k
        X_reconstructed = U_k * temp
        
        # Calculate compression ratio and error
        original_size = length(X_original)
        compressed_size = length(U_k) + length(S_k) + length(Vt_k)
        compression_ratio = compressed_size / original_size
        reconstruction_error = norm(X_original - X_reconstructed) / norm(X_original)
        
        return X_reconstructed, compression_ratio, reconstruction_error
    end
    
    # Experiment with different k values
    k_values = [1, 5, 10, 20, 50, 100, 200, 300, 400]
    compression_ratios = Float64[]
    reconstruction_errors = Float64[]
    
    println("Testing compression with k values: ", k_values)
    
    for k in k_values
        if k <= length(S)
            X_recon, cr, error = compress_and_reconstruct(k, U, S, Vt, X)
            push!(compression_ratios, cr)
            push!(reconstruction_errors, error)
            println("k=$k: compression ratio=$(round(cr, digits=4)), error=$(round(error, digits=4))")
        end
    end
    
    # Task 5: Visualize and compare original vs reconstructed images
    sample_indices = [1, 2, 3]
    demo_k_values = [1, 10, 50, 200]
    
    for sample_idx in sample_indices
        original_image = reshape(X[:, sample_idx], 28, 28)
        original_label = train_labels[sample_idx]
        
        row_plots = []
        
        # Original image
        p_original = heatmap(original_image, 
                           aspect_ratio=1, 
                           color=:grays, 
                           title="Original (label: $original_label)",
                           showaxis=false, 
                           colorbar=false)
        push!(row_plots, p_original)
        
        # Reconstructed images at different k values
        for k in demo_k_values
            if k <= length(S)
                X_recon, _, _ = compress_and_reconstruct(k, U, S, Vt, X)
                reconstructed_image = reshape(X_recon[:, sample_idx], 28, 28)
                
                p_recon = heatmap(reconstructed_image, 
                                aspect_ratio=1, 
                                color=:grays, 
                                title="k = $k",
                                showaxis=false, 
                                colorbar=false)
                push!(row_plots, p_recon)
            end
        end
        
        row_plot = plot(row_plots..., 
                       layout=(1, length(row_plots)), 
                       size=(800, 200))
        println("Sample $sample_idx reconstruction:")
        display(row_plot)
    end
    
    # Task 6: Plot compression ratio and reconstruction error as function of k
    p1 = plot(k_values, compression_ratios, 
             xlabel="Number of singular values (k)", 
             ylabel="Compression Ratio", 
             title="Compression Ratio vs k",
             marker=:circle, 
             linewidth=2, 
             legend=false)
    
    p2 = plot(k_values, reconstruction_errors, 
             xlabel="Number of singular values (k)", 
             ylabel="Relative Reconstruction Error", 
             title="Reconstruction Error vs k",
             marker=:circle, 
             linewidth=2, 
             legend=false,
             yscale=:log10)
    
    compression_plot = plot(p1, p2, layout=(1,2), size=(1000, 400))
    println("Compression analysis:")
    display(compression_plot)
    
    # Save results
    results = Dict(
        "k_values" => k_values,
        "compression_ratios" => compression_ratios,
        "reconstruction_errors" => reconstruction_errors,
        "singular_values" => S
    )
    
    save("svd_compression_results.jld2", results)
    
    return results
end

# Run the main function
results = main()