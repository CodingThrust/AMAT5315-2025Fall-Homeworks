
using LinearAlgebra
using Images
using Plots
include("download_mnist.jl")
# Download MNIST dataset
train_images, train_labels = download_mnist(:train)
test_images, test_labels = download_mnist(:test)

println("Train set: ", size(train_images))
println("Test set: ", size(test_images))

# Flatten images and perform SVD
flattened_images = reshape(train_images, 28 * 28, :)
U, D, V = svd(flattened_images)

# Test different compression dimensions
k = [10, 50, 100, 200]
err = zeros(4)
compressed_ratio = zeros(4)

for i in eachindex(k)
    dim = k[i]
    
    # Calculate compression ratio
    m, n = size(flattened_images)
    compressed_ratio[i] = (dim * (1 + m + n)) / (m * n)
    
    # Compress and reconstruct
    compressed_D = D[1:dim]
    compressed_images = U[:,1:dim] * Diagonal(compressed_D) * V[:,1:dim]'
    reconstructed_images = reshape(compressed_images, 28, 28, :)
    
    # Calculate reconstruction error on test set
    for j in 1:size(test_images, 3)
        test_vec = reshape(test_images[:,:,j], :)
        compressed_test = U[:,1:dim] * (U[:,1:dim]' * test_vec)
        err[i] += norm(test_vec - compressed_test, 2)
    end
    err[i] /= size(test_images, 3)
    
    
        example_idx = 20
        original_img = Gray.(test_images[:,:,example_idx])
        compressed_vec = U[:,1:dim] * (U[:,1:dim]' * reshape(test_images[:,:,example_idx], :))
        compressed_img = Gray.(clamp.(reshape(compressed_vec, 28, 28), 0.0, 1.0))
        
        save("D://juliahw//hw5//original.png", original_img)
        save("D://juliahw//hw5//compressed_k$(dim).png", compressed_img)
    
end

# Plot and save results
p1 = plot(k, compressed_ratio, xlabel="k", ylabel="Compression Ratio", 
          title="Compression Ratio vs k", marker=:circle, linewidth=2, label=false)
savefig("D://juliahw//hw5//compressed_ratio.png")

p2 = plot(k, err, xlabel="k", ylabel="Reconstruction Error", 
          title="Reconstruction Error vs k", marker=:circle, linewidth=2, label=false)
savefig("D://juliahw//hw5//reconstructed_error.png")

# Print results
println("\nResults:")
for i in eachindex(k)
    println("k=$(k[i]): Ratio=$(round(compressed_ratio[i], digits=4)), Error=$(round(err[i], digits=4))")
end

