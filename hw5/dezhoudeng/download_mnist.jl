"""
MNIST Dataset Downloader

Simple function to download and load the MNIST dataset using MLDatasets.jl
Data is cached locally to avoid repeated downloads.

Usage:
    include("download_mnist.jl")
    train_images, train_labels = download_mnist(:train)
    test_images, test_labels = download_mnist(:test)
"""

using MLDatasets
using JLD2

"""
    download_mnist(split=:train)

Download and return the MNIST dataset. Data is cached locally in the 
`mnist_data/` directory to avoid repeated downloads.

# Arguments
- `split::Symbol`: Either `:train` or `:test` (default: `:train`)

# Returns
- `images`: 28×28×N array of images (N=60000 for train, N=10000 for test)
- `labels`: N-element vector of labels (0-9)

# Example
```julia
# Download training data
train_images, train_labels = download_mnist(:train)
println("Training set: ", size(train_images))  # (28, 28, 60000)

# Download test data
test_images, test_labels = download_mnist(:test)
println("Test set: ", size(test_images))  # (28, 28, 10000)

# Vectorize for SVD (each column is a flattened image)
X = reshape(train_images, 28*28, :)  # 784×60000 matrix
```
"""
function download_mnist(split::Symbol=:train)
    # Create cache directory if it doesn't exist
    cache_dir = "mnist_data"
    if !isdir(cache_dir)
        mkdir(cache_dir)
    end
    
    # Define cache file path
    cache_file = joinpath(cache_dir, "mnist_$(split).jld2")
    
    # Check if data already exists locally
    if isfile(cache_file)
        println("Loading MNIST $(split) data from cache: $cache_file")
        data = load(cache_file)
        return data["images"], data["labels"]
    else
        # Download data from MLDatasets
        println("Downloading MNIST $(split) data...")
        data = MNIST(split=split)
        images = data.features
        labels = data.targets
        
        # Save to cache
        println("Saving to cache: $cache_file")
        save(cache_file, Dict("images" => images, "labels" => labels))
        
        println("Download complete! Data cached for future use.")
        return images, labels
    end
end
