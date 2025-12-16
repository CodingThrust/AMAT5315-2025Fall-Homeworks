# download_mnist.jl
#
# MNIST Dataset Downloader
#
# Simple function to download and load the MNIST dataset using MLDatasets.jl.
# Data is cached locally (using JLD2) to avoid repeated downloads.
#
# Usage:
#     include("download_mnist.jl")
#     train_images, train_labels = download_mnist(:train)
#     test_images,  test_labels  = download_mnist(:test)

using MLDatasets
using JLD2
using FileIO  # for load/save via JLD2

"""
    download_mnist(split::Symbol = :train)

Download and return the MNIST dataset. Data is cached locally in the
`mnist_data/` directory (next to this file) to avoid repeated downloads.

# Arguments
- `split::Symbol`: Either `:train` or `:test` (default: `:train`)

# Returns
- `images`: 28×28×N array of images (N=60000 for train, N=10000 for test)
- `labels`: N-element vector of labels (0-9)
"""
function download_mnist(split::Symbol = :train)
    @assert split == :train || split == :test "split must be :train or :test"

    # Cache directory placed next to this file, e.g. hw5/JizheLai/mnist_data
    cache_dir = joinpath(@__DIR__, "mnist_data")
    isdir(cache_dir) || mkpath(cache_dir)

    # Cache file path, e.g. mnist_data/mnist_train.jld2
    cache_file = joinpath(cache_dir, "mnist_$(split).jld2")

    # If JLD2 cache exists, load directly
    if isfile(cache_file)
        println("Loading MNIST $(split) data from cache: $cache_file")
        data = load(cache_file)
        return data["images"], data["labels"]
    end

    # Otherwise download via MLDatasets.
    # We also tell MLDatasets to use our cache_dir for raw files.
    println("Downloading MNIST $(split) data via MLDatasets...")
    mnist = MNIST(split; dir = cache_dir)

    # Newer MLDatasets returns an object with .features and .targets
    images = mnist.features
    labels = mnist.targets

    # Save to JLD2 cache for future runs
    println("Saving MNIST $(split) data to cache: $cache_file")
    save(cache_file, Dict("images" => images, "labels" => labels))

    println("Download complete! Data cached for future use.")
    return images, labels
end