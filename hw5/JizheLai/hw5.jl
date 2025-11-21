#!/usr/bin/env julia

############################################################
# AMAT5315 — HW5
# Student: Jizhe Lai
#
# This script:
#   1. Downloads MNIST (train + test) via download_mnist.jl
#   2. Vectorizes images into 784×N matrix
#   3. Computes SVD / PCA on the training data
#   4. Prints basic information (singular values, variance ratios)
############################################################

using LinearAlgebra
using Statistics

# Load the MNIST downloader (must be in the same directory)
include("download_mnist.jl")

############################
# 1. Load MNIST data
############################

println("=== Loading MNIST data ===")

train_images, train_labels = download_mnist(:train)
test_images,  test_labels  = download_mnist(:test)

println("Train images size: ", size(train_images))   # (28, 28, 60000)
println("Train labels size: ", size(train_labels))   # (60000,)
println("Test  images size: ", size(test_images))    # (28, 28, 10000)
println("Test  labels size: ", size(test_labels))    # (10000,)

############################
# 2. Vectorize images
############################
# Each column is a flattened 28×28 image → 784×N

println("\n=== Vectorizing data ===")

X_train = reshape(train_images, :, size(train_images, 3))  # 784×60000
X_test  = reshape(test_images,  :, size(test_images, 3))   # 784×10000

println("X_train size: ", size(X_train))
println("X_test  size: ", size(X_test))

############################
# 3. Mean-centering (for PCA)
############################

println("\n=== Mean-centering training data ===")

μ = mean(X_train; dims = 2)        # 784×1 mean vector
Xc_train = X_train .- μ            # centered train data
Xc_test  = X_test  .- μ            # center test using train mean

############################
# 4. SVD on centered training data
############################

println("\n=== Computing SVD on centered train data ===")

@time U, S, Vt = svd(Xc_train; full = false)

println("U size: ", size(U))       # 784×784
println("S length: ", length(S))   # 784
println("Vt size: ", size(Vt))     # 784×60000

############################
# 5. PCA / variance information
############################

println("\n=== PCA / SVD summary ===")

# Explained variance (proportional to singular values squared)
explained_var   = S.^2
total_var       = sum(explained_var)
explained_ratio = explained_var ./ total_var

# Cumulative explained variance
cum_explained = cumsum(explained_ratio)

k90 = findfirst(>=(0.90), cum_explained)
k95 = findfirst(>=(0.95), cum_explained)

println("First 10 singular values: ", S[1:10])
println("First 10 explained variance ratios: ", explained_ratio[1:10])
println("Cumulative explained variance (first 10): ", cum_explained[1:10])
println("Number of principal components for ≥90% variance: ", k90)
println("Number of principal components for ≥95% variance: ", k95)

println("\n=== HW5 script finished successfully ===")

# Train images size: (28, 28, 60000)
# Train labels size: (60000,)
# Test  images size: (28, 28, 10000)
# Test  labels size: (10000,)

# === Vectorizing data ===
# X_train size: (784, 60000)
# X_test  size: (784, 10000)

# === Mean-centering training data ===

# === Computing SVD on centered train data ===
#   3.048989 seconds (282.18 k allocations: 385.241 MiB, 0.20% gc time, 3.30% compilation time)
# U size: (784, 784)
# S length: 784
# Vt size: (60000, 784)

# === PCA / SVD summary ===
# First 10 singular values: Float32[554.08746, 473.7967, 441.77908, 412.91513, 392.46338, 369.35358, 321.72803, 302.04852, 295.59808, 273.0654]
# First 10 explained variance ratios: Float32[0.09704665, 0.070959084, 0.061692767, 0.05389463, 0.048688028, 0.04312296, 0.032719128, 0.028838811, 0.02762022, 0.023569874]
# Cumulative explained variance (first 10): Float32[0.09704665, 0.16800573, 0.22969851, 0.28359312, 0.33228114, 0.3754041, 0.40812322, 0.43696204, 0.46458226, 0.48815215]
# Number of principal components for ≥90% variance: 87
# Number of principal components for ≥95% variance: 154

# === HW5 script finished successfully ===