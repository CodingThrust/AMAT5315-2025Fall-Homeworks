using MLDatasets
using LinearAlgebra
using Statistics
using Plots

include("download_mnist.jl")
train_images, train_labels = download_mnist(:train)
test_images, test_labels = download_mnist(:test)

# 数据预处理
X = Float32.(reshape(train_images, 28*28, :))
mean_image_X = vec(mean(X, dims=2))
X_centered = X .- mean_image_X

# SVD分解
U_X, S_X, Vt_X = svd(X_centered)

# 使用200个主成分重建图像
k = 200
U_k = U_X[:, 1:k]
S_k = Diagonal(S_X[1:k])
Vt_k = Vt_X[1:k, :]

X_reconstructed = U_k * S_k * Vt_k .+ mean_image_X

# 计算解释方差
total_variance = sum(S_X.^2)
explained_variance = sum(S_X[1:k].^2) / total_variance
println("使用 k=200 个主成分的解释方差比例: $(round(explained_variance, digits=4))")

# 可视化比较
function compare_images(original, reconstructed, labels, n=5)
    max_idx = min(size(original, 2), size(reconstructed, 2))
    indices = rand(1:max_idx, n)
    
    plots = []
    for idx in indices
        # 原始图像
        p1 = heatmap(reshape(original[:, idx], 28, 28), 
                    aspect_ratio=:equal, color=:grays,
                    title="Original: $(labels[idx])", 
                    showaxis=false)
        
        # 重建图像
        p2 = heatmap(reshape(reconstructed[:, idx], 28, 28), 
                    aspect_ratio=:equal, color=:grays,
                    title="Reconstructed: $(labels[idx])",
                    showaxis=false)
        
        push!(plots, p1, p2)
    end
    
    plot(plots..., layout=(n, 2), size=(600, 200*n))
end

# 显示结果
compare_images(X, X_reconstructed, train_labels)