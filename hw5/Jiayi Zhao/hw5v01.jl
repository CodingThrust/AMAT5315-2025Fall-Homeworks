# ==========================================
# HW5: SVD and FFT Compression (Fixed Version)
# ==========================================

# 1. 设置自动同意下载，跳过 [y/n] 询问
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# 2. 引入必要的包
using LinearAlgebra
using Statistics
using Plots
using MLDatasets
using Images
using FFTW
using FileIO
using ColorTypes

# ==========================================
# PART 1: MNIST SVD Compression
# ==========================================
println("--- Part 1: MNIST SVD ---")

# 下载并加载数据 (会自动跳过询问)
train_x, train_y = MNIST(split=:train)[:] 

# 向量化: (28, 28, 60000) -> (784, 60000)
img_h, img_w, total_imgs = size(train_x)
dim = img_h * img_w
data_matrix = reshape(train_x, dim, total_imgs)

# 取前 2000 张做实验 (加快速度)
num_samples = 2000
X = Float64.(data_matrix[:, 1:num_samples])
println("Data Matrix Shape: ", size(X)) # 应该是 (784, 2000)

# SVD 分解
println("Computing SVD...")
F = svd(X)
U, S, Vt = F.U, F.S, F.Vt

# 定义修正后的重建函数
function reconstruct_svd(U, S, Vt, k)
    # 修正点：Vt 已经是 V^T 了，不需要再转置
    # 我们取 Vt 的前 k 行 (对应 V 的前 k 列)
    return U[:, 1:k] * Diagonal(S[1:k]) * Vt[1:k, :]
end

k_values = [10, 50, 100, 200]
errors = Float64[]

# 准备可视化
sample_idx = 1 #以此图为例
original_img = reshape(X[:, sample_idx], img_h, img_w)'
plts = []
push!(plts, heatmap(original_img, c=:grays, title="Original", axis=false, legend=false, yflip=true))

println("Reconstructing images...")
for k in k_values
    # 重建
    X_rec = reconstruct_svd(U, S, Vt, k)
    
    # 计算误差 (Frobenius norm)
    curr_error = norm(X - X_rec) / norm(X)
    push!(errors, curr_error)
    println("k=$k, Error=$curr_error")
    
    # 添加到绘图列表
    rec_img = reshape(X_rec[:, sample_idx], img_h, img_w)'
    push!(plts, heatmap(rec_img, c=:grays, title="k=$k", axis=false, legend=false, yflip=true))
end

# 保存 SVD 对比图
p_svd = plot(plts..., layout=(1, 5), size=(1000, 250))
savefig("mnist_svd.png")

# 保存误差曲线图
p_err = plot(k_values, errors, marker=:o, label="Error", 
             xlabel="k", ylabel="Frobenius Error", title="SVD Compression Error")
savefig("mnist_error.png")
println("Part 1 Completed. Files saved: mnist_svd.png, mnist_error.png")

# ==========================================
# PART 2: Cat Image FFT (RGB vs HSV)
# ==========================================
println("\n--- Part 2: Cat Image FFT ---")

# 下载图片 (如果不存在)
if !isfile("cat.png")
    download("https://raw.githubusercontent.com/CodingThrust/AMAT5315-2025Fall-Homeworks/main/hw5/cat.png", "cat.png")
end
img = load("cat.png")

# FFT 压缩函数
function fft_compress_channel(channel_data, keep_ratio=0.01)
    f_transform = fft(channel_data)
    magnitudes = abs.(f_transform)
    sorted_mags = sort(vec(magnitudes), rev=true)
    # 找到前 1% 的阈值
    threshold = sorted_mags[Int(floor(length(sorted_mags) * keep_ratio))]
    # 掩码操作
    mask = magnitudes .>= threshold
    return real.(ifft(f_transform .* mask))
end

# --- A. RGB 压缩 ---
println("Processing RGB...")
r = fft_compress_channel(Float64.(red.(img)))
g = fft_compress_channel(Float64.(green.(img)))
b = fft_compress_channel(Float64.(blue.(img)))
img_rgb_rec = colorview(RGB, r, g, b)
save("cat_rgb_rec.png", clamp01nan.(img_rgb_rec))

# --- B. HSV 压缩 ---
println("Processing HSV...")
img_hsv = HSV.(img)
h = fft_compress_channel(Float64.(getfield.(img_hsv, :h)))
s = fft_compress_channel(Float64.(getfield.(img_hsv, :s)))
v = fft_compress_channel(Float64.(getfield.(img_hsv, :v)))
# 转换回 RGB 保存
img_hsv_rec = HSV.(h, s, v)
img_hsv_rec_rgb = RGB.(img_hsv_rec)
save("cat_hsv_rec.png", clamp01nan.(img_hsv_rec_rgb))

# --- 可视化对比 ---
p_cat = plot(
    plot(img, title="Original", axis=false),
    plot(img_rgb_rec, title="RGB FFT (1%)", axis=false),
    plot(img_hsv_rec_rgb, title="HSV FFT (1%)", axis=false),
    layout=(1,3), size=(900,300)
)
savefig("cat_compare.png")

println("Part 2 Completed. Files saved: cat_rgb_rec.png, cat_hsv_rec.png, cat_compare.png")
println("All tasks finished!")