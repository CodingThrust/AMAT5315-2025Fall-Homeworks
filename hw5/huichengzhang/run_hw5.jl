#!/usr/bin/env julia

# AMAT5315 - Homework 5 (Julia)
# Author: huichengzhang

using LinearAlgebra
using Statistics
using Printf
using Random
using DelimitedFiles
using Colors
using FFTW
using Images
using ImageIO
using FileIO
using Plots

include(joinpath(@__DIR__, "..", "download_mnist.jl"))

function ensure_dir(path::AbstractString)
    if !isdir(path)
        mkpath(path)
    end
end

function vectorize_images(images::AbstractArray)
    # images: 28×28×N -> 784×N
    @assert ndims(images) == 3 "Expect 28×28×N array"
    h, w, n = size(images)
    @assert h == 28 && w == 28 "MNIST images must be 28×28"
    reshape(Float64.(images), h*w, n)
end

function compute_compression_ratio(m::Int, n::Int, k::Int)
    # data matrix size: m×n (784×N). Full storage = m*n.
    # Rank-k SVD storage ~ k*(m + n + 1) (U: m×k, V: n×k, Σ: k)
    full = m * n
    comp = k * (m + n + 1)
    full / comp
end

function svd_compress_and_reconstruct(X::AbstractMatrix{<:Real}, ks::AbstractVector{<:Integer})
    # Centering optional; here we keep original grayscale range [0,1] if input scaled
    F = Float64.(X)
    U, S, V = svd(F)
    reconstructions = Dict{Int, Matrix{Float64}}()
    errors = Dict{Int, Float64}()
    ratios = Dict{Int, Float64}()
    m, n = size(F)
    for k in ks
        Uk = @view U[:, 1:k]
        Sk = @view S[1:k]
        Vk = @view V[:, 1:k]
        Xk = Uk * Diagonal(Sk) * Vk'
        reconstructions[k] = Xk
        # Frobenius relative error
        err = sqrt(sum(abs2, F .- Xk)) / sqrt(sum(abs2, F))
        errors[k] = err
        ratios[k] = compute_compression_ratio(m, n, k)
    end
    reconstructions, errors, ratios
end

function save_mnist_visualizations(save_dir::AbstractString, X::AbstractMatrix, recon::Dict{Int, Matrix{Float64}}, ks::Vector{Int};
                                   num_examples::Int=8)
    ensure_dir(save_dir)
    m, n = size(X)
    h = 28; w = 28
    # Normalize images to 0..1 for saving
    toimg = x -> Gray.(clamp01.(reshape(x, h, w)))
    idxs = collect(1:min(num_examples, n))
    # save original examples
    for i in idxs
        img = toimg(X[:, i])
        save(joinpath(save_dir, @sprintf("mnist_original_%02d.png", i)), img)
    end
    # save reconstructions per k
    for k in ks
        Xk = recon[k]
        for i in idxs
            img = toimg(Xk[:, i])
            save(joinpath(save_dir, @sprintf("mnist_recon_k%03d_%02d.png", k, i)), img)
        end
    end
end

function run_task1_mnist_svd(output_root::AbstractString; ks::Vector{Int}=[10, 50, 100, 200], use_split::Symbol=:train, sample_n::Int=2000)
    println("== Task 1: SVD Compression on MNIST ==")
    # ensure non-interactive DataDeps download
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    ensure_dir(output_root)
    # Load data
    images, labels = download_mnist(use_split)
    # Scale to [0,1] if needed; MNIST from MLDatasets is UInt8 0..255
    Xfull = vectorize_images(images) ./ 255.0
    # Optional subsample for speed
    n_total = size(Xfull, 2)
    n_use = min(sample_n, n_total)
    X = Xfull[:, 1:n_use]
    println(@sprintf("Data matrix size: %d×%d (using first %d of %d)", size(X,1), size(X,2), n_use, n_total))

    recon, errors, ratios = svd_compress_and_reconstruct(X, ks)

    # Save visualizations
    save_dir = joinpath(output_root, "mnist_svd")
    save_mnist_visualizations(save_dir, X, recon, ks)

    # Plot error and compression ratio vs k
    ks_sorted = sort(collect(keys(errors)))
    errs = [errors[k] for k in ks_sorted]
    rts = [ratios[k] for k in ks_sorted]

    p1 = plot(ks_sorted, errs, xlabel="k", ylabel="Relative Frobenius Error", title="MNIST SVD Reconstruction Error", marker=:o)
    savefig(p1, joinpath(save_dir, "mnist_svd_error.png"))

    p2 = plot(ks_sorted, rts, xlabel="k", ylabel="Compression Ratio (full/comp)", title="MNIST SVD Compression Ratio", marker=:o)
    savefig(p2, joinpath(save_dir, "mnist_svd_ratio.png"))

    # Save CSV of metrics
    metrics_path = joinpath(save_dir, "mnist_svd_metrics.csv")
    open(metrics_path, "w") do io
        println(io, "k,error,ratio")
        for k in ks_sorted
            @printf(io, "%d,%.6f,%.6f\n", k, errors[k], ratios[k])
        end
    end
    println("Saved MNIST SVD outputs to: ", save_dir)
end

function fft_keep_top_percent!(F::AbstractMatrix{<:Complex}, percent::Float64)
    # Keep top P% coefficients by magnitude, zero others
    @assert 0 < percent <= 100
    mags = abs.(F)
    flat = vec(mags)
    total = length(flat)
    k = max(1, round(Int, total * percent / 100))
    # threshold = k-th largest magnitude
    topk = partialsort(flat, 1:k; rev=true)
    thresh = topk[end]
    F .*= (mags .>= thresh)
    F
end

function compress_image_fft_rgb(img_rgb; percent::Float64=1.0)
    # accept any AbstractMatrix of colorant; convert to RGB{Float64}
    img_rgb = RGB{Float64}.(img_rgb)
    h, w = size(img_rgb)
    channels = (channelview(img_rgb))  # 3×H×W
    out = similar(img_rgb)
    for c in 1:3
        plane = Array{Float64}(reshape(channels[c, :, :], h, w))
        F = fft(plane)
        F = fft_keep_top_percent!(F, percent)
        rec = real(ifft(F))
        out[:, :] .= RGB.(clamp01.(rec), clamp01.(rec), clamp01.(rec)) # placeholder overwritten below
        # replace just channel c after we rebuild channels
        channels[c, :, :] .= rec
    end
    # rebuild RGB from channels
    recons = colorview(RGB, channels)
    clamp01.(recons)
end

function compress_image_fft_hsv(img_rgb; percent::Float64=1.0)
    img_hsv = HSV.(RGB{Float64}.(img_rgb))
    h, w = size(img_hsv)
    Hc = map(c -> c.h, img_hsv)
    Sc = map(c -> c.s, img_hsv)
    Vc = map(c -> c.v, img_hsv)
    for comp in (Hc, Sc, Vc)
        F = fft(comp)
        fft_keep_top_percent!(F, percent)
        rec = real(ifft(F))
        comp .= clamp01.(rec)
    end
    recons = [HSV(Hc[i], Sc[i], Vc[i]) for i in eachindex(Hc)]
    recons_img = reshape(recons, size(img_hsv))
    RGB.(recons_img)
end

function mse_rgb(img1, img2)
    A = channelview(RGB{Float64}.(img1))
    B = channelview(RGB{Float64}.(img2))
    d = A .- B
    mse_total = mean(d .^ 2)
    mse_r = mean((A[1, :, :] .- B[1, :, :]) .^ 2)
    mse_g = mean((A[2, :, :] .- B[2, :, :]) .^ 2)
    mse_b = mean((A[3, :, :] .- B[3, :, :]) .^ 2)
    return mse_total, (mse_r, mse_g, mse_b)
end

function save_diff_image(img1, img2, outpath::AbstractString)
    A = channelview(RGB{Float64}.(img1))
    B = channelview(RGB{Float64}.(img2))
    dR = abs.(A[1, :, :] .- B[1, :, :])
    dG = abs.(A[2, :, :] .- B[2, :, :])
    dB = abs.(A[3, :, :] .- B[3, :, :])
    rms = sqrt.((dR .^ 2 .+ dG .^ 2 .+ dB .^ 2) ./ 3)
    save(outpath, Gray.(clamp01.(rms)))
end

function run_task2_fft_cat(output_root::AbstractString; percent::Float64=1.0)
    println("== Task 2: FFT Compression on cat.png ==")
    ensure_dir(output_root)
    input_path = joinpath(@__DIR__, "..", "cat.png")
    @assert isfile(input_path) "cat.png not found at $(input_path)"
    img = load(input_path)
    img_rgb = RGB.(float.(img))

    save_dir = joinpath(output_root, "cat_fft")
    ensure_dir(save_dir)

    # Save original
    save(joinpath(save_dir, "cat_original.png"), img_rgb)

    # RGB FFT 1%
    rec_rgb = compress_image_fft_rgb(img_rgb; percent=percent)
    save(joinpath(save_dir, @sprintf("cat_rgb_fft_%0.1fpercent.png", percent)), rec_rgb)

    # HSV FFT 1%
    rec_hsv = compress_image_fft_hsv(img_rgb; percent=percent)
    save(joinpath(save_dir, @sprintf("cat_hsv_fft_%0.1fpercent.png", percent)), rec_hsv)

    # Metrics and comparison
    mse_rgb_total, (mse_rr, mse_rg, mse_rb) = mse_rgb(img_rgb, rec_rgb)
    mse_hsv_total, (mse_hr, mse_hg, mse_hb) = mse_rgb(img_rgb, rec_hsv)
    mse_between_total, (mse_br, mse_bg, mse_bb) = mse_rgb(rec_rgb, rec_hsv)

    # Save difference image between two reconstructions
    save_diff_image(rec_rgb, rec_hsv, joinpath(save_dir, "cat_diff_rgb_vs_hsv.png"))

    # Save metrics and brief explanation
    metrics_path = joinpath(save_dir, "cat_fft_metrics.txt")
    open(metrics_path, "w") do io
        println(io, "Cat FFT Compression (keep $(percent)% coefficients)")
        println(io, "MSE original→RGB: ", @sprintf("%.6e (R=%.6e, G=%.6e, B=%.6e)", mse_rgb_total, mse_rr, mse_rg, mse_rb))
        println(io, "MSE original→HSV: ", @sprintf("%.6e (R=%.6e, G=%.6e, B=%.6e)", mse_hsv_total, mse_hr, mse_hg, mse_hb))
        println(io, "MSE RGB↔HSV:     ", @sprintf("%.6e (R=%.6e, G=%.6e, B=%.6e)", mse_between_total, mse_br, mse_bg, mse_bb))
        println(io)
        println(io, "Observation:")
        println(io, "- HSV 通道分离了亮度与色度，固定保留比例下，通常比直接在 RGB 通道分别截断频谱更少出现彩色振铃与偏色，亮度结构更稳定。")
        println(io, "- 若 MSE(original→HSV) < MSE(original→RGB)，说明 HSV 方案在色彩/亮度方面更稳健；若相反，则说明该图在 RGB 各通道频谱更集中。")
        println(io, "- 可结合 cat_diff_rgb_vs_hsv.png 的差异图观察两种重建的局部差异分布（越亮差异越大）。")
    end

    println("Saved FFT outputs to: ", save_dir)
end

function main()
    outdir = joinpath(@__DIR__, "outputs")
    ensure_dir(outdir)
    # Task 1
    run_task1_mnist_svd(outdir; ks=[10,50,100,200], use_split=:train, sample_n=2000)
    # Task 2
    run_task2_fft_cat(outdir; percent=1.0)
    println("All tasks completed. Outputs saved under: ", outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


