using LinearAlgebra
using Images
using FFTW
using Plots
using Statistics

imag_name = "cat.png"
println("Loading image: ", imag_name)

original_img = Images.load(imag_name)
println("Loaded image with size: ", size(original_img))

println("\nProcessing in HSV channel...")

imag_hsv = HSV.(original_img)
original_size_hsv = size(imag_hsv)
channels_hsv = channelview(imag_hsv)
n_channels_hsv = size(channels_hsv, 1)

compressed_channels_hsv = Vector{Matrix{Float64}}(undef, n_channels_hsv)

for i in 1:n_channels_hsv
    Frequency_domain_hsv = fftshift(fft(channels_hsv[i, :, :]))
    magnitudes = abs.(Frequency_domain_hsv)
    threshold_idx = size(magnitudes)[1] * size(magnitudes)[2] ÷ 100
    threshold = sort(vec(magnitudes), rev=true)[threshold_idx]
    mask = magnitudes .> threshold
    compressed_frequency = mask .* Frequency_domain_hsv
    spatial_domain = ifft(ifftshift(compressed_frequency))
    real_spatial = real.(spatial_domain)
    
    if i == 1
        compressed_channels_hsv[i] = clamp.(real_spatial, 0.0, 1.0)
    else
        compressed_channels_hsv[i] = clamp.(real_spatial, 0.0, 1.0)
    end
end

compressed_hsv = colorview(HSV, 
    compressed_channels_hsv[1],
    compressed_channels_hsv[2], 
    compressed_channels_hsv[3]
)

Images.save("compressed_cat_hsv.png", compressed_hsv)
println("HSV compressed image saved as compressed_cat_hsv.png")

println("Processing in RGB channel...")

imag_rgb = original_img
channels_rgb = channelview(imag_rgb)
n_channels_rgb = size(channels_rgb, 1)

if n_channels_rgb == 4
    compressed_channels_rgb = Vector{Matrix{Float64}}(undef, n_channels_rgb)
    for i in 1:n_channels_rgb
        Frequency_domain_rgb = fftshift(fft(channels_rgb[i, :, :]))
        magnitudes = abs.(Frequency_domain_rgb)
        threshold_idx = size(magnitudes)[1] * size(magnitudes)[2] ÷ 100
        threshold = sort(vec(magnitudes), rev=true)[threshold_idx]
        mask = magnitudes .> threshold
        compressed_frequency = mask .* Frequency_domain_rgb
        spatial_domain = ifft(ifftshift(compressed_frequency))
        real_spatial = real.(spatial_domain)
        compressed_channels_rgb[i] = clamp.(real_spatial, 0.0, 1.0)
    end
    
    compressed_rgb = colorview(RGBA, 
        compressed_channels_rgb[1],
        compressed_channels_rgb[2], 
        compressed_channels_rgb[3],
        compressed_channels_rgb[4]
    )
else
    compressed_channels_rgb = Vector{Matrix{Float64}}(undef, n_channels_rgb)
    for i in 1:n_channels_rgb
        Frequency_domain_rgb = fftshift(fft(channels_rgb[i, :, :]))
        magnitudes = abs.(Frequency_domain_rgb)
        threshold_idx = size(magnitudes)[1] * size(magnitudes)[2] ÷ 100
        threshold = sort(vec(magnitudes), rev=true)[threshold_idx]
        mask = magnitudes .> threshold
        compressed_frequency = mask .* Frequency_domain_rgb
        spatial_domain = ifft(ifftshift(compressed_frequency))
        real_spatial = real.(spatial_domain)
        compressed_channels_rgb[i] = clamp.(real_spatial, 0.0, 1.0)
    end
    
    compressed_rgb = colorview(RGB, 
        compressed_channels_rgb[1],
        compressed_channels_rgb[2],
        compressed_channels_rgb[3]
    )
end

Images.save("compressed_cat_rgb.png", compressed_rgb)
println("RGB compressed image saved as compressed_cat_rgb.png")

function calculate_image_difference(img1, img2)
    arr1 = Float64.(channelview(img1))
    arr2 = Float64.(channelview(img2))
    mse = Statistics.mean((arr1 .- arr2).^2)
    max_val = 1.0
    psnr = 10 * log10(max_val^2 / mse)
    return mse, psnr
end

hsv_mse, hsv_psnr = calculate_image_difference(original_img, compressed_hsv)
rgb_mse, rgb_psnr = calculate_image_difference(original_img, compressed_rgb)

comparison_mse, comparison_psnr = calculate_image_difference(compressed_hsv, compressed_rgb)

println("\nCompression Metrics:")
println("HSV Compression - MSE: $(round(hsv_mse, digits=6)), PSNR: $(round(hsv_psnr, digits=2)) dB")
println("RGB Compression - MSE: $(round(rgb_mse, digits=6)), PSNR: $(round(rgb_psnr, digits=2)) dB")
println("HSV vs RGB Difference - MSE: $(round(comparison_mse, digits=6)), PSNR: $(round(comparison_psnr, digits=2)) dB")

println("\nCreating visualization...")
p_original = plot(original_img, title="Original", aspect_ratio=:equal)
p_hsv = plot(compressed_hsv, title="HSV Compressed (1%)", aspect_ratio=:equal)
p_rgb = plot(compressed_rgb, title="RGB Compressed (1%)", aspect_ratio=:equal)

comparison_plot = plot(p_original, p_hsv, p_rgb, layout=(1,3), size=(1200, 400))
println("FFT compression comparison:")
display(comparison_plot)

println("\nAnalysis:")
println("HSV vs RGB compression differences:")
println("- HSV compression preserves color relationships better because it separates intensity (V) from color (H,S)")
println("- RGB compression treats each color channel independently, which may lead to different visual artifacts")
println("- The difference in results depends on the image content and how color information is distributed")
println("- HSV often provides better perceptual quality for the same compression ratio")
println("- In HSV, only the most significant 1% of frequency components in each channel are preserved")
println("- In RGB, only the most significant 1% of frequency components in each channel are preserved")

println("\nTask 2 completed successfully!")