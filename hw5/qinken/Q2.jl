using Images, FFTW, LinearAlgebra, Statistics
using FileIO, Plots

function load_and_prepare_image(image_path)
    img = load(image_path)
    h, w = size(img)
    if h % 2 != 0 || w % 2 != 0
        img = img[1:2*(h÷2), 1:2*(w÷2)]
    end
    return img
end

function compress_with_fft(channel, keep_ratio=0.01)
    fft_channel = fftshift(fft(channel))
    magnitudes = abs.(fft_channel)
    sorted_mags = sort(magnitudes[:], rev=true)
    threshold_idx = Int(ceil(keep_ratio * length(sorted_mags)))
    threshold = sorted_mags[threshold_idx]
    mask = magnitudes .>= threshold
    compressed_fft = fft_channel .* mask
    compressed_channel = real.(ifft(ifftshift(compressed_fft)))
    return compressed_channel
end

function compress_rgb(image, keep_ratio=0.01)
    rgb_channels = channelview(image)
    compressed_channels = similar(rgb_channels)
    for i in 1:size(rgb_channels, 1)
        compressed_channels[i, :, :] = compress_with_fft(rgb_channels[i, :, :], keep_ratio)
    end
    return colorview(RGB, compressed_channels)
end

function compress_hsv(image, keep_ratio=0.01)
    hsv_img = HSV.(image)
    H_channel = [c.h for c in hsv_img]
    S_channel = [c.s for c in hsv_img]  
    V_channel = [c.v for c in hsv_img]
    
    H_compressed = compress_with_fft(H_channel, keep_ratio)
    S_compressed = compress_with_fft(S_channel, keep_ratio)
    V_compressed = compress_with_fft(V_channel, keep_ratio)
    
    hsv_compressed = [HSV(H_compressed[i], S_compressed[i], V_compressed[i]) 
                      for i in eachindex(H_compressed)]
    
    return RGB.(hsv_compressed)
end

function calculate_differences(img1, img2)
    mse = mean((img1 .- img2).^2)
    psnr = 10 * log10(1 / mse)
    return mse, psnr
end

function main()
    # Load image
    image_path = "cat.png"
    original_img = load_and_prepare_image(image_path)
    
    # Compress in RGB space
    rgb_compressed = compress_rgb(original_img, 0.01)
    
    # Compress in HSV space
    hsv_compressed = compress_hsv(original_img, 0.01)
    
    # Calculate differences
    rgb_mse, rgb_psnr = calculate_differences(float.(original_img), float.(rgb_compressed))
    hsv_mse, hsv_psnr = calculate_differences(float.(original_img), float.(hsv_compressed))
    rgb_hsv_mse, rgb_hsv_psnr = calculate_differences(float.(rgb_compressed), float.(hsv_compressed))
    
    # Display results
    p1 = plot(original_img, title="Original", aspect_ratio=:equal)
    p2 = plot(rgb_compressed, title="RGB Compressed", aspect_ratio=:equal)
    p3 = plot(hsv_compressed, title="HSV Compressed", aspect_ratio=:equal)
    diff_rgb_hsv = abs.(float.(rgb_compressed) - float.(hsv_compressed))
    p4 = heatmap(diff_rgb_hsv, title="RGB vs HSV Difference", aspect_ratio=:equal, color=:hot)
    
    results_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
    display(results_plot)
    
    # Print metrics
    println("RGB Compression - MSE: ", round(rgb_mse, digits=6), ", PSNR: ", round(rgb_psnr, digits=2), " dB")
    println("HSV Compression - MSE: ", round(hsv_mse, digits=6), ", PSNR: ", round(hsv_psnr, digits=2), " dB")
    println("RGB vs HSV Difference - MSE: ", round(rgb_hsv_mse, digits=6), ", PSNR: ", round(rgb_hsv_psnr, digits=2), " dB")
    
    return Dict(
        "original" => original_img,
        "rgb_compressed" => rgb_compressed, 
        "hsv_compressed" => hsv_compressed,
        "rgb_mse" => rgb_mse,
        "hsv_mse" => hsv_mse,
        "rgb_psnr" => rgb_psnr,
        "hsv_psnr" => hsv_psnr
    )
end

# Execute
results = main()

# HSV compression works better:

# Brightness priority - HSV separates color and brightness. Our eyes are more sensitive to brightness, and Fourier transform keeps the important brightness info better.

# Color separation - RGB mixes all color information together, so compression causes more color distortion. HSV handles colors separately, keeping them more accurate.

# Eye-friendly - Human vision cares more about brightness changes than color changes, and HSV matches how we actually see.