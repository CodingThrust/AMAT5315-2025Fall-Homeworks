using Images, FFTW

# Load image
imag = load("D:/juliahw/hw5/cat.png")

# HSV channel compression
imag_hsv = HSV.(imag)
channels_hsv = channelview(imag_hsv)
compressed_channels_hsv = similar(channels_hsv)

for i in 1:size(channels_hsv, 1)
    freq_domain = fft(channels_hsv[i, :, :])
    magnitudes = abs.(freq_domain)
    threshold = sort(vec(magnitudes), rev=true)[length(magnitudes) ÷ 100]
    compressed_freq = freq_domain .* (magnitudes .> threshold)
    spatial_domain = real.(ifft(compressed_freq))
    
    # Clamp based on channel type
    if i == 1  # Hue: 0-360
        compressed_channels_hsv[i, :, :] = clamp.(spatial_domain, 0.0f0, 360.0f0)
    else  # Saturation/Value: 0-1
        compressed_channels_hsv[i, :, :] = clamp.(spatial_domain, 0.0f0, 1.0f0)
    end
end

compressed_hsv = colorview(HSV, compressed_channels_hsv)
save("D:/juliahw/hw5/hsv.png", compressed_hsv)

# RGB channel compression  
channels_rgb = channelview(imag)
compressed_channels_rgb = similar(channels_rgb)

for i in 1:size(channels_rgb, 1)
    freq_domain = fft(channels_rgb[i, :, :])
    magnitudes = abs.(freq_domain)
    threshold = sort(vec(magnitudes), rev=true)[length(magnitudes) ÷ 100]
    compressed_freq = freq_domain .* (magnitudes .> threshold)
    compressed_channels_rgb[i, :, :] = clamp.(real.(ifft(compressed_freq)), 0.0f0, 1.0f0)
end

compressed_rgb = colorview(RGB, compressed_channels_rgb[1:3, :, :])  # Use only RGB channels
save("D:/juliahw/hw5/rgb.png", compressed_rgb)