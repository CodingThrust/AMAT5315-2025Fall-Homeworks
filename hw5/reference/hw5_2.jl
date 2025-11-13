using LinearAlgebra
using Images, ImageView
using Plots
using FFTW

# ---------------------------------------------------------------------- #

# code for hsv channel

imag_name = "hw5/cat.png"

imag_hsv = HSV.(Images.load(imag_name))
original_size_hsv = size(imag_hsv)
channels_hsv = channelview(imag_hsv)
n_channels_hsv = size(channels_hsv, 1)# three channels of HSV: Hue, Saturation and value

compressed_channels = Array{Float32}(undef,n_channels_hsv, original_size_hsv...)

for i in 1:n_channels_hsv
    Frequency_domian_hsv = fftshift(fft(channels_hsv[i,:,:]))# using fft to transofrom into frequency domian and shifts zero frequency to center
    magnitudes = abs.(Frequency_domian_hsv)
    thresold = sort(vec(magnitudes),rev=true)[size(magnitudes)[1]*size(magnitudes)[2]÷100] # compute the thresold of the frequency magnitude, top 1%
    mask = magnitudes.>thresold # high pass filter
    compressed_frequency = mask.*Frequency_domian_hsv # apply the filter
    spatial_domain = ifft(ifftshift(compressed_frequency)) # using inverse fourier transform to get the real domain information
    real_spatial = real.(spatial_domain)
    # Clamp values to valid range for HSV (H: 0-360, S: 0-1, V: 0-1)
    if i == 1  # Hue channel: 0-360
        compressed_channels[i,:,:] = clamp.(real_spatial, 0.0f0, 360.0f0)
    else  # Saturation and Value channels: 0-1
        compressed_channels[i,:,:] = clamp.(real_spatial, 0.0f0, 1.0f0)
    end
end
compressed_hsv = colorview(HSV, 
        compressed_channels
)

Images.save("hw5/reference/compressed_cat_hsv.png",compressed_hsv)

# ---------------------------------------------------------------------- #

# code for rgba channel

imag_rgba = Images.load(imag_name)
original_size_rgba = size(imag_rgba)
channels_rgba = channelview(imag_rgba)
n_channels_rgba = size(channels_rgba, 1)# four channels of RGBA

compressed_channels = Array{Float32}(undef,n_channels_rgba, original_size_rgba...)

for i in 1:n_channels_rgb
    Frequency_domian_rgb = fftshift(fft(channels_rgb[i,:,:]))# using fft to transofrom into frequency domian and shifts zero frequency to center
    magnitudes = abs.(Frequency_domian_rgb)
    thresold = sort(vec(magnitudes),rev=true)[size(magnitudes)[1]*size(magnitudes)[2]÷100] # compute the thresold of the frequency magnitude, top 1%
    mask = magnitudes.>thresold # high pass filter
    compressed_frequency = mask.*Frequency_domian_rgb # apply the filter
    spatial_domain = ifft(ifftshift(compressed_frequency)) # using inverse fourier transform to get the real domain information
    real_spatial = real.(spatial_domain)
    # Clamp values to valid range [0, 1] for RGB channels
    compressed_channels[i,:,:] = clamp.(real_spatial, 0.0f0, 1.0f0)
end
compressed_rgba = colorview(RGBA, 
        compressed_channels
    )

Images.save("hw5/reference/compressed_cat_rgba.png",compressed_rgba)
