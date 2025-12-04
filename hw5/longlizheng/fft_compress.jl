using FFTW
using Images
using ColorTypes
using CairoMakie
using LinearAlgebra

function fft_compress_channel(
    channel::AbstractMatrix{T},
    percentage::Float64
) where T <: Real
    frequency_domain = fftshift(fft(channel))
    magnitudes = abs.(frequency_domain)
    num_components = size(magnitudes, 1) * size(magnitudes, 2)
    threshold_idx = max(1, num_components ÷ Int(1 / percentage))
    threshold = sort(vec(magnitudes), rev=true)[threshold_idx]
    mask = magnitudes .> threshold
    compressed_frequency = mask .* frequency_domain
    spatial_domain = ifft(ifftshift(compressed_frequency))
    reconstructed = real.(spatial_domain)
    return reconstructed
end

function fft_compress_hsv(
    hsv_image::AbstractArray{<:HSV},
    percentage::Float64
)
    channels = channelview(hsv_image)
    n_channels = size(channels, 1)
    compressed_channels = Vector{Matrix{Float32}}(undef, n_channels)
    
    for i in 1:n_channels
        channel = Float32.(channels[i, :, :])
        compressed_channel = fft_compress_channel(channel, percentage)
        # Handle HSV channels: Hue (0-360), Saturation/Value (0-1)
        if i == 1  # Hue channel
            compressed_channel = clamp.(compressed_channel, 0.0f0, 360.0f0)
        else  # Saturation and Value channels
            compressed_channel = clamp.(compressed_channel, 0.0f0, 1.0f0)
        end
        compressed_channels[i] = compressed_channel
    end
    compressed_image = colorview(HSV, 
        compressed_channels[1],  # Hue
        compressed_channels[2],  # Saturation
        compressed_channels[3]   # Value
    )
    
    return compressed_image
end

function fft_compress_rgb(
    rgb_image::AbstractArray,
    percentage::Float64
)
    channels = channelview(rgb_image)
    n_channels = size(channels, 1)
    compressed_channels = Vector{Matrix{Float32}}(undef, n_channels)
    
    for i in 1:n_channels
        channel = Float32.(channels[i, :, :])
        compressed_channel = fft_compress_channel(channel, percentage)
        # RGB channels: all clamped to 0-1
        compressed_channel = clamp.(compressed_channel, 0.0f0, 1.0f0)
        compressed_channels[i] = compressed_channel
    end
    compressed_image = colorview(eltype(rgb_image), 
        compressed_channels[1],  # Red
        compressed_channels[2],  # Green
        compressed_channels[3]   # Blue
    )
    
    return compressed_image
end

function main()
    cat_path = joinpath(@__DIR__, "..", "cat.png")
    original_image = load(cat_path)
    original_image = RGB.(original_image)
    percentage = 0.01  # Keep top 1% of components
    
    # Compress in HSV
    hsv_image = HSV.(original_image)
    compressed_hsv = fft_compress_hsv(hsv_image, percentage)
    compressed_hsv_rgb = RGB.(compressed_hsv)
    
    # Compress in RGB
    compressed_rgb = fft_compress_rgb(original_image, percentage)
    
    fig = Figure(figsize=(1200, 300))
    # Original image
    ax1 = CairoMakie.Axis(fig[1, 1], title="Original Image", aspect=AxisAspect(1))
    image!(ax1, rotr90(original_image))
    hidedecorations!(ax1)
    # HSV compressed
    ax2 = CairoMakie.Axis(fig[1, 2], title="HSV Compressed (top 1%)", aspect=AxisAspect(1))
    image!(ax2, rotr90(compressed_hsv))
    hidedecorations!(ax2)
    # RGB compressed
    ax3 = CairoMakie.Axis(fig[1, 3], title="RGB Compressed (top 1%)", aspect=AxisAspect(1))
    image!(ax3, rotr90(compressed_rgb))
    hidedecorations!(ax3)
    
    save(joinpath(@__DIR__, "fft_compression_comparison.png"), fig)
end

# Explanation:
# HSV (Hue, Saturation, Value) separates color information from brightness.
# When compressing in HSV:
#   - The Value (brightness) channel typically contains most of the energy
#   - Hue and Saturation channels may have different frequency characteristics

# RGB compresses each color channel independently:
#   - Each channel (R, G, B) is compressed separately
#   - Color information is distributed across all three channels

# The difference arises because:
#   - HSV separates luminance (V) from chrominance (H, S), which can lead
#      to different frequency domain characteristics
#   - RGB treats all channels equally, potentially preserving color relationships
#      differently than HSV
#   - The 1% threshold selection is based on magnitude, which may favor
#      different components in HSV vs RGB space