using Images, FFTW

# Load image
img = load("cat.png")
h, w = size(img)
println("Image dimensions: $h x $w")

# Convert to HSV color space
hsv_img = HSV.(img)
channels = channelview(float.(hsv_img))

# Compress single channel using FFT
function compress_1percent(channel)
    # Apply FFT
    fft_data = fft(channel)
    magnitudes = abs.(fft_data)
    
    # Calculate threshold to keep top 1% of frequencies
    total = length(magnitudes)
    keep_count = max(1, Int(floor(0.01 * total)))
    threshold = sort(magnitudes[:], rev=true)[keep_count]
    
    # Create mask and apply compression
    mask = magnitudes .> threshold
    compressed = real.(ifft(fft_data .* mask))
    
    # Ensure values are in valid range
    return clamp.(compressed, 0.0, 1.0)
end

# Compress each HSV channel
H_compressed = compress_1percent(channels[1,:,:])
S_compressed = compress_1percent(channels[2,:,:]) 
V_compressed = compress_1percent(channels[3,:,:])

# Reconstruct compressed image using array approach
compressed_array = zeros(Float32, 3, h, w)
compressed_array[1, :, :] = H_compressed
compressed_array[2, :, :] = S_compressed
compressed_array[3, :, :] = V_compressed

# Create and save compressed image
compressed_hsv_img = colorview(HSV, compressed_array)
save("compressed_result.png", RGB.(compressed_hsv_img))