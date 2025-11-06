using Images, FFTW

# Load image
img = load("cat.png")
println("Image dimensions: ", size(img))

# Direct RGB channel processing
function compress_rgb_image(image_path, output_path)
    img = load(image_path)
    h, w = size(img)
    
    # Extract RGB channels
    channels = channelview(float.(img))
    
    # Compression function
    function compress_channel(channel)
        fft_data = fft(channel)
        mags = abs.(fft_data)
        total = length(mags)
        keep_num = max(1, Int(floor(0.01 * total)))
        threshold = sort(mags[:], rev=true)[keep_num]
        mask = mags .> threshold
        return clamp.(real.(ifft(fft_data .* mask)), 0.0, 1.0)
    end
    
    # Compress RGB channels
    R_comp = compress_channel(channels[1,:,:])
    G_comp = compress_channel(channels[2,:,:])
    B_comp = compress_channel(channels[3,:,:])
    
    # Reconstruct image
    compressed_img = colorview(RGB, R_comp, G_comp, B_comp)
    save(output_path, compressed_img)
    
    return compressed_img
end

# Usage
compressed_img = compress_rgb_image("cat.png", "compressed_rgb.png")