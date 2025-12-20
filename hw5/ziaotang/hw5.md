## Problem 1

```julia

using LinearAlgebra
using Images
using Plots
include("download_mnist.jl")
# Download MNIST dataset
train_images, train_labels = download_mnist(:train)
test_images, test_labels = download_mnist(:test)

println("Train set: ", size(train_images))
println("Test set: ", size(test_images))

# Flatten images and perform SVD
flattened_images = reshape(train_images, 28 * 28, :)
U, D, V = svd(flattened_images)

# Test different compression dimensions
k = [10, 50, 100, 200]
err = zeros(4)
compressed_ratio = zeros(4)

for i in eachindex(k)
    dim = k[i]
    
    # Calculate compression ratio
    m, n = size(flattened_images)
    compressed_ratio[i] = (dim * (1 + m + n)) / (m * n)
    
    # Compress and reconstruct
    compressed_D = D[1:dim]
    compressed_images = U[:,1:dim] * Diagonal(compressed_D) * V[:,1:dim]'
    reconstructed_images = reshape(compressed_images, 28, 28, :)
    
    # Calculate reconstruction error on test set
    for j in 1:size(test_images, 3)
        test_vec = reshape(test_images[:,:,j], :)
        compressed_test = U[:,1:dim] * (U[:,1:dim]' * test_vec)
        err[i] += norm(test_vec - compressed_test, 2)
    end
    err[i] /= size(test_images, 3)
    
    
        example_idx = 20
        original_img = Gray.(test_images[:,:,example_idx])
        compressed_vec = U[:,1:dim] * (U[:,1:dim]' * reshape(test_images[:,:,example_idx], :))
        compressed_img = Gray.(clamp.(reshape(compressed_vec, 28, 28), 0.0, 1.0))
        
        save("D://juliahw//hw5//original.png", original_img)
        save("D://juliahw//hw5//compressed_k$(dim).png", compressed_img)
    
end

# Plot and save results
p1 = plot(k, compressed_ratio, xlabel="k", ylabel="Compression Ratio", 
          title="Compression Ratio vs k", marker=:circle, linewidth=2, label=false)
savefig("D://juliahw//hw5//compressed_ratio.png")

p2 = plot(k, err, xlabel="k", ylabel="Reconstruction Error", 
          title="Reconstruction Error vs k", marker=:circle, linewidth=2, label=false)
savefig("D://juliahw//hw5//reconstructed_error.png")

# Print results
println("\nResults:")
for i in eachindex(k)
    println("k=$(k[i]): Ratio=$(round(compressed_ratio[i], digits=4)), Error=$(round(err[i], digits=4))")
end


```



Plots:

![reconstructed_error](D:\juliahw\hw5\reconstructed_error.png)

![compressed_ratio](D:\juliahw\hw5\compressed_ratio.png)

Comparsion:

original:

![original](D:\juliahw\hw5\original.png)

$k=10$ :



![compressed_k10](D:\juliahw\hw5\compressed_k10.png)

$k=50$:

![compressed_k50](D:\juliahw\hw5\compressed_k50.png)

$k=100$:

![compressed_k100](D:\juliahw\hw5\compressed_k100.png)

$k=200$:

![compressed_k200](D:\juliahw\hw5\compressed_k200.png)





## Problem 2

```julia
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
```



original:



![cat](D:\juliahw\hw5\cat.png)

hsv:

![hsv](D:\juliahw\hw5\hsv.png)

rgb:

![rgb](D:\juliahw\hw5\rgb.png)



Color performance is better for RGB channel. This is because RGB mimics how digital sensors and displays actually works, where each pixel is a combination of red, green, and blue components. Keeping the strongest frequency components in RGB naturally preserves color relationships.
