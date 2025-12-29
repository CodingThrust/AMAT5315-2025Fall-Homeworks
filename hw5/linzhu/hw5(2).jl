using Images
using FFTW
using LinearAlgebra
using Plots

# =============================================================
# Helper Function: FFT Compression (keep top p fraction)
# =============================================================
"""
    compress_fft_channel(channel, p)

Apply FFT → keep top p fraction of frequencies (by magnitude) → IFFT.
Returns the reconstructed real-valued channel.
"""
function compress_fft_channel(channel::AbstractMatrix, p::Float64)
    # FFT and shift
    F = fftshift(fft(channel))

    # Magnitude for threshold selection
    mags = abs.(F)
    total = length(mags)
    keep = max(round(Int, total * p), 1)

    # threshold for top-p fraction
    sorted = sort(vec(mags), rev = true)
    threshold = sorted[keep]

    # keep only large coefficients
    mask = mags .>= threshold
    F_compressed = mask .* F

    # IFFT → real image
    spatial = real.(ifft(ifftshift(F_compressed)))
    return spatial
end

# =============================================================
# Load cat image
# =============================================================
img_path = "hw5/cat.png"
img = load(img_path)

h, w = size(img)[1:2]

# =============================================================
# 1. Fourier Compression in HSV Space
# =============================================================
img_hsv = HSV.(img)
hsv_channels = channelview(img_hsv)   # 3×H×W
compressed_hsv_channels = similar(hsv_channels)

p = 0.01   # keep top 1%

for i in 1:3
    ch = Float64.(hsv_channels[i, :, :])

    # compress
    comp = compress_fft_channel(ch, p)

    # clamp per HSV rules:
    if i == 1        # Hue: [0, 360]
        compressed_hsv_channels[i, :, :] = clamp.(comp, 0, 360)
    else             # S,V: [0,1]
        compressed_hsv_channels[i, :, :] = clamp.(comp, 0, 1)
    end
end

compressed_hsv_img = colorview(HSV,
    compressed_hsv_channels[1, :, :],
    compressed_hsv_channels[2, :, :],
    compressed_hsv_channels[3, :, :]
)

save("hw5/linzhu/compressed_cat_hsv.png", compressed_hsv_img)


# =============================================================
# 2. Fourier Compression in RGB Space
# =============================================================
img_rgb = img
rgb_channels = channelview(img_rgb)    # 3×H×W
compressed_rgb_channels = similar(rgb_channels)

for i in 1:3
    ch = Float64.(rgb_channels[i, :, :])

    comp = compress_fft_channel(ch, p)

    # clamp RGB to [0,1]
    compressed_rgb_channels[i, :, :] = clamp.(comp, 0, 1)
end

compressed_rgb_img = colorview(RGB,
    compressed_rgb_channels[1, :, :],
    compressed_rgb_channels[2, :, :],
    compressed_rgb_channels[3, :, :]
)

save("hw5/linzhu/compressed_cat_rgb.png", compressed_rgb_img)


# =============================================================
# 3. Optional: Compute difference image
# =============================================================
diff_img = abs.(float.(compressed_hsv_img) .- float.(compressed_rgb_img))
save("hw5/linzhu/difference_hsv_vs_rgb.png", diff_img)

println("Saved: compressed_cat_hsv.png, compressed_cat_rgb.png, difference_hsv_vs_rgb.png")
