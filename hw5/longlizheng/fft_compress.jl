using FFTW
using Images
using SparseArrays

function truncate_k(
    mat::AbstractMatrix{T},
    nx::Int,
    ny::Int
) where T
    _nx, _ny = min(size(mat, 1), nx), min(size(mat,2), ny)
    cx, cy = (size(mat, 1) + 1) ÷ 2, (size(mat, 2) + 1) ÷ 2
    lx, ly = cx - (_nx - 1) ÷ 2, cy - (_ny - 1) ÷ 2
    return mat[lx:lx + _nx - 1, ly:ly + _ny - 1]
end

function truncated_fft(
    mat::AbstractMatrix{T},
    nx::Int,
    ny::Int
) where T
    _mat = fftshift(fft(mat))
    return truncate_k(_mat, nx, ny)
end

function padding(
    mat::AbstractMatrix{T},
    Nx::Int,
    Ny::Int
) where T
    output = zeros(T, (Nx, Ny))
    nx, ny = size(mat)
    cx, cy = (Nx + 1) ÷ 2, (Ny + 1) ÷ 2
    lx, ly = cx - (nx - 1) ÷ 2, cy - (ny - 1) ÷ 2
    for i in lx:lx + nx - 1, j in ly:ly + ny - 1
        output[i, j] = mat[i - lx + 1, j - ly + 1]
    end
    return output
end

function fft_compress(
    image::AbstractMatrix{CT},
    ratio::Float64
) where CT
    nx, ny = √ratio * size(image)
    nx, ny = Int(nx), Int(ny)
    original_channels = channelview(image)
    compressed_channels = zero(original_channels)
    compressed_channels = mapslices(
        m -> truncated_fft(m, nx, ny),

    )
end