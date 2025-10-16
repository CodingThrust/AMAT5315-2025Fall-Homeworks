using LinearAlgebra
using CairoMakie

include("../download_mnist.jl")

build_data_mat(data::AbstractArray{T, 3}) where T = dropdims(mapslices(fig -> reshape(fig, :), data; dims=(1, 2)); dims=2)

function compress(mat::AbstractMatrix{T}, k::Int) where T
    u, s, v = svd(mat)
    _k = min(k, length(s))
    return u[:, 1:_k] * Diagonal(s[1:_k]) * v'[1:_k, :]
end

function plot_images(
    original_mat::AbstractMatrix{T},
    compressed_mat::AbstractMatrix{T},
    indices::AbstractVector{Int}
) where T
    fig = Figure(figsize=(300, 150 * length(indices)))
    for (i, idx) in enumerate(indices)
        ax_orig = Axis(fig[i, 1], aspect=AxisAspect(1), title="original")
        heatmap!(
            ax_orig,
            1:28, 28:-1:1,
            reshape(view(original_mat, :, idx), (28, 28)),
            colormap=:grays
        )

        ax_comp = Axis(fig[i, 2], aspect=AxisAspect(1), title="compressed")
        heatmap!(
            ax_comp,
            1:28, 28:-1:1,
            reshape(view(compressed_mat, :, idx), (28, 28)),
            colormap=:grays
        )
    end
    return fig
end

function main()
    train_images, train_labels = download_mnist(:train)
    test_images, test_labels = download_mnist(:test)

    data_mat = build_data_mat(train_images)

    _, s, _ = svd(data_mat)
    sfig = Figure()
    sax = Axis(sfig[1, 1], title="Singular Values", xlabel="index", ylabel="s", yscale=log10)
    lines!(sax, 1:length(s), s)
    save(joinpath(@__DIR__, "s.png"), sfig)

    ks = [10, 50, 100, 200]
    errors = Float32[]

    for k in ks
        compressed_data_mat = compress(data_mat, k)
        fig = plot_images(data_mat, compressed_data_mat, [64, 128, 256])
        save(joinpath(@__DIR__, "top_$(k)_compressed.png"), fig)
        err = sum(map(norm, eachcol(compressed_data_mat - data_mat))) / size(data_mat, 2)
        push!(errors, err)
    end

    efig = Figure()
    eax = Axis(efig[1, 1], title="Errors", xlabel="k", ylabel="error", yscale=log10)
    lines!(eax, ks, errors)
    save(joinpath(@__DIR__, "errors.png"), efig)
end