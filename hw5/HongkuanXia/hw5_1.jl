include("../download_mnist.jl")
using LinearAlgebra
using Images, ImageView
using Plots
test_images, test_labels = download_mnist(:test)
println("Test set: ", size(test_images))  
flattened_image = reshape(train_images, 28*28, :)
U,D,V = svd(flattened_image)
maxDim = 50
compressed_D = D[1:maxDim]
compressed_image = reshape(U[:,1:maxDim]*diagm(compressed_D)*V[:,1:maxDim]',28,28,10000)
gray_img_ini = Gray.(test_images[:,:,1])
gray_img_compressed = Gray.(compressed_image[:,:,1])
save("original_image.png",gray_img_ini)
save("compressed_image.png",gray_img_compressed)

maxDim = [10,50,100,200]
err = [0.0,0.0,0.0,0.0]
compressed_ratio = [0.0,0.0,0.0,0.0]
for i in eachindex(maxDim)
    dim = maxDim[i]
    compressed_D = D[1:dim]
    m, n = size(U, 1), size(V, 2)
    rank = length(compressed_D)
    compressed_ratio[i] = (rank*(1+m+n))/(m*n) 
    compressed_image = reshape(U[:,1:dim]*diagm(compressed_D)*V[:,1:dim]',28,28,10000)
    for j in eachindex(test_images[1,1,:])
        err[i] += norm(test_images[:,:,j]-compressed_image[:,:,j],2)
    end
end
err=err./length(test_images[1,1,:])
plot(maxDim,compressed_ratio,xlabel = "k",ylabel = "ratio",label=false,title="compressed ratio")
savefig("compressed_ratio.png")
plot(maxDim,err,xlabel="k",ylabel = "error",label=false,title="reconstructed error")
savefig("reconstructed_error.png")