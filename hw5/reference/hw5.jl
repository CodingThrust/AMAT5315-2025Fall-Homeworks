include("../download_mnist.jl")

train_images, train_labels = download_mnist(:train)
test_images, test_labels = download_mnist(:test)