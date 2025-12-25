#### output_task1 ####
=== MNIST SVD Compression Analysis ===
Downloading MNIST training data...
Loading MNIST train data from cache: mnist_data/mnist_train.jld2
Vectorizing images...
Data matrix dimensions: (784, 1000)
Computing SVD...
Singular values computed. Total: 784
Testing different compression levels...
k = 5: Error = 1.2551, Ratio = 0.01
k = 15: Error = 1.3117, Ratio = 0.03
k = 30: Error = 1.3415, Ratio = 0.06
k = 50: Error = 1.3583, Ratio = 0.1001
k = 100: Error = 1.3739, Ratio = 0.2001
k = 150: Error = 1.3795, Ratio = 0.3002
k = 200: Error = 1.3822, Ratio = 0.4003
Creating visualization of compression quality...
Sample image 1 reconstruction comparison:
Sample image 100 reconstruction comparison:
Sample image 500 reconstruction comparison:
Compression metrics visualization:
Cumulative energy preservation:
Results saved to mnist_svd_analysis.jld2

=== SVD Compression Summary ===
Original data size: 784 × 1000
Total singular values: 784
Energy preserved by top 50 components: 90.57%
Energy preserved by top 100 components: 95.74%

SVD compression analysis completed successfully!

