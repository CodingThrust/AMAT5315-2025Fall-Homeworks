using Einsum, LinearAlgebra

# Example matrices for demonstration
println("=== Matrix Operations using Einsum.jl ===\n")

# 1. Matrix Multiplication with Transpose: C = A * B'
println("1. Matrix Multiplication with Transpose: C = A * B'")
A = [1 2 3; 4 5 6]
B = [7 8 9; 10 11 12]

@einsum C[i,j] := A[i,k] * B[j,k]  # Note: B[j,k] instead of B[k,j]

println("Matrix A:")
println(A)
println("\nMatrix B:")
println(B)
println("\nEinsum result C = A * B':")
println(C)
println("Verification (A * B'):")
println(A * B')
println()

# 2. Summing over all elements in a matrix
println("2. Summing over all matrix elements")
A = [1 2 3; 4 5 6]

@einsum total := A[i,j]

println("Matrix A:")
println(A)
println("Einsum sum result: ", total)
println("Verification (sum(A)): ", sum(A))
println()

# 3. Element-wise multiplication of three matrices: D = A ⊙ B ⊙ C
println("3. Element-wise multiplication of three matrices")
A = [1 2; 3 4]
B = [5 6; 7 8] 
C = [9 10; 11 12]

@einsum D[i,j] := A[i,j] * B[i,j] * C[i,j]

println("Matrix A:"); println(A)
println("Matrix B:"); println(B)
println("Matrix C:"); println(C)
println("Einsum element-wise product:")
println(D)
println("Verification (A .* B .* C):")
println(A .* B .* C)
println()

# 4. Kronecker Product: D = A ⊗ B ⊗ C
println("4. Kronecker Product of three matrices")
A = [1 2; 3 4]
B = [5 6; 7 8]
C = [9 10; 11 12]

# Calculate block structure using Einsum
@einsum D_blocks[i,k,m,j,l,n] := A[i,j] * B[k,l] * C[m,n]

# Reshape to final matrix
D = reshape(D_blocks, 
            size(A,1)*size(B,1)*size(C,1), 
            size(A,2)*size(B,2)*size(C,2))

println("Matrix A:"); println(A)
println("Matrix B:"); println(B)  
println("Matrix C:"); println(C)
println("Einsum Kronecker product result:")
println(D)
println("Verification (kron(kron(A, B), C)):")
println(kron(kron(A, B), C))
