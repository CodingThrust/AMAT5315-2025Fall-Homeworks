# Task 1: Sparse Matrix Construction
using SparseArrays

println("=== Task 1: Sparse Matrix Construction ===")

# Define input arrays derived from CSC structure
rowindices = [3, 1, 1, 4, 5]  # Row indices of non-zeros (matches rowval)
colindices = [1, 2, 3, 3, 4]  # Column indices of non-zeros
data = [0.799, 0.942, 0.848, 0.164, 0.637]  # Values of non-zeros (matches nzval)

# Construct 5x5 sparse matrix in CSC format
sp = sparse(rowindices, colindices, data, 5, 5)

# Verify results
println("=== Verification ===")
println("colptr: ", sp.colptr, " (expected: [1,2,3,5,6,6])")
println("rowval: ", sp.rowval, " (expected: [3,1,1,4,5])")
println("nzval: ", sp.nzval, " (expected: [0.799,0.942,0.848,0.164,0.637])")
println("Rows (m): ", sp.m, " (expected: 5)")
println("Cols (n): ", sp.n, " (expected: 5)")

println("\nMatrix reconstruction explanation:")
println("The matrix A is 5×5 with non-zero entries:")
println("A[3,1] = 0.799 (from rowval[1]=3, nzval[1]=0.799, colptr[1]=1 to colptr[2]-1=1)")
println("A[1,2] = 0.942 (from rowval[2]=1, nzval[2]=0.942, colptr[2]=2 to colptr[3]-1=2)")
println("A[1,3] = 0.848 (from rowval[3]=1, nzval[3]=0.848, colptr[3]=3 to colptr[4]-1=4)")
println("A[4,3] = 0.164 (from rowval[4]=4, nzval[4]=0.164, colptr[3]=3 to colptr[4]-1=4)")
println("A[5,4] = 0.637 (from rowval[5]=5, nzval[5]=0.637, colptr[4]=5 to colptr[5]-1=5)")

# Display the full matrix for visualization
println("\nFull matrix representation:")
display(Matrix(sp))
