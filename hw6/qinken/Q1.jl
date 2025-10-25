using SparseArrays

# Core input arrays (derived from CSC format attributes)
row_indices = [3, 1, 1, 4, 5]
col_indices = [1, 2, 3, 3, 4]
values = [0.799, 0.942, 0.848, 0.164, 0.637]

# Initialize 5x5 CSC sparse matrix
sparse_mat = sparse(row_indices, col_indices, values, 5, 5)

# Define expected attributes for validation
expected_colptr = [1, 2, 3, 5, 6, 6]
expected_rowval = [3, 1, 1, 4, 5]
expected_nzval = [0.799, 0.942, 0.848, 0.164, 0.637]
expected_dims = (5, 5)

# Validate and print results with match checks
println("Sparse Matrix Validation Results")
println("--------------------------------")
println("colptr match: ", sparse_mat.colptr == expected_colptr)
println("rowval match: ", sparse_mat.rowval == expected_rowval)
println("nzval match: ", isapprox(sparse_mat.nzval, expected_nzval))  # Handle float precision
println("dimensions match: ", (sparse_mat.m, sparse_mat.n) == expected_dims)