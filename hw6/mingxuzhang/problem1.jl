# Problem 1: Sparse Matrix Construction
# Understanding CSC format:
# - colptr[j] to colptr[j+1]-1 gives indices in rowval/nzval for column j
# - rowval contains row indices
# - nzval contains the corresponding values

# Given CSC structure:
# colptr = [1, 2, 3, 5, 6, 6]
# rowval = [3, 1, 1, 4, 5]
# nzval = [0.799, 0.942, 0.848, 0.164, 0.637]

# Decoding the structure:
# Column 1: indices 1:1 -> row 3, value 0.799
# Column 2: indices 2:2 -> row 1, value 0.942
# Column 3: indices 3:4 -> rows 1, 4, values 0.848, 0.164
# Column 4: indices 5:5 -> row 5, value 0.637
# Column 5: indices 6:5 (empty, no elements)

# This gives us the following non-zero entries:
# (row, col) = value
# (3, 1) = 0.799
# (1, 2) = 0.942
# (1, 3) = 0.848
# (4, 3) = 0.164
# (5, 4) = 0.637

using SparseArrays

rowindices = [3, 1, 1, 4, 5]
colindices = [1, 2, 3, 3, 4]
data = [0.799, 0.942, 0.848, 0.164, 0.637]

# Verify the solution
sp = sparse(rowindices, colindices, data, 5, 5)

println("Verification:")
println("colptr: ", sp.colptr)
println("rowval: ", sp.rowval)
println("nzval: ", sp.nzval)
println("m: ", sp.m)
println("n: ", sp.n)

