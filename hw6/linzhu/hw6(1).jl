rowindices = [3, 1, 1, 4, 5]
colindices = [1, 2, 3, 4, 5]
data       = [0.799, 0.942, 0.848, 0.164, 0.637]

# colptr = [1,2,3,4,5,6] → each column contains exactly one non-zero.

# The i-th entry in rowval and nzval (i=1..5) is the nonzero for column i.

# Therefore, COO format is obtained directly:

# rowindices = rowval

# colindices = column numbers 1..5

# data = nzval