using LinearAlgebra,SparseArrays

rowindices = [3,1,1,4,5]
colindices = [1,2,3,3,4]
data = [0.799,0.942,0.848,0.164,0.637]
sp = sparse(rowindices, colindices, data, 5, 5)