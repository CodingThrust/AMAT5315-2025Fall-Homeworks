# Homework 6
1. Using the following julia code:
```julia
using LinearAlgebra,SparseArrays

rowindices = [3,1,1,4,5]
colindices = [1,2,3,3,4]
data = [0.799,0.942,0.848,0.164,0.637]
sp = sparse(rowindices, colindices, data, 5, 5)
```
this code reproduce the correct sparse matrix sp:
```julia-repl
julia> sp.colptr
6-element Vector{Int64}:
 1
 2
 3
 5
 6
 6
 julia> sp.rowval
5-element Vector{Int64}:
 3
 1
 1
 4
 5
 sp.nzval
5-element Vector{Float64}:
 0.799
 0.942
 0.848
 0.164
 0.637
 julia> sp.m
5

julia> sp.n
5
```
sp.rowval is the row indices of the sparse matrix, and sp.colptr[j]-sp.colptr[j-1] is the number of the nonzero elements in j-1th column, sp.nzval is the nonzero values of the sparse matrix.

2. I complete the code as: 
```julia
using Graphs, Random, KrylovKit
Random.seed!(42)
g = random_regular_graph(100000, 3)

A = laplacian_matrix(g)

q1 = randn(100000)

vals, vecs, info = eigsolve(A, q1, 2, :SR)
```
and the values of vals are:
```julia-repl
vals
2-element Vector{Float64}:
 -4.6226092074930866e-15
  0.17173162534019853
```
Since the degenracy of the zero eigenvalue is the number of the connected component, so the number of connected component of the random 3-regular graph is 1.

3. I write the following code for implementing the restarting lanzcos algorithm, the original code and test code are in "HongkuanXia/hw6_3.jl", and I get the test passed:
```julia-repl
Test Summary:      | Pass  Total  Time
restarting_lanczos |    1      1  0.8s
Test.DefaultTestSet("restarting_lanczos", Any[], 1, false, false, true, 1.76269782892631e9, 1.762697829736629e9, false, "/Users/hongkuanhisa/amat5135/AMAT5315-2025Fall-Homeworks/hw6/HongkuanXia/hw6_3.jl")
```