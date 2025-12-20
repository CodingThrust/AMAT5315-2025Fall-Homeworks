# Homework 7

1. (Ground state energy) What is the ground state energy of the following anti-ferromagnetic Ising model on the Fullerene graph?
```math
H = \sum_{ij \in E} \sigma_i \sigma_j
```
   where $\sigma_i = \pm 1$ is the spin of the $i$-th site.
   ![](images/c60.svg)

 The graph topology is constructed by the following code:
 ```julia
 julia> using Graphs, ProblemReductions
 julia> function fullerene()  # construct the fullerene graph in 3D space
         th = (1+sqrt(5))/2
         res = NTuple{3,Float64}[]
         for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
             for (a, b, c) in ((x,y,z), (y,z,x), (z,x,y))
                 for loc in ((a,b,c), (a,b,-c), (a,-b,c), (a,-b,-c), (-a,b,c), (-a,b,-c), (-a,-b,c), (-a,-b,-c))
                     if loc ∉ res
                         push!(res, loc)
                     end
                 end
             end
         end
         return res
     end
 julia> fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5)); # construct the unit disk graph
 ```
 It is encouraged to use simulated annealing to find the ground state energy.

Answer: Ground state energy (best found): -66

2. (Spectral gap) Given an anti-ferromagnetic Ising model ($J = 1$) with different graph topology. Complete the following tasks:
   1. Analyse the spectral gap v.s. at different temperature $T$ from $0.1$ to $2.0$.
   2. Analyse the spectral gap v.s. the system size $N$ at $T = 0.1$.

   The following graph topologies up to $18$ nodes are considered:
   
   ![](images/topologies.svg)
   
   Hint: use sparse matrices and dominant eigenvalue solver to find the spectral gap!

Results:
```
Problem 2.1: Spectral gap vs temperature
diamond(m=3)
  T=0.1  gap=0.017124
  T=0.2  gap=0.000000
  T=0.3  gap=0.000000
  T=0.4  gap=0.000003
  T=0.5  gap=0.000026
  T=0.6  gap=0.000099
  T=0.7  gap=0.000263
  T=0.8  gap=0.000556
  T=0.9  gap=0.001015
  T=1.0  gap=0.001673
  T=1.1  gap=0.002554
  T=1.2  gap=0.003673
  T=1.3  gap=0.005031
  T=1.4  gap=0.006621
  T=1.5  gap=0.008424
  T=1.6  gap=0.010419
  T=1.7  gap=0.012580
  T=1.8  gap=0.014879
  T=1.9  gap=0.017290
  T=2.0  gap=0.019788
square(4x2)
  T=0.1  gap=0.000000
  T=0.2  gap=0.000000
  T=0.3  gap=0.000000
  T=0.4  gap=0.000000
  T=0.5  gap=0.000002
  T=0.6  gap=0.000015
  T=0.7  gap=0.000065
  T=0.8  gap=0.000196
  T=0.9  gap=0.000471
  T=1.0  gap=0.000950
  T=1.1  gap=0.001688
  T=1.2  gap=0.002720
  T=1.3  gap=0.004064
  T=1.4  gap=0.005717
  T=1.5  gap=0.007663
  T=1.6  gap=0.009874
  T=1.7  gap=0.012321
  T=1.8  gap=0.014968
  T=1.9  gap=0.017782
  T=2.0  gap=0.020730
triangular(4x2)
  T=0.1  gap=0.000000
  T=0.2  gap=0.000005
  T=0.3  gap=0.000149
  T=0.4  gap=0.000802
  T=0.5  gap=0.002241
  T=0.6  gap=0.004538
  T=0.7  gap=0.007637
  T=0.8  gap=0.011415
  T=0.9  gap=0.015719
  T=1.0  gap=0.020385
  T=1.1  gap=0.025264
  T=1.2  gap=0.030230
  T=1.3  gap=0.035187
  T=1.4  gap=0.040065
  T=1.5  gap=0.044819
  T=1.6  gap=0.049424
  T=1.7  gap=0.053864
  T=1.8  gap=0.058135
  T=1.9  gap=0.062239
  T=2.0  gap=0.066180

Problem 2.2: Spectral gap vs size at T=0.1
diamond(m)
  N=7  gap=0.054567
  N=10  gap=0.017124
  N=13  gap=0.007121
  N=16  gap=0.003589
square(m,2)
  N=4  gap=0.156930
  N=6  gap=-0.000000
  N=8  gap=0.000000
  N=10  gap=0.000000
  N=12  gap=0.000000
  N=14  gap=0.000000
  N=16  gap=0.000000
  N=18  gap=0.000000
triangular(m,2)
  N=4  gap=0.000000
  N=6  gap=0.000000
  N=8  gap=0.000000
  N=10  gap=0.000000
  N=12  gap=0.000000
  N=14  gap=0.000000
  N=16  gap=0.001360
  N=18  gap=0.000929
```