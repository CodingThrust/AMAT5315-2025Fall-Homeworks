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

 Tips

2. (Spectral gap) Given an anti-ferromagnetic Ising model ($J = 1$) with different graph topology. Complete the following tasks:
   1. Analyse the spectral gap v.s. at different temperature $T$ from $0.1$ to $2.0$.
   2. Analyse the spectral gap v.s. the system size $N$ at $T = 0.1$.

   The following graph topologies up to $18$ nodes are considered:
   
   ![](images/topologies.svg)
   
   Hint: use sparse matrices and dominant eigenvalue solver to find the spectral gap!

Tips: Transfer matrix $T$ is defined as a $2^N\times 2^N$ sparse matrix which has $\approx N+1$ non-zero entries per column. For those $\sigma_\text{new}$ which differ only on one spin with $\sigma_\text{old}$, $$
    T_{(\vec{\sigma_\text{new}}, \vec{\sigma_\text{old}})} = \frac{\min\{1, \exp(-\beta *[ E(\vec{\sigma_\text{new}})- E(\vec{\sigma_\text{old}})])\}}{N}
$$

$$
T_{\vec{\sigma_\text{old}},\vec{\sigma_\text{old}}} = 1-\sum_{} T_{\vec{\sigma_\text{new}},\vec{\sigma_\text{old}}}
$$

With $N=18$, about $2^{18}\times 19=262144\times 19$ non-zero entries.