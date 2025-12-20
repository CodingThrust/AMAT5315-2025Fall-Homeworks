# Homework 8

1. (Einsum notation) Write the einsum notation for the following operations:
    - Matrix multiplication with transpose: $C = A B^T$
    - Summing over all elements in a matrix: $\sum_{i,j} A_{i,j}$
    - Multiplying three matrices element-wise: $D = A \odot B \odot C$
    - Kronecker product: $D = A \otimes B \otimes C$

Answer:
 - $C = A B^T$: `ik,jk->ij`
 - $\sum_{i,j} A_{i,j}$: `ij->`
 - $D = A \odot B \odot C$: `ij,ij,ij->ij`
 - $D = A \otimes B \otimes C$: `ij,kl,mn->ikm jln`

2. (Contraction order) What is the optimal contraction order for the following tensor network?
   
  ![](images/order.svg)
  
  where $T_i$ are tensors, $A - H$ are indices.

Answer:
Using `OMEinsum` to optimize the contraction order, one optimal order is:
```julia
using OMEinsum
code = ein"abc, fbg, dce, hge -> afdh"
abc, fbg, dce, hge -> afdh
optcode = optimize_code(code, uniformsize(code, 2), TreeSA())

abde, hefb -> afdh
├─ abc, dce -> abde
│  ├─ abc
│  └─ dce
└─ hge, fbg -> hefb
   ├─ hge
   └─ fbg
```

3. (Partition function) Compute the partition function $Z$ for the AFM (anti-ferromagnetic) Ising model on the Fullerene graph. Please scan the inverse temperature $\beta$ from $0.1$ to $2.0$ with step $0.1$. For the information needed to construct the Fullerene graph, please refer to Homework 7.
   
  ![](images/c60.svg)

Answer:
See `hw8.jl` for the exact variable-elimination contraction that computes $Z(\beta)$ and prints the scan for $\beta=0.1:0.1:2.0$.

```
beta=0.1  Z=1.806610940398e+18
beta=0.2  Z=6.875665762147e+18
beta=0.3  Z=6.151122581721e+19
beta=0.4  Z=1.230212310828e+21
beta=0.5  Z=5.144558716182e+22
beta=0.6  Z=4.111628289399e+24
beta=0.7  Z=5.612702683290e+26
beta=0.8  Z=1.162248872680e+29
beta=0.9  Z=3.285330233501e+31
beta=1.0  Z=1.166740897055e+34
beta=1.1  Z=4.898099153436e+36
beta=1.2  Z=2.327394879240e+39
beta=1.3  Z=1.213677663873e+42
beta=1.4  Z=6.793958994978e+44
beta=1.5  Z=4.017291439227e+47
beta=1.6  Z=2.479402296713e+50
beta=1.7  Z=1.582884234995e+53
beta=1.8  Z=1.038099917657e+56
beta=1.9  Z=6.956429030632e+58
beta=2.0  Z=4.743082955474e+61
```
