# Homework 8

1. (Einsum notation) Write the einsum notation for the following operations:
    - Matrix multiplication with transpose: $C = A B^T$
    - Summing over all elements in a matrix: $\sum_{i,j} A_{i,j}$
    - Multiplying three matrices element-wise: $D = A \odot B \odot C$
    - Kronecker product: $D = A \otimes B \otimes C$
  
  Ans:
  - `ein"ik, jk -> ij"(A, B)`
  - `ein"ij ->"(A)`
  - `ein"ij, ij, ij -> ij"(A, B, C)`
  - `ein"ij, kl, mn -> ikmjln"(A, B, C)`

2. (Contraction order) What is the optimal contraction order for the following tensor network?
  ![](images/order.svg)
  where $T_i$ are tensors, $A - H$ are indices.

  Ans:
  ```julia
  julia> using OMEinsum

   julia> code = ein"abc, fbg, dce, hge -> afdh"
   abc, fbg, dce, hge -> afdh

   julia> optcode = optimize_code(code, uniformsize(code, 2), TreeSA())
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

  Ans:
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
   fullerene (generic function with 1 method)

   julia> fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5));

   julia> using OMEinsum

   julia> code = EinCode([[e.src, e.dst] for e in edges(fullerene_graph)], Int[]);

   julia> optcode = optimize_code(code, uniformsize(code, 2), TreeSA());

   julia> sitetensors(β::Real) = [[exp(-β) exp(β); exp(β) exp(-β)] for _ in 1:ne(fullerene_graph)]
   sitetensors (generic function with 1 method)

   julia> partition_func(β::Real) = only(optcode(sitetensors(β)...))
   partition_func (generic function with 1 method)

   julia> Zs = [partition_func(β) for β in 0.1:0.1:2.0]
   20-element Vector{Float64}:
    1.8066109403976804e18
    6.875665762146549e18
    6.151122581721418e19
    1.2302123108278568e21
    5.144558716182042e22
    4.111628289398971e24
    5.6127026832899126e26
    1.1622488726804851e29
    3.2853302335009414e31
    1.1667408970547078e34
    4.898099153436166e36
    2.327394879239758e39
    1.2136776638732638e42
    6.793958994977612e44
    4.017291439227067e47
    2.4794022967126456e50
    1.58288423499531e53
    1.0380999176571235e56
    6.956429030632339e58
    4.7430829554740573e61
  ```

1. (Challenge) Develop a better algorithm to compute the contraction order. If you can beat all algoirthms in [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl), you will get an A+.
A good starting point is this benchmark repository: [OMEinsumContractionOrdersBenchmark](https://github.com/TensorBFS/OMEinsumContractionOrdersBenchmark)
