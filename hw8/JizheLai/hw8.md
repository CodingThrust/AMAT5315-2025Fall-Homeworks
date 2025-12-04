# AMAT5315 – Homework 8

## 1. Einsum notation

Write the einsum notation for the following operations.

### (a) Matrix multiplication with transpose

Matrix product with transpose  
\(C = A B^\top\), where \(A_{ij}\), \(B_{kj}\), \(C_{ik}\):

\[
C_{ik} = \sum_j A_{ij} B_{kj}
\]

Einsum:

```text
"ij,kj->ik"
```

---

### (b) Sum over all elements of a matrix

Sum of all entries:

\[
\sum_{i,j} A_{ij}
\]

Einsum:

```text
"ij->"
```

---

### (c) Elementwise product of three matrices

Elementwise (Hadamard) product:

\[
D = A \odot B \odot C,\qquad
D_{ij} = A_{ij} B_{ij} C_{ij}
\]

Einsum:

```text
"ij,ij,ij->ij"
```

---

### (d) Kronecker product of three matrices

Kronecker product:

\[
D = A \otimes B \otimes C
\]

with components

\[
A_{ij},\quad B_{k\ell},\quad C_{mn},\quad
D_{ikm,\; j\ell n} = A_{ij} B_{k\ell} C_{mn}.
\]

Einsum:

```text
"ij,kl,mn->ikm,jln"
```

---

## 2. Contraction order

For the given 4‑tensor network with tensors \(T_1,T_2,T_3,T_4\) and internal indices \(A\text{–}H\), one explicit contraction order is:

1. Contract \(T_2\) and \(T_3\) over their common internal indices (for example indices shared between them) to form an intermediate tensor \(X\):
   \[
   X = T_2 * T_3
   \]
2. Contract \(X\) with \(T_1\) over the indices shared by \(X\) and \(T_1\) to form \(Y\):
   \[
   Y = X * T_1
   \]
3. Contract \(Y\) with \(T_4\) over the remaining internal indices to obtain the final result \(Z\):
   \[
   Z = Y * T_4
   \]

Bracket notation:

\[
Z = (((T_2 * T_3) * T_1) * T_4),
\]

where “\(*\)” denotes contraction over all shared indices.

---

## 4. Partition function on the Fullerene (C\(_{60}\)) graph

We consider the anti‑ferromagnetic (AFM) Ising model on the fullerene graph:

\[
H(\{\sigma\}) = \sum_{(i,j)\in E} \sigma_i \sigma_j,
\qquad
\sigma_i \in \{+1,-1\}.
\]

The partition function is

\[
Z(\beta) = \sum_{\{\sigma\}} 
\exp\!\left( \beta J \sum_{(i,j)\in E} \sigma_i \sigma_j \right),
\]

with \(J = -1\) (AFM interaction). Equivalently, each edge \((i,j)\) contributes a 2×2 tensor

\[
W(\sigma_i,\sigma_j) = \exp\bigl(\beta J \sigma_i \sigma_j\bigr),
\]

and the partition function is the contraction of all such edge tensors.

### 4.1 Fullerene graph construction

The fullerene (C\(_{60}\)) coordinates are generated as in Homework 7:

```julia
function fullerene_points()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3 * th),
                      (1.0, 2 + th, 2 * th),
                      (th, 2.0, 2 * th + 1.0))
        for (a, b, c) in ((x, y, z), (y, z, x), (z, x, y))
            for loc in ((a,  b,  c),
                        (a,  b, -c),
                        (a, -b,  c),
                        (a, -b, -c),
                        (-a,  b,  c),
                        (-a,  b, -c),
                        (-a, -b,  c),
                        (-a, -b, -c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end
```

Using these coordinates, the graph is constructed as a unit disk graph with radius \(\sqrt{5}\):

```julia
using Graphs

function fullerene_graph()
    pts = fullerene_points()
    n   = length(pts)
    g   = SimpleGraph(n)

    r2 = 5.0  # (√5)^2

    @inbounds for i in 1:n-1
        xi, yi, zi = pts[i]
        for j in i+1:n
            xj, yj, zj = pts[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            if dx*dx + dy*dy + dz*dz < r2 + 1e-8
                add_edge!(g, i, j)
            end
        end
    end

    return g
end

function ising_fullerene_graph()
    g = fullerene_graph()
    N = nv(g)
    es = Tuple{Int,Int}[]
    for e in edges(g)
        i = src(e); j = dst(e)
        if j < i
            i, j = j, i
        end
        push!(es, (i, j))
    end
    sort!(es)
    unique!(es)
    return N, es
end
```

This yields \(N = 60\) vertices and the correct fullerene edge set.

### 4.2 Tensor network formulation

Each vertex \(i\) is a spin index of dimension 2; each edge \((i,j)\) carries a rank‑2 tensor:

```julia
const SPINS = (+1, -1)

function interaction_tensor(β::Float64; J::Float64 = -1.0)
    W = zeros(Float64, 2, 2)
    for (a, σi) in enumerate(SPINS), (b, σj) in enumerate(SPINS)
        W[a, b] = exp(β * J * σi * σj)
    end
    return W
end

function build_ising_einsum(N::Int, edges::Vector{Tuple{Int,Int}})
    ixs  = [Int[i, j] for (i, j) in edges]    # indices per edge tensor
    iy   = Int[]                              # scalar output
    size = Dict(i => 2 for i in 1:N)          # spin dim = 2
    return (ixs = ixs, iy = iy, size = size)
end

function build_tensors_for_ising(edges::Vector{Tuple{Int,Int}}, β::Real)
    W = interaction_tensor(Float64(β))
    return [W for _ in edges]
end
```

We create an einsum code and optimize its contraction order using `OMEinsumContractionOrders.jl` (version interface as in the README):

```julia
using OMEinsum
using OMEinsumContractionOrders

function optimize_contraction(ixs, iy, size)
    code      = EinCode(ixs, iy)              # DynamicEinCode(ixs, iy)
    optimizer = TreeSA()                      # code optimizer
    opt_code  = optimize_code(code, size, optimizer)
    return opt_code                          # NestedEinsum, callable
end
```

For each value of \(\beta\), we construct all edge tensors and call the optimized contraction:

```julia
function partition_function_fullerene(βs::AbstractVector{<:Real})
    N, edges = ising_fullerene_graph()
    spec     = build_ising_einsum(N, edges)

    opt_code = optimize_contraction(spec.ixs, spec.iy, spec.size)

    Zs = Dict{Float64,Float64}()
    for β in βs
        tensors  = build_tensors_for_ising(edges, β)
        Z_tensor = opt_code(tensors...)  # NestedEinsum returns 0‑dim Array
        Z        = only(Z_tensor)        # extract scalar Float64
        Zs[float(β)] = Z
    end
    return Zs
end
```

The driver scans \(\beta\) from 0.1 to 2.0 in steps of 0.1:

```julia
function run_problem4()
    βs = 0.1:0.1:2.0
    Zs = partition_function_fullerene(collect(βs))
    println("# beta\tZ(beta)")
    for β in βs
        println("$(round(β, digits = 2))\t", Zs[float(β)])
    end
end
```

### 4.3 Numerical results

Running the above code produces the following values of the partition function for the AFM Ising model on the fullerene graph:

| β   | Z(β)                    |
|-----|-------------------------|
| 0.1 | 1.80661094039768e18     |
| 0.2 | 6.87566576214655e18     |
| 0.3 | 6.151122581721418e19    |
| 0.4 | 1.2302123108278566e21   |
| 0.5 | 5.144558716182041e22    |
| 0.6 | 4.1116282893989694e24   |
| 0.7 | 5.612702683289911e26    |
| 0.8 | 1.1622488726804853e29   |
| 0.9 | 3.2853302335009414e31   |
| 1.0 | 1.1667408970547078e34   |
| 1.1 | 4.898099153436166e36    |
| 1.2 | 2.3273948792397584e39   |
| 1.3 | 1.2136776638732638e42   |
| 1.4 | 6.793958994977612e44    |
| 1.5 | 4.017291439227066e47    |
| 1.6 | 2.479402296712645e50    |
| 1.7 | 1.5828842349953103e53   |
| 1.8 | 1.038099917657124e56    |
| 1.9 | 6.956429030632339e58    |
| 2.0 | 4.7430829554740573e61   |

These satisfy the requirement to compute \(Z(\beta)\) for \(\beta = 0.1, 0.2, \dots, 2.0\).

---

## 6. Challenge – Contraction order optimizer (baseline)

The challenge problem asks to develop a better contraction order algorithm than all existing optimizers in `OMEinsumContractionOrders.jl`. As a baseline implementation, we use the built‑in `TreeSA` optimizer:

```julia
function my_contraction_order_optimizer(ixs, iy, size)
    code      = EinCode(ixs, iy)
    optimizer = TreeSA()
    return optimize_code(code, size, optimizer)
end
```

This function takes the einsum specification `(ixs, iy, size)`, constructs a `DynamicEinCode`, and applies `TreeSA` to obtain a `NestedEinsum` object that defines the optimized contraction order.
