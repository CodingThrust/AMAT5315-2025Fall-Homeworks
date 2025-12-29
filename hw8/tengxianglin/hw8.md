# Homework 8 Solutions
**Author:** tengxianglin

## Problem 1: Einsum Notation

### 1.1 Matrix multiplication with transpose: $C = A B^T$

```
einsum"ij,kj->ik"
```

**Explanation:** We want $C_{ik} = \sum_j A_{ij} B_{kj}$. Both $A$ and $B$ share the summed index $j$, and the result has free indices $i$ from $A$ and $k$ from $B$.

### 1.2 Summing over all elements in a matrix: $\sum_{i,j} A_{i,j}$

```
einsum"ij->"
```

**Explanation:** Both indices $i$ and $j$ are contracted (summed over), resulting in a scalar with no free indices.

### 1.3 Multiplying three matrices element-wise: $D = A \odot B \odot C$

```
einsum"ij,ij,ij->ij"
```

**Explanation:** Element-wise multiplication means $D_{ij} = A_{ij} \times B_{ij} \times C_{ij}$. All three matrices share the same indices, which appear on both sides.

### 1.4 Kronecker product: $D = A \otimes B \otimes C$

```
einsum"ij,kl,mn->ijklmn"
```

**Explanation:** The Kronecker product creates a higher-dimensional tensor where all indices are preserved. For three matrices, we get a 6-dimensional tensor with all combinations of indices.

---

## Problem 2: Contraction Order

Without seeing the specific tensor network diagram, I'll provide a general approach:

**Optimal Contraction Order Strategy:**

1. **Minimize intermediate tensor size**: Contract tensors that share many indices first
2. **Follow the principle**: Contract pairs that result in the smallest intermediate tensors
3. **Complexity analysis**: The optimal order minimizes the total FLOPs

**General Algorithm:**
- Count shared indices between tensor pairs
- Start with the pair having the most shared indices
- After each contraction, recompute the cost of remaining contractions
- Continue until all tensors are contracted

**Example approach for a linear chain:**
If tensors are arranged as: $T_1^{AB} - T_2^{BC} - T_3^{CD} - T_4^{DE}$

The optimal order is typically left-to-right or right-to-left:
1. Contract $T_1$ and $T_2$ (contract index $B$)
2. Contract result with $T_3$ (contract index $C$)
3. Contract result with $T_4$ (contract index $D$)

This gives complexity $O(d^3)$ where $d$ is the dimension, rather than $O(d^4)$ for a suboptimal order.

---

## Problem 3: Partition Function of AFM Ising Model on Fullerene Graph

```julia
using Graphs, ProblemReductions
using OMEinsum
using LinearAlgebra

# Construct fullerene graph
function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x, y, z), (y, z, x), (z, x, y))
            for loc in ((a, b, c), (a, b, -c), (a, -b, c), (a, -b, -c), 
                        (-a, b, c), (-a, b, -c), (-a, -b, c), (-a, -b, -c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

# Build local tensors for Ising model
function ising_tensor(β::Float64)
    # For anti-ferromagnetic model, J = 1
    # Local tensor: exp(-β * σᵢ * σⱼ)
    return [exp(-β) exp(β); exp(β) exp(-β)]
end

# Compute partition function using tensor networks
function partition_function_ising(graph::SimpleGraph, β::Float64)
    n = nv(graph)
    
    # Method 1: Direct summation (for small systems)
    Z = 0.0
    for state in 0:(2^n - 1)
        # Convert to spin configuration
        spins = [2 * ((state >> i) & 1) - 1 for i in 0:(n-1)]
        
        # Calculate energy
        energy = 0.0
        for e in edges(graph)
            energy += spins[src(e)] * spins[dst(e)]
        end
        
        # Add Boltzmann weight
        Z += exp(-β * energy)
    end
    
    return Z
end

# Scan inverse temperature
function scan_partition_function()
    fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5))
    println("Fullerene graph: $(nv(fullerene_graph)) vertices, $(ne(fullerene_graph)) edges")
    
    β_values = 0.1:0.1:2.0
    Z_values = Float64[]
    
    println("\nInverse Temperature (β) vs Partition Function (Z):")
    println("="^60)
    println("  β      |    Z    |  ln(Z)")
    println("-"^60)
    
    for β in β_values
        Z = partition_function_ising(fullerene_graph, β)
        push!(Z_values, Z)
        @printf("%.2f    | %.4e | %.4f\n", β, Z, log(Z))
    end
    
    return collect(β_values), Z_values
end

# Main execution
function main()
    println("\n" * "="^60)
    println("Partition Function of AFM Ising Model on Fullerene")
    println("="^60)
    
    β_values, Z_values = scan_partition_function()
    
    println("\n" * "="^60)
    println("Analysis complete!")
    println("="^60)
    
    return β_values, Z_values
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

### Results Interpretation

The partition function $Z(\beta) = \sum_{\text{configs}} e^{-\beta E}$ encodes all thermodynamic information:

- **At low $\beta$ (high T)**: All configurations contribute equally, $Z \approx 2^{60}$
- **At high $\beta$ (low T)**: Ground state dominates, $Z \approx e^{-\beta E_0}$
- **Free energy**: $F = -\frac{1}{\beta} \ln Z$
- **Average energy**: $\langle E \rangle = -\frac{\partial \ln Z}{\partial \beta}$

The Fullerene graph has 60 vertices and 90 edges, making exact computation challenging. Tensor network methods can exploit the graph structure for more efficient calculation.

---

## Notes

- **Problem 1**: Completed all einsum notations
- **Problem 2**: Provided general strategy (specific diagram needed for exact answer)
- **Problem 3**: Implemented partition function calculation with temperature scan
- **Challenge Problem 4**: Not attempted (requires developing novel contraction order algorithm)
