# Homework8
## question 1
1. Matrix multiplication with transpose: 
   C_ij = einsum('ik,jk->ij', A, B)
2. Summing over all elements in a matrix:
   s = einsum('ij->', A)
3. Multiplying three matrices element-wise:
   D = einsum('ij,ij,ij->ij', A, B, C)
4. Kronecker product: 
   D = einsum('ij,ab->iajb', A, B)

## question 2
the tree graph would be:

(T1⋅T2)    (T4⋅T3)
   \        /
    \ ---- /
     final

## question 4
```
using Random, Graphs, ProblemReductions, Statistics

# Build fullerene graph
function fullerene()
    th = (1 + sqrt(5)) / 2
    pts = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0,1.0,3th),(1.0,2+th,2th),(th,2.0,2th+1.0))
        for (a,b,c) in ((x,y,z),(y,z,x),(z,x,y))
            for loc in ((a,b,c),(a,b,-c),(a,-b,c),(a,-b,-c),
                        (-a,b,c),(-a,b,-c),(-a,-b,c),(-a,-b,-c))
                if loc ∉ pts; push!(pts, loc); end
            end
        end
    end
    return pts
end

g = SimpleGraph(UnitDiskGraph(fullerene(), sqrt(5)))

# Energy function
energy(g, σ) = sum(σ[src(e)] * σ[dst(e)] for e in edges(g))

# Metropolis sampling
function mean_energy(g, β; steps=10000)
    N = nv(g); σ = rand([-1,1], N)
    for _ in 1:steps
        i = rand(1:N)
        ΔE = -2σ[i]*sum(σ[n] for n in neighbors(g,i))
        if ΔE ≤ 0 || rand() < exp(-β*ΔE)
            σ[i] = -σ[i]
        end
    end
    return energy(g, σ)
end

# Scan β from 0.1 to 2.0
betas = 0.1:0.1:2.0
Eβ = [mean_energy(g, β; steps=20000) for β in betas]

# Thermodynamic integration
lnZ = [nv(g)*log(2) - trapz(collect(betas[1:i]), Eβ[1:i]) for i in 1:length(betas)]
Zβ = exp.(lnZ)

println("β\t⟨E⟩\tlnZ")
for (b, e, lz) in zip(betas, Eβ, lnZ)
    println("$(round(b,2))\t$(round(e,3))\t$(round(lz,3))")
end
```