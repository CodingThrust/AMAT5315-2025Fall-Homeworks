# HW8
## Task1
1. C_ij = einsum('ik,jk->ij', A, B)
2. s = einsum('ij->', A)
3. D = einsum('ij,ij,ij->ij', A, B, C)
4. D = einsum('ij,ab,kl->iakjbl', A, B, C)

## Task2
X = einsum('ABC,BFG->ACFG', T1, T2)
Y = einsum('CDE,GEH->CDGH', T4, T3)
Z = einsum('ACFG,CDGH->AFDH', X, Y)

This order minimizes intermediate tensor ranks — first forming two rank-4 tensors (top and bottom), then combining them. Any other order would create higher-rank intermediates and increase computational cost.

## Task4
```
# partition_fullerene.jl
using Random, LinearAlgebra, Statistics
using Graphs
using ProblemReductions   # for UnitDiskGraph
# optionally: using ProgressMeter

# ---------- build fullerene graph  ----------
function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x,y,z),(y,z,x),(z,x,y))
            for loc in ((a,b,c),(a,b,-c),(a,-b,c),(a,-b,-c),(-a,b,c),(-a,b,-c),(-a,-b,c),(-a,-b,-c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

points = fullerene()
ug = UnitDiskGraph(points, sqrt(5))
g = SimpleGraph(ug)   # Graph with nv(g)=60, ne(g)=90
N = nv(g)
M = ne(g)
println("N = $N, E = $M")

# ---------- energy functions ----------
# spins: Vector{Int8} with ±1
function energy(g::Graph, σ::Vector{Int8})
    s = 0
    for e in edges(g)
        s += σ[src(e)] * σ[dst(e)]
    end
    return s
end

function delta_energy_flip(g::Graph, σ::Vector{Int8}, i::Int)
    s = Int8(0)
    for nb in neighbors(g, i)
        s += σ[nb]
    end
    return -2 * σ[i] * s
end

# ---------- Metropolis MCMC at fixed beta ----------
"""
mcmc_sample_meanE(g, beta; nequil=2000, nsamples=2000, sweep=1, rng=GLOBAL_RNG)

Run Metropolis single-spin-flip MCMC at inverse temperature `beta`.

- nequil: number of sweeps for equilibration (each sweep = N attempted flips)
- nsamples: number of samples to collect
- sweep: number of sweeps between recorded samples (to reduce autocorrelation)
Returns (meanE, std_err, all_sampledE)
"""
function mcmc_sample_meanE(g::Graph, beta; nequil=2000, nsamples=2000, sweep=1, rng=Random.GLOBAL_RNG)
    N = nv(g)
    # initial random spins ±1
    σ = rand(rng, [-1, 1], N) .|> Int8
    # equilibration
    for t in 1:nequil
        for _ in 1:N
            i = rand(rng, 1:N)
            Δ = delta_energy_flip(g, σ, i)
            if Δ <= 0 || rand(rng) < exp(-beta*Δ)
                σ[i] = -σ[i]
            end
        end
    end

    samples = Float64[]
    for sidx in 1:nsamples
        for t in 1:sweep
            for _ in 1:N
                i = rand(rng, 1:N)
                Δ = delta_energy_flip(g, σ, i)
                if Δ <= 0 || rand(rng) < exp(-beta*Δ)
                    σ[i] = -σ[i]
                end
            end
        end
        push!(samples, energy(g, σ))
    end

    meanE = mean(samples)
    stderr = std(samples) / sqrt(length(samples))
    return meanE, stderr, samples
end

# ---------- main scan over beta and TI integration ----------
function compute_partition_function(g::Graph;
        betas = collect(0.0:0.1:2.0),
        nequil=20000, nsamples=10000, sweep_between=2, repeats=3, rngseed=42)

    Random.seed!(rngseed)
    N = nv(g)
    lnZ0 = N * log(2)   # ln Z at beta=0
    nb = length(betas)
    meanEs = zeros(nb)
    stderrEs = zeros(nb)

    for (k, β) in enumerate(betas)
        if isapprox(β, 0.0; atol=1e-12)
            meanEs[k] = 0.0
            stderrEs[k] = 0.0
            println("β=0.0: ⟨E⟩ = 0 (exact)")
            continue
        end
        # do several independent repeats, average them to reduce bias
        means_rep = Float64[]
        for r in 1:repeats
            rng = MersenneTwister(rand(UInt))
            me, se, _ = mcmc_sample_meanE(g, β; nequil=nequil, nsamples=nsamples, sweep=sweep_between, rng=rng)
            push!(means_rep, me)
            println("β=$(round(β, digits=3)) rep $r: ⟨E⟩=$(round(me,digits=4)) (stderr sample ≈ $(round(se,digits=4)))")
        end
        meanEs[k] = mean(means_rep)
        stderrEs[k] = std(means_rep) / sqrt(length(means_rep))
        println("=> β=$(round(β,3)): mean ⟨E⟩=$(round(meanEs[k],digits=5)) ± $(round(stderrEs[k],digits=5))")
    end

    # Numerical integration of ⟨E⟩ from 0 to β: use trapezoid
    lnZ = Float64[]
    for i in 1:nb
        βs = betas[1:i]
        Es = meanEs[1:i]
        # trapezoidal integral of E dβ
        integral = 0.0
        for j in 1:(length(βs)-1)
            h = βs[j+1] - βs[j]
            integral += 0.5 * h * (Es[j] + Es[j+1])
        end
        lnZ_i = lnZ0 - integral
        push!(lnZ, lnZ_i)
    end

    return betas, meanEs, stderrEs, lnZ
end

# ---------- run (defaults) ----------
betas = collect(0.0:0.1:2.0)
# WARNING: default nequil and nsamples set for reasonable accuracy but will cost time.
# Adjust nequil/nsamples/repeats smaller for quick test.
betas, meanEs, stderrEs, lnZ = compute_partition_function(g;
    betas=betas, nequil=20000, nsamples=8000, sweep_between=2, repeats=3, rngseed=1234)

# output table
println("\nβ\t⟨E⟩\tstderr\tlnZ")
for i in 1:length(betas)
    println("$(round(betas[i],digits=2))\t$(round(meanEs[i],digits=5))\t$(round(stderrEs[i],digits=5))\t$(round(lnZ[i],digits=6))")
end

# If you want final Z values:
Z_vals = exp.(lnZ)
```