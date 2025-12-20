using SparseArrays, LinearAlgebra, Arpack, Random, Graphs, Plots

# --- ΔE for flipping spin i ---
function local_delta_energy(g::Graph, σ::Vector{Int8}, i::Int)
    s = Int8(0)
    for nb in neighbors(g, i)
        s += σ[nb]
    end
    return -2 * σ[i] * s
end

# --- Encode / decode spin states ---
spins_from_index(idx::Int, N::Int) = [((idx >> (k-1)) & 1) == 1 ? Int8(1) : Int8(-1) for k in 1:N]
function index_from_spins(σ::Vector{Int8})
    idx = 0
    for k in 1:length(σ)
        if σ[k] == 1
            idx |= (1 << (k-1))
        end
    end
    return idx
end

# --- Build sparse Metropolis transition matrix ---
function build_metropolis_P(g::Graph, T::Float64)
    N = nv(g); Nstates = 1 << N
    rows = Int[]; cols = Int[]; vals = Float64[]
    for idx in 0:(Nstates-1)
        σ = spins_from_index(idx, N)
        row_sum = 0.0
        for i in 1:N
            Δ = local_delta_energy(g, σ, i)
            p = (1 / N) * min(1.0, exp(-Δ / T))
            if p > 0
                σ[i] = -σ[i]; jidx = index_from_spins(σ); σ[i] = -σ[i]
                push!(rows, idx+1); push!(cols, jidx+1); push!(vals, p)
                row_sum += p
            end
        end
        push!(rows, idx+1); push!(cols, idx+1); push!(vals, 1 - row_sum)
    end
    sparse(rows, cols, vals, Nstates, Nstates)
end

# --- Compute spectral gap (1 - λ₂) ---
function spectral_gap(P::SparseMatrixCSC; nev=4)
    vals, = eigs(P; nev=nev, which=:LR, maxiter=500, tol=1e-8)
    λ = sort(real(vals); rev=true)
    1 - λ[2]
end

# --- Spectral gap vs temperature ---
function gap_vs_temperature(g::Graph, Ts)
    [spectral_gap(build_metropolis_P(g, T)) for T in Ts]
end

# --- Spectral gap vs system size ---
function gap_vs_size(f::Function, Ns, T)
    [spectral_gap(build_metropolis_P(f(N), T)) for N in Ns]
end

# --- Example demo ---
function main()
    g = cycle_graph(12)
    Ts = range(0.1, 2.0; length=20)
    gaps_T = gap_vs_temperature(g, Ts)
    plot(Ts, gaps_T, xlabel="T", ylabel="Gap", title="Spectral gap vs T (cycle 12)", marker=:circle)

    Ns = 6:2:18
    gaps_cycle = gap_vs_size(cycle_graph, Ns, 0.1)
    gaps_path = gap_vs_size(path_graph, Ns, 0.1)
    plot(Ns, gaps_path, label="path", marker=:diamond, xlabel="N", ylabel="Gap", title="Gap vs N at T=0.1")
    plot!(Ns, gaps_cycle, label="cycle", marker=:utriangle)
end

main()
