############################################################
# Problem 2 – Spectral gap of anti-ferromagnetic Ising model
# J = 1, Metropolis single-spin flip dynamics
############################################################

using Graphs
using SparseArrays
using LinearAlgebra
using KrylovKit
using Random

# ----------------------------------------------------------
# Helpers: graph families (up to N ≤ 18)
# ----------------------------------------------------------

# map (row, col) -> vertex index
idx(row, col) = 2*(col-1) + row   # row = 1 or 2

"""
    triangles_graph(N)

2×L ladder with one diagonal per cell (triangular strip).
N must be even and N = 2L.
"""
function triangles_graph(N::Int)
    @assert iseven(N) "N must be even for triangles_graph"
    L = N ÷ 2
    g = SimpleGraph(N)

    # horizontal edges
    for c in 1:L-1
        add_edge!(g, idx(1,c), idx(1,c+1))
        add_edge!(g, idx(2,c), idx(2,c+1))
    end
    # vertical edges
    for c in 1:L
        add_edge!(g, idx(1,c), idx(2,c))
    end
    # one diagonal per square (say from top-left to bottom-right)
    for c in 1:L-1
        add_edge!(g, idx(1,c), idx(2,c+1))
    end
    return g
end

"""
    squares_graph(N)

2×L ladder of squares (no diagonals).
N must be even and N = 2L.
"""
function squares_graph(N::Int)
    @assert iseven(N) "N must be even for squares_graph"
    L = N ÷ 2
    g = SimpleGraph(N)

    # horizontal edges
    for c in 1:L-1
        add_edge!(g, idx(1,c), idx(1,c+1))
        add_edge!(g, idx(2,c), idx(2,c+1))
    end
    # vertical edges
    for c in 1:L
        add_edge!(g, idx(1,c), idx(2,c))
    end
    return g
end

"""
    diamonds_graph(N)

2×L ladder with both diagonals per square ("diamond chain").
N must be even and N = 2L.
"""
function diamonds_graph(N::Int)
    @assert iseven(N) "N must be even for diamonds_graph"
    L = N ÷ 2
    g = SimpleGraph(N)

    # horizontal edges
    for c in 1:L-1
        add_edge!(g, idx(1,c), idx(1,c+1))
        add_edge!(g, idx(2,c), idx(2,c+1))
    end
    # vertical edges
    for c in 1:L
        add_edge!(g, idx(1,c), idx(2,c))
    end
    # both diagonals per cell
    for c in 1:L-1
        add_edge!(g, idx(1,c), idx(2,c+1))
        add_edge!(g, idx(2,c), idx(1,c+1))
    end
    return g
end

# ----------------------------------------------------------
# Ising / Metropolis machinery
# ----------------------------------------------------------

# spin at site i from integer label s (0-based), bits 0..N-1
@inline spin_of_state(s::Int, i::Int) = ((s >> (i-1)) & 1) == 1 ? 1 : -1

"""
    build_transition_matrix(g, T)

Construct sparse Metropolis transition matrix P(T) for
single-spin flip dynamics on graph g at temperature T.

State space: all σ ∈ {±1}^N, encoded as integers 0..2^N-1.
"""
function build_transition_matrix(g::SimpleGraph, T::Float64)
    N = nv(g)
    nstates = 1 << N
    β = 1.0 / T

    # pre-build neighbor list (1-based vertex indices)
    neigh = [collect(neighbors(g, v)) for v in 1:N]

    # we'll collect I, J, V for sparse(I,J,V,nstates,nstates)
    row_idx = Int[]
    col_idx = Int[]
    vals    = Float64[]

    # loop over all configurations (encoded as 0..2^N-1)
    for s in 0:nstates-1
        row = s + 1
        outgoing = 0.0

        # try flipping each spin
        for i in 1:N
            si = spin_of_state(s, i)
            # local field at i: sum_{j~i} σ_j
            h = 0
            for j in neigh[i]
                h += spin_of_state(s, j)
            end
            ΔE = 2.0 * si * h        # J=1, anti-ferro H = Σ σ_i σ_j
            # Metropolis acceptance prob
            a = ΔE <= 0 ? 1.0 : exp(-β * ΔE)
            p = (1.0 / N) * a
            outgoing += p

            # index of flipped state
            sflipped = s ⊻ (1 << (i-1))    # xor the bit i-1
            col = sflipped + 1

            push!(row_idx, row)
            push!(col_idx, col)
            push!(vals, p)
        end

        # self-loop probability
        push!(row_idx, row)
        push!(col_idx, row)
        push!(vals, 1.0 - outgoing)
    end

    P = sparse(row_idx, col_idx, vals, nstates, nstates)
    return P
end

"""
    spectral_gap(g, T; nev=2)

Compute spectral gap γ(T) = 1 - λ₂ for Metropolis chain on g at temperature T.
Uses KrylovKit to find the top two eigenvalues of P(T).
"""
function spectral_gap(g::SimpleGraph, T::Float64)
    P = build_transition_matrix(g, T)

    # largest-magnitude eigenvalues
    vals, _, _ = eigsolve(P, 2, :LM; tol=1e-8, maxiter=500)
    # sort so λ1 ≈ 1, λ2 ≤ λ1
    λ = sort(real.(vals), rev=true)
    λ1, λ2 = λ[1], λ[2]
    return 1.0 - λ2, λ1, λ2
end

# ----------------------------------------------------------
# Experiments
# ----------------------------------------------------------

function main()
    Random.seed!(0)

    println("=== Spectral gap vs T for N = 18 ===")
    N = 18
    g_tri  = triangles_graph(N)
    g_sq   = squares_graph(N)
    g_dia  = diamonds_graph(N)

    Ts = 0.1:0.1:2.0

    println("\nTriangles:")
    for T in Ts
        gap, λ1, λ2 = spectral_gap(g_tri, T)
        println("T = $(round(T, digits=2)), gap = $(round(gap, digits=5))")
    end

    println("\nSquares:")
    for T in Ts
        gap, λ1, λ2 = spectral_gap(g_sq, T)
        println("T = $(round(T, digits=2)), gap = $(round(gap, digits=5))")
    end

    println("\nDiamonds:")
    for T in Ts
        gap, λ1, λ2 = spectral_gap(g_dia, T)
        println("T = $(round(T, digits=2)), gap = $(round(gap, digits=5))")
    end

    # ------------------------------------------------------
    # Spectral gap vs system size at fixed T = 0.1
    # ------------------------------------------------------
    println("\n=== Spectral gap vs N at T = 0.1 ===")
    Tfixed = 0.1

    println("\nTriangles:")
    for N in 4:2:18        # even N
        g = triangles_graph(N)
        gap, _, _ = spectral_gap(g, Tfixed)
        println("N = $N, gap = $(round(gap, digits=5))")
    end

    println("\nSquares:")
    for N in 4:2:18
        g = squares_graph(N)
        gap, _, _ = spectral_gap(g, Tfixed)
        println("N = $N, gap = $(round(gap, digits=5))")
    end

    println("\nDiamonds:")
    for N in 4:2:18
        g = diamonds_graph(N)
        gap, _, _ = spectral_gap(g, Tfixed)
        println("N = $N, gap = $(round(gap, digits=5))")
    end
end

main()
