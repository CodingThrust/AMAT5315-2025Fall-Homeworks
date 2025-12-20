using LinearAlgebra
using SparseArrays
using Random
using Graphs
using KrylovKit
using Printf

function fullerene_positions()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3, Float64}[]
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

function unit_disk_graph(points, r; atol=1e-10)
    n = length(points)
    g = SimpleGraph(n)
    r2 = r^2 + atol
    for i in 1:(n - 1)
        xi, yi, zi = points[i]
        for j in (i + 1):n
            xj, yj, zj = points[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            if dx * dx + dy * dy + dz * dz <= r2
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

function fullerene_graph()
    points = fullerene_positions()
    return unit_disk_graph(points, sqrt(5))
end

function triangular_grid_graph(m, n)
    g = Graphs.grid([m, n])
    for j in 1:(n - 1)
        for i in 1:(m - 1)
            u = i + (j - 1) * m
            v = (i + 1) + j * m
            add_edge!(g, u, v)
        end
    end
    return g
end

function diamond_graph(m)
    g = SimpleGraph(3 * m + 1)
    for i in 1:(3 * m)
        if i % 3 == 1
            add_edge!(g, i, i + 1)
            add_edge!(g, i, i + 2)
        elseif i % 3 == 2
            add_edge!(g, i, i + 2)
        else
            add_edge!(g, i, i + 1)
        end
    end
    return g
end

function ising_energy(spins, g)
    e = 0
    for ed in edges(g)
        i = src(ed)
        j = dst(ed)
        e += spins[i] * spins[j]
    end
    return e
end

function neighbors_list(g)
    return [collect(neighbors(g, i)) for i in 1:nv(g)]
end

function simulated_annealing_ground_state(g; rng=Random.GLOBAL_RNG, restarts=40,
                                          sweeps_per_temp=40, t_start=5.0, t_end=0.05,
                                          temps=60)
    n = nv(g)
    nbrs = neighbors_list(g)
    temps_list = exp.(range(log(t_start), log(t_end), length=temps))

    best_energy = Inf
    best_spins = zeros(Int8, n)

    for _ in 1:restarts
        spins = rand(rng, (-1):2:1, n)
        e = ising_energy(spins, g)

        for T in temps_list
            beta = 1 / T
            for _ in 1:sweeps_per_temp
                for _ in 1:n
                    i = rand(rng, 1:n)
                    sumn = 0
                    for j in nbrs[i]
                        sumn += spins[j]
                    end
                    dE = -2 * spins[i] * sumn
                    if dE <= 0 || rand(rng) < exp(-beta * dE)
                        spins[i] = -spins[i]
                        e += dE
                    end
                end
            end
        end

        improved = true
        while improved
            improved = false
            for i in 1:n
                sumn = 0
                for j in nbrs[i]
                    sumn += spins[j]
                end
                dE = -2 * spins[i] * sumn
                if dE < 0
                    spins[i] = -spins[i]
                    e += dE
                    improved = true
                end
            end
        end

        if e < best_energy
            best_energy = e
            best_spins = copy(spins)
        end
    end

    return best_energy, best_spins
end

function metropolis_transition_matrix(g, T)
    n = nv(g)
    nstates = 1 << n
    beta = 1 / T
    nbrs = neighbors_list(g)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    nfloat = float(n)

    for s in 0:(nstates - 1)
        self_prob = 0.0
        for i in 1:n
            si = ((s >> (i - 1)) & 0x1) == 0 ? 1 : -1
            sumn = 0
            for j in nbrs[i]
                sj = ((s >> (j - 1)) & 0x1) == 0 ? 1 : -1
                sumn += sj
            end
            dE = -2 * si * sumn
            a = dE <= 0 ? 1.0 : exp(-beta * dE)
            p = a / nfloat
            sp = s ⊻ (1 << (i - 1))
            push!(rows, s + 1)
            push!(cols, sp + 1)
            push!(vals, p)
            self_prob += (1.0 / nfloat) * (1.0 - a)
        end
        push!(rows, s + 1)
        push!(cols, s + 1)
        push!(vals, self_prob)
    end

    return sparse(rows, cols, vals, nstates, nstates)
end

function spectral_gap(P; tol=1e-10, maxiter=4000, krylovdim=40)
    v0 = randn(size(P, 1))
    vals, _, _ = eigsolve(P, v0, 2, :LR; tol=tol, maxiter=maxiter, krylovdim=krylovdim)
    gap = 1.0 - real(vals[2])
    return gap, vals
end

function gap_vs_temperature()
    temps = 0.1:0.1:2.0
    graphs = Dict(
        "triangular(4x2)" => triangular_grid_graph(4, 2),
        "diamond(m=3)" => diamond_graph(3),
        "square(4x2)" => Graphs.grid([4, 2]),
    )
    results = Dict{String, Vector{Float64}}()

    for (name, g) in graphs
        gaps = Float64[]
        for T in temps
            P = metropolis_transition_matrix(g, T)
            gap, _ = spectral_gap(P)
            push!(gaps, gap)
        end
        results[name] = gaps
    end
    return temps, results
end

function gap_vs_size(; T=0.1)
    results = Dict{String, Vector{Tuple{Int, Float64}}}()

    tri = Tuple{Int, Float64}[]
    for m in 2:9
        # println("  triangular m=", m)
        g = triangular_grid_graph(m, 2)
        gap, _ = spectral_gap(metropolis_transition_matrix(g, T))
        push!(tri, (nv(g), gap))
    end
    results["triangular(m,2)"] = tri

    sq = Tuple{Int, Float64}[]
    for m in 2:9
        # println("  square m=", m)
        g = Graphs.grid([m, 2])
        gap, _ = spectral_gap(metropolis_transition_matrix(g, T))
        push!(sq, (nv(g), gap))
    end
    results["square(m,2)"] = sq

    dia = Tuple{Int, Float64}[]
    for m in 2:5
        # println("  diamond m=", m)
        g = diamond_graph(m)
        gap, _ = spectral_gap(metropolis_transition_matrix(g, T))
        push!(dia, (nv(g), gap))
    end
    results["diamond(m)"] = dia

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(2024)

    println("Problem 1: Fullerene ground state energy (simulated annealing)")
    gfull = fullerene_graph()
    best_energy, _ = simulated_annealing_ground_state(gfull)
    println("Ground state energy (best found): ", best_energy)

    println("\nProblem 2.1: Spectral gap vs temperature")
    temps, gaps = gap_vs_temperature()
    for (name, vals) in sort(collect(gaps); by=x -> x[1])
        println(name)
        for (T, gap) in zip(temps, vals)
            @printf("  T=%.1f  gap=%.6f\n", T, gap)
        end
    end

    println("\nProblem 2.2: Spectral gap vs size at T=0.1")
    size_gaps = gap_vs_size()
    for (name, vals) in sort(collect(size_gaps); by=x -> x[1])
        println(name)
        for (n, gap) in vals
            @printf("  N=%d  gap=%.6f\n", n, gap)
        end
    end
end
