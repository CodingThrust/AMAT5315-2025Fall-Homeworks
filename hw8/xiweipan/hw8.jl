using LinearAlgebra
using Graphs
using Printf

struct Factor
    vars::Vector{Int}
    table::Vector{Float64}
end

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

function edge_factor(i, j, beta)
    table = Vector{Float64}(undef, 4)
    for mask in 0:3
        si = (mask & 0x1) == 0 ? 1 : -1
        sj = (mask & 0x2) == 0 ? 1 : -1
        table[mask + 1] = exp(-beta * si * sj)
    end
    return Factor([i, j], table)
end

function multiply(f1::Factor, f2::Factor)
    vars = copy(f1.vars)
    for v in f2.vars
        if v ∉ vars
            push!(vars, v)
        end
    end
    k = length(vars)
    table = zeros(Float64, 1 << k)

    pos = Dict{Int, Int}()
    for (idx, v) in enumerate(vars)
        pos[v] = idx
    end
    f1pos = [pos[v] for v in f1.vars]
    f2pos = [pos[v] for v in f2.vars]

    for mask in 0:(1 << k) - 1
        idx1 = 0
        for (i, p) in enumerate(f1pos)
            bit = (mask >> (p - 1)) & 0x1
            idx1 |= bit << (i - 1)
        end
        idx2 = 0
        for (i, p) in enumerate(f2pos)
            bit = (mask >> (p - 1)) & 0x1
            idx2 |= bit << (i - 1)
        end
        table[mask + 1] = f1.table[idx1 + 1] * f2.table[idx2 + 1]
    end

    return Factor(vars, table)
end

function sum_out(f::Factor, v::Int)
    idx = findfirst(==(v), f.vars)
    idx === nothing && return f
    new_vars = [x for x in f.vars if x != v]
    k = length(f.vars)
    k2 = length(new_vars)
    table = zeros(Float64, 1 << k2)

    for mask in 0:(1 << k) - 1
        new_mask = 0
        bitpos = 0
        for p in 1:k
            if p == idx
                continue
            end
            bit = (mask >> (p - 1)) & 0x1
            new_mask |= bit << bitpos
            bitpos += 1
        end
        table[new_mask + 1] += f.table[mask + 1]
    end

    return Factor(new_vars, table)
end

function eliminate_variable(factors, v)
    idxs = findall(f -> v in f.vars, factors)
    isempty(idxs) && return factors
    combined = factors[idxs[1]]
    for k in idxs[2:end]
        combined = multiply(combined, factors[k])
    end
    reduced = sum_out(combined, v)
    remaining = [factors[i] for i in 1:length(factors) if i ∉ idxs]
    push!(remaining, reduced)
    return remaining
end

function minfill_order(g)
    n = nv(g)
    adj = [Set(neighbors(g, i)) for i in 1:n]
    remaining = Set(1:n)
    order = Int[]

    while !isempty(remaining)
        best = 0
        best_fill = typemax(Int)
        best_deg = typemax(Int)
        for v in remaining
            neigh = intersect(adj[v], remaining)
            neigh_vec = collect(neigh)
            fill = 0
            for i in 1:length(neigh_vec)
                for j in (i + 1):length(neigh_vec)
                    u = neigh_vec[i]
                    w = neigh_vec[j]
                    if !(w in adj[u])
                        fill += 1
                    end
                end
            end
            deg = length(neigh_vec)
            if fill < best_fill || (fill == best_fill && deg < best_deg)
                best = v
                best_fill = fill
                best_deg = deg
            end
        end

        neigh = intersect(adj[best], remaining)
        neigh_vec = collect(neigh)
        for i in 1:length(neigh_vec)
            for j in (i + 1):length(neigh_vec)
                u = neigh_vec[i]
                w = neigh_vec[j]
                if !(w in adj[u])
                    push!(adj[u], w)
                    push!(adj[w], u)
                end
            end
        end

        push!(order, best)
        delete!(remaining, best)
    end

    return order
end

function partition_function(g, beta; order=minfill_order(g))
    factors = Factor[]
    for e in edges(g)
        i = src(e)
        j = dst(e)
        push!(factors, edge_factor(i, j, beta))
    end

    for v in order
        factors = eliminate_variable(factors, v)
    end

    combined = factors[1]
    for k in 2:length(factors)
        combined = multiply(combined, factors[k])
    end
    @assert isempty(combined.vars)
    return combined.table[1]
end

function scan_partition_function(; beta_start=0.1, beta_end=2.0, beta_step=0.1)
    g = fullerene_graph()
    order = minfill_order(g)
    betas = collect(beta_start:beta_step:beta_end)
    Zs = Float64[]
    for beta in betas
        Z = partition_function(g, beta; order=order)
        push!(Zs, Z)
    end
    return betas, Zs
end

if abspath(PROGRAM_FILE) == @__FILE__
    betas, Zs = scan_partition_function()
    for (beta, Z) in zip(betas, Zs)
        @printf("beta=%.1f  Z=%.12e\n", beta, Z)
    end
end
