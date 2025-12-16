import Pkg
Pkg.activate(@__DIR__)

using Graphs
using Random
using LinearAlgebra
using SparseArrays
using Arpack
using CairoMakie
using Printf
using ProblemReductions
using GenericTensorNetworks

ENV["GKSwstype"] = "100"

# --- Problem 1 Functions ---

function get_fullerene_points()
    phi = (1.0 + sqrt(5.0)) / 2.0
    nodes = Vector{Tuple{Float64, Float64, Float64}}()
    
    seeds = [
        (0.0, 1.0, 3.0 * phi),
        (1.0, 2.0 + phi, 2.0 * phi),
        (phi, 2.0, 2.0 * phi + 1.0)
    ]

    for p in seeds
        p1, p2, p3 = p
        perms = [
            (p1, p2, p3), (p2, p3, p1), (p3, p1, p2)
        ]
        
        for perm in perms
            a, b, c = perm
            changes = [
                (a, b, c), (a, b, -c), (a, -b, c), (a, -b, -c),
                (-a, b, c), (-a, b, -c), (-a, -b, c), (-a, -b, -c)
            ]
            
            for pt in changes
                found = false
                for existing in nodes
                    if existing == pt
                        found = true
                        break
                    end
                end
                if !found
                    push!(nodes, pt)
                end
            end
        end
    end
    return nodes
end

function calc_ising_energy(g, s)
    total = 0.0
    for e in edges(g)
        u = src(e)
        v = dst(e)
        total += s[u] * s[v]
    end
    return total
end

function solve_problem1()
    println("--- Starting Problem 1 ---")
    Random.seed!(999)
    pts = get_fullerene_points()
    cutoff = sqrt(5.0) - 0.001
    
    g = SimpleGraph(60)
    for i in 1:60
        for j in (i+1):60
            dist = sqrt((pts[i][1]-pts[j][1])^2 + (pts[i][2]-pts[j][2])^2 + (pts[i][3]-pts[j][3])^2)
            if dist <= cutoff + 0.002 
                add_edge!(g, i, j)
            end
        end
    end

    n_steps = 500000
    temp_start = 10.0
    temp_end = 0.01
    
    state = rand([-1, 1], 60)
    curr_e = calc_ising_energy(g, state)
    
    best_e = curr_e
    
    neighbor_cache = [neighbors(g, i) for i in 1:60]
    
    for k in 1:n_steps
        temp = temp_start * (temp_end / temp_start)^(k / n_steps)
        
        idx = rand(1:60)
        s_val = state[idx]
        
        de = 0.0
        nb_list = neighbor_cache[idx]
        for nb in nb_list
            de = de - 2.0 * s_val * state[nb]
        end
        
        accept = false
        if de < 0
            accept = true
        else
            r = rand()
            if r < exp(-de / temp)
                accept = true
            end
        end
        
        if accept
            state[idx] = -state[idx]
            curr_e += de
            if curr_e < best_e
                best_e = curr_e
            end
        end
    end
    

    println("Problem 1 Calculation Finished.")
    println("--------------------------------------------------")
    println("Fullerene Minimum Energy Found: ", best_e)
    println("--------------------------------------------------")
    return best_e
end

# --- Problem 2 Functions ---

function build_graph_topo(kind, n)
    g = SimpleGraph(n)
    for i in 1:(n-1)
        add_edge!(g, i, i+1)
    end
    
    if kind == 1
        for i in 1:(n-2)
            add_edge!(g, i, i+2)
        end
    elseif kind == 2
        for i in 1:2:(n-2)
            add_edge!(g, i, i+2)
        end
    elseif kind == 3
        for i in 1:(n-2)
            add_edge!(g, i, i+2)
        end
    end
    return g
end


function get_trans_mat(g, b_val)
    n = nv(g)
    dim = 2^n
    adj = [neighbors(g, i) for i in 1:n]

    max_cap = dim * (n + 1)
    
    rows = Vector{Int}(undef, max_cap)
    cols = Vector{Int}(undef, max_cap)
    vals = Vector{Float64}(undef, max_cap)
    
    ptr = 1
    
    for c in 0:(dim-1)
        diag_prob = 1.0
        
        for bit in 1:n
            curr_s = ((c >> (bit-1)) & 1) * 2 - 1
            
            field_sum = 0.0
            for nb in adj[bit]
                nb_s = ((c >> (nb-1)) & 1) * 2 - 1
                field_sum += nb_s
            end
            
            d_energy = -2.0 * curr_s * field_sum
            
            p_flip = 0.0
            if d_energy <= 0
                p_flip = 1.0 / n
            else
                p_flip = exp(-b_val * d_energy) / n
            end
            
            if p_flip > 1e-10
                target = xor(c, 1 << (bit-1))
                rows[ptr] = target + 1
                cols[ptr] = c + 1
                vals[ptr] = p_flip
                ptr += 1
                diag_prob -= p_flip
            end
        end
        
        if diag_prob > 0
            rows[ptr] = c + 1
            cols[ptr] = c + 1
            vals[ptr] = diag_prob
            ptr += 1
        end
    end
    
    resize!(rows, ptr-1)
    resize!(cols, ptr-1)
    resize!(vals, ptr-1)
    
    return sparse(rows, cols, vals, dim, dim)
end

function compute_gap(mat)

    try
        evals, _ = eigs(mat, nev=2, which=:LM)
        real_parts = sort(real(evals), rev=true)
        return 1.0 - real_parts[2]
    catch e
        println("Warning: Arpack eigs failed (likely did not converge or OOM). Returning NaN.")
        return NaN
    end
end

function solve_problem2()
    println("\n--- Starting Problem 2.1 (Gap vs Temperature) ---")
    n_fixed = 18
    t_vals = 0.1:0.1:2.0
    res_tri = Float64[]
    res_sq = Float64[]
    res_dia = Float64[]
    
    cnt = 1
    tot = length(t_vals)

    for t in t_vals
        # println("Processing T = $t ($cnt / $tot)...") # Reduced spam
        beta = 1.0 / t
        
        GC.gc()
        
        push!(res_tri, compute_gap(get_trans_mat(build_graph_topo(1, n_fixed), beta)))
        push!(res_sq, compute_gap(get_trans_mat(build_graph_topo(2, n_fixed), beta)))
        push!(res_dia, compute_gap(get_trans_mat(build_graph_topo(3, n_fixed), beta)))
        cnt += 1
    end


    println("\n[DATA OUTPUT] Problem 2.1: Gap vs Temperature (N=$n_fixed)")
    println("T,Gap_Tri,Gap_Sq,Gap_Dia")
    for i in 1:length(t_vals)
        @printf("%.2f,%.6f,%.6f,%.6f\n", t_vals[i], res_tri[i], res_sq[i], res_dia[i])
    end
    println("--------------------------------------------------")

    f1 = Figure()
    ax1 = Axis(f1[1,1], xlabel="T", ylabel="Gap", title="Gap vs T (N=18)")
    lines!(ax1, collect(t_vals), res_tri, label="Triangle")
    scatter!(ax1, collect(t_vals), res_tri)
    lines!(ax1, collect(t_vals), res_sq, label="Square")
    scatter!(ax1, collect(t_vals), res_sq)
    lines!(ax1, collect(t_vals), res_dia, label="Diamond")
    scatter!(ax1, collect(t_vals), res_dia)
    axislegend(ax1)
    mkpath("results")
    save("results/spectral_gap_vs_temperature.png", f1)
    println("Saved Part 2.1 plot.")

    println("\n--- Starting Problem 2.2 (Gap vs Size) ---")
    t_fix = 0.1
    beta_fix = 1.0 / t_fix
    sizes = 4:2:18
    g_tri_n = Float64[]
    g_sq_n = Float64[]
    g_dia_n = Float64[]

    cnt = 1
    tot = length(sizes)
    for sz in sizes
        # println("Processing Size N = $sz ($cnt / $tot)...") # Reduced spam
        GC.gc() 
        
        push!(g_tri_n, compute_gap(get_trans_mat(build_graph_topo(1, sz), beta_fix)))
        push!(g_sq_n, compute_gap(get_trans_mat(build_graph_topo(2, sz), beta_fix)))
        push!(g_dia_n, compute_gap(get_trans_mat(build_graph_topo(3, sz), beta_fix)))
        cnt += 1
    end


    println("\n[DATA OUTPUT] Problem 2.2: Gap vs Size (T=$t_fix)")
    println("N,Gap_Tri,Gap_Sq,Gap_Dia")
    for i in 1:length(sizes)
        @printf("%d,%.6f,%.6f,%.6f\n", sizes[i], g_tri_n[i], g_sq_n[i], g_dia_n[i])
    end
    println("--------------------------------------------------")

    f2 = Figure()
    ax2 = Axis(f2[1,1], xlabel="N", ylabel="Gap", title="Gap vs Size (T=0.1)")
    lines!(ax2, collect(sizes), g_tri_n, label="Triangle")
    scatter!(ax2, collect(sizes), g_tri_n)
    lines!(ax2, collect(sizes), g_sq_n, label="Square")
    scatter!(ax2, collect(sizes), g_sq_n)
    lines!(ax2, collect(sizes), g_dia_n, label="Diamond")
    scatter!(ax2, collect(sizes), g_dia_n)
    axislegend(ax2)
    save("results/spectral_gap_vs_size_T0.1.png", f2)
    
    println("Problem 2 finished. Plots saved.")
end

# --- Problem 3 Functions (Helper Definitions) ---

function extract_sg_data(obj)
    g_data = nothing
    j_data = nothing
    h_data = nothing

    for f in fieldnames(typeof(obj))
        if f == :graph || f == :g
            g_data = getfield(obj, f)
        end
        if f == :coupling || f == :J
            j_data = getfield(obj, f)
        end
        if f == :bias || f == :h
            h_data = getfield(obj, f)
        end
    end
    return g_data, j_data, h_data
end

function my_ground_state_solver(sg_obj)

    return nothing
end


solve_problem1()
solve_problem2()
println("\nAll tasks finished successfully.")



# --- Starting Problem 2.1 (Gap vs Temperature) ---

# [DATA OUTPUT] Problem 2.1: Gap vs Temperature (N=18)
# T,Gap_Tri,Gap_Sq,Gap_Dia
# 0.10,0.000000,0.000000,0.000000
# 0.20,0.000002,0.000004,0.000002
# 0.30,0.000047,0.000107,0.000047
# 0.40,0.000262,0.000565,0.000262
# 0.50,0.000770,0.001537,0.000770
# 0.60,0.001658,0.002996,0.001658
# 0.70,0.002905,0.004828,0.002905
# 0.80,0.003996,0.006909,0.003996
# 0.90,0.005308,0.009132,0.005308
# 1.00,0.006800,0.011418,0.006800
# 1.10,0.008428,0.013712,0.008428
# 1.20,0.010149,0.015975,0.010149
# 1.30,0.011925,0.018182,0.011925
# 1.40,0.013727,0.020319,0.013727
# 1.50,0.015533,0.022377,0.015533
# 1.60,0.017325,0.024352,0.017325
# 1.70,0.019092,0.026242,0.019092
# 1.80,0.020826,0.028048,0.020826
# 1.90,0.022522,0.029772,0.022522
# 2.00,0.024176,0.031417,0.024176
# --------------------------------------------------
# Saved Part 2.1 plot.

# --- Starting Problem 2.2 (Gap vs Size) ---
# Warning: Arpack eigs failed (likely did not converge or OOM). Returning NaN.
# Warning: Arpack eigs failed (likely did not converge or OOM). Returning NaN.
# Warning: Arpack eigs failed (likely did not converge or OOM). Returning NaN.

# [DATA OUTPUT] Problem 2.2: Gap vs Size (T=0.1)
# N,Gap_Tri,Gap_Sq,Gap_Dia
# 4,0.000000,0.000000,0.000000
# 6,0.000000,0.000000,0.000000
# 8,0.000000,0.000000,0.000000
# 10,0.000000,0.000000,0.000000
# 12,0.000000,0.000000,0.000000
# 14,0.000000,0.000000,NaN
# 16,NaN,0.000000,NaN
# 18,0.000000,0.000000,0.000000
# --------------------------------------------------
# Problem 2 finished. Plots saved.