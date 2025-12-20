###############################
# hw7.jl  —  AMAT5315 HW7
###############################

import Pkg
Pkg.activate(@__DIR__)

using Graphs
using Random
using LinearAlgebra
using SparseArrays
using Arpack
using CairoMakie
using ProblemReductions
using GenericTensorNetworks
using Printf

ENV["GKSwstype"] = "100"  # 让 CairoMakie 在无图形界面环境下也能画图

############################################################
# Problem 1  —  Fullerene + anti-ferro Ising + SA
############################################################

"""
生成 C60 fullerene 的 60 个顶点坐标（题目给的构造）。
"""
function fullerene()
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

"""
反铁磁 Ising 能量:
    H(σ) = ∑_{<ij>} σ_i σ_j,  σ_i ∈ {±1}
图的边集合由 graph 给出。
"""
function ising_energy(graph::AbstractGraph, spins::AbstractVector{<:Integer})
    energy = 0.0
    for e in edges(graph)
        energy += spins[src(e)] * spins[dst(e)]
    end
    return energy
end

"""
在给定图 graph 上，对反铁磁 Ising 模型做模拟退火，寻找近似基态能。

参数：
- graph::AbstractGraph : 图
- steps::Int           : Monte Carlo 更新步数
- T0, Tf               : 温度从 T0 退火到 Tf（指数退火）

返回：
- best_energy::Float64
- best_spins::Vector{Int}
"""
function simulated_annealing(
    graph::AbstractGraph;
    steps::Int = 1_000_000,
    T0::Float64 = 10.0,
    Tf::Float64 = 0.01,
)
    n = nv(graph)
    # 随机初态
    spins = rand([-1, 1], n)
    energy = ising_energy(graph, spins)

    best_energy = energy
    best_spins = copy(spins)

    # 预先计算邻居列表加速
    neighbor_lists = [collect(neighbors(graph, i)) for i in 1:n]

    for step in 1:steps
        # 指数退火：T(step) = T0 * (Tf/T0)^(step/steps)
        T = T0 * (Tf / T0)^(step / steps)

        # 随机选择一个自旋并尝试翻转
        site = rand(1:n)

        # 计算翻转 site 处自旋的 ΔE = E_new - E_old
        flip_ΔE = 0.0
        for nb in neighbor_lists[site]
            # 每条边 (i, j) 的能量项为 σ_i σ_j
            # 翻转 σ_i → -σ_i 时，该项变化：
            #   old = σ_i σ_j
            #   new = (-σ_i) σ_j = -σ_i σ_j
            #   ΔE_edge = new - old = -2 σ_i σ_j
            flip_ΔE -= 2 * spins[site] * spins[nb]
        end

        # Metropolis 接受率
        if flip_ΔE < 0 || rand() < exp(-flip_ΔE / T)
            spins[site] *= -1
            energy += flip_ΔE

            if energy < best_energy
                best_energy = energy
                copy!(best_spins, spins)
            end
        end
    end

    return best_energy, best_spins
end

"""
求解 Problem 1:
- 构造 fullerene 60 点；
- 根据题目要求用 UnitDiskGraph(radius = sqrt(5)) 连边；
- 在该图上做模拟退火并打印近似基态能。
"""
function solve_problem1()
    Random.seed!(1234)  # 可复现

    points = fullerene()
    # 题目中给出的 fullerene 连接方式：距离 < sqrt(5) 视为一条边
    fullerene_graph = UnitDiskGraph(points, sqrt(5))

    best_E, best_spins = simulated_annealing(
        fullerene_graph;
        steps = 500_000,
        T0 = 10.0,
        Tf = 0.01,
    )

    @info "Problem 1: Fullerene approximate ground-state energy" best_E
    return best_E, best_spins
end

############################################################
# Problem 2  —  三种拓扑 + Metropolis 链 + 谱隙
############################################################

"Triangle topology graph（按题目图 1）：链 + 所有 i 连到 i+1 和 i+2。"
function triangle_graph(n::Int)
    g = SimpleGraph(n)
    for i in 1:n-1
        add_edge!(g, i, i+1)
    end
    for i in 1:n-2
        add_edge!(g, i, i+2)
    end
    return g
end

"Square topology graph：链 + 只有奇数点 i 连到 i+2。"
function square_graph(n::Int)
    g = SimpleGraph(n)
    for i in 1:n-1
        add_edge!(g, i, i+1)
    end
    for i in 1:2:n-2
        add_edge!(g, i, i+2)
    end
    return g
end

"Diamond topology graph：链 + 每个点 i（除了最后两个）都连到 i+2。"
function diamond_graph(n::Int)
    g = SimpleGraph(n)
    for i in 1:n-1
        add_edge!(g, i, i+1)
    end
    for i in 1:n-2
        add_edge!(g, i, i+2)
    end
    return g
end

"""
给定状态向量 state (σ_i = ±1) 和要翻转的 site，
在反铁磁 Ising 能量 H = ∑ σ_i σ_j 下，
计算翻转该点自旋的 ΔE = E_new - E_old。
"""
function calc_energy_diff(
    state::AbstractVector{<:Integer},
    site::Int,
    neighbor_lists::Vector{Vector{Int}},
)
    ΔE = 0.0
    σi = state[site]
    for nb in neighbor_lists[site]
        # ΔE_edge = -2 σ_i σ_j
        ΔE -= 2 * σi * state[nb]
    end
    return ΔE
end

"""
构造反铁磁 Ising Metropolis Markov 链的转移矩阵（稀疏，列随机）：
- 图 g 上有 N 个自旋；
- 状态空间为 {±1}^N，总数 2^N；
- 每一步随机选择一个自旋尝试翻转：
    P(σ → σ^flip_k) = (1/N) * min(1, exp(-β ΔE_k))；
  留在原状态的概率补齐到 1。

返回：P::SparseMatrixCSC，总维度 2^N × 2^N。
"""
function transition_matrix_sparse(g::AbstractGraph, β::Float64)
    N = nv(g)
    total_states = 2^N

    # 预分配每一列 N 个 off-diagonal + 1 个对角元素
    I = Vector{Int}(undef, total_states * (N + 1))
    J = Vector{Int}(undef, total_states * (N + 1))
    V = Vector{Float64}(undef, total_states * (N + 1))
    idx = 1

    # 邻居表
    neighbor_lists = [collect(neighbors(g, i)) for i in 1:N]

    # 状态编码：用 0...(2^N-1) 对应二进制，自最低位起表示自旋是否为 +1
    readbit(cfg::Int, i::Int) = (cfg >> (i - 1)) & 1
    int2cfg(cfg::Int) = [2 * readbit(cfg, i) - 1 for i in 1:N]

    for j_col in 0:total_states-1
        state_j = int2cfg(j_col)
        sum_prob = 0.0

        # 对每个自旋尝试翻转（对应 N 个可能跃迁）
        for k in 1:N
            # 翻 k-th 位：用 XOR 翻转那位比特
            i_row = j_col ⊻ (1 << (k - 1))

            ΔE = calc_energy_diff(state_j, k, neighbor_lists)
            prob = min(1.0, exp(-β * ΔE)) / N

            I[idx] = i_row + 1
            J[idx] = j_col + 1
            V[idx] = prob
            idx += 1

            sum_prob += prob
        end

        # 留在原状态的概率
        I[idx] = j_col + 1
        J[idx] = j_col + 1
        V[idx] = 1.0 - sum_prob
        idx += 1
    end

    return sparse(I[1:idx-1], J[1:idx-1], V[1:idx-1], total_states, total_states)
end

"""
谱隙定义为 gap = 1 - λ₂，其中 λ₁ = 1 ≥ λ₂ ≥ ...
P 可以是行随机或列随机的 Markov 矩阵，特征值集合一样。
"""
function spectral_gap(P::SparseMatrixCSC)
    vals, _ = eigs(P; nev = 2, which = :LM)  # largest magnitude
    vals = sort(real(vals); rev = true)      # 降序
    return 1.0 - vals[2]
end

"""
在一系列温度 T_list 上计算给定图 g 的谱隙。
"""
function run_vs_temperature(g::AbstractGraph, T_list::Vector{Float64})
    gaps = similar(T_list)
    for (i, T) in pairs(T_list)
        β = 1.0 / T
        P = transition_matrix_sparse(g, β)
        gaps[i] = spectral_gap(P)
        @info "N=$(nv(g))  T=$(round(T, digits=3))  β=$(round(β, digits=3))  gap=$(round(gaps[i], digits=6))"
    end
    return gaps
end

"""
在固定温度 T 下，随系统尺寸 N_list 分析谱隙，
graph_constructor 是 triangle_graph / square_graph / diamond_graph 等。
"""
function run_vs_size(
    graph_constructor::Function,
    N_list::AbstractVector{<:Integer},
    T::Float64,
)
    β = 1.0 / T
    gaps = Vector{Float64}(undef, length(N_list))
    for (i, N) in pairs(N_list)
        g = graph_constructor(N)
        P = transition_matrix_sparse(g, β)
        gaps[i] = spectral_gap(P)
        @info "Topo=$(graph_constructor)  N=$N  T=$(round(T, digits=3))  gap=$(round(gaps[i], digits=6))"
    end
    return gaps
end

"""
画一张图并保存到 filename（PNG），用于题 2.1 和 2.2。
"""
function plot_and_save(x, y_list, labels, xlabel_str, ylabel_str, title_str, filename)
    fig = Figure()
    ax = Axis(fig[1, 1],
              xlabel = xlabel_str,
              ylabel = ylabel_str,
              title = title_str)

    for (i, y) in enumerate(y_list)
        scatter!(ax, x, y, markersize = 6, label = labels[i])
        lines!(ax, x, y)
    end
    axislegend(ax, position = :lt)

    dir = dirname(filename)
    if dir != "."
        mkpath(dir)
    end

    CairoMakie.save(filename, fig)
    @info "Saved figure to $filename"
end

"""
求解 Problem 2：
- 2.1 固定 N=18，T 从 0.1 到 2.0，三种拓扑的谱隙 vs T；
- 2.2 固定 T=0.1，N 从 4 到 18（偶数），三种拓扑的谱隙 vs N；
- 结果画图并保存为 PNG。
"""
function solve_problem2()
    # ---------- 2.1: spectral gap vs temperature ----------
    N = 18
    g_tri = triangle_graph(N)
    g_sq  = square_graph(N)
    g_dia = diamond_graph(N)

    T_list = collect(0.1:0.1:2.0)
    @info "Problem 2.1: spectral gap vs T, N=$N"

    gaps_tri = run_vs_temperature(g_tri, T_list)
    gaps_sq  = run_vs_temperature(g_sq,  T_list)
    gaps_dia = run_vs_temperature(g_dia, T_list)

    plot_and_save(
        T_list,
        [gaps_tri, gaps_sq, gaps_dia],
        ["Triangle", "Square", "Diamond"],
        "Temperature T",
        "Spectral Gap  (1 - λ₂)",
        "Spectral Gap vs Temperature (N = $N)",
        "results/spectral_gap_vs_temperature.png",
    )

    # ---------- 2.2: spectral gap vs system size ----------
    T_fixed = 0.1
    N_list = collect(4:2:18)  # 4, 6, ..., 18

    @info "Problem 2.2: spectral gap vs N, T=$T_fixed"

    gaps_tri_size = run_vs_size(triangle_graph, N_list, T_fixed)
    gaps_sq_size  = run_vs_size(square_graph,   N_list, T_fixed)
    gaps_dia_size = run_vs_size(diamond_graph,  N_list, T_fixed)

    plot_and_save(
        N_list,
        [gaps_tri_size, gaps_sq_size, gaps_dia_size],
        ["Triangle", "Square", "Diamond"],
        "System size N",
        "Spectral Gap  (1 - λ₂)",
        "Spectral Gap vs System Size (T = $T_fixed)",
        "results/spectral_gap_vs_size_T0.1.png",
    )

    return (T_list, gaps_tri, gaps_sq, gaps_dia,
            N_list, gaps_tri_size, gaps_sq_size, gaps_dia_size)
end

############################################################
# Problem 3 — Spin glass ground state via Parallel Tempering
############################################################

# ---------- 辅助：安全读取 SpinGlass 字段（兼容不同定义） ----------

# 尝试按多个候选名字取字段，如失败则报错
function _getfield_any(sg, names::NTuple{N,Symbol}) where {N}
    fns = fieldnames(typeof(sg))
    for nm in names
        if nm in fns
            return getfield(sg, nm)
        end
    end
    error("SpinGlass type $(typeof(sg)) does not have any of fields $(names)")
end

# 从 GenericTensorNetworks.SpinGlass 或其它兼容类型提取 graph / coupling / bias
_get_graph(sg)    = _getfield_any(sg, (:graph, :g))
_get_coupling(sg) = _getfield_any(sg, (:coupling, :J))
_get_bias(sg)     = _getfield_any(sg, (:bias, :h))

# ---------- 并行回火核心代码 ----------

"生成一个随机的 ±1 自旋配置"
function rand_spins(n::Int)
    s = Vector{Int8}(undef, n)
    @inbounds for i in 1:n
        s[i] = rand(Bool) ? Int8(1) : Int8(-1)
    end
    return s
end

"""
为 SpinGlass sg 预处理：
- 提取 graph / coupling / bias
- 构造：每个点 i 的邻居列表 neighbors[i] 及对应耦合 Jnb[i]
- 校准我们写的局部能量和库里 energy(sg, ·) 的能量符号（只关心正负号）

返回：
- neighbors::Vector{Vector{Int}}
- Jnb::Vector{Vector{Float64}}
- h::Vector{Float64}
- signcorr::Float64   # = ±1，用来保证 Metropolis 的 ΔE 和 energy(sg,·) 一致
"""
function build_local_structures(sg)
    g      = _get_graph(sg)              # SimpleGraph
    J_edge = _get_coupling(sg)           # Vector, 长度 ne(g)
    h      = Float64.(_get_bias(sg))     # Vector, 长度 nv(g)

    n = nv(g)

    neighbors = [Int[] for _ in 1:n]
    Jnb       = [Float64[] for _ in 1:n]

    # 假设 coupling 的顺序与 edges(g) 一一对应
    idx = 1
    for e in edges(g)
        i = src(e)
        j = dst(e)
        Jij = float(J_edge[idx])
        push!(neighbors[i], j); push!(Jnb[i], Jij)
        push!(neighbors[j], i); push!(Jnb[j], Jij)
        idx += 1
    end

    # 局部能量：H_loc(σ) = ∑_{i<j} J_ij σ_i σ_j + ∑_i h_i σ_i
    local_energy = function (s::AbstractVector{<:Integer})
        E = 0.0
        @inbounds for i in 1:n
            E += h[i] * s[i]
            ni = neighbors[i]; Ji = Jnb[i]
            len = length(ni)
            for t in 1:len
                j = ni[t]
                if j > i
                    E += Ji[t] * s[i] * s[j]
                end
            end
        end
        return E
    end

    # 用两组随机自旋，校准我们写的能量和库里的 energy(sg,·) 之间的正负号
    s1 = rand_spins(n)
    s2 = rand_spins(n)

    El1 = local_energy(s1)
    El2 = local_energy(s2)
    Er1 = energy(sg, s1)
    Er2 = energy(sg, s2)

    diff_local = El2 - El1
    diff_real  = Er2 - Er1

    # diff_local ≈ 0 的概率极低，如发生就再随机几次
    tries = 0
    while diff_local == 0 && tries < 5
        s2 = rand_spins(n)
        El2 = local_energy(s2)
        Er2 = energy(sg, s2)
        diff_local = El2 - El1
        diff_real  = Er2 - Er1
        tries += 1
    end

    signcorr = (diff_local * diff_real < 0) ? -1.0 : 1.0

    return neighbors, Jnb, h, signcorr
end

"单点翻转的局部能量差（只根据 neighbors / Jnb / h 计算）"
@inline function deltaE_local(
    s::AbstractVector{<:Integer},
    i::Int,
    neighbors::Vector{Vector{Int}},
    Jnb::Vector{Vector{Float64}},
    h::Vector{Float64},
)
    σi = s[i]
    field = h[i]
    ni = neighbors[i]; Ji = Jnb[i]
    len = length(ni)
    @inbounds @simd for t in 1:len
        j = ni[t]
        field += Ji[t] * s[j]
    end
    # 对 Hamiltonian: H = ∑_{i<j} J_ij σ_i σ_j + ∑_i h_i σ_i
    # 翻转 σ_i -> -σ_i 时，ΔE_local = H_new - H_old = -2 σ_i * (h_i + Σ_j J_ij σ_j)
    return -2.0 * σi * field
end

"根据 neighbors / Jnb / h 计算局部能量（和 build_local_structures 里一致）"
function local_energy_from_struct(
    s::AbstractVector{<:Integer},
    neighbors::Vector{Vector{Int}},
    Jnb::Vector{Vector{Float64}},
    h::Vector{Float64},
)
    n = length(s)
    E = 0.0
    @inbounds for i in 1:n
        E += h[i] * s[i]
        ni = neighbors[i]; Ji = Jnb[i]
        len = length(ni)
        for t in 1:len
            j = ni[t]
            if j > i
                E += Ji[t] * s[i] * s[j]
            end
        end
    end
    return E
end

"""
并行回火求解 SpinGlass 的近似基态。

返回：一组自旋配置 σ::Vector{Int8}，使得 energy(sg, σ) 尽量小。
"""
function my_ground_state_solver(sg::SpinGlass)
    # 预处理图结构和局部能量
    neighbors, Jnb, h, signcorr = build_local_structures(sg)

    n = length(h)

    # ====== 并行回火参数（可根据时间调得大/小一点） ======
    n_replicas = 16          # 温度数
    T_min      = 0.3
    T_max      = 5.0
    n_sweeps   = 1200        # 全局 sweep 次数（每次 sweep 包含 n 次单点更新）

    # 几何级的温度序列 [T_min, T_max]
    Ts = [T_min * (T_max / T_min)^((r - 1) / (n_replicas - 1)) for r in 1:n_replicas]
    betas = 1.0 ./ Ts

    # 初始化各个温度下的副本
    spins  = [rand_spins(n) for _ in 1:n_replicas]
    Elocal = [local_energy_from_struct(spins[r], neighbors, Jnb, h)
              for r in 1:n_replicas]

    # 记录目前为止的最好解（真正的 energy(sg, ·)）
    best_spins = copy(spins[1])
    best_E_real = energy(sg, best_spins)

    # ====== 主循环：并行回火 ======
    for sweep in 1:n_sweeps
        # 1）对每个温度的副本做一轮 Metropolis 更新
        @inbounds for r in 1:n_replicas
            β = betas[r]
            s = spins[r]
            E_loc = Elocal[r]

            for _ in 1:n
                i = rand(1:n)
                ΔE_loc = deltaE_local(s, i, neighbors, Jnb, h)
                # 真实能量差：ΔE_real = signcorr * ΔE_loc
                Δ = signcorr * ΔE_loc
                if Δ <= 0.0 || rand() < exp(-β * Δ)
                    s[i] = -s[i]
                    E_loc += ΔE_loc       # 只更新局部能量（未乘 signcorr）
                end
            end

            Elocal[r] = E_loc
        end

        # 2）每隔几轮做一次 replica 交换
        if sweep % 5 == 0
            @inbounds for r in 1:n_replicas-1
                β1 = betas[r]
                β2 = betas[r + 1]

                # 我们的 Elocal 是“局部能量”，真实能量乘个 signcorr 即可
                E1 = signcorr * Elocal[r]
                E2 = signcorr * Elocal[r + 1]

                # 交换接受率：min(1, exp( (β1-β2)*(E2-E1) ))
                Δswap = (β1 - β2) * (E2 - E1)
                if Δswap <= 0.0 || rand() < exp(-Δswap)
                    spins[r], spins[r + 1] = spins[r + 1], spins[r]
                    Elocal[r], Elocal[r + 1] = Elocal[r + 1], Elocal[r]
                end
            end
        end

        # 3）定期用最低温副本更新“当前最好解”
        if sweep % 10 == 0
            cand = spins[1]  # 温度最低的那条链
            E_cand = energy(sg, cand)
            if E_cand < best_E_real
                best_E_real = E_cand
                best_spins = copy(cand)
            end
        end
    end

    return best_spins
end

############################################################
# main: 统一调用前两道题（Problem 3 老师会单独测）
############################################################

function main()
    # Problem 1
    best_E, _ = solve_problem1()
    println("Problem 1 — Fullerene approximate ground-state energy ≈ $best_E")

    # Problem 2
    T_list, gaps_tri, gaps_sq, gaps_dia,
    N_list, gaps_tri_size, gaps_sq_size, gaps_dia_size = solve_problem2()

    println("Problem 2.1 — Spectral gaps vs T (N = 18):")
    println("  T       Triangle      Square        Diamond")
    for i in eachindex(T_list)
        @printf("  %-5.2f  %10.6f  %10.6f  %10.6f\n",
                T_list[i], gaps_tri[i], gaps_sq[i], gaps_dia[i])
    end

    println("\nProblem 2.2 — Spectral gaps vs N (T = 0.1):")
    println("  N       Triangle      Square        Diamond")
    for i in eachindex(N_list)
        @printf("  %-3d    %10.6f  %10.6f  %10.6f\n",
                N_list[i], gaps_tri_size[i], gaps_sq_size[i], gaps_dia_size[i])
    end

    println("\nFigures saved to:")
    println("  results/spectral_gap_vs_temperature.png")
    println("  results/spectral_gap_vs_size_T0.1.png")

    # Problem 3: 老师会单独调用 my_ground_state_solver(sg) 来评分
end

# 执行 main
main()

#   Activating project at `~/Desktop/Desktop/HKUST/scientific-computing/AMAT5315-2025Fall-Homeworks/hw7/JizheLai`
# ┌ Info: Problem 1: Fullerene approximate ground-state energy
# └   best_E = -66.0
# Problem 1 — Fullerene approximate ground-state energy ≈ -66.0
# [ Info: Problem 2.1: spectral gap vs T, N=18
# [ Info: N=18  T=0.1  β=10.0  gap=0.0
# [ Info: N=18  T=0.2  β=5.0  gap=2.0e-6
# [ Info: N=18  T=0.3  β=3.333  gap=4.7e-5
# [ Info: N=18  T=0.4  β=2.5  gap=0.000262
# [ Info: N=18  T=0.5  β=2.0  gap=0.00077
# [ Info: N=18  T=0.6  β=1.667  gap=0.001658
# [ Info: N=18  T=0.7  β=1.429  gap=0.002905
# [ Info: N=18  T=0.8  β=1.25  gap=0.003996
# [ Info: N=18  T=0.9  β=1.111  gap=0.005308
# [ Info: N=18  T=1.0  β=1.0  gap=0.0068
# [ Info: N=18  T=1.1  β=0.909  gap=0.008428
# [ Info: N=18  T=1.2  β=0.833  gap=0.010149
# [ Info: N=18  T=1.3  β=0.769  gap=0.011925
# [ Info: N=18  T=1.4  β=0.714  gap=0.013727
# [ Info: N=18  T=1.5  β=0.667  gap=0.015533
# [ Info: N=18  T=1.6  β=0.625  gap=0.017325
# [ Info: N=18  T=1.7  β=0.588  gap=0.019092
# [ Info: N=18  T=1.8  β=0.556  gap=0.020826
# [ Info: N=18  T=1.9  β=0.526  gap=0.022522
# [ Info: N=18  T=2.0  β=0.5  gap=0.024176
# [ Info: N=18  T=0.1  β=10.0  gap=0.0
# [ Info: N=18  T=0.2  β=5.0  gap=4.0e-6
# [ Info: N=18  T=0.3  β=3.333  gap=0.000107
# [ Info: N=18  T=0.4  β=2.5  gap=0.000565
# [ Info: N=18  T=0.5  β=2.0  gap=0.001537
# [ Info: N=18  T=0.6  β=1.667  gap=0.002996
# [ Info: N=18  T=0.7  β=1.429  gap=0.004828
# [ Info: N=18  T=0.8  β=1.25  gap=0.006909
# [ Info: N=18  T=0.9  β=1.111  gap=0.009132
# [ Info: N=18  T=1.0  β=1.0  gap=0.011418
# [ Info: N=18  T=1.1  β=0.909  gap=0.013712
# [ Info: N=18  T=1.2  β=0.833  gap=0.015975
# [ Info: N=18  T=1.3  β=0.769  gap=0.018182
# [ Info: N=18  T=1.4  β=0.714  gap=0.020319
# [ Info: N=18  T=1.5  β=0.667  gap=0.022377
# [ Info: N=18  T=1.6  β=0.625  gap=0.024352
# [ Info: N=18  T=1.7  β=0.588  gap=0.026242
# [ Info: N=18  T=1.8  β=0.556  gap=0.028048
# [ Info: N=18  T=1.9  β=0.526  gap=0.029772
# [ Info: N=18  T=2.0  β=0.5  gap=0.031417
# [ Info: N=18  T=0.1  β=10.0  gap=0.0
# [ Info: N=18  T=0.2  β=5.0  gap=2.0e-6
# [ Info: N=18  T=0.3  β=3.333  gap=4.7e-5
# [ Info: N=18  T=0.4  β=2.5  gap=0.000262
# [ Info: N=18  T=0.5  β=2.0  gap=0.00077
# [ Info: N=18  T=0.6  β=1.667  gap=0.001658
# [ Info: N=18  T=0.7  β=1.429  gap=0.002905
# [ Info: N=18  T=0.8  β=1.25  gap=0.003996
# [ Info: N=18  T=0.9  β=1.111  gap=0.005308
# [ Info: N=18  T=1.0  β=1.0  gap=0.0068
# [ Info: N=18  T=1.1  β=0.909  gap=0.008428
# [ Info: N=18  T=1.2  β=0.833  gap=0.010149
# [ Info: N=18  T=1.3  β=0.769  gap=0.011925
# [ Info: N=18  T=1.4  β=0.714  gap=0.013727
# [ Info: N=18  T=1.5  β=0.667  gap=0.015533
# [ Info: N=18  T=1.6  β=0.625  gap=0.017325
# [ Info: N=18  T=1.7  β=0.588  gap=0.019092
# [ Info: N=18  T=1.8  β=0.556  gap=0.020826
# [ Info: N=18  T=1.9  β=0.526  gap=0.022522
# [ Info: N=18  T=2.0  β=0.5  gap=0.024176
# [ Info: Saved figure to results/spectral_gap_vs_temperature.png
# [ Info: Problem 2.2: spectral gap vs N, T=0.1
# [ Info: Topo=triangle_graph  N=4  T=0.1  gap=0.0
# [ Info: Topo=triangle_graph  N=6  T=0.1  gap=0.0
# [ Info: Topo=triangle_graph  N=8  T=0.1  gap=0.0
# [ Info: Topo=triangle_graph  N=10  T=0.1  gap=0.0
# [ Info: Topo=triangle_graph  N=12  T=0.1  gap=0.0
# [ Info: Topo=triangle_graph  N=14  T=0.1  gap=0.0
# [ Info: Topo=triangle_graph  N=16  T=0.1  gap=0.0
# [ Info: Topo=triangle_graph  N=18  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=4  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=6  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=8  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=10  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=12  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=14  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=16  T=0.1  gap=0.0
# [ Info: Topo=square_graph  N=18  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=4  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=6  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=8  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=10  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=12  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=14  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=16  T=0.1  gap=0.0
# [ Info: Topo=diamond_graph  N=18  T=0.1  gap=0.0
# [ Info: Saved figure to results/spectral_gap_vs_size_T0.1.png
# Problem 2.1 — Spectral gaps vs T (N = 18):
#   T       Triangle      Square        Diamond
#   0.10     0.000000    0.000000    0.000000
#   0.20     0.000002    0.000004    0.000002
#   0.30     0.000047    0.000107    0.000047
#   0.40     0.000262    0.000565    0.000262
#   0.50     0.000770    0.001537    0.000770
#   0.60     0.001658    0.002996    0.001658
#   0.70     0.002905    0.004828    0.002905
#   0.80     0.003996    0.006909    0.003996
#   0.90     0.005308    0.009132    0.005308
#   1.00     0.006800    0.011418    0.006800
#   1.10     0.008428    0.013712    0.008428
#   1.20     0.010149    0.015975    0.010149
#   1.30     0.011925    0.018182    0.011925
#   1.40     0.013727    0.020319    0.013727
#   1.50     0.015533    0.022377    0.015533
#   1.60     0.017325    0.024352    0.017325
#   1.70     0.019092    0.026242    0.019092
#   1.80     0.020826    0.028048    0.020826
#   1.90     0.022522    0.029772    0.022522
#   2.00     0.024176    0.031417    0.024176

# Problem 2.2 — Spectral gaps vs N (T = 0.1):
#   N       Triangle      Square        Diamond
#   4        0.000000    0.000000    0.000000
#   6        0.000000    0.000000    0.000000
#   8        0.000000    0.000000    0.000000
#   10       0.000000    0.000000    0.000000
#   12       0.000000    0.000000    0.000000
#   14       0.000000    0.000000    0.000000
#   16       0.000000    0.000000    0.000000
#   18       0.000000    0.000000    0.000000

# Figures saved to:
#   results/spectral_gap_vs_temperature.png
#   results/spectral_gap_vs_size_T0.1.png
