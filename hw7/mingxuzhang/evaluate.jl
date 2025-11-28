# Homework 7 - 反铁磁Ising模型和自旋玻璃问题求解
# 作者: 张明旭

using Graphs, ProblemReductions
using LinearAlgebra, SparseArrays
using Random, Statistics
using GenericTensorNetworks
using Test

println("开始求解作业7...")

# ============================================================================
# 问题1: 富勒烯图上反铁磁Ising模型的基态能量
# ============================================================================

println("\n=== 问题1: 富勒烯图基态能量 ===")

# 构建富勒烯图
function fullerene()  # construct the fullerene graph in 3D space
    th = (1+sqrt(5))/2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3th), (1.0, 2 + th, 2th), (th, 2.0, 2th + 1.0))
        for (a, b, c) in ((x,y,z), (y,z,x), (z,x,y))
            for loc in ((a,b,c), (a,b,-c), (a,-b,c), (a,-b,-c), (-a,b,c), (-a,b,-c), (-a,-b,c), (-a,-b,-c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5))

println("富勒烯图节点数: ", nv(fullerene_graph))
println("富勒烯图边数: ", ne(fullerene_graph))

# 模拟退火算法求解基态能量
function simulated_annealing_ising(graph, max_iter=10000, T_init=2.0, T_final=0.01)
    n = nv(graph)
    # 随机初始化自旋配置
    spins = rand([-1, 1], n)
    
    # 计算当前能量
    function energy(spins)
        E = 0.0
        for edge in edges(graph)
            i, j = src(edge), dst(edge)
            E += spins[i] * spins[j]  # 反铁磁相互作用
        end
        return E
    end
    
    current_energy = energy(spins)
    best_spins = copy(spins)
    best_energy = current_energy
    
    # 模拟退火过程
    for iter in 1:max_iter
        # 线性降温
        T = T_init * (T_final/T_init)^(iter/max_iter)
        
        # 随机翻转一个自旋
        i = rand(1:n)
        old_spin = spins[i]
        spins[i] = -spins[i]
        
        # 计算能量变化
        ΔE = 0.0
        for neighbor in neighbors(graph, i)
            ΔE += 2 * old_spin * spins[neighbor]
        end
        
        new_energy = current_energy + ΔE
        
        # Metropolis准则
        if ΔE < 0 || rand() < exp(-ΔE/T)
            current_energy = new_energy
            if current_energy < best_energy
                best_energy = current_energy
                best_spins = copy(spins)
            end
        else
            spins[i] = old_spin  # 恢复原状态
        end
    end
    
    return best_energy, best_spins
end

# 多次运行模拟退火以获得更好的结果
best_energy = Inf
best_config = nothing
num_runs = 20

println("运行模拟退火算法...")
for run in 1:num_runs
    energy, config = simulated_annealing_ising(fullerene_graph, 20000)
    if energy < best_energy
        best_energy = energy
        best_config = config
    end
    if run % 5 == 0
        println("运行 $run/$num_runs, 当前最佳能量: $best_energy")
    end
end

println("富勒烯图反铁磁Ising模型基态能量: $best_energy")

# ============================================================================
# 问题2: 光谱间隙分析
# ============================================================================

println("\n=== 问题2: 光谱间隙分析 ===")

# 构建不同拓扑的图
function create_triangle_graph(n)
    vertices = []
    edges = []
    for i in 0:(n-1)
        push!(vertices, (i, 0))
        push!(vertices, (i, 1))
        push!(edges, (2*i+1, 2*i+2))
        if i > 0
            push!(edges, (2*i-1, 2*i+1))
            push!(edges, (2*i, 2*i+1))
            push!(edges, (2*i, 2*i+2))
        end
    end
    
    g = SimpleGraph(length(vertices))
    for (i, j) in edges
        add_edge!(g, i, j)
    end
    return g
end

function create_square_graph(n)
    vertices = []
    edges = []
    for i in 0:(n-1)
        push!(vertices, (i, 0))
        push!(vertices, (i, 1))
        push!(edges, (2*i+1, 2*i+2))
        if i > 0
            push!(edges, (2*i-1, 2*i+1))
            push!(edges, (2*i, 2*i+2))
        end
    end
    
    g = SimpleGraph(length(vertices))
    for (i, j) in edges
        add_edge!(g, i, j)
    end
    return g
end

function create_diamond_graph(n)
    vertices = [(0, 0)]
    edges = []
    
    for i in 0:(n-1)
        push!(vertices, (i + 0.5, 0.5))
        push!(vertices, (i + 0.5, -0.5))
        push!(vertices, (i + 1, 0))
        
        push!(edges, (3*i+1, 3*i+2))
        push!(edges, (3*i+1, 3*i+3))
        push!(edges, (3*i+2, 3*i+4))
        push!(edges, (3*i+3, 3*i+4))
    end
    
    g = SimpleGraph(length(vertices))
    for (i, j) in edges
        if i <= nv(g) && j <= nv(g)
            add_edge!(g, i, j)
        end
    end
    return g
end

# 计算转移矩阵的光谱间隙
function compute_spectral_gap(graph, T)
    n = nv(graph)
    β = 1.0 / T
    
    # 构建转移矩阵 (使用稀疏矩阵)
    # 状态空间大小为2^n，对于大图需要近似方法
    if n > 20
        # 对于大图，使用蒙特卡洛方法估计光谱间隙
        return estimate_spectral_gap_mc(graph, T)
    end
    
    # 对于小图，精确计算
    N_states = 2^n
    H = spzeros(N_states, N_states)
    
    # 构建哈密顿矩阵
    for state in 0:(N_states-1)
        spins = [(state >> i) & 1 == 1 ? 1 : -1 for i in 0:(n-1)]
        energy = 0.0
        
        for edge in edges(graph)
            i, j = src(edge), dst(edge)
            energy += spins[i] * spins[j]
        end
        
        H[state+1, state+1] = energy
    end
    
    # 转移矩阵 T = exp(-βH)
    T_matrix = exp(-β * Matrix(H))
    
    # 计算特征值
    eigenvals = eigvals(T_matrix)
    eigenvals = real(eigenvals)
    sort!(eigenvals, rev=true)
    
    # 光谱间隙是最大特征值和第二大特征值的差
    if length(eigenvals) >= 2
        return eigenvals[1] - eigenvals[2]
    else
        return 0.0
    end
end

# 蒙特卡洛估计光谱间隙（用于大图）
function estimate_spectral_gap_mc(graph, T, num_samples=1000)
    # 简化的估计方法：基于能量涨落
    n = nv(graph)
    β = 1.0 / T
    
    energies = Float64[]
    
    for _ in 1:num_samples
        spins = rand([-1, 1], n)
        energy = 0.0
        for edge in edges(graph)
            i, j = src(edge), dst(edge)
            energy += spins[i] * spins[j]
        end
        push!(energies, energy)
    end
    
    # 基于能量方差的简单估计
    energy_var = var(energies)
    return energy_var * β^2  # 简化的关系
end

# 分析不同温度下的光谱间隙
println("分析光谱间隙与温度的关系...")

temperatures = 0.1:0.1:2.0
graph_types = [
    ("Triangle", create_triangle_graph(3)),
    ("Square", create_square_graph(3)),
    ("Diamond", create_diamond_graph(2))
]

spectral_gaps_vs_T = Dict()

for (name, graph) in graph_types
    println("分析 $name 图...")
    gaps = Float64[]
    
    for T in temperatures
        gap = compute_spectral_gap(graph, T)
        push!(gaps, gap)
    end
    
    spectral_gaps_vs_T[name] = gaps
    println("$name 图完成")
end

# 分析不同系统大小下的光谱间隙 (T = 0.1)
println("\n分析光谱间隙与系统大小的关系...")
T_fixed = 0.1
sizes = 2:6
spectral_gaps_vs_N = Dict()

for graph_type in ["Triangle", "Square"]
    println("分析 $graph_type 图的大小依赖性...")
    gaps = Float64[]
    system_sizes = Int[]
    
    for size in sizes
        if graph_type == "Triangle"
            graph = create_triangle_graph(size)
        else
            graph = create_square_graph(size)
        end
        
        if nv(graph) <= 18  # 只分析小于18个节点的图
            gap = compute_spectral_gap(graph, T_fixed)
            push!(gaps, gap)
            push!(system_sizes, nv(graph))
        end
    end
    
    spectral_gaps_vs_N[graph_type] = (system_sizes, gaps)
    println("$graph_type 图大小依赖性分析完成")
end

# ============================================================================
# 问题4: 并行回火算法求解自旋玻璃问题 (挑战题)
# ============================================================================

println("\n=== 问题4: 并行回火算法 (挑战题) ===")

# 强乘积图构造
function strong_product(g1, g2)
    vs = [(v1, v2) for v1 in vertices(g1), v2 in vertices(g2)]
    graph = SimpleGraph(length(vs))
    for (i, vi) in enumerate(vs), (j, vj) in enumerate(vs)
        if (vi[1] == vj[1] && has_edge(g2, vi[2], vj[2])) ||
                (vi[2] == vj[2] && has_edge(g1, vi[1], vj[1])) ||
                (has_edge(g1, vi[1], vj[1]) && has_edge(g2, vi[2], vj[2]))
            add_edge!(graph, i, j)
        end
    end
    return graph
end

strong_power(g, k::Int) = k == 1 ? g : strong_product(g, strong_power(g, k - 1))

function spin_glass_c(n::Int, k::Int)
    g1 = Graphs.cycle_graph(n)
    g = strong_power(g1, k)
    coupling = fill(1, ne(g))
    bias = 1 .- degree(g)
    return SpinGlass(g, coupling, bias)
end

# 并行回火算法实现
function parallel_tempering_spin_glass(sg, num_replicas=8, max_iter=50000)
    n = nv(sg.graph)
    
    # 设置温度梯度
    T_min, T_max = 0.1, 3.0
    temperatures = [T_min * (T_max/T_min)^(i/(num_replicas-1)) for i in 0:(num_replicas-1)]
    
    # 初始化副本
    replicas = [rand([-1, 1], n) for _ in 1:num_replicas]
    energies = [energy(sg, replica) for replica in replicas]
    
    best_energy = minimum(energies)
    best_config = copy(replicas[argmin(energies)])
    
    # 交换尝试间隔
    exchange_interval = 100
    
    println("开始并行回火算法...")
    println("副本数: $num_replicas, 最大迭代数: $max_iter")
    
    for iter in 1:max_iter
        # 对每个副本进行蒙特卡洛更新
        for r in 1:num_replicas
            T = temperatures[r]
            β = 1.0 / T
            
            # 随机选择一个自旋翻转
            i = rand(1:n)
            old_spin = replicas[r][i]
            replicas[r][i] = -replicas[r][i]
            
            # 计算能量变化
            ΔE = 0.0
            # 耦合项
            for neighbor in neighbors(sg.graph, i)
                ΔE += 2 * old_spin * sg.coupling[1] * replicas[r][neighbor]
            end
            # 偏置项
            ΔE += 2 * old_spin * sg.bias[i]
            
            new_energy = energies[r] + ΔE
            
            # Metropolis准则
            if ΔE < 0 || rand() < exp(-β * ΔE)
                energies[r] = new_energy
                if new_energy < best_energy
                    best_energy = new_energy
                    best_config = copy(replicas[r])
                end
            else
                replicas[r][i] = old_spin  # 恢复
            end
        end
        
        # 副本交换
        if iter % exchange_interval == 0
            for r in 1:(num_replicas-1)
                β1, β2 = 1.0/temperatures[r], 1.0/temperatures[r+1]
                ΔE = energies[r+1] - energies[r]
                
                if rand() < exp((β1 - β2) * ΔE)
                    # 交换配置
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                    energies[r], energies[r+1] = energies[r+1], energies[r]
                end
            end
        end
        
        if iter % 10000 == 0
            println("迭代 $iter: 最佳能量 = $best_energy")
        end
    end
    
    return best_config
end

# 自定义求解器
function my_ground_state_solver(sg)
    return parallel_tempering_spin_glass(sg)
end

# 测试用例
println("测试用例1...")
sg1 = spin_glass_c(5, 2)
result1 = my_ground_state_solver(sg1)
energy1 = energy(sg1, result1)
println("测试1结果: 能量 = $energy1 (目标: -85)")

println("测试用例2...")
sg2 = spin_glass_c(7, 4)
result2 = my_ground_state_solver(sg2)
energy2 = energy(sg2, result2)
println("测试2结果: 能量 = $energy2 (目标: < -93855)")

# 运行测试
try
    @test energy(sg1, my_ground_state_solver(sg1)) == -85
    println("✓ 测试1通过!")
catch e
    println("✗ 测试1失败: $e")
end

try
    @test energy(sg2, my_ground_state_solver(sg2)) < -93855
    println("✓ 测试2通过!")
catch e
    println("✗ 测试2失败: $e")
end

# ============================================================================
# 结果总结
# ============================================================================

println("\n" * "="^60)
println("作业7结果总结")
println("="^60)
println("1. 富勒烯图反铁磁Ising模型基态能量: $best_energy")
println("2. 光谱间隙分析:")
for (name, gaps) in spectral_gaps_vs_T
    println("   $name 图: 温度范围 0.1-2.0 的光谱间隙已计算")
end
println("3. 系统大小依赖性分析完成")
println("4. 并行回火算法:")
println("   测试1能量: $energy1")
println("   测试2能量: $energy2")
println("="^60)
