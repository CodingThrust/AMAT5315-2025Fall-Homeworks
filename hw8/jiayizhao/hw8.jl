using LinearAlgebra
using Graphs
using Plots
using OMEinsum
using OMEinsumContractionOrders
using Printf

# ==============================================================================
# Homework 8 (Corrected AFM Physics)
# Name: Jiayi Zhao
# ==============================================================================

println(">>> HW8 Script Started (Fixed AFM Logic)...")

# ------------------------------------------------------------------------------
# Problem 1: Einsum Notation (Text Answers)
# ------------------------------------------------------------------------------
# (a) C = A * B^T       => "ij,kj->ik"
# (b) Sum elements      => "ij->"
# (c) Element-wise mul  => "ij,ij,ij->ij"
# (d) Kronecker product => "ij,kl,mn->ikm,jln"


# ------------------------------------------------------------------------------
# Problem 2: Contraction Order
# ------------------------------------------------------------------------------
# Order: ((T2 * T3) * T1) * T4
# This contracts the inner loop first, then propagates out.


# ------------------------------------------------------------------------------
# Problem 3 & 4: C60 AFM Ising Model
# ------------------------------------------------------------------------------

# --- 1. Geometry: Generate C60 coordinates ---
function get_c60_coords()
    phi = (1.0 + sqrt(5.0)) / 2.0
    base_pts = [
        (0.0, 1.0, 3*phi),
        (1.0, 2.0+phi, 2*phi),
        (phi, 2.0, 2*phi+1.0)
    ]
    coords = Vector{Tuple{Float64,Float64,Float64}}()
    for (x, y, z) in base_pts
        for (a, b, c) in [(x,y,z), (y,z,x), (z,x,y)]
            for sa in [1.0, -1.0], sb in [1.0, -1.0], sc in [1.0, -1.0]
                p = (sa*a, sb*b, sc*c)
                is_unique = true
                for existing in coords
                    if norm([existing[k]-p[k] for k in 1:3]) < 1e-5
                        is_unique = false; break
                    end
                end
                if is_unique; push!(coords, p); end
            end
        end
    end
    return coords
end

# --- 2. Topology: Build Graph ---
function build_fullerene_graph()
    points = get_c60_coords()
    g = SimpleGraph(60)
    threshold_sq = 2.0^2 + 0.1 
    for i in 1:60
        for j in (i+1):60
            dist_sq = sum((points[i] .- points[j]).^2)
            if dist_sq <= threshold_sq
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

# --- 3. Physics: AFM Interaction Tensor (FIXED) ---
function get_bond_tensor(beta; J = -1.0)
    # AFM Interaction: J = -1.0
    # Hamiltonian contribution per edge: -J * s_i * s_j (Standard Convention)
    # OR: J * s_i * s_j (Problem Convention)
    
    # Let's follow the Reference Logic strictly:
    # W[a, b] = exp(beta * J * spin_a * spin_b)
    
    # Case 1: Same Spins (1,1 or -1,-1) -> product = 1
    # W_same = exp(beta * -1.0 * 1) = exp(-beta)
    val_same = exp(beta * J) 
    
    # Case 2: Diff Spins (1,-1 or -1,1) -> product = -1
    # W_diff = exp(beta * -1.0 * -1) = exp(beta)
    val_diff = exp(-beta * J)
    
    return [val_same val_diff; val_diff val_same]
end

# --- 4. Tensor Network: Optimize & Contract ---
function solve_c60_ising(beta_list)
    g = build_fullerene_graph()
    N = nv(g)
    edge_list = edges(g)
    
    # Map graph to Tensor Network indices
    ixs = [[src(e), dst(e)] for e in edge_list]
    iy = Int[]
    size_dict = Dict(i => 2 for i in 1:N)
    
    println("  Graph constructed: $N vertices, $(length(edge_list)) edges.")
    println("  Optimizing contraction order (TreeSA)...")
    
    # Use TreeSA for finding optimal path on C60
    code = EinCode(ixs, iy)
    optimizer = TreeSA(ntrials=5, niters=20, βs=0.1:0.1:10)
    opt_code = optimize_code(code, size_dict, optimizer)
    
    println("  Contraction starts...")
    results = Float64[]
    
    for beta in beta_list
        W = get_bond_tensor(beta, J=-1.0)
        tensors = [W for _ in 1:length(edge_list)]
        z_tensor = opt_code(tensors...)
        push!(results, z_tensor[])
    end
    
    return results
end

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

beta_range = 0.1:0.1:2.0
println("Running simulation...")

Zs = solve_c60_ising(beta_range)

println("\n=== Results Table (AFM) ===")
println("Beta \t log(Z)")
println("-"^20)
for (b, z) in zip(beta_range, Zs)
    @printf("%.1f \t %.4f\n", b, log(z))
end

println("\nGenerating plot...")
p = plot(beta_range, log.(Zs),
    label = "AFM C60 (TN Exact)",
    xlabel = "Inverse Temperature (β)",
    ylabel = "log(Z)",
    title = "AFM Ising Partition Function on Fullerene",
    lw = 2,
    marker = :circle,
    color = :purple
)

# High-T Limit: N*log(2) ~ 41.58
hline!(p, [60 * log(2)], label="High-T Limit", linestyle=:dash, color=:grey)

savefig("hw8_c60_result.png")
println("Plot saved to 'hw8_c60_result.png'")



# Beta     log(Z)
# --------------------
# 0.1      42.0380
# 0.2      43.3745
# 0.3      45.5658
# 0.4      48.5615
# 0.5      52.2948
# 0.6      56.6759
# 0.7      61.5922
# 0.8      66.9253
# 0.9      72.5696
# 1.0      78.4421
# 1.1      84.4819
# 1.2      90.6456
# 1.3      96.9022
# 1.4      103.2298
# 1.5      109.6121
# 1.6      116.0373
# 1.7      122.4963
# 1.8      128.9822
# 1.9      135.4896
# 2.0      142.0144
