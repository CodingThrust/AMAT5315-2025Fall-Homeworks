# Quick test of Parallel Tempering implementation
using Graphs
using Random
using Printf

include("hw7.jl")

println("\n" * "="^70)
println("Quick Test: Parallel Tempering on Small Problem")
println("="^70)

# Test on very small problem first
println("\n[Quick Test] C4^2 (should be fast)")
sg_small = spin_glass_c(4, 2)
println("Graph: $(nv(sg_small.graph)) vertices, $(ne(sg_small.graph)) edges")

config, E = parallel_tempering(sg_small, 
                               n_replicas=8, 
                               T_min=0.5, 
                               T_max=10.0,
                               n_sweeps=1000, 
                               swap_interval=10)

println("\nResult: E = $E")
println("="^70)
