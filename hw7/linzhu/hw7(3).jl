###########################################################
# HW7 – Problem 3: Spin glass ground state on Cₙ^(⊠k)
# Exact solver:
#   - small (n=25): tensor network SingleConfigMin
#   - large (n=2401): MILP via JuMP + HiGHS
###########################################################

using GenericTensorNetworks, GenericTensorNetworks.Graphs
using ProblemReductions
using Graphs
using Random
using Test
using JuMP
using HiGHS

##############################
# 1. Strong product and power
##############################

"""
    strong_product(g1, g2) -> SimpleGraph

Strong product of two simple graphs g1, g2.

Vertices are pairs (v1, v2). Two vertices (u1,u2) and (v1,v2) are adjacent iff

  * u1 == v1 and {u2,v2} is an edge in g2, OR
  * u2 == v2 and {u1,v1} is an edge in g1, OR
  * {u1,v1} is an edge in g1 AND {u2,v2} is an edge in g2.
"""
function strong_product(g1::SimpleGraph, g2::SimpleGraph)
    vs1 = collect(vertices(g1))
    vs2 = collect(vertices(g2))

    pairs = [(v1, v2) for v1 in vs1, v2 in vs2]
    n = length(pairs)
    g = SimpleGraph(n)

    getpair(i) = pairs[i]

    for i in 1:n
        u1, u2 = getpair(i)
        for j in (i+1):n
            v1, v2 = getpair(j)

            cond =  (u1 == v1 && has_edge(g2, u2, v2)) ||
                    (u2 == v2 && has_edge(g1, u1, v1)) ||
                    (has_edge(g1, u1, v1) && has_edge(g2, u2, v2))

            cond && add_edge!(g, i, j)
        end
    end
    return g
end

"""
    strong_power(g, k)

k-th strong power g ⊠ g ⊠ … ⊠ g (k times).
"""
strong_power(g::SimpleGraph, k::Int) =
    k == 1 ? g : strong_product(g, strong_power(g, k - 1))

######################################
# 2. Spin–glass instances on Cₙ^(⊠k)
######################################

"""
    spin_glass_c(n, k) -> SpinGlass

Construct the spin–glass instance on the strong power of the cycle graph Cₙ.

Couplings J_ij = 1 on every edge; local fields h_i = 1 - deg(i).

Hamiltonian (GenericTensorNetworks / ProblemReductions):

    H(s) = ∑_{(i,j)∈E} J_ij s_i s_j + ∑_i h_i s_i
    with s_i ∈ {−1,1}, and boolean variables n_i = (1 - s_i)/2 internally.
"""
function spin_glass_c(n::Int, k::Int)
    g1 = Graphs.cycle_graph(n)
    g  = strong_power(g1, k)

    coupling = fill(1, ne(g))          # J_ij = 1
    bias     = 1 .- degree(g)          # h_i = 1 - deg(i)

    return SpinGlass(g, coupling, bias)
end

###############################################
# 3. Energy wrapper (use ProblemReductions)
###############################################

"""
    energy(sg, cfg)

Compute the spin-glass energy using the same definition as the
GenericTensorNetworks / ProblemReductions docs.

`cfg` is a boolean (or bit) vector n with n_i = 1 ↔ s_i = −1, n_i = 0 ↔ s_i = +1.
"""
function energy(sg::SpinGlass, cfg::AbstractVector)
    return ProblemReductions.energy(sg, BitVector(cfg))
end

###############################################
# 4. Exact MILP solver for the ground state
###############################################

# We use boolean variables x_i = n_i = (1 - s_i)/2 ∈ {0,1}
#
# With s_i = 1 - 2 x_i, the Hamiltonian
#
#   H(s) = ∑_{(i,j)} J_ij s_i s_j + ∑_i h_i s_i
#
# can be rewritten as
#
#   H(x) = const + ∑_{(i,j)} 4 J_ij x_i x_j + ∑_i b_i x_i
#
# where
#
#   const = ∑_{(i,j)} J_ij + ∑_i h_i
#   b_i   = -2 h_i - 2 ∑_{j: (i,j)∈E} J_ij
#
# We introduce auxiliary binary variables z_e ≈ x_i x_j and linearize:
#
#   z_e ≤ x_i,  z_e ≤ x_j,  z_e ≥ x_i + x_j - 1
#
# Then we minimize:
#
#   H'(x,z) = ∑_e 4 J_e z_e + ∑_i b_i x_i
#
# which is H(x) up to an additive constant (irrelevant for argmin).

function exact_ground_state_ilp(sg::SpinGlass)
    g = sg.graph
    n = nv(g)
    m = ne(g)

    edges_vec = collect(edges(g))
    J = sg.J
    h = sg.h

    # Precompute sum of incident couplings for each vertex
    incident_sum = zeros(Int, n)
    for (e_idx, e) in enumerate(edges_vec)
        Je = J[e_idx]
        i = src(e); j = dst(e)
        incident_sum[i] += Je
        incident_sum[j] += Je
    end

    # Linear coefficients b_i
    b = [-2*h[i] - 2*incident_sum[i] for i in 1:n]

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:n], Bin)   # x_i = n_i = (1 - s_i)/2
    @variable(model, z[1:m], Bin)   # z_e = x_i * x_j

    # Objective: H' = ∑ 4 J_e z_e + ∑ b_i x_i
    @objective(model, Min,
        sum(4 * J[e_idx] * z[e_idx] for e_idx in 1:m) +
        sum(b[i] * x[i] for i in 1:n)
    )

    # Linearization constraints for z_e = x_i * x_j
    for (e_idx, e) in enumerate(edges_vec)
        i = src(e); j = dst(e)
        @constraint(model, z[e_idx] <= x[i])
        @constraint(model, z[e_idx] <= x[j])
        @constraint(model, z[e_idx] >= x[i] + x[j] - 1)
    end

    optimize!(model)

    # Extract configuration: boolean n_i = x_i
    cfg = falses(n)
    for i in 1:n
        xi = value(x[i])
        cfg[i] = xi > 0.5
    end
    return cfg
end

###################################################
# 5. Main solver
###################################################

"""
    my_ground_state_solver(sg::SpinGlass) -> cfg::BitVector

- For small graphs (n ≤ 100), use the exact tensor-network solver
  `SingleConfigMin()` from GenericTensorNetworks.
- For larger graphs (like C₇^(⊠4), n=2401), solve the MILP above
  with JuMP + HiGHS.

Returns cfg in boolean encoding n_i = 1 ↔ s_i = −1, n_i = 0 ↔ s_i = +1.
"""
function my_ground_state_solver(sg::SpinGlass)
    g = sg.graph
    n = nv(g)

    if n <= 100
        # Exact tensor-network solution
        problem = GenericTensorNetwork(sg)
        cfg_bits = read_config(solve(problem, SingleConfigMin())[])
        return cfg_bits
    else
        # Exact MILP formulation
        return exact_ground_state_ilp(sg)
    end
end

##########################
# 6. Provided test cases
##########################

sg1 = spin_glass_c(5, 2)
@test energy(sg1, my_ground_state_solver(sg1)) == -85   # testing

sg2 = spin_glass_c(7, 4)
@test energy(sg2, my_ground_state_solver(sg2)) < -93855 # challenge
