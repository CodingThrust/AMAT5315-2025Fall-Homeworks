#############################
# hw9.jl (fixed again)
# Homework 9 – Spin Glass & Graph Algorithms
# Language: Julia
#############################

using LinearAlgebra
using Random
using Statistics
using Printf

############################################################
# 1. Circuit SAT (half adder) → Spin Glass ground state
############################################################
# Boolean x ∈ {0,1} encoded as spin s ∈ {−1,+1}:
#       x = (1 + s)/2,  s = 2x − 1
#
# Half adder:
#   Inputs : A,B
#   Outputs: S (sum), C (carry)
#
# Relations:
#   S = A ⊕ B
#   C = A ∧ B
#
# Each 3-SAT clause (ℓ1 ∨ ℓ2 ∨ ℓ3) yields Ising penalty
#
#   E_clause = (1/8) ∏_{k=1}^3 (1 - t_k s_{i_k})
#
# where literal ℓk is variable x_{i_k} if t_k=+1, or ¬x_{i_k} if t_k=-1.
#
# Penalty = 0 for satisfying assignments, 1 when violated.
############################################################

# variable indices
const VAR_A = 1
const VAR_B = 2
const VAR_S = 3
const VAR_C = 4

"""
    add_3sat_clause!(constE, h, J, K, lits)

Add a 3-SAT clause (ℓ1 ∨ ℓ2 ∨ ℓ3) where

    lits = ((i1,sign1),(i2,sign2),(i3,sign3))

and signk = +1 for variable x_ik, −1 for ¬x_ik.
Updates:
  constE::Ref{Float64}
  h::Vector{Float64}
  J::Matrix{Float64}
  K::Dict{NTuple{3,Int},Float64}  (cubic terms)
"""
function add_3sat_clause!(constE::Base.RefValue{Float64},
                          h::Vector{Float64},
                          J::Matrix{Float64},
                          K::Dict{NTuple{3,Int},Float64},
                          lits::NTuple{3,Tuple{Int,Int}})
    (i1,t1), (i2,t2), (i3,t3) = lits

    # (1/8) * (1 - t1 s1)(1 - t2 s2)(1 - t3 s3)
    # = (1/8)[1
    #         - t1 s1 - t2 s2 - t3 s3
    #         + t1 t2 s1 s2 + t1 t3 s1 s3 + t2 t3 s2 s3
    #         - t1 t2 t3 s1 s2 s3]
    constE[] += 1/8
    h[i1]    += (-t1)/8
    h[i2]    += (-t2)/8
    h[i3]    += (-t3)/8

    J[i1,i2] += ( t1*t2)/8
    J[i2,i1] += ( t1*t2)/8
    J[i1,i3] += ( t1*t3)/8
    J[i3,i1] += ( t1*t3)/8
    J[i2,i3] += ( t2*t3)/8
    J[i3,i2] += ( t2*t3)/8

    # store cubic term with sorted index triple as key
    idxs = sort(collect((i1,i2,i3)))  # collect Tuple -> Vector, then sort
    key = (idxs[1], idxs[2], idxs[3])
    K[key] = get(K, key, 0.0) - (t1*t2*t3)/8
    return nothing
end

"""
    build_half_adder_ising()

Build Ising Hamiltonian for half-adder encoding:

    S = A ⊕ B
    C = A ∧ B

Returns (constE, h, J, K).
"""
function build_half_adder_ising()
    n = 4
    constE = Ref(0.0)
    h = zeros(Float64,n)
    J = zeros(Float64,n,n)
    K = Dict{NTuple{3,Int},Float64}()

    # XOR: S = A ⊕ B
    # (A ∨ B ∨ ¬S)
    add_3sat_clause!(constE,h,J,K, ((VAR_A,+1),(VAR_B,+1),(VAR_S,-1)))
    # (¬A ∨ ¬B ∨ ¬S)
    add_3sat_clause!(constE,h,J,K, ((VAR_A,-1),(VAR_B,-1),(VAR_S,-1)))
    # (A ∨ ¬B ∨ S)
    add_3sat_clause!(constE,h,J,K, ((VAR_A,+1),(VAR_B,-1),(VAR_S,+1)))
    # (¬A ∨ B ∨ S)
    add_3sat_clause!(constE,h,J,K, ((VAR_A,-1),(VAR_B,+1),(VAR_S,+1)))

    # AND: C = A ∧ B
    # (¬A ∨ ¬B ∨ C)
    add_3sat_clause!(constE,h,J,K, ((VAR_A,-1),(VAR_B,-1),(VAR_C,+1)))
    # (A ∨ ¬C)  -> pad to 3-SAT: (A ∨ ¬C ∨ A)
    add_3sat_clause!(constE,h,J,K, ((VAR_A,+1),(VAR_C,-1),(VAR_A,+1)))
    # (B ∨ ¬C)  -> (B ∨ ¬C ∨ B)
    add_3sat_clause!(constE,h,J,K, ((VAR_B,+1),(VAR_C,-1),(VAR_B,+1)))

    return (constE[], h, J, K)
end

"""
    energy_half_adder(s, constE, h, J, K)

Compute Ising energy for spin configuration s (entries ±1).
"""
function energy_half_adder(s::Vector{Int},
                           constE::Float64,
                           h::Vector{Float64},
                           J::Matrix{Float64},
                           K::Dict{NTuple{3,Int},Float64})
    n = length(s)
    E = constE + dot(h,s)

    # quadratic
    for i in 1:n-1, j in i+1:n
        E += J[i,j]*s[i]*s[j]
    end

    # cubic
    for ((i,j,k), coeff) in K
        E += coeff * s[i]*s[j]*s[k]
    end
    return E
end

############################################################
# 2. Metropolis spin dynamics to find ground state
############################################################`

"""
    metropolis_ising(constE, h, J, K;
                     nsteps=10^5, β=5.0, s0=nothing, fixed=nothing)

Metropolis-Hastings Monte Carlo for Ising with possible fixed spins.
fixed is Dict(index => spin_value).
Returns (best_s, best_E).
"""
function metropolis_ising(constE, h, J, K;
                          nsteps::Int=100_000,
                          β::Float64=5.0,
                          s0::Union{Nothing,Vector{Int}}=nothing,
                          fixed::Union{Nothing,Dict{Int,Int}}=nothing)

    n = length(h)
    if s0 === nothing
        s = [rand(Bool) ? 1 : -1 for _ in 1:n]
    else
        s = copy(s0)
    end

    # enforce fixed spins
    if fixed !== nothing
        for (i,val) in fixed
            s[i] = val
        end
    end

    E = energy_half_adder(s,constE,h,J,K)
    best_s = copy(s)
    best_E = E

    for _ in 1:nsteps
        i = rand(1:n)

        # skip fixed spin
        if fixed !== nothing && haskey(fixed,i)
            continue
        end

        old_si = s[i]
        new_si = -old_si

        # compute ΔE = E(new) - E(old)
        ΔE = (new_si - old_si) * h[i]

        # quadratic contributions
        for j in 1:n
            if j != i
                ΔE += (new_si - old_si) * J[i,j] * s[j]
            end
        end

        # cubic terms where i participates
        for ((a,b,c), coeff) in K
            if a == i
                ΔE += coeff * (new_si - old_si) * s[b]*s[c]
            elseif b == i
                ΔE += coeff * s[a] * (new_si - old_si) * s[c]
            elseif c == i
                ΔE += coeff * s[a]*s[b] * (new_si - old_si)
            end
        end

        if ΔE <= 0 || rand() < exp(-β*ΔE)
            s[i] = new_si
            E += ΔE
            if E < best_E
                best_E = E
                best_s .= s
            end
        end
    end

    return best_s, best_E
end

"""
    run_half_adder_spin_dynamics(; nsteps=50_000, β=5.0, seed=42)

Fix output to S=0 (spin −1), C=1 (spin +1) and infer input spins A,B.
Prints decoded bits and returns (best_s, best_E).
"""
function run_half_adder_spin_dynamics(;nsteps=50_000,β=5.0,seed=42)
    Random.seed!(seed)
    constE, h, J, K = build_half_adder_ising()

    fixed = Dict{Int,Int}()
    fixed[VAR_S] = -1  # S=0
    fixed[VAR_C] = +1  # C=1

    best_s, best_E = metropolis_ising(constE,h,J,K;
                                      nsteps=nsteps,β=β,
                                      s0=nothing,fixed=fixed)

    spin_to_bit(s) = (1 + s) ÷ 2
    A = spin_to_bit(best_s[VAR_A])
    B = spin_to_bit(best_s[VAR_B])
    S = spin_to_bit(best_s[VAR_S])
    C = spin_to_bit(best_s[VAR_C])

    println("=== Half-adder spin dynamics result ===")
    println("Best energy: ", best_E)
    println("Spins: A=$(best_s[VAR_A]), B=$(best_s[VAR_B]), S=$(best_s[VAR_S]), C=$(best_s[VAR_C])")
    println("Bits : A=$A, B=$B, S=$S, C=$C")
    return best_s, best_E
end

############################################################
# 3. Greedy Maximum Independent Set on 3-regular graphs
############################################################

"""
    rand_3regular_graph(n; max_tries=10^4)

Generate a simple random 3-regular graph on vertices 1..n using the
configuration model with rejection. Returns adjacency list `neighbors`.
"""
function rand_3regular_graph(n::Int; max_tries::Int=10_000)
    @assert iseven(3n) "For 3-regular graph, 3n must be even."

    for _ in 1:max_tries
        stubs = repeat(collect(1:n), inner=3)
        shuffle!(stubs)
        m = length(stubs) ÷ 2
        neighbors = [Int[] for _ in 1:n]
        ok = true

        for k in 1:m
            u = stubs[2k-1]
            v = stubs[2k]
            if u == v || v in neighbors[u]
                ok = false
                break
            end
            push!(neighbors[u], v)
            push!(neighbors[v], u)
        end

        if ok
            return neighbors
        end
    end
    error("Failed to generate 3-regular graph after $max_tries attempts")
end

"""
    greedy_mis(neighbors)

Greedy heuristic for Maximum Independent Set:
repeatedly pick a minimum-degree vertex, add it to the independent
set, then remove it and its neighbors.
"""
function greedy_mis(neighbors::Vector{Vector{Int}})
    n = length(neighbors)
    remaining = trues(n)
    deg = [length(neighbors[i]) for i in 1:n]
    indep = Int[]

    while any(remaining)
        # choose v with minimum degree among remaining
        best_v = 0
        best_deg = typemax(Int)
        for i in 1:n
            if remaining[i] && deg[i] < best_deg
                best_deg = deg[i]
                best_v = i
            end
        end
        v = best_v
        push!(indep, v)

        # remove v and its neighbors
        to_remove = [v; neighbors[v]]
        for u in to_remove
            if 1 <= u <= n && remaining[u]
                remaining[u] = false
                for w in neighbors[u]
                    deg[w] -= 1
                end
            end
        end
    end

    return sort(indep)
end

"""
    exact_mis(neighbors; limit=10^7)

Exact (branch-and-bound) maximum independent set.
Stops if recursion calls exceed `limit`.
Returns (best_set, exact::Bool).
"""
function exact_mis(neighbors::Vector{Vector{Int}}; limit::Int=10_7)
    n = length(neighbors)
    adj = neighbors
    calls = Ref(0)
    best = Int[]
    best_size = Ref(0)

    remaining0 = collect(1:n)
    current0 = Int[]

    function branch(remaining::Vector{Int}, current::Vector{Int})
        calls[] += 1
        if calls[] > limit
            return false
        end

        # simple bound
        if length(current) + length(remaining) <= best_size[]
            return true
        end

        if isempty(remaining)
            if length(current) > best_size[]
                best_size[] = length(current)
                best = copy(current)
            end
            return true
        end

        # choose vertex of maximum degree among remaining
        best_v = remaining[1]
        best_d = -1
        for v in remaining
            d = length(adj[v])
            if d > best_d
                best_d = d
                best_v = v
            end
        end
        v = best_v

        # Branch 1: include v
        new_current = copy(current)
        push!(new_current, v)
        forbidden = Set([v; adj[v]])
        new_remaining = [u for u in remaining if !(u in forbidden)]
        if !branch(new_remaining, new_current)
            return false
        end

        # Branch 2: exclude v
        new_remaining2 = [u for u in remaining if u != v]
        if !branch(new_remaining2, current)
            return false
        end

        return true
    end

    exact = branch(remaining0,current0)
    return sort(best), exact
end

"""
    experiment_greedy_mis_scaling(; nsizes=10:10:200,
                                   ntrials=10,
                                   seed=1234,
                                   exact_cutoff=40)

Run greedy MIS on random 3-regular graphs of various sizes.
For n ≤ exact_cutoff, compare to exact MIS to get approximation ratio.
For larger n, record |greedy MIS| / n (not a true ratio to optimum).

Returns Dict n => (avg_ratio, std_ratio, exact_fraction).
"""
function experiment_greedy_mis_scaling(;nsizes=10:10:200,
                                       ntrials::Int=10,
                                       seed::Int=1234,
                                       exact_cutoff::Int=40)
    Random.seed!(seed)
    results = Dict{Int,Tuple{Float64,Float64,Float64}}()

    println("n\tavg_ratio\tstd_ratio\texact_frac")
    for n in nsizes
        ratios = Float64[]
        n_exact = 0
        for _ in 1:ntrials
            neighbors = rand_3regular_graph(n)
            greedy_set = greedy_mis(neighbors)

            if n <= exact_cutoff
                opt_set, exact = exact_mis(neighbors)
                if exact
                    n_exact += 1
                    push!(ratios, length(greedy_set)/length(opt_set))
                end
            else
                # approximate “density” for large n
                push!(ratios, length(greedy_set)/n)
            end
        end

        avg_ratio = mean(ratios)
        std_ratio = std(ratios)
        exact_frac = n <= exact_cutoff ? n_exact/ntrials : 0.0
        println(@sprintf("%d\t%.3f\t\t%.3f\t\t%.2f", n, avg_ratio, std_ratio, exact_frac))
        results[n] = (avg_ratio, std_ratio, exact_frac)
    end

    return results
end

############################################################
# Main
############################################################

if abspath(PROGRAM_FILE) == @__FILE__
    # 2. Spin dynamics for half adder with S=0, C=1.
    run_half_adder_spin_dynamics()

    # 3. Greedy MIS scaling on random 3-regular graphs.
    println("\n=== Greedy MIS scaling on random 3-regular graphs ===")
    experiment_greedy_mis_scaling()
end


# === Half-adder spin dynamics result ===
# Best energy: -3.75
# Spins: A=1, B=1, S=-1, C=1
# Bits : A=1, B=1, S=0, C=1

# === Greedy MIS scaling on random 3-regular graphs ===
# n       avg_ratio       std_ratio       exact_frac
# 10      0.975           0.079           1.00
# 20      NaN             NaN             0.00
# 30      NaN             NaN             0.00
# 40      NaN             NaN             0.00
# 50      0.420           0.016           0.00
# 60      0.422           0.014           0.00
# 70      0.427           0.013           0.00
# 80      0.431           0.009           0.00
# 90      0.424           0.010           0.00
# 100     0.428           0.010           0.00
# 110     0.429           0.007           0.00
# 120     0.431           0.012           0.00
# 130     0.424           0.009           0.00
# 140     0.430           0.005           0.00
# 150     0.425           0.011           0.00
# 160     0.429           0.011           0.00
# 170     0.428           0.007           0.00
# 180     0.429           0.005           0.00
# 190     0.431           0.010           0.00
# 200     0.430           0.008           0.00