using LinearAlgebra
using Random
using Statistics
using Printf


# Part 1: Ising Model, Circuit SAT Setup

struct IsingModel
    n::Int
    h::Vector{Float64}
    J::Matrix{Float64}
    K::Dict{Tuple{Int,Int,Int}, Float64}
    offset::Ref{Float64}
end

function IsingModel(n::Int)
    return IsingModel(
        n, 
        zeros(n), 
        zeros(n, n), 
        Dict{Tuple{Int,Int,Int}, Float64}(), 
        Ref(0.0)
    )
end

function add_clause!(model::IsingModel, lits)
    (i1, s1), (i2, s2), (i3, s3) = lits
    

    
    model.offset[] += 0.125
    
    w = -0.125
    model.h[i1] += w * s1
    model.h[i2] += w * s2
    model.h[i3] += w * s3

    w_pair = 0.125
    model.J[i1, i2] += w_pair * s1 * s2
    model.J[i2, i1] += w_pair * s1 * s2
    model.J[i1, i3] += w_pair * s1 * s3
    model.J[i3, i1] += w_pair * s1 * s3
    model.J[i2, i3] += w_pair * s2 * s3
    model.J[i3, i2] += w_pair * s2 * s3
    

    key = Tuple(sort([i1, i2, i3]))
    w_cubic = -0.125
    current_val = get(model.K, key, 0.0)
    model.K[key] = current_val + w_cubic * s1 * s2 * s3
end

function build_half_adder()

    A, B, S, C = 1, 2, 3, 4
    model = IsingModel(4)


    add_clause!(model, ((A, 1), (B, 1), (S, -1)))
    add_clause!(model, ((A, -1), (B, -1), (S, -1)))
    add_clause!(model, ((A, 1), (B, -1), (S, 1)))
    add_clause!(model, ((A, -1), (B, 1), (S, 1)))
    
    add_clause!(model, ((A, -1), (B, -1), (C, 1)))
    add_clause!(model, ((A, 1), (C, -1), (A, 1)))
    add_clause!(model, ((B, 1), (C, -1), (B, 1)))
    
    return model
end

function compute_energy(model::IsingModel, s::Vector{Int})
    E = model.offset[]
    n = model.n
    
    # Linear
    E += dot(model.h, s)
    
    for i in 1:n-1
        for j in i+1:n
            E += model.J[i, j] * s[i] * s[j]
        end
    end
    
    # Cubic
    for ((i, j, k), val) in model.K
        E += val * s[i] * s[j] * s[k]
    end
    
    return E
end

# Part 2: Metropolis Solver


function solve_ground_state(model::IsingModel; 
                            steps=100_000, 
                            beta=5.0, 
                            fix_spins::Dict{Int,Int}=Dict{Int,Int}())
    n = model.n

    s = rand([-1, 1], n)

    for (idx, val) in fix_spins
        s[idx] = val
    end
    
    current_E = compute_energy(model, s)
    best_s = copy(s)
    best_E = current_E
    
    for _ in 1:steps

        i = rand(1:n)

        if haskey(fix_spins, i)
            continue
        end

        old_si = s[i]
        new_si = -old_si
        ds = new_si - old_si
        

        dE = model.h[i] * ds
        
        for j in 1:n
            if i != j
                dE += model.J[i, j] * s[j] * ds
            end
        end
        
        for ((u, v, w), k_val) in model.K
            if u == i
                dE += k_val * ds * s[v] * s[w]
            elseif v == i
                dE += k_val * s[u] * ds * s[w]
            elseif w == i
                dE += k_val * s[u] * s[v] * ds
            end
        end
        
        if dE <= 0 || rand() < exp(-beta * dE)
            s[i] = new_si
            current_E += dE
            
            if current_E < best_E
                best_E = current_E
                best_s .= s
            end
        end
    end
    
    return best_s, best_E
end

function run_circuit_demo()
    println(">>> Running Half-Adder Inverse Problem...")
    

    Random.seed!(42)
    
    model = build_half_adder()
    

    constraints = Dict(3 => -1, 4 => 1) # S(3)=-1, C(4)=+1
    
    sol, e_min = solve_ground_state(model, steps=50_000, beta=5.0, fix_spins=constraints)
    

    to_bit(x) = (x + 1) ÷ 2
    
    A, B = to_bit(sol[1]), to_bit(sol[2])
    S, C = to_bit(sol[3]), to_bit(sol[4])
    
    println("Result Energy: $e_min")
    println("Spins: $sol")
    println("Bits : A=$A, B=$B => Sum=$S, Carry=$C")
    println("---------------------------------------------------")
end


# Part 3: Graph Algorithms

function generate_regular_graph(n::Int, k=3; max_attempts=10_000)
    if (n * k) % 2 != 0
        error("n * k must be even")
    end
    
    for _ in 1:max_attempts

        stubs = repeat(1:n, inner=k)
        shuffle!(stubs)
        
        adj = [Int[] for _ in 1:n]
        valid_graph = true
        

        while length(stubs) >= 2
            u = pop!(stubs)
            v = pop!(stubs)
            
            if u == v || v in adj[u]
                valid_graph = false
                break
            end
            
            push!(adj[u], v)
            push!(adj[v], u)
        end
        
        if valid_graph
            return adj
        end
    end
    error("Could not generate graph after $max_attempts tries.")
end

function greedy_mis_solver(adj::Vector{Vector{Int}})
    n = length(adj)
    active = trues(n)
    degrees = length.(adj)
    independent_set = Int[]
    
    while any(active)

        min_d = typemax(Int)
        u = -1
        
        for i in 1:n
            if active[i]
                if degrees[i] < min_d
                    min_d = degrees[i]
                    u = i
                end
            end
        end
        
        push!(independent_set, u)
        

        
        nodes_to_remove = [u; adj[u]]
        
        for v in nodes_to_remove
            if 1 <= v <= n && active[v]
                active[v] = false
                # Reduce degree of neighbors of removed nodes
                for neighbor in adj[v]
                    degrees[neighbor] -= 1
                end
            end
        end
    end
    
    return sort(independent_set)
end

# Exact solver: Branch and Bound
function exact_mis_solver(adj::Vector{Vector{Int}}; recursion_limit=10^7)
    best_set_size = Ref(0)
    best_set = Int[]
    calls = Ref(0)
    n = length(adj)
    
    # Recursive inner function
    function search(candidates::Vector{Int}, current_set::Vector{Int})
        calls[] += 1
        if calls[] > recursion_limit
            return false # Abort
        end
        
        # Pruning: if what we have + what's left <= best found, give up
        if length(current_set) + length(candidates) <= best_set_size[]
            return true
        end
        
        if isempty(candidates)
            if length(current_set) > best_set_size[]
                best_set_size[] = length(current_set)
                best_set = copy(current_set)
            end
            return true
        end
        
        # Heuristic branching: pick max degree node in candidates
        # This usually prunes the tree faster
        pivot = candidates[1]
        max_d = -1
        for v in candidates
            d = length(adj[v]) # Static degree is usually fine for pivot choice
            if d > max_d
                max_d = d
                pivot = v
            end
        end
        
        # Branch 1: Include pivot (must exclude neighbors)
        # Filter candidates: remove pivot AND its neighbors
        forbidden = Set(adj[pivot])
        push!(forbidden, pivot)
        next_candidates_in = [x for x in candidates if !(x in forbidden)]
        
        new_set = copy(current_set)
        push!(new_set, pivot)
        
        if !search(next_candidates_in, new_set)
            return false
        end
        
        # Branch 2: Exclude pivot
        next_candidates_out = [x for x in candidates if x != pivot]
        if !search(next_candidates_out, current_set)
            return false
        end
        
        return true
    end
    
    is_complete = search(collect(1:n), Int[])
    return sort(best_set), is_complete
end

function run_mis_experiment()
    println("\n>>> MIS Scaling Experiment (Greedy vs Exact)")
    println("N\tAvg Ratio\tStd Dev\t\tExact %")
    
    sizes = 10:10:200
    trials = 10
    exact_limit = 40
    
    Random.seed!(1234)
    
    results = Dict()
    
    for n in sizes
        ratios = Float64[]
        exact_solved_count = 0
        
        for _ in 1:trials
            g = generate_regular_graph(n, 3)
            
            # Greedy solution
            set_greedy = greedy_mis_solver(g)
            
            val_denom = 0.0
            
            if n <= exact_limit
                set_opt, success = exact_mis_solver(g)
                if success
                    exact_solved_count += 1
                    val_denom = length(set_opt)
                end
            else

                val_denom = Float64(n)
            end
            
            if val_denom > 0
                push!(ratios, length(set_greedy) / val_denom)
            end
        end
        
        avg = mean(ratios)
        sd = std(ratios)
        frac_exact = (n <= exact_limit) ? (exact_solved_count / trials) : 0.0
        
        @printf("%d\t%.3f\t\t%.3f\t\t%.2f\n", n, avg, sd, frac_exact)
        results[n] = (avg, sd, frac_exact)
    end
    return results
end



run_circuit_demo()
run_mis_experiment()




# Result Energy: -5.5
# Spins: [1, 1, -1, 1]
# Bits : A=1, B=1 => Sum=0, Carry=1
# ---------------------------------------------------

# >>> MIS Scaling Experiment (Greedy vs Exact)
# N       Avg Ratio       Std Dev         Exact %
# 10      0.975           0.079           1.00
# 20      0.989           0.035           1.00
# 30      0.962           0.041           1.00
# 40      0.966           0.029           1.00
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