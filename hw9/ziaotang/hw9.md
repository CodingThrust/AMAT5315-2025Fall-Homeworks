# HW9

## Q1

The cost function is given as 
$$
H = (S - A - B + 2AB)^2 + (C - AB)^2.
$$

* (H = 0) if the logic constraints are satisfied.
* (H > 0) otherwise.
  Thus, finding the ground state solves the circuit SAT.

Using $x = (1 - \sigma)/2$, we can rewrite as
$$
H = J_1(1 - \sigma_A\sigma_B\sigma_S) + J_2(1 - \sigma_A\sigma_B)(1 - \sigma_C).
$$

# Q2

```julia
using Random, Statistics


function H_bool(x::NTuple{4,Int})
    A,B,S,C = x
    xorgate = (S - A - B + 2*A*B)
    andgate = (C - A*B)
    return xorgate^2 + andgate^2
end


Random.seed!(12345)


function simulated_annealing()
    max_iters=10000 
    T0=1.0
    Tf=1e-5 
    restarts=50
    best = 0
    bestE = 1e10
    for r in 1:restarts
        A = rand([0,1])
        B = rand([0,1])
        S = 0
        C = 1
        x = (A,B,S,C)
        E = H_bool(x)

        for t in 1:max_iters
            T = T0 * (Tf/T0)^(t/max_iters)
            # propose flip A or B
            if rand() < 0.5
                A2 = 1 - A
                x2 = (A2, B, S, C)
            else
                B2 = 1 - B
                x2 = (A, B2, S, C)
            end
            E2 = H_bool(x2)
            Δ = E2 - E
            if Δ <= 0 || rand() < exp(-Δ / max(T,1e-12))
                # accept
                A, B = x2[1], x2[2]
                x = x2
                E = E2
            end
            # quick exit if zero energy found
            if E == 0
                break
            end
        end
        if E < bestE
            bestE = E
            best = x
        end
    end
    return best,bestE
end
best,bestE = simulated_annealing()
println("(A,B,S,C) = ", best, "  Energy = ", bestE)
```

Answer gives

```
(A,B,S,C) = (1, 1, 0, 1)  Energy = 0
```





# Q3

```julia
using Graphs, Statistics, Random

function generate_3_regular(n)
    n % 2 == 0 || error("n must be even")
    g = SimpleGraph(n)
    max_attempts = 50
    
    for _ in 1:max_attempts
        g = SimpleGraph(n)
        stubs = vcat([fill(i, 3) for i in 1:n]...)
        shuffle!(stubs)
        valid = true
        
        for i in 1:2:length(stubs)
            u, v = stubs[i], stubs[i+1]
            if u == v || has_edge(g, u, v)
                valid = false
                break
            end
            add_edge!(g, u, v)
        end
        
        valid && return g
    end
    return g
end

function greedy_mis(g)
    active = trues(nv(g))
    mis = Int[]
    
    while any(active)
        min_deg, candidates = 4, Int[]
        for v in 1:nv(g)
            active[v] || continue
            deg = count(n -> active[n], neighbors(g, v))
            if deg < min_deg
                min_deg, candidates = deg, [v]
            elseif deg == min_deg
                push!(candidates, v)
            end
        end
        v = rand(candidates)
        push!(mis, v)
        active[v] = false
        for n in neighbors(g, v)
            active[n] = false
        end
    end
    return mis
end

function run_analysis()
    sizes = 10:10:200
    results = []
    
    for n in sizes
        ratios = Float64[]
        for _ in 1:5
            g = generate_3_regular(n)
            mis = greedy_mis(g)
            # Theoretical lower bound for 3-regular graphs
            lower_bound = n / 4
            ratio = length(mis) / lower_bound
            push!(ratios, ratio)
        end
        avg = mean(ratios)
        std_dev = std(ratios)
        push!(results, (n, avg, std_dev))
        println("n=$n: ratio=$(round(avg, digits=3)) ± $(round(std_dev, digits=3))")
    end
    return results
end

Random.seed!(42)
run_analysis()
```



```
n=10: ratio=1.6 ± 0.0
n=20: ratio=1.56 ± 0.089
n=30: ratio=1.68 ± 0.073
n=40: ratio=1.74 ± 0.055
n=50: ratio=1.648 ± 0.072
n=60: ratio=1.693 ± 0.076
n=70: ratio=1.749 ± 0.031
n=80: ratio=1.7 ± 0.035
n=90: ratio=1.716 ± 0.051
n=100: ratio=1.704 ± 0.036
n=110: ratio=1.738 ± 0.03
n=120: ratio=1.713 ± 0.03
n=130: ratio=1.698 ± 0.026
n=140: ratio=1.709 ± 0.042
n=150: ratio=1.717 ± 0.03
n=160: ratio=1.73 ± 0.011
n=170: ratio=1.741 ± 0.044
n=180: ratio=1.711 ± 0.022
n=190: ratio=1.68 ± 0.018
n=200: ratio=1.7 ± 0.024
```

The approximation ratio is nearly the same