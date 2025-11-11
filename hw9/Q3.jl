using Graphs, Random, Statistics


function greedy_mis(g::SimpleGraph)
    indep = Int[]
    remaining = Set(vertices(g))
    while !isempty(remaining)
        # pick vertex with minimum degree
        v = argmin(u -> degree(g, u), remaining)
        push!(indep, v)
        # remove v and its neighbors
        delete!(remaining, v)
        for nb in neighbors(g, v)
            delete!(remaining, nb)
        end
    end
    return indep
end


function test_scaling(ns = 10:10:200; trials=10)
    ratios = Float64[]
    for n in ns
        local_r = Float64[]
        for _ in 1:trials
            g = random_regular_graph(n, 3)
            sol = greedy_mis(g)
            approx_ratio = length(sol) / (n/2)   # n/2 ≈ theoretical upper bound
            push!(local_r, approx_ratio)
        end
        push!(ratios, mean(local_r))
    end
    return collect(ns), ratios
end

ns, ratios = test_scaling()
for (n,r) in zip(ns, ratios)
    println("n = $n,  average ratio = $(round(r, digits=3))")
end
