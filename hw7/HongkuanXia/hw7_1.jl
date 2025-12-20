using Graphs, ProblemReductions
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

function annealing(nrun::Int, sg::SpinGlass, tempscales::Vector{Float64}, num_update_each_temp::Int)
    
    local opt_config, opt_cost
    for r = 1:nrun
        initial_config = rand([-1, 1], nv(sg.graph))
        cost, config = annealing_singlerun!(sg, initial_config, tempscales, num_update_each_temp)
        if r == 1 || cost < opt_cost
            opt_cost = cost
            opt_config = config
        end
    end
    return opt_cost, opt_config
end

function annealing_singlerun!(sg::SpinGlass, spins::Vector{Int}, tempscales::Vector{Float64}, num_update_each_temp::Int)
    cost = energy(sg, spins)
    
    opt_config = deepcopy(spins)
    opt_cost = cost

    for β in 1 ./ tempscales
        for _ in 1:num_update_each_temp
            flip_spin = rand(1:length(spins))
            ΔE = energy_change(sg,spins,flip_spin)
            if exp(-β*ΔE) > rand()  # Accept move
                spins[flip_spin] = -spins[flip_spin]
                cost += ΔE
                # Track the best configuration seen
                if cost < opt_cost
                    opt_cost = cost
                    opt_config = deepcopy(spins)
                end
            end
        end
    end
    return opt_cost, opt_config
end

"""
    energy(sg::SpinGlass,spins::Vector{Int}) -> Float64

Calculate the total energy of the spinglass

#Arguments
- `sg::SpinGlass`: SpinGlass config with coupling constants
- `spins::Vector{Int}`: spin configuration with 1 indicates up spin, -1 indicates down spin


#Returns

- Energy value: E = Σᵢⱼ J_{ij}σᵢσⱼ

#Examples
```julia
```
"""
function energy(sg::SpinGlass,spins::Vector{Int})
    E = 0.0
    J = sg.J
    h = sg.h
    for (idx,edge) in enumerate(edges(sg.graph))
        i,j = src(edge),dst(edge)
        E += J[idx]*spins[i]*spins[j]
    end

    E += spins'* h

    return E
end

function energy_change(sg::SpinGlass,spins::Vector{Int},spin_idx::Int)
    
    ΔE = 0.0
    for neighbor in neighbors(sg.graph, spin_idx)
        
        edge_idx = find_edge_index(sg.graph, spin_idx, neighbor)
        if edge_idx !== nothing
            ΔE -= 2 * sg.J[edge_idx] * spins[spin_idx] * spins[neighbor]
        end
    end

    ΔE -= 2*spins[spin_idx]*sg.h[spin_idx]

    return ΔE
end
function find_edge_index(graph, i::Int, j::Int)
    for (idx, edge) in enumerate(edges(graph))
        if (src(edge) == i && dst(edge) == j) || (src(edge) == j && dst(edge) == i)
            return idx
        end
    end
    return nothing
end


fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5)); # construct the unit disk graph

spin_glass = SpinGlass(
            fullerene_graph,   # graph
           ones(Int, ne(fullerene_graph)),     # J, in order of edges
           zeros(Int, nv(fullerene_graph))     # h, in order of vertices
       )


tempscales = collect(10 .- (1:64 .- 1) .* 0.15)
gs_energy,_ = annealing(20,spin_glass,tempscales,4000)
@show gs_energy
