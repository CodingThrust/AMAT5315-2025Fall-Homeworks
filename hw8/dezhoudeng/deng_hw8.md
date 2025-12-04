# Homework 8

## Problem 1

1. ik,jk -> ij

2. ij ->

3. ij,ij,ij -> ij

4. ij,kl,mn  -> ikmjln

## Problem 2

```

using OMEinsum

function main()
    println("=== Tensor Network Contraction Order Optimization ===")
    println()

    # Define Einstein summation notation
    # Based on the network connectivity:
    # T1 connects: A, B, C → indices: a,b,c
    # T2 connects: B, F, G → indices: b,f,g
    # T3 connects: G, E, H → indices: g,e,h
    # T4 connects: C, D, E → indices: c,d,e
    code = ein"abc,bfg,geh,cde -> afdh"

    println("1. Original contraction order:")
    println("   ", code)
    println()

    # Optimize contraction order
    println("2. Optimized contraction order:")
    optcode = optimize_code(code, uniformsize(code, 2), TreeSA())
    println("   ", optcode)

    println("=== Optimization Complete ===")
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

```

=== Tensor Network Contraction Order Optimization ===

1. Original contraction order:
   abc, bfg, geh, cde -> afdh

2. Optimized contraction order:
   SlicedEinsum{Char, DynamicNestedEinsum{Char}}(Char[], acfg, cdgh -> afdh
├─ abc, bfg -> acfg
│  ├─ abc
│  └─ bfg
└─ cde, geh -> cdgh
   ├─ cde
   └─ geh
)
=== Optimization Complete ===

```

## Problem 3

```
julia> using Graphs, ProblemReductions

julia> function fullerene()  # construct the fullerene graph in 3D space
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
fullerene (generic function with 1 method)

julia> fullerene_graph = UnitDiskGraph(fullerene(), sqrt(5));

julia> using OMEinsum, ProblemReductions, Graphs

julia> rawcode = EinCode([[e.src, e.dst] for e in edges(fullerene_graph)], Int[])
1∘3, 1∘37, 1∘41, 2∘4, 2∘38, 2∘42, 3∘39, 3∘43, 4∘40, 4∘44, 5∘7, 5∘45, 5∘46, 6∘8, 6∘47, 6∘48, 7∘49, 7∘50, 8∘51, 8∘52, 9∘10, 9∘53, 9∘55, 10∘54, 10∘56, 11∘12, 11∘57, 11∘59, 12∘58, 12∘60, 13∘17, 13∘37, 13∘45, 14∘18, 14∘38, 14∘46, 15∘19, 15∘39, 15∘47, 16∘20, 16∘40, 16∘48, 17∘41, 17∘49, 18∘42, 18∘50, 19∘43, 19∘51, 20∘44, 20∘52, 21∘22, 21∘45, 21∘53, 22∘46, 22∘54, 23∘24, 23∘47, 23∘55, 24∘48, 24∘56, 25∘26, 25∘49, 25∘57, 26∘50, 26∘58, 27∘28, 27∘51, 27∘59, 28∘52, 28∘60, 29∘31, 29∘37, 29∘53, 30∘32, 30∘38, 30∘54, 31∘39, 31∘55, 32∘40, 32∘56, 33∘35, 33∘41, 33∘57, 34∘36, 34∘42, 34∘58, 35∘43, 35∘59, 36∘44, 36∘60 ->

julia> optcode = optimize_code(rawcode, uniformsize(rawcode, 2), TreeSA())
SlicedEinsum{Int64, DynamicNestedEinsum{Int64}}(Int64[], 2∘38, 2∘38 ->
├─ 14∘38, 38∘2∘14 -> 2∘38
│  ├─ 14∘38
│  └─ 38∘54∘32, 54∘2∘14∘32 -> 38∘2∘14
│     ├─ 30∘38, 54∘32∘30 -> 38∘54∘32
│     │  ├─ 30∘38
│     │  └─ 30∘54, 30∘32 -> 54∘32∘30
│     │     ├─ 30∘54
│     │     └─ 30∘32
│     └─ 14∘7∘17∘1∘32∘47∘24∘54∘31, 31∘24∘32∘14∘7∘17∘2∘47∘1 -> 54∘2∘14∘32
│        ├─ 14∘7∘54∘17∘31∘1∘53, 53∘32∘54∘31∘47∘24 -> 14∘7∘17∘1∘32∘47∘24∘54∘31
│        │  ├─ 14∘7∘54∘53∘45, 45∘17∘31∘53∘1 -> 14∘7∘54∘17∘31∘1∘53
│        │  │  ⋮
│        │  │
│        │  └─ 53∘24∘32∘54∘9, 31∘9∘47∘24 -> 53∘32∘54∘31∘47∘24
│        │     ⋮
│        │
│        └─ 35∘51∘47∘31∘1, 47∘24∘32∘14∘7∘1∘17∘35∘51∘2 -> 31∘24∘32∘14∘7∘17∘2∘47∘1
│           ├─ 35∘43, 51∘47∘31∘1∘43 -> 35∘51∘47∘31∘1
│           │  ⋮
│           │
│           └─ 51∘47∘24∘2∘32∘36∘52, 14∘2∘7∘52∘51∘36∘1∘17∘35 -> 47∘24∘32∘14∘7∘1∘17∘35∘51∘2
│              ⋮
│
└─ 2∘38
)

julia> sitetensors(β::Real) = [[exp(-β) exp(β); exp(β) exp(-β)] for _ in 1:ne(fullerene_graph)]
sitetensors (generic function with 1 method)

julia> partition_func(β::Real) = only(optcode(sitetensors(β)...))
partition_func (generic function with 1 method)

julia> Zs = [partition_func(β) for β in 0.1:0.1:2.0]
20-element Vector{Float64}:
 1.8066109403976796e18
 6.87566576214655e18
 6.15112258172142e19
 1.2302123108278566e21
 5.14455871618204e22
 4.1116282893989694e24
 5.612702683289911e26
 1.1622488726804851e29
 3.285330233500941e31
 1.1667408970547078e34
 4.898099153436166e36
 2.3273948792397587e39
 1.2136776638732638e42
 6.793958994977614e44
 4.017291439227067e47
 2.479402296712646e50
 1.5828842349953101e53
 1.0380999176571235e56
 6.95642903063234e58
 4.7430829554740573e61

```

