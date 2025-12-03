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
