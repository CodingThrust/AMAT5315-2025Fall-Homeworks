#!/usr/bin/env julia
"""
Quick runner script for Homework 5 Solution

This script checks prerequisites and runs the complete homework solution.

Usage:
    julia run_solution.jl
"""

println("=" ^ 70)
println("Homework 5 Solution Runner")
println("=" ^ 70)

# Check if required packages are available
println("\n[1] Checking required packages...")
required_packages = [
    "MLDatasets", "LinearAlgebra", "Images", "Plots", 
    "Statistics", "FileIO", "ColorTypes", "JLD2", "FFTW"
]

missing_packages = String[]
for pkg in required_packages
    try
        eval(Meta.parse("using $pkg"))
        println("  ✓ $pkg")
    catch
        println("  ✗ $pkg (MISSING)")
        push!(missing_packages, pkg)
    end
end

if !isempty(missing_packages)
    println("\n❌ Missing packages detected!")
    println("Please install them by running:")
    println("  julia install_packages.jl")
    println("\nOr manually:")
    println("  using Pkg")
    for pkg in missing_packages
        println("  Pkg.add(\"$pkg\")")
    end
    exit(1)
end

println("\n✓ All required packages are available!")

# Check if cat.png exists
println("\n[2] Checking required files...")
cat_image_path = "../cat.png"
if isfile(cat_image_path)
    println("  ✓ cat.png found")
else
    println("  ✗ cat.png NOT FOUND at: $cat_image_path")
    println("\n❌ Required file missing!")
    exit(1)
end

println("\n✓ All required files are available!")

# Run the solution
println("\n[3] Running homework solution...")
println("This may take several minutes...\n")

try
    include("homework5_solution.jl")
    main()
    
    println("\n" * "=" ^ 70)
    println("✓✓✓ HOMEWORK 5 COMPLETED SUCCESSFULLY! ✓✓✓")
    println("=" ^ 70)
    println("\nResults have been saved in:")
    println("  - output_problem1/")
    println("  - output_problem2/")
    println("\nPlease review the generated images and analysis files.")
    
catch e
    println("\n" * "=" ^ 70)
    println("❌ Error during execution:")
    println("=" ^ 70)
    println(e)
    println("\nStack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

