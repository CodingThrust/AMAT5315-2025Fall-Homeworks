"""
Package Installation Script for Homework 5
==========================================

This script installs all required Julia packages for the homework solution.
Run this once before running the main solution.

Usage:
    julia install_packages.jl
"""

using Pkg

println("=" ^ 70)
println("Installing required packages for Homework 5...")
println("=" ^ 70)

# List of required packages
packages = [
    "MLDatasets",
    "Images", 
    "Plots",
    "FileIO",
    "ColorTypes",
    "JLD2",
    "FFTW"
]

println("\nPackages to install:")
for pkg in packages
    println("  - $pkg")
end

println("\nInstalling packages...")
println("This may take a few minutes...\n")

try
    Pkg.add(packages)
    println("\n" * "=" ^ 70)
    println("✓ All packages installed successfully!")
    println("=" ^ 70)
    println("\nYou can now run the homework solution:")
    println("  julia homework5_solution.jl")
    println("\nOr in Julia REPL:")
    println("  include(\"homework5_solution.jl\")")
    println("  main()")
catch e
    println("\n" * "=" ^ 70)
    println("❌ Error during installation:")
    println("=" ^ 70)
    println(e)
    println("\nPlease try installing packages manually:")
    println("  using Pkg")
    for pkg in packages
        println("  Pkg.add(\"$pkg\")")
    end
end

