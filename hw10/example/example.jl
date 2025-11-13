using SCIP  # or other integer programming solvers

include("dataset.jl")
include("factoring.jl")

"""
    factoring_file(filepath::String, optimizer; line_number::Union{Int, Nothing}=nothing)

This is an example function to read the provided dataset file and factor the numbers in the file.

# Arguments
- `filepath::String`: the path to the dataset file
- `optimizer`: the optimizer to use
- `line_number::Union{Int, Nothing}=nothing`: the line number to factor, if not provided, all lines will be factored

# Available Optimizers <: MathOptInterface.AbstractOptimizer
- SCIP.Optimizer (Open Source)
- HiGHS.Optimizer (Open Source)
- Gurobi.Optimizer (Need License)
- CPLEX.Optimizer (Need License)
"""
function factoring_file(filepath::String, optimizer; line_number::Union{Int, Nothing}=nothing)
    open(filepath, "r") do io
        for (line_num, line) in enumerate(eachline(io))
            if !isnothing(line_number) && line_num != line_number
                continue
            end
            
            line = strip(line)
            isempty(line) && continue
            
            parts = split(line)
            if length(parts) < 3
                @warn "Skipping line $line_num: invalid format"
                continue
            end
            
            m = parse(Int, parts[1])
            n = parse(Int, parts[2])
            N = parse(BigInt, parts[3])
            
            @info "Factoring line $line_num: m=$m, n=$n, N=$N"
            
            factoring(m, n, N, optimizer; verbose=false)  # No detailed output
        end
    end
end

"""
    random_factoring(m::Int, n::Int, optimizer; verbose::Bool=false)

Factorize a random semiprime number using the method of Integer Programming.

# Arguments
- `m`: the number of bit length of the first prime
- `n`: the number of bit length of the second prime
- `optimizer`: the optimizer to use

# Keyword Arguments
- `verbose`: whether to print the verbose output of the IP solver
"""
function random_factoring(m::Int, n::Int, optimizer; verbose::Bool=false)
    _, _, N = random_semiprime(m, n)
    factoring(m, n, N, optimizer; verbose)
end


# Example usage:
# Factoring the numbers in the dataset file
path = joinpath(@__DIR__, "data", "numbers_12x12.txt")
results = factoring_file(path, SCIP.Optimizer)  # Process all lines
results = factoring_file(path, SCIP.Optimizer; line_number=1)  # Process only line 1

# Factoring a random semiprime number
results = random_factoring(15, 15, SCIP.Optimizer; verbose=false)

