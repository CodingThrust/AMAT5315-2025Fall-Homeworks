using SparseArrays, LinearAlgebra

"""
    function restarting_lanczos(A, q1::AbstractVector{T}; s::Int=20,
                            maxiter::Int=20,abstol::Real=1e-10 ) -> (Real,Vector)
Restarting Lanczos algorithm to find the largest eigenvalue of a Hermitian matrix.

# Algorithm:
The restatring lanczos algorithm builds an orthonormal basis for the Krylov subspace
K_m(A, q₁) = span{q₁, Aq₁, A²q₁, ..., A^(m-1)q₁} and produces a tridiagonal
matrix T that approximates A in this subspace.

# Arguments:
- `A`: Hermitian matrix (can be sparse)
- `q1::AbstractVector`: Initial vector
- `s::Int`: Number of Lanczos iteration steps per restart (default: 20)
- `maxiter::Int`: Maximum number of iterations
- `abstol::Real`: Absolute tolerance (default: 1e-10)

# Returns:
- maximal eigenvalue of A

# Example:
```julia
A = Symmetric(rand(100,100))
q1 = randn(100)
λ_max = restarting_lanczos(A,q1)
```
"""
function restarting_lanczos(A, q1::AbstractVector{T}; s::Int=20,
                            maxiter::Int=100,abstol::Real=1e-10 ) where T

    row_number = length(q1)

    # Normalize the initial vector
    q1 = normalize(q1)

    # Initialize storage for basis vectors and tridiagonal matrix elements
    q = zeros(T,row_number,min(row_number, s))    # Orthonormal basis vectors
    q[:,1] = q1
    α = zeros(T,min(row_number, s))  # Diagonal elements of tridiagonal matrix 
    α[1] = q1'*(A*q1)

    # Compute the first residual
    rk = A*q1 - α[1] .*q1
    β = zeros(T,min(row_number, s)-1) # Off-diagonal elements of tridiagonal matrix
    β[1] = norm(rk)

    # convergence check varibles
    convergence_condition = abs(β[1])
    iteration = 0

    # Main Lanczos iteration
    while convergence_condition > abstol || iteration < maxiter
        for k = 2:min(length(q1), s)
            # Compute next basis vector: q_k = r_{k-1}/β_{k-1}
            q[:,k] = rk ./ β[k-1]

            # medium varible
            Aqk = A*q[:,k]

            # Compute diagonal element: α_k = q_k' * A * q_k

            α[k] = q[:,k]' * Aqk

            # Compute residual: r_k = A*q_k - α_k*q_k - β_{k-1}*q_{k-1}
            # This enforces orthogonality to the previous two vectors
            rk = Aqk -  α[k] .*q[:,k] - β[k-1]*q[:,k-1]
            # Compute the residual norm 
            nrk = norm(rk)
            convergence_condition = abs(nrk)
            if convergence_condition < abstol
                return eigen(SymTridiagonal(α, β)).values[end]
            end

            if k < min(row_number,s)
                β[k] = nrk
            end
        end
        iteration +=1
        if iteration == maxiter
            return eigen(SymTridiagonal(α, β)).values[end]
        end
        Ts =  q'*A*q
        u1 = eigen(Ts).vectors[:,end]
        q[:,1] = normalize(q*u1)
        α[1] = q[:,1]'*(A*q[:,1])

        # Compute the first residual
        rk = A*q[:,1] - α[1] .*q[:,1]
        β[1] = norm(rk)
    end

    return eigen(SymTridiagonal(α, β)).values[end]
end


using Test
@testset "restarting_lanczos" begin
    A = Symmetric(rand(100,100))
    q1 = randn(100)
    @test restarting_lanczos(A,q1) ≈ eigen(A).values[end]
end