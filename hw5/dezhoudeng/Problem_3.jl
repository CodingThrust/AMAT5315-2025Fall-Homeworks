using LinearAlgebra

function restarting_lanczos(A, q1::AbstractVector{T}; s::Int, max_restarts::Int=100) where T
    """
    Parameter:
    A - Hermitian Matrix
    q1 - initial vector
    s - subspace size
    max_restarts - max restarts times
    
    return:
    λ - estimated the largest eigenvalue
    v - eigenvector
    """
    
    # normalization
    q_current = normalize(q1)
    
    for restart in 1:max_restarts
        q_basis = [q_current]  
        α = [real(dot(q_current, A * q_current))]  
        β = T[]  
        
        # Lanczos iteration
        for k in 2:s
            if k == 2
                # Compute first residual: r₁ = Aq₁ - α₁q₁
                Aq1 = A * q_basis[1]
                rk = Aq1 .- α[1] .* q_basis[1]
            else
                # Compute diagonal element: α_k = q_k' * A * q_k
                Aqk = A * q_basis[k-1]
                rk = Aqk .- α[k-1] .* q_basis[k-1] .- β[k-2] * q_basis[k-2]
            end
            
            
            βk = norm(rk)
            push!(β, βk)
            
            
            push!(q_basis, rk ./ βk)
            
          
            Aqk_new = A * q_basis[k]
            push!(α, real(dot(q_basis[k], Aqk_new)))
        end
        
        
        T_matrix = SymTridiagonal(α, β)
        
        
        F = eigen(Matrix(T_matrix))
        θ = real(F.values)
        U = F.vectors
        
        perm = sortperm(θ, rev=true)
        θ_sorted = θ[perm]
        U_sorted = U[:, perm]
        
        Q_matrix = hcat(q_basis...)  
        q_new = Q_matrix * U_sorted[:, 1]
        q_new = normalize(q_new)
        
        q_current = q_new
    end
    
    v_final = q_current
    λ_final = real(dot(v_final, A * v_final))
    
    return λ_final, v_final
end