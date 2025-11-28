using LinearAlgebra, Random

function lanczos_tridiag(A, q1, s; reorth=true)
    n = length(q1)
    Q = zeros(eltype(q1), n, s)
    α = zeros(eltype(q1), s)
    β = zeros(eltype(q1), max(0, s-1))

    Q[:,1] .= q1 ./ norm(q1)
    v = similar(q1)

    for j in 1:s
        v .= A * Q[:,j]
        α[j] = dot(conj(Q[:,j]), v)
        v .-= α[j] .* Q[:,j]
        if j > 1
            v .-= β[j-1] .* Q[:,j-1]
        end

        if reorth
            for k in 1:j
                coeff = dot(conj(Q[:,k]), v)
                v .-= coeff .* Q[:,k]
            end
        end

        if j < s
            βj = norm(v)
            if βj == 0
                return Q[:,1:j], α[1:j], β[1:j-1]
            end
            β[j] = βj
            Q[:, j+1] .= v ./ βj
        end
    end
    return Q, α, β
end


function restarted_lanczos_maxeig(A; s=20, max_restarts=100, tol=1e-10, q0=nothing, verbose=false)
    n = size(A,1)
    q = q0 === nothing ? randn(ComplexF64, n) : q0
    q ./= norm(q)

    prev_theta = -Inf
    history = Float64[]

    for restart in 1:max_restarts
        Q, α, β = lanczos_tridiag(A, q, s; reorth=true)
        m = length(α)
        T = zeros(eltype(A), m, m)
        for i in 1:m
            T[i,i] = α[i]
            if i < m
                T[i,i+1] = β[i]
                T[i+1,i] = conj(β[i])
            end
        end

        eigT = eigen(Hermitian(T))
        θ = eigT.values[end]
        u = eigT.vectors[:, end]
        push!(history, real(θ))

        q = Q * u
        q ./= norm(q)

        if verbose
            println("Restart $restart: θ_max = $(real(θ)), Δ = $(abs(real(θ)-prev_theta)))")
        end

        if restart > 1 && abs(real(θ) - prev_theta) < tol
            return real(θ), q, Dict(:restarts=>restart, :history=>history)
        end
        prev_theta = real(θ)
    end

    return real(prev_theta), q, Dict(:restarts=>max_restarts, :history=>history)
end


function test_restarted_lanczos()
    Random.seed!(123)
    n = 200
    # Test 1: diagonal matrix
    D = Diagonal(collect(1.0:n))
    λ_true = maximum(diag(D))
    λ_est, v, info = restarted_lanczos_maxeig(D; s=20, tol=1e-12)
    println("Diagonal test: true=$λ_true, est=$λ_est, error=$(abs(λ_true-λ_est))")

    # Test 2: random Hermitian
    U = qr(randn(n,n)).Q
    vals = range(1.0, 10.0, length=n)
    A = U * Diagonal(vals) * U'
    λ_true = maximum(vals)
    λ_est, v, info = restarted_lanczos_maxeig(A; s=30, tol=1e-10, verbose=true)
    println("Hermitian test: true=$λ_true, est=$λ_est, error=$(abs(λ_true-λ_est))")
end

test_restarted_lanczos()