using Random, Statistics


function H_bool(x::NTuple{4,Int})
    A,B,S,C = x
    term_xor = (S - A - B + 2*A*B)
    term_and = (C - A*B)
    return term_xor^2 + term_and^2
end


function find_inputs_with_clamped_outputs(S_clamp::Int, C_clamp::Int; 
        max_iters=20000, T0=1.0, Tf=1e-3, restarts=50, rngseed=1234)

    Random.seed!(rngseed)
    best = nothing
    bestE = Inf
    for r in 1:restarts
        # random initial A,B
        A = rand([0,1])
        B = rand([0,1])
        S = S_clamp
        C = C_clamp
        x = (A,B,S,C)
        E = H_bool(x)
        # geometric cooling
        for t in 1:max_iters
            T = T0 * (Tf/T0)^(t/max_iters)
            # propose flip A or B
            if rand() < 0.5
                A2 = 1 - A
                x2 = (A2, B, S, C)
            else
                B2 = 1 - B
                x2 = (A, B2, S, C)
            end
            E2 = H_bool(x2)
            Δ = E2 - E
            if Δ <= 0 || rand() < exp(-Δ / max(T,1e-12))
                # accept
                A, B = x2[1], x2[2]
                x = x2
                E = E2
            end
            # quick exit if zero energy found
            if E == 0
                break
            end
        end
        if E < bestE
            bestE = E
            best = x
        end
    end
    return best, bestE
end


best, bestE = find_inputs_with_clamped_outputs(0, 1; max_iters=20000, restarts=100)
println("Best config (A,B,S,C) = ", best, "  Energy = ", bestE)
