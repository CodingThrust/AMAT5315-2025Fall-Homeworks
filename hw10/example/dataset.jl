using Random
using Primes

function random_prime_bits(bits::Int; rng=Random.GLOBAL_RNG)
    # This function is NOT suitable for generating large primes (more than ~120 bits)
    if bits <= 1
        return 2
    end
    min_val = BigInt(2^(bits-1))
    max_val = BigInt(2^bits - 1)
    for _ in 1:1000
        candidate = rand(rng, min_val:max_val)
        if candidate % 2 == 0
            candidate += 1
        end
        if isprime(candidate)
            return BigInt(candidate)
        end
    end
    candidate = rand(rng, min_val:max_val)
    return nextprime(candidate)
end

function random_semiprime(m::Int, n::Int; rng=Random.GLOBAL_RNG, distinct::Bool=true)
    # distinct = true means p ≠ q
    p = random_prime_bits(m; rng)
    q = random_prime_bits(n; rng)

    if distinct && p == q
        while q == p
            q = random_prime_bits(n; rng)
        end
    end
    N = p * q
    return p, q, N
end
