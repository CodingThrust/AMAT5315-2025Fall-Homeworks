# Copied from hw4/longlizheng

using LinearAlgebra
using Makie, CairoMakie

N = 1024

K = zeros(Int, (N, N));
K[1, 1] = -2; K[1, 2] = 1; K[1, N] = 1;
K[N, N] = -2; K[N, N-1] = 1; K[N, 1] = 1;
for i in 2:N-1
    K[i, i-1] = 1;
    K[i, i] = -2;
    K[i, i+1] = 1;
end

# dual species
M = Diagonal(collect(i%2 + 1 for i in 1:N))
# single species
M = Diagonal(collect(1 for _ in 1:N))

omega = .√abs.(eigen(K, -M).values)

C = 1.0  # Spring constant

# Function to create the dynamical matrix for dual species spring chain
function dual_species_matrix(N, C)
    H = zeros(N, N)
    
    for i in 1:N
        # Mass: 1 for even sites, 2 for odd sites
        m_i = (i % 2 == 0) ? 1.0 : 2.0
        
        # Diagonal term
        H[i, i] = 2*C / m_i
        
        # Off-diagonal terms with periodic boundary conditions
        j_next = (i % N) + 1
        j_prev = (i == 1) ? N : i - 1
        
        m_next = (j_next % 2 == 0) ? 1.0 : 2.0
        m_prev = (j_prev % 2 == 0) ? 1.0 : 2.0
        
        H[i, j_next] = -C / sqrt(m_i * m_next)
        H[i, j_prev] = -C / sqrt(m_i * m_prev)
    end
    
    return H
end

# Function for single species spring chain (for comparison)
function single_species_matrix(N, C)
    H = zeros(N, N)
    
    for i in 1:N
        H[i, i] = 2*C
        
        j_next = (i % N) + 1
        j_prev = (i == 1) ? N : i - 1
        
        H[i, j_next] = -C
        H[i, j_prev] = -C
    end
    
    return H
end

eigenvals_dual = eigvals(dual_species_matrix(N, C))
eigenvals_single = eigvals(single_species_matrix(N, C))