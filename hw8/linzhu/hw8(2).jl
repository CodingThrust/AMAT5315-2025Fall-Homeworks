using OMEinsum, LinearAlgebra

# Random example tensors (all legs dimension 2 for convenience)
T1 = randn(2, 2, 2)  # indices a,b,c
T2 = randn(2, 2, 2)  # f,b,g
T3 = randn(2, 2, 2)  # h,g,e
T4 = randn(2, 2, 2)  # d,c,e

# Direct contraction in one shot
C_direct = ein"abc, fbg, dce, hge -> afdh"(T1, T2, T4, T3)

# Optimal contraction order found by OMEinsum:
# 1) abc, dce -> abde
X = ein"abc, dce -> abde"(T1, T4)

# 2) hge, fbg -> hefb
Y = ein"hge, fbg -> hefb"(T3, T2)

# 3) abde, hefb -> afdh
C_opt = ein"abde, hefb -> afdh"(X, Y)

@assert C_direct ≈ C_opt
println("Contractions match, optimal order implemented correctly.")
