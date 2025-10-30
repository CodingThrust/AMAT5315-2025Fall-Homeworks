# HW4
## Task1
```
  using LinearAlgebra

  A = [10^10 0; 0 10^-10]
  B = [10^10 0; 0 10^10]
  C = [10^-10 0; 0 10^-10]
  D = [1 2; 2 4]

  cond_A = cond(A)
  cond_B = cond(B)
  cond_C = cond(C)
  cond_D = cond(D)

  println("(a) : ", cond_A, " → ", cond_A > 1e10 ? "ill-conditioned" : "well-conditioned")
  println("(b) : ", cond_B, " → ", cond_B ≈ 1 ? "well-conditioned" : "ill-conditioned")
  println("(c) : ", cond_C, " → ", cond_C ≈ 1 ? "well-conditioned" : "ill-conditioned")
  println("(d) : ", cond_D, " → ", isinf(cond_D) ? "ill-conditioned" : "well-conditioned")
```
result:
a: ill-conditioned
b: well-conditioned
c: well-conditioned
d: ill-conditioned

## Task2
```
using LinearAlgebra

A = [
    2   1  -1   0   1
    1   3   1  -1   0
    0   1   4   1  -1
   -1   0   1   3   1
    1  -1   0   1   2
]

b = [4, 6, 2, 5, 3]

x = A \ b

println()
for i in 1:5
    println("x_$i = $(x[i])")
end
```
result:
```
  x_1 = -0.04651162790697683
  x_2 = 2.186046511627907
  x_3 = 0.30232558139534904
  x_4 = 0.8139534883720929
  x_5 = 2.2093023255813957
```

## Task3
in hw4.jl and population_fit.png