# HW9

## Task1

**1. Boolean (0/1) formulation**

Let (A, B, S, C \in {0,1}).
Define the energy function:
[
\boxed{H = (S - A - B + 2AB)^2 + (C - AB)^2.}
]

* (H = 0) if the logic constraints are satisfied.
* (H > 0) otherwise.
  Thus, minimizing (H) (finding the ground state) solves the circuit SAT.

---

**2. Ising (±1) spin form**

Using (x = (1 - \sigma)/2), we can express it as:
[
H = J_1(1 - \sigma_A\sigma_B\sigma_S) + J_2(1 - \sigma_A\sigma_B)(1 - \sigma_C),
]
where (\sigma_i \in {\pm 1}) and (J_1, J_2 > 0).

---

**3. Summary**

This Hamiltonian’s ground states correspond exactly to the valid half-adder truth table:

| A | B | S | C |
| - | - | - | - |
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 |

Hence, the **half-adder SAT** problem is successfully reduced to finding the **spin-glass ground state** of (H).
