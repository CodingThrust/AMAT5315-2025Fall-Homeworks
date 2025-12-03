# Homework 9 – Mingxu Zhang-50046133

Code: `PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw9/mingxuzhang/hw9.jl`.

Run with:

```bash
cd PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw9/mingxuzhang
julia --project=. hw9.jl
```

This script reproduces every numerical result below and writes the MIS statistics to `mis_results.csv`.

---

## 1. Circuit-SAT → Spin Glass Ground State

A half-adder over Boolean bits (0/1) satisfies:

* Sum bit: `S = A ⊕ B = A + B − 2AB`
* Carry bit: `C = A ∧ B = AB`

Introduce binary penalty terms that vanish on every satisfying assignment and stay positive otherwise:

```
E_sum   = (S - A - B + 2AB)^2
E_carry = (C - AB)^2
```

Map each Boolean bit to an Ising spin via `b = (1 - s)/2` so that `b=0` ↔ `s=+1` and `b=1` ↔ `s=-1`. After substituting and simplifying both penalties, the total Ising Hamiltonian becomes

```
H(s) = 3
     + s_A + s_B - s_S - 2 s_C
     + s_A s_B - s_A s_S - 2 s_A s_C - s_B s_S - 2 s_B s_C + 2 s_S s_C
```

Written compactly,

- Constant shift: `c0 = 3`
- Local fields: `h_A = h_B = +1`, `h_S = -1`, `h_C = -2`
- Couplers: `J_AB=+1`, `J_AS=J_BS=-1`, `J_AC=J_BC=-2`, `J_SC=+2`.

Evaluating all 16 spin assignments shows that the four logical half-adder inputs (A,B) ∈ {00,01,10,11} with corresponding outputs (S,C) = (0,0),(1,0),(1,0),(0,1) are exactly the ground states with energy −1. All other assignments sit at energy +1 or higher, so minimizing the spin-glass energy solves the original circuit-SAT instance.

---

## 2. Spin Dynamics Readout with Fixed Output

I used a zero-temperature Glauber dynamics (greedy spin-flip) to relax the Ising model while pinning the output spins to S=0 and C=1 (i.e., s_S=+1, s_C=−1). Implementation details:

* Random initial spins for \(A\) and \(B\)
* Repeated sweeps (30 sweeps × 200 random restarts) updating each unfixed spin to the sign that lowers its local field
* Acceptance only for energy-decreasing moves.

Result (see solver log):

| Quantity | Value |
| --- | --- |
| Best constrained energy | -1 |
| Spin configuration | s_A=s_B=-1, s_S=+1, s_C=-1 |
| Bits (after inverse map) | A=1, B=1, S=0, C=1 |

Thus the dynamics recovers the only input consistent with the fixed output (S,C)=(0,1), namely A=B=1.

---

## 3. Greedy MIS on Random 3-Regular Graphs

Algorithm (implemented in Julia):

1. Generate 3-regular simple graphs via the configuration model with rejection until every node has degree 3.
2. **Greedy MIS:** While vertices remain, select a minimum-degree vertex (breaking ties uniformly at random), add it to the independent set, and delete it plus its neighbors.
3. **Exact MIS size:** Solve a binary ILP with JuMP + HiGHS to obtain the optimal independent set for comparison.
4. Repeat for each size n = 10, 20, …, 200 using 3 random graphs per size.

Summary statistics (means taken over the 3 samples/size):

| n | avg. ratio | std. ratio | avg. greedy | avg. optimal |
| --- | --- | --- | --- | --- |
| 10 | 1.000 | 0.000 | 4.00 | 4.00 |
| 20 | 0.963 | 0.064 | 8.33 | 8.67 |
| 30 | 0.952 | 0.082 | 13.00 | 13.67 |
| 40 | 1.000 | 0.000 | 17.67 | 17.67 |
| 50 | 0.970 | 0.026 | 21.00 | 21.67 |
| 60 | 0.974 | 0.022 | 25.67 | 26.33 |
| 70 | 0.968 | 0.032 | 29.67 | 30.67 |
| 80 | 0.953 | 0.017 | 34.00 | 35.67 |
| 90 | 0.967 | 0.038 | 39.00 | 40.33 |
| 100 | 0.962 | 0.026 | 42.67 | 44.33 |
| 110 | 0.946 | 0.011 | 46.33 | 49.00 |
| 120 | 0.962 | 0.000 | 51.00 | 53.00 |
| 130 | 0.960 | 0.009 | 56.00 | 58.33 |
| 140 | 0.962 | 0.009 | 59.67 | 62.00 |
| 150 | 0.956 | 0.025 | 64.33 | 67.33 |
| 160 | 0.958 | 0.014 | 67.67 | 70.67 |
| 170 | 0.956 | 0.015 | 72.33 | 75.67 |
| 180 | 0.929 | 0.019 | 74.33 | 80.00 |
| 190 | 0.976 | 0.012 | 82.33 | 84.33 |
| 200 | 0.955 | 0.000 | 85.00 | 89.00 |

Observations:

* The greedy heuristic consistently attains >92% of optimum, with fluctuations of ~±3%.
* No evidence of a size-independent plateau; the ratio drifts downward slightly (e.g., dip near n=180).

---

All raw data (ratios, greedy sizes, optimal sizes) are stored in `mis_results.csv` for plotting or further analysis.
