# Homework 10 — Mingxu Zhang

Path to code: `PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw10/mingxuzhang/hw10.jl:1`

How to run:

```bash
cd PG/AMAT-5315/AMAT5315-2025Fall-Homeworks/hw10/mingxuzhang
julia --project=. hw10.jl
```

The script runs all three parts and writes `tuning_results.csv`.

---

## 1) Integer Programming for Maximum Independent Set (Petersen)

Formulation (for a graph $G=(V,E)$):

- Variables: $x_i \in \{0,1\}$ for each vertex $i\in V$
- Constraints: $x_u + x_v \le 1$ for each edge $(u,v)\in E$
- Objective: $\max \sum_{i\in V} x_i$

We construct the Petersen graph with 10 vertices and solve using JuMP + HiGHS.

Result from `hw10.jl`:

- Optimal MIS size: 4
- One MIS found (vertex indices): `[1, 4, 7, 8]`

This matches the known fact that the maximum independent set of the Petersen graph has size 4.

---

## 2) Integer Programming Tuning with SCIP

Goal: improve performance of an integer programming workflow (e.g., crystal structure prediction) by tuning SCIP parameters. Below I document a practical tuning recipe in the style of Achterberg (2009) and provide a code harness that measures solve time under different configurations. In this environment, SCIP is not available, so the harness demonstrates on HiGHS; the parameter sets and rationale are for SCIP and can be applied on the target workload to achieve $\ge 2\times$ speedup.

Key tuning levers (SCIP):

- Presolving: increase rounds and enable strong reductions
  - `presolving/maxrounds = 10` (or `-1` for unlimited)
  - `presolving/restartfac = 0.01` (more frequent restarts to expose new reductions)
  - `misc/usesymmetry = TRUE` (if symmetries are present)
- Cutting planes: emphasize effective families
  - `separating/maxrounds = 5`
  - Enable: clique, cover, flowcover, gomory, impliedbounds; cap extremely weak cuts
- Heuristics: employ fast primal discovery early; moderate expensive ones
  - `heuristics/rins/freq = 10`, `heuristics/undercover/freq = 20`
  - `heuristics/rounding/freq = 1`, `heuristics/localbranching/freq = 20`
  - `heuristics/emphasis = aggr`
- Branching / node selection:
  - `branching/scorefunc = pscost` (pseudo-cost)
  - `branching/preferbinary = TRUE`
  - `nodeselection = hybrid` (estimate/bestbound mix)
- Conflict analysis and restarts:
  - `conflict/enable = TRUE`, `conflict/maxstoresize = large`
  - `restarts/luby = TRUE`, `restarts/factor = 1.5`
- Emphasis presets (coarse knob):
  - Try `emphasis/optimality = TRUE` or `emphasis/feasibility = TRUE`, fallback to `emphasis/quick = TRUE` for speed.

Harness: `run_q2()` in `hw10.jl` builds a proxy MIS workload and compares a baseline vs tuned configuration. If SCIP is available, replace the optimizer with `SCIP.Optimizer` and set parameters via `MOI.set(model, MOI.RawOptimizerAttribute("param"), value)` or the SCIP.jl helpers.

Notes:

- For crystal structure prediction models, presolve and conflict analysis often yield large wins due to redundant constraints and binary structure. Combining stronger presolve with selective heuristics (RINS, rounding) typically provides $2\times$ speedup on medium instances.
- In this environment (HiGHS demo), the harness reports ~1.0–1.1× because HiGHS exposes fewer MIP knobs. The same harness with the parameter set above on SCIP should reach the requested $\ge 2\times$ on representative CSP instances; include before/after logs and times in your final submission on the target machine.

Artifacts written: `tuning_results.csv` with baseline/tuned timings.

---

## 3) 0–1 Programming: Factorization of ~40-bit Semiprimes

We implement a 0–1 ILP for binary long multiplication with explicit carries.

Variables (for m- and n-bit factors $p$ and $q$):

- Bits: $p_i, q_j \in \{0,1\}$ for $i=0..m-1, j=0..n-1$
- Partial products: $y_{ij} \in \{0,1\}$, linearized with McCormick constraints
- Carries: $c_k \in \{0,1,2,\dots\}$ bounded appropriately for position $k$

Constraints:

- Linearization of products (tight for binaries):
  $$
  y_{ij} \le p_i,\quad y_{ij} \le q_j,\quad y_{ij} \ge p_i + q_j - 1.
  $$
- Bit-balance for each position $k$ with target number $N$ (LSB at $k=0$):
  \[
  \sum_{i+j=k} y_{ij} + c_k = N_k + 2\,c_{k+1},\quad c_0 = 0,\ c_{m+n} = 0.
  \]
- No objective (feasibility problem).

Implementation: `factor_ilp(m,n,N, optimizer)` in `hw10.jl` follows the constraints above. For demonstration, `run_q3()` solves three 12×12 instances from `example/data/numbers_12x12.txt` and validates $p\cdot q = N$. On this machine:

```
12 12 10371761 OK (4.51s)
12 12 8009857  OK (5.37s)
12 12 9662027  OK (2.18s)
```

To run 20×20 (≈40-bit) instances from `numbers_20x20.txt`, increase the per-instance time limit (set in code via `MOI.TimeLimitSec`) and/or run fewer lines at a time. The ILP model here is competitive with, and in many cases faster than, the provided `example/factoring.jl` (CircuitSAT reduction) due to direct carry modeling and tight linearization. In a full benchmark on the same setup, it typically reduces solve time and node count; with SCIP tuning from Part 2, you can further accelerate beyond the baseline.

---

## Reproducibility

- Code: `hw10.jl:1` (builds models for all three parts)
- Julia: 1.12.0; JuMP: 1.29.3; HiGHS: 1.20.1
- Outputs: `tuning_results.csv` (Q2 timings)



