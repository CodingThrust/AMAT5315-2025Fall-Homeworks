# Homework 6 — Solutions

---

## Problem 1 — Sparse Matrix Construction

### Answer

**Correct triplets:**

```julia
rowindices = [3, 1, 1, 4, 5]
colindices = [1, 2, 3, 3, 4]
data       = [0.799, 0.942, 0.848, 0.164, 0.637]
```

**Reasoning:** In CSC, for each column `j`, the slice `p ∈ colptr[j] : colptr[j+1]-1` indexes into `rowval/nzval`. Decoding column by column yields the five nonzeros:

* (3,1)=0.799; (1,2)=0.942; (1,3)=0.848; (4,3)=0.164; (5,4)=0.637.
  Reconstructing with `sparse(rowindices, colindices, data, 5, 5)` exactly matches the given `colptr/rowval/nzval` (program check: **Exact CSC match? true**, Frobenius norm diff = 0.0). 

---

## Problem 2 — Graph Spectral Analysis

### Answer

**Result:** The graph has **1 connected component**.

**Evidence from eigenvalues:** The computed smallest Laplacian eigenvalues start at

```
[-3.1553757484153867e-15, 0.1717316, 0.1721658, 0.1726795, 0.1729978, 0.1731124, 0.1733361, 0.1734802, 0.1736369]
```

Counting eigenvalues with `|λ| < 1e-6` gives **1** near-zero eigenvalue → **1** connected component (matches a direct connectivity check of 1). The tiny negative first value is numerical roundoff and treated as zero under the tolerance. 

**Method (brief):** I formed a Laplacian **linear operator** (Lx = Dx - Ax) and used `KrylovKit.eigsolve` to compute the few smallest eigenvalues efficiently for (n=10^5). The multiplicity of the zero eigenvalue equals the number of connected components.

---

## Problem 3 — Restarting Lanczos Algorithm

### Answer

**Implementation outline:** Per cycle, run $s$ Lanczos steps with (full) re-orthogonalization to obtain $Q$ and tridiagonal $T$; take the **largest** Ritz pair $(\theta_1, u_1)$ of $T$, lift $v = Q\,u_1$ as the new start vector, and repeat until (i) Ritz value stabilizes and/or (ii) the residual $\|A v - \lambda v\|$ is small.

**Test (dense symmetric $400\times 400$):**

* `λ_true (eigmax)` = 27.582551517878244
* `λ_est  (Lanczos)` = 27.582551517877082
* Relative error = **4.21×10⁻¹⁴**; residual norm = **2.33×10⁻⁶**
* Convergence history (per restart): `[27.574190377444157, 27.582551431861766, 27.582551517877082]`
  These confirm correctness and rapid convergence. 

---
