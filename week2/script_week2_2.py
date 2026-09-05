import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee

def poisson_2d(n, scaled=True):
    # A_2D on an n x n interior grid, lexicographic ordering, as a Kronecker sum.
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    I = sp.identity(n, format="csr")
    # format="csr" matters: the default lets scipy return BSR, which stores whole
    # dense blocks and so carries explicit zeros. eliminate_zeros drops any that survive
    # the addition, so nnz counts below are the real structural ones.
    T2D = sp.kron(I, T, format="csr") + sp.kron(T, I, format="csr")
    h = 1 / (n + 1)
    A = (T2D / h**2 if scaled else T2D).tocsr()
    A.eliminate_zeros()
    return A


def poisson_2d_by_hand(n):
    # The same matrix, assembled by looping over the grid and writing the stencil.
    N = n * n
    A = sp.lil_matrix((N, N))
    idx = lambda i, j: j * n + i          # p = (j-1)n + i, zero-based
    for j in range(n):
        for i in range(n):
            p = idx(i, j)
            A[p, p] = 4.0
            if i > 0:     A[p, idx(i - 1, j)] = -1.0
            if i < n - 1: A[p, idx(i + 1, j)] = -1.0
            if j > 0:     A[p, idx(i, j - 1)] = -1.0
            if j < n - 1: A[p, idx(i, j + 1)] = -1.0
    return A.tocsr()


n = 6
diff = (poisson_2d(n, scaled=False) - poisson_2d_by_hand(n))
print(f"n = {n}:  max |Kronecker - by hand| = {abs(diff).max() if diff.nnz else 0.0}")

A = poisson_2d(4, scaled=False)
print(f"\nn = 4 (N = 16): nnz = {A.nnz} (expected {5*16 - 4*4}), "
      f"max nonzeros in a row = {np.diff(A.indptr).max()}")
print("\nleading 8x8 block (the first two grid rows):")
print(A.toarray()[:8, :8].astype(int))

n = 8
h = 1 / (n + 1)
A = poisson_2d(n).toarray()

theta = 4 * np.sin(np.arange(1, n + 1) * np.pi / (2 * (n + 1)))**2
predicted = np.sort(np.add.outer(theta, theta).ravel() / h**2)
computed = np.sort(np.linalg.eigvalsh(A))

print(f"n = {n} (N = {n*n} eigenvalues)")
print(f"max |computed - predicted| = {np.max(np.abs(computed - predicted)):.3e}")
print(f"all eigenvalues positive?    {np.all(computed > 0)}   -> SPD")

T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).toarray()
lam_T = np.linalg.eigvalsh(T)
print(f"\nkappa(A_2D) = {computed[-1] / computed[0]:.6f}")
print(f"kappa(T)    = {lam_T[-1] / lam_T[0]:.6f}   <- the same number")

# And the eigenvector claim: v^(k) tensor v^(l) really is an eigenvector.
v = np.linalg.eigh(T)[1]
w = np.kron(v[:, 2], v[:, 5])
print(f"\nresidual ||A(v2 x v5) - lambda (v2 x v5)|| = "
      f"{np.linalg.norm(A @ w - (theta[2] + theta[5]) / h**2 * w):.3e}")

fill_rows = []
for n in [10, 20, 30, 40, 60, 80, 120, 160]:
    A = poisson_2d(n).tocsc()
    N = n * n
    # diag_pivot_thresh=0 turns pivoting off: the matrix is SPD, so it is safe, and it
    # lets us see the fill the ordering alone produces.
    nat = spla.splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    red = spla.splu(A, permc_spec="COLAMD", diag_pivot_thresh=0.0)
    fill_rows.append((n, N, A.nnz, nat.L.nnz + nat.U.nnz, red.L.nnz + red.U.nnz))

print(f"{'n':>5} {'N':>7} {'nnz(A)':>9} {'fill: natural':>14} {'fill: COLAMD':>13} {'N^1.5':>10}")
for n, N, nnz, f_nat, f_red in fill_rows:
    print(f"{n:>5} {N:>7} {nnz:>9} {f_nat:>14} {f_red:>13} {N**1.5:>10.0f}")

Ns = np.array([r[1] for r in fill_rows], float)
for k, name in [(3, "natural"), (4, "COLAMD ")]:
    y = np.array([r[k] for r in fill_rows], float)
    slope = np.polyfit(np.log(Ns[-4:]), np.log(y[-4:]), 1)[0]
    print(f"\nfitted exponent, {name}: fill ~ N^{slope:.3f}")

f_nat = np.array([r[3] for r in fill_rows], float)
f_red = np.array([r[4] for r in fill_rows], float)
nnzA = np.array([r[2] for r in fill_rows], float)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
# The reference goes on FIRST, thick and pale, so the measured curve can sit on top of
# it: the natural-ordering fill lands on the theory line, which is easier to see as a
# halo than as a dashed line hidden underneath.
ax.loglog(Ns, f_nat[-1] * (Ns / Ns[-1])**1.5, "-", color="0.62", lw=7, alpha=0.55,
          solid_capstyle="round", zorder=1, label=r"reference $N^{3/2}$")
ax.loglog(Ns, f_nat, "-o", lw=2, ms=7, zorder=3, label="fill, natural ordering")
ax.loglog(Ns, f_red, "-s", lw=2, ms=7, zorder=3, label="fill, COLAMD ordering")
ax.loglog(Ns, nnzA, "-^", lw=2, ms=7, zorder=3, label="nnz($A$) itself")

ax.set_xlabel("$N = n^2$ unknowns")
ax.set_ylabel("nonzeros")
ax.set_title("The matrix stays sparse; its factors do not")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.show()

def to_banded_upper(A, b):
    # Upper-triangular banded storage for solveh_banded: ab[b + i - j, j] = A[i, j], i <= j.
    A = A.tocoo()
    ab = np.zeros((b + 1, A.shape[0]))
    m = A.col >= A.row
    ab[b + A.row[m] - A.col[m], A.col[m]] = A.data[m]
    return ab


def timeit(f, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


ns_dense = [10, 20, 30, 40, 60]

print(f"{'n':>5} {'N':>7} {'sparse (s)':>12} {'banded (s)':>12} {'dense (s)':>11} {'dense MB':>10}")
for n in [10, 20, 30, 40, 60, 80, 120, 160]:
    A = poisson_2d(n)
    N = n * n
    rhs = np.ones(N)
    t_sp = timeit(lambda: spla.spsolve(A.tocsc(), rhs))
    ab = to_banded_upper(A, n)
    t_ba = timeit(lambda: sla.solveh_banded(ab, rhs))
    if n in ns_dense:
        Ad = A.toarray()
        t_de = f"{timeit(lambda: np.linalg.solve(Ad, rhs)):11.4f}"
    else:
        t_de = "        ---"
    print(f"{n:>5} {N:>7} {t_sp:>12.4f} {t_ba:>12.4f} {t_de} {N*N*8/1e6:>10.1f}")

print(f"\nA dense matrix at n = 120 would need {(120**2)**2 * 8 / 1e9:.2f} GB just to store.")

A = poisson_2d(1000).tocsr()
A.indices = A.indices.astype(np.int32)
A.indptr = A.indptr.astype(np.int32)

data_mb = A.data.nbytes / 1e6
ind_mb = A.indices.nbytes / 1e6
ptr_mb = A.indptr.nbytes / 1e6
print(f"N = {A.shape[0]:,}   nnz = {A.nnz:,}  (~{A.nnz / A.shape[0]:.1f} per row)")
print(f"  values   (float64): {data_mb:6.1f} MB")
print(f"  col_ind  (int32)  : {ind_mb:6.1f} MB")
print(f"  row_ptr  (int32)  : {ptr_mb:6.1f} MB")
print(f"  total             : {data_mb + ind_mb + ptr_mb:6.1f} MB")
print(f"\nthe same matrix dense : {A.shape[0]**2 * 8 / 1e12:.0f} TB")
print(f"its LU factors, natural ordering, ~O(N^1.5) entries: "
      f"{A.shape[0]**1.5 * 8 / 1e9:.0f} GB")
del A

def half_bandwidth(A):
    A = A.tocoo()
    return int(np.max(np.abs(A.row - A.col)))


def chol_fill(A):
    # Number of nonzeros in the Cholesky factor (dense factorisation, small n only).
    L = np.linalg.cholesky(A.toarray())
    return int((np.abs(L) > 1e-12).sum())


n = 30
A_nat = poisson_2d(n)
N = n * n

rng = np.random.default_rng(0)
p = rng.permutation(N)
A_rand = A_nat[p][:, p]

q = reverse_cuthill_mckee(sp.csr_matrix(A_rand), symmetric_mode=True)
A_rcm = A_rand[q][:, q]

tri = N * (N + 1) // 2
print(f"n = {n}, N = {N}, nnz = {A_nat.nnz}  (identical for all three orderings)")
print(f"a full lower triangle would be {tri} entries\n")
print(f"{'ordering':>14} {'half-bandwidth b':>18} {'nnz in chol(A)':>16} {'% of full':>11}")
for name, M in [("natural", A_nat), ("random", A_rand), ("RCM of random", A_rcm)]:
    f = chol_fill(M)
    print(f"{name:>14} {half_bandwidth(M):>18} {f:>16} {f/tri:>10.1%}")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, (name, M) in zip(axes, [("natural", A_nat), ("random", A_rand), ("RCM of random", A_rcm)]):
    ax.spy(M, markersize=0.7, color="C0")
    ax.set_title(f"{name}\n$b$ = {half_bandwidth(M)}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()
