"""
Week 3.3 - Checking the spectral claims of Chapter 3.

Five short numerical experiments, one per claim made in the text:

  1. the Gauss-Seidel spectrum and its multiplicities
  2. non-normality: rho^k versus ||G^k|| for optimal SOR
  3. why undamped Jacobi is not a smoother
  4. random walk: P averages functions, P^T moves distributions
  5. the deflated power iteration and the Fiedler vector

Each experiment prints a PASS/FAIL line so the script can be run as a check.
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
results = []


def check(name, ok):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def poisson(n):
    """Unscaled 1D Dirichlet Laplacian tridiag(-1, 2, -1)."""
    return (np.diag(2 * np.ones(n))
            + np.diag(-np.ones(n - 1), 1)
            + np.diag(-np.ones(n - 1), -1))


def splitting(A):
    """A = D - L - U with L, U the *positive* strict triangular parts."""
    D = np.diag(np.diag(A))
    return D, -np.tril(A, -1), -np.triu(A, 1)


# ----------------------------------------------------------------------
# 1. Gauss-Seidel spectrum
# ----------------------------------------------------------------------
# The characteristic polynomial is
#     det(lambda I - G_GS) = lambda^ceil(n/2) * prod_{j=1}^{floor(n/2)}
#                            (lambda - cos^2(j pi / (n+1))).
# So the nonzero eigenvalues are simple and 0 has algebraic multiplicity
# ceil(n/2).  Reading cos^2(k pi/(n+1)) over k = 1..n instead double-counts
# every positive value, because cos^2(k pi/(n+1)) = cos^2((n+1-k) pi/(n+1)).

print("\n1. Gauss-Seidel spectrum")

def G_GS(n):
    A = poisson(n)
    D, L, U = splitting(A)
    return np.linalg.solve(D - L, U)


def charpoly_certificate(n, lam):
    """lambda^ceil(n/2) * prod_j (lambda - cos^2(j pi/(n+1)))."""
    val = lam ** ((n + 1) // 2)
    for j in range(1, n // 2 + 1):
        val *= lam - np.cos(j * np.pi / (n + 1)) ** 2
    return val


ok = True
for n in range(1, 13):
    G = G_GS(n)
    for lam in (0.3, 1.7, -0.9, 2.5):
        lhs = np.linalg.det(lam * np.eye(n) - G)
        ok &= abs(lhs - charpoly_certificate(n, lam)) < 1e-9 * max(1.0, abs(lhs))
check("char. poly matches the certificate for n = 1..12", ok)

n = 5
print(f"\n  n = {n}:")
print("    true spectrum            :", np.sort(np.linalg.eigvals(G_GS(n)).real))
print("    cos^2(k pi/(n+1)), k=1..n:", np.sort([np.cos(k * np.pi / (n + 1)) ** 2
                                                 for k in range(1, n + 1)]))
print("    -> the second list double-counts the positive values and loses two zeros")

# kernel, normality, diagonalisability
print("\n   n  nullity  ker=span{e1}  normal  diagonalisable")
for n in range(1, 8):
    G = G_GS(n)
    _, S, Vt = np.linalg.svd(G)
    nullity = int(np.sum(S < 1e-10))
    ker = Vt[len(S) - nullity:].T
    is_e1 = nullity == 1 and abs(abs(ker[0, 0]) - 1) < 1e-9
    normal = np.allclose(G @ G.T, G.T @ G, atol=1e-12)
    diag = np.linalg.matrix_rank(np.linalg.eig(G)[1], tol=1e-8) == n
    print(f"  {n:2d}  {nullity:^7d}  {str(is_e1):^12}  {str(normal):^6}  {str(diag):^14}")
check("non-normal for n >= 2", all(
    not np.allclose(G_GS(n) @ G_GS(n).T, G_GS(n).T @ G_GS(n), atol=1e-12)
    for n in range(2, 8)))
check("diagonalisable for n = 1, 2 and not for n >= 3", all(
    (np.linalg.matrix_rank(np.linalg.eig(G_GS(n))[1], tol=1e-8) == n) == (n <= 2)
    for n in range(1, 8)))

# ----------------------------------------------------------------------
# 2. Non-normality: rho^k is not a guarantee
# ----------------------------------------------------------------------
# rho(G) governs the asymptotic rate, but for a non-normal G the transient
# ||G^k|| can sit orders of magnitude above rho^k.  With n = 49 the text's
# k = 110 satisfies rho^k < 1e-6 yet does not reduce every unit initial
# error by 1e-6.

print("\n2. Optimal SOR at n = 49: rho^k versus ||G^k||")

n = 49
A = poisson(n)
D, L, U = splitting(A)
omega = 2 / (1 + np.sin(np.pi / (n + 1)))
G = np.linalg.solve(D - omega * L, (1 - omega) * D + omega * U)
rho = max(abs(np.linalg.eigvals(G)))

print(f"  omega_opt = {omega:.6f},  rho = {rho:.6f},  ||G||_2 = {np.linalg.norm(G, 2):.4f}")
G110 = np.linalg.matrix_power(G, 110)
print(f"  rho^110            = {rho**110:.6e}")
print(f"  ||G^110 e_49||_2   = {np.linalg.norm(G110[:, n-1]):.6e}")
print(f"  ||G^110||_2        = {np.linalg.norm(G110, 2):.6e}")
check("rho^110 < 1e-6", rho ** 110 < 1e-6)
check("but ||G^110 e_49||_2 > 1e-6", np.linalg.norm(G110[:, n - 1]) > 1e-6)

k_uniform = 110
while np.linalg.norm(np.linalg.matrix_power(G, k_uniform), 2) > 1e-6:
    k_uniform += 1
print(f"  smallest k with ||G^k||_2 <= 1e-6 : {k_uniform}")

ks = np.arange(0, 200)
norms = [np.linalg.norm(np.linalg.matrix_power(G, int(k)), 2) for k in ks]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(ks, rho ** ks, label=r"$\rho(G)^k$")
ax.semilogy(ks, norms, label=r"$\|G^k\|_2$")
ax.axhline(1e-6, ls=":", color="grey", label=r"$10^{-6}$")
ax.axvline(110, ls="--", color="crimson", lw=1)
ax.set_xlabel("k"); ax.set_ylabel("error factor")
ax.set_title("Optimal SOR, n = 49: spectral radius vs operator norm")
ax.legend(); fig.tight_layout()

# ----------------------------------------------------------------------
# 3. Undamped Jacobi is not a smoother
# ----------------------------------------------------------------------
# mu_j = cos(j pi/(n+1)) satisfies |mu_{n+1-j}| = |mu_j|, so the most
# oscillatory mode decays exactly as slowly as the smoothest one.  Damping
# with omega = 2/3 gives |mu_j| <= 1/3 on the whole upper half.

print("\n3. Jacobi smoothing")

n = 19
j = np.arange(1, n + 1)
mu = np.cos(j * np.pi / (n + 1))
mu23 = 1 - 2 * (2 / 3) * np.sin(j * np.pi / (2 * (n + 1))) ** 2
upper = j >= (n + 1) / 2

print(f"  n = {n}:  |mu_1| = {abs(mu[0]):.6f},  |mu_n| = {abs(mu[-1]):.6f}")
print(f"  max |mu_j| over the upper half   (omega = 1)   : {abs(mu[upper]).max():.6f}")
print(f"  max |mu_j| over the upper half   (omega = 2/3) : {abs(mu23[upper]).max():.6f}")
check("|mu_n| = |mu_1| exactly", np.isclose(abs(mu[-1]), abs(mu[0])))
check("undamped smoothing factor is close to 1", abs(mu[upper]).max() > 0.9)
check("damped smoothing factor is 1/3", np.isclose(abs(mu23[upper]).max(), 1 / 3))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(j, abs(mu), "o-", label=r"$|\mu_j|$, $\omega=1$")
ax.plot(j, abs(mu23), "s-", label=r"$|\mu_j|$, $\omega=2/3$")
ax.axhline(1 / 3, ls=":", color="grey")
ax.axvline((n + 1) / 2, ls="--", color="grey", lw=1)
ax.set_xlabel("mode index j"); ax.set_ylabel(r"$|\mu_j|$")
ax.set_title("Jacobi damping factors (grey line: upper half of the spectrum)")
ax.legend(); fig.tight_layout()

# ----------------------------------------------------------------------
# 4. P averages functions, P^T moves distributions
# ----------------------------------------------------------------------

print("\n4. Random walk on the path P_4")

A4 = np.array([[0, 1, 0, 0],
               [1, 0, 1, 0],
               [0, 1, 0, 1],
               [0, 0, 1, 0]], float)
d4 = A4.sum(1)
D4 = np.diag(d4)
P4 = np.linalg.solve(D4, A4)

x0 = np.array([1.0, 0, 0, 0])
print(f"  P x0    = {P4 @ x0}   total mass {(P4 @ x0).sum():.3f}   (function averaging)")
print(f"  P^T x0  = {P4.T @ x0}   total mass {(P4.T @ x0).sum():.3f}   (distribution)")
check("P is row-stochastic", np.allclose(P4 @ np.ones(4), 1))
check("P is not column-stochastic", not np.allclose(P4.T @ np.ones(4), 1))
check("P^T conserves mass, P does not",
      np.isclose((P4.T @ x0).sum(), 1) and not np.isclose((P4 @ x0).sum(), 1))

# ----------------------------------------------------------------------
# 5. Deflated power iteration and the Fiedler vector
# ----------------------------------------------------------------------
# On a bipartite graph mu_n = -1, so the alternating mode is never damped.
# On P_4 it is q = (1,-1,1,-1), which is D-orthogonal to 1: the projection
# does not remove it and the iterates alternate.  The lazy walk maps that
# mode to 0 and repairs the scheme.

print("\n5. Deflated power iteration")


def fiedler(A):
    """Fiedler direction D^{-1/2} v_2 of the normalised Laplacian."""
    d = A.sum(1)
    S = np.diag(1 / np.sqrt(d)) @ A @ np.diag(1 / np.sqrt(d))
    w, V = np.linalg.eigh(S)
    order = np.argsort(-w)
    f = np.diag(1 / np.sqrt(d)) @ V[:, order[1]]
    return w[order], f / np.linalg.norm(f)


def deflated_power(A, lazy, steps=400, seed=0):
    d = A.sum(1)
    D = np.diag(d)
    P = np.linalg.solve(D, A)
    M = 0.5 * (np.eye(len(d)) + P) if lazy else P
    one = np.ones(len(d))
    x = np.random.default_rng(seed).standard_normal(len(d))
    for _ in range(steps):
        x = M @ x
        x = x - ((x @ D @ one) / (one @ D @ one)) * one
        x = x / np.sqrt(x @ D @ x)
    return x / np.linalg.norm(x)


q = np.array([1.0, -1, 1, -1])
print(f"  P_4:  P q = {P4 @ q}   equals -q: {np.allclose(P4 @ q, -q)}")
print(f"        q^T D 1 = {q @ D4 @ np.ones(4):.1f}  -> the projection does not remove q")
check("Pq = -q on P_4", np.allclose(P4 @ q, -q))
check("q is D-orthogonal to 1", np.isclose(q @ D4 @ np.ones(4), 0))

mu4, f4 = fiedler(A4)
print(f"  spectrum of P     : {mu4}")
print(f"  spectrum of (I+P)/2: {0.5 * (1 + mu4)}")

x_plain = deflated_power(A4, lazy=False)
x_lazy = deflated_power(A4, lazy=True)
print(f"  plain walk, |cos| with Fiedler vector : {abs(x_plain @ f4):.6f}")
print(f"  lazy  walk, |cos| with Fiedler vector : {abs(x_lazy @ f4):.6f}")
check("plain walk fails on the bipartite P_4", abs(x_plain @ f4) < 0.5)
check("lazy walk converges on P_4", abs(x_lazy @ f4) > 0.999)

# The chapter's two-triangle graph is not bipartite, and there the plain walk
# happens to work -- but only because it satisfies mu_2 > |mu_n|.  That is a
# property of the example, not a general fact: on a random connected graph the
# plain walk usually fails too.
A6 = np.array([[0, 1, 1, 0, 0, 0],
               [1, 0, 1, 0, 0, 0],
               [1, 1, 0, 1, 0, 0],
               [0, 0, 1, 0, 1, 1],
               [0, 0, 0, 1, 0, 1],
               [0, 0, 0, 1, 1, 0]], float)
mu6, f6 = fiedler(A6)
print(f"\n  two triangles (not bipartite), spectrum of P: {mu6}")
print(f"  mu_2 = {mu6[1]:.4f} > |mu_n| = {abs(mu6[-1]):.4f}  -> the lucky case")
print(f"  plain walk, |cos| with Fiedler vector : {abs(deflated_power(A6, lazy=False) @ f6):.6f}")
print(f"  stated rate |mu_3/mu_2|              : {abs(mu6[2] / mu6[1]):.4f}")
print(f"  true  rate max_i>=3 |mu_i| / |mu_2|  : {max(abs(mu6[2:])) / abs(mu6[1]):.4f}")
check("plain walk happens to work here, where mu_2 > |mu_n|",
      abs(deflated_power(A6, lazy=False) @ f6) > 0.999)
check("stated rate |mu_3/mu_2| differs from the true rate",
      not np.isclose(abs(mu6[2] / mu6[1]), max(abs(mu6[2:])) / abs(mu6[1])))

# random connected non-bipartite graphs: the plain walk is unreliable, the lazy
# walk is not.  This is the reason the chapter uses damped Jacobi throughout.
print("\n  random connected graphs (none bipartite):")
rng = np.random.default_rng(3)
plain_ok, lazy_ok = 0, 0
for trial in range(6):
    n_v = int(rng.integers(6, 14))
    M = (rng.random((n_v, n_v)) < 0.35).astype(float)
    M = np.triu(M, 1); M = M + M.T
    for k in range(n_v - 1):
        M[k, k + 1] = M[k + 1, k] = 1.0          # force connectivity
    mu, fv = fiedler(M)
    if abs(mu[1] - mu[2]) < 1e-8:
        continue                                  # mu_2 not simple, skip
    ap = abs(deflated_power(M, lazy=False, seed=trial) @ fv)
    al = abs(deflated_power(M, lazy=True, seed=trial) @ fv)
    plain_ok += ap > 0.999
    lazy_ok += al > 0.999
    print(f"    n = {n_v:2d}   plain |cos| = {ap:.4f}   lazy |cos| = {al:.4f}")
check("the lazy walk succeeds on every random graph tried", lazy_ok == 6)
check("the plain walk does not", plain_ok < 6)

# ----------------------------------------------------------------------
print("\n" + "-" * 60)
n_pass = sum(ok for _, ok in results)
print(f"{n_pass}/{len(results)} checks passed")
for name, ok in results:
    if not ok:
        print(f"  FAILED: {name}")

plt.show()
