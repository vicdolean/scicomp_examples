import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg, LinearOperator
import matplotlib.pyplot as plt

def poisson1d(n):
    h = 1.0 / (n + 1)
    e = np.ones(n)
    A = sp.spdiags([-e, 2 * e, -e], [-1, 0, 1], n, n, format='csr') / h**2
    return A, h

def subdomains(n, ja, jb):
    """Omega_1 = (0, beta) and Omega_2 = (alpha, 1) with alpha = ja*h < beta = jb*h.
    Returns the 0-based index sets of the interior nodes of each subdomain."""
    I1 = np.arange(0, jb - 1)      # nodes 1 .. jb-1
    I2 = np.arange(ja, n)          # nodes ja+1 .. n
    return I1, I2

# ---------------------------------------------------------------- setup
n = 127
A, h = poisson1d(n)
x = np.arange(1, n + 1) * h
ja, jb = 50, 80                       # alpha = 50h, beta = 80h
alpha, beta = ja * h, jb * h
I1, I2 = subdomains(n, ja, jb)
A1 = A[I1, :][:, I1].tocsc()
A2 = A[I2, :][:, I2].tocsc()

rho_theory = alpha * (1 - beta) / (beta * (1 - alpha))
print('alpha = %.4f, beta = %.4f, overlap delta = %.4f' % (alpha, beta, beta - alpha))
print('theoretical rho = alpha(1-beta)/(beta(1-alpha)) = %.6f' % rho_theory)

uex = np.sin(2 * np.pi * x)
b = (2 * np.pi) ** 2 * uex            # -u'' = f with u = sin(2 pi x)
ustar = spsolve(A.tocsc(), b)         # discrete exact solution

# ------------------------------------------- (a) parallel Schwarz iteration
u1 = np.zeros(len(I1))
u2 = np.zeros(len(I2))
e_if = []                             # |e_2(beta)| at the interface beta
for k in range(40):
    g1 = b[I1].copy()
    g1[-1] += (u2[jb - 1 - ja] if True else 0.0) / h**2   # value at node jb from Omega_2
    g2 = b[I2].copy()
    g2[0] += u1[ja - 1] / h**2                            # value at node ja from Omega_1
    u1_new = spsolve(A1, g1)
    u2_new = spsolve(A2, g2)
    u1, u2 = u1_new, u2_new
    e_if.append(abs(ustar[jb - 1] - u2[jb - 1 - ja]))     # e_2 at the node just left of beta

e_if = np.array(e_if)
r2 = e_if[2:12] / e_if[0:10]          # two-step ratios
print('\n(a) parallel Schwarz, measured two-step ratios:')
print('   ', np.array2string(r2, precision=5))
print('    mean two-step ratio = %.6f  vs  rho = %.6f' % (r2.mean(), rho_theory))
print('    per-iteration factor = %.6f  vs  sqrt(rho) = %.6f' % (np.sqrt(r2.mean()), np.sqrt(rho_theory)))

# -------------------------------- (b) additive Schwarz as a CG preconditioner
def asm(n, A, I1, I2, weights=False):
    """Additive Schwarz.
    weights=False: M^{-1} = sum_i R_i^T A_i^{-1} R_i, symmetric positive definite,
                   the form that may be used inside CG.
    weights=True : M^{-1} = sum_i R_i^T D_i A_i^{-1} R_i (restricted additive Schwarz).
                   The partition of unity makes the stationary iteration converge,
                   but D_i A_i^{-1} is NOT symmetric, so this form is for GMRES, not CG."""
    A1 = A[I1, :][:, I1].tocsc(); A2 = A[I2, :][:, I2].tocsc()
    w1 = np.ones(n); w2 = np.ones(n)
    if weights:
        ov = np.intersect1d(I1, I2)
        w1[ov] = 0.5; w2[ov] = 0.5    # sum_i R_i^T D_i R_i = I
    def mv(v):
        y = np.zeros(n)
        y[I1] += w1[I1] * spsolve(A1, v[I1])
        y[I2] += w2[I2] * spsolve(A2, v[I2])
        return y
    return LinearOperator((n, n), matvec=mv)

def is_symmetric(M, n, tol=1e-8):
    E = np.eye(n)
    Md = np.column_stack([M @ E[:, j] for j in range(n)])
    return np.linalg.norm(Md - Md.T) / np.linalg.norm(Md) < tol

def cg_iters(A, b, M=None, rtol=1e-10):
    it = [0]
    cg(A, b, x0=np.zeros_like(b), rtol=rtol, maxiter=2000, M=M,
       callback=lambda xk: it.__setitem__(0, it[0] + 1))
    return it[0]

def strips(n, N, ov):
    """N overlapping strips of the index set 0..n-1, each extended by ov nodes
    on either side (ov = 0 gives a non-overlapping block Jacobi decomposition)."""
    cuts = np.linspace(0, n, N + 1).round().astype(int)
    return [np.arange(max(cuts[i] - ov, 0), min(cuts[i + 1] + ov, n)) for i in range(N)]

def asm_N(n, A, Is):
    """M^{-1} = sum_i R_i^T A_i^{-1} R_i over an arbitrary list of index sets."""
    locs = [(I, A[I, :][:, I].tocsc()) for I in Is]
    def mv(v):
        y = np.zeros(n)
        for I, Ai in locs:
            y[I] += spsolve(Ai, v[I])
        return y
    return LinearOperator((n, n), matvec=mv)

rng = np.random.default_rng(0)
bb = rng.standard_normal(n)
print('\n(b) CG vs PCG with the two-subdomain additive Schwarz preconditioner (n = %d)' % n)
print('    symmetric?   sum R_i^T A_i^{-1} R_i      : %s' % is_symmetric(asm(n, A, I1, I2), n))
print('    symmetric?   sum R_i^T D_i A_i^{-1} R_i  : %s  (so it belongs in GMRES, not CG)'
      % is_symmetric(asm(n, A, I1, I2, weights=True), n))
print('    CG                  : %4d iterations' % cg_iters(A, bb))
print('    PCG (Schwarz, 2 sub): %4d iterations' % cg_iters(A, bb, M=asm(n, A, I1, I2)))

# ---------------------------------- (c) many subdomains: no coarse space yet
print('\n(c) one-level ASM with N subdomains, overlap 2 nodes (n = 511)')
n2 = 511
A2g, h2 = poisson1d(n2)
b2 = np.random.default_rng(0).standard_normal(n2)
print('      N    PCG its')
for N in (2, 4, 8, 16, 32):
    print('    %3d     %4d' % (N, cg_iters(A2g, b2, M=asm_N(n2, A2g, strips(n2, N, 2)))))
print('    the count grows with N: nothing in the preconditioner transports')
print('    information across the whole domain. This is what a coarse space fixes.')

# ---------------------------------------- (d) mesh independence for fixed N
print('\n(d) overlap and mesh dependence, N = 8 subdomains')
print('      n     overlap (nodes)   PCG its')
for n_ in (127, 255, 511, 1023):
    A_, _ = poisson1d(n_)
    b_ = np.random.default_rng(0).standard_normal(n_)
    ov = max(1, n_ // 64)              # overlap a fixed fraction of the domain
    print('    %4d        %3d            %4d' % (n_, ov, cg_iters(A_, b_, M=asm_N(n_, A_, strips(n_, 8, ov)))))
print('    for a fixed number of subdomains and a fixed relative overlap the count')
print('    does not grow with n: Schwarz is mesh independent, unlike Jacobi.')

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(np.arange(1, len(e_if) + 1), e_if / e_if[0], 'o-', ms=3,
            label='parallel Schwarz: $|e_2(\\beta)|$')
kk = np.arange(1, len(e_if) + 1)
ax.semilogy(kk, np.sqrt(rho_theory) ** (kk - 1), 'k--', label=r'$\sqrt{\rho}^{\,k}$ (theory)')
ax.set_xlabel('iteration $k$'); ax.set_ylabel('interface error (normalised)')
ax.grid(True, which='both', alpha=.3); ax.legend(loc='lower left')
plt.tight_layout(); plt.show()
