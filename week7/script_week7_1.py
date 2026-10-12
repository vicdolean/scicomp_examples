import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt

def poisson1d(n):
    """A_h = tridiag(-1,2,-1)/h^2 for -u'' on (0,1), n interior points."""
    h = 1.0 / (n + 1)
    e = np.ones(n)
    A = sp.spdiags([-e, 2 * e, -e], [-1, 0, 1], n, n, format='csr') / h**2
    return A, h

def badly_scaled(n, s):
    """A = S A_h S with S = diag(s): the same PDE in badly chosen units."""
    A, h = poisson1d(n)
    S = sp.diags(s)
    return (S @ A @ S).tocsr(), h

def run_cg(A, b, M=None, rtol=1e-10, maxiter=5000):
    """CG (or PCG) returning the history of relative residual norms."""
    nb = np.linalg.norm(b)
    hist = [1.0]
    A_ = A
    def cb(xk):
        hist.append(np.linalg.norm(b - A_ @ xk) / nb)
    x, info = cg(A, b, x0=np.zeros_like(b), rtol=rtol, maxiter=maxiter, M=M, callback=cb)
    return np.array(hist), info

def jacobi(A):
    d = A.diagonal()
    return LinearOperator(A.shape, matvec=lambda r: r / d)

def cond2(A):
    ev = np.linalg.eigvalsh(A.toarray())
    return ev.max() / ev.min()

def cond_prec(A):
    """kappa(D^{-1}A) = kappa(D^{-1/2} A D^{-1/2}): same spectrum, symmetric."""
    d = A.diagonal()
    S = sp.diags(1 / np.sqrt(d))
    return cond2(S @ A @ S)

# ---------------------------------------------------------------- Part 1
# Constant diagonal: Jacobi can do nothing (Chapter 6, Section 6.1).
n = 400
A, h = poisson1d(n)
x = np.arange(1, n + 1) * h
rng = np.random.default_rng(0)      # fixed seed: reproducible numbers
b = rng.standard_normal(n)          # a generic right-hand side, not a single eigenmode

D = A.diagonal()
print('Part 1: 1D Poisson, n = %d' % n)
print('  diag(A) constant?  min = %.4e  max = %.4e' % (D.min(), D.max()))
print('  kappa(A)      = %.3e' % cond2(A))
print('  kappa(D^-1 A) = %.3e   (identical: D = (2/h^2) I)' % cond_prec(A))

hist_cg, _ = run_cg(A, b)
hist_pcg, _ = run_cg(A, b, M=jacobi(A))
print('  CG  iterations to 1e-10 : %d' % (len(hist_cg) - 1))
print('  PCG iterations to 1e-10 : %d   (no gain, as predicted)' % (len(hist_pcg) - 1))

# ---------------------------------------------------------------- Part 2
# Variable coefficient: the diagonal now varies over several decades,
# and this is exactly the situation diagonal scaling was made for.
s_scale = 10.0 ** (3.0 * np.sin(6 * np.pi * x))   # unit of unknown i varies over 6 decades
Av, hv = badly_scaled(n, s_scale)
Dv = Av.diagonal()
bv = rng.standard_normal(n)

print('\nPart 2: badly scaled unknowns, A = S A_h S, s_i = 10^{3 sin(6 pi x_i)}, n = %d' % n)
print('  diag(A) spread: max/min = %.3e' % (Dv.max() / Dv.min()))
print('  kappa(A)      = %.3e' % cond2(Av))
print('  kappa(D^-1 A) = %.3e' % cond_prec(Av))

hist_cgv, _ = run_cg(Av, bv)
hist_pcgv, _ = run_cg(Av, bv, M=jacobi(Av))
print('  CG  iterations to 1e-10 : %d' % (len(hist_cgv) - 1))
print('  PCG iterations to 1e-10 : %d' % (len(hist_pcgv) - 1))

# ---------------------------------------------------------------- Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
ax[0].semilogy(hist_cg, 'o-', ms=3, label='CG')
ax[0].semilogy(hist_pcg, '*--', ms=4, label='PCG (Jacobi)')
ax[0].set_title(r'constant diagonal: $-u^{\prime\prime}$')
ax[1].semilogy(hist_cgv, 'o-', ms=3, label='CG')
ax[1].semilogy(hist_pcgv, '*--', ms=4, label='PCG (Jacobi)')
ax[1].set_title(r'varying diagonal: $A = S A_h S$')
for a_ in ax:
    a_.set_xlabel('iteration $k$'); a_.grid(True, which='both', alpha=.3); a_.legend(loc='lower left')
ax[0].set_ylabel(r'$\|b-Ax_k\|_2/\|b\|_2$')
plt.tight_layout()
plt.show()
