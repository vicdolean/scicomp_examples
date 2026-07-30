import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr, cg, LinearOperator
import matplotlib.pyplot as plt

# Example 4 — Tall least squares: LSQR vs. forming normal equations
# Construct a sparse, mildly ill-conditioned tall system; compare solvers.
m = 4000
n = 1000
density = 0.01

# sprandn in MATLAB creates a sparse matrix with normally distributed non-zero entries.
A = sp.random(m, n, density=density, format='csr', data_rvs=np.random.randn)

# sprandn(n,1,0.05) creates a sparse vector.
x_true_sparse = sp.random(n, 1, density=0.05, format='csc', data_rvs=np.random.randn)
x_true = x_true_sparse.toarray().flatten()

b = A @ x_true + 1e-3 * np.random.randn(m)
tol = 1e-8
maxit = 2000

# --- LSQR (matrix-free; uses A and A')
# Note: scipy.sparse.linalg.lsqr does not return a residual history vector.
x_lsqr, istop, itn_l, r1norm = lsqr(A, b, atol=tol, btol=tol, iter_lim=maxit)[:4]

# --- Normal equations with CG (matrix-free)
AtA_op = LinearOperator((n, n), matvec=lambda x: A.T @ (A @ x), rmatvec=lambda x: A.T @ (A @ x))
rhs = A.T @ b

res_c = []
def callback_cg(xk):
    res_c.append(np.linalg.norm(rhs - AtA_op @ xk))

x0_cg = np.zeros(n)
res_c.append(np.linalg.norm(rhs - AtA_op @ x0_cg))
x_cgne, flag_c = cg(AtA_op, rhs, x0=x0_cg, tol=tol, maxiter=maxit, callback=callback_cg)

# --- Plot residual norms
plt.semilogy(np.arange(len(res_c)), res_c, 'x-', label='CG on A^T A (NE residual)')
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.title('LSQR vs. CG on normal equations')
plt.legend(loc='southwest')
plt.grid(True)
plt.show()

print("Note: scipy.sparse.linalg.lsqr does not provide residual history, so it is not plotted.")

# Compare solutions
print(f'Rel. solution error (LSQR): {np.linalg.norm(x_lsqr - x_true) / np.linalg.norm(x_true):.2e}')
print(f'Rel. solution error (CGNE): {np.linalg.norm(x_cgne - x_true) / np.linalg.norm(x_true):.2e}')