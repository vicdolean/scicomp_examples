import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, cg, lsqr, LinearOperator
import matplotlib.pyplot as plt

# %% Example 1 — GMRES vs. CG-on-normal-equations for a non-symmetric PDE matrix
# Model: 1D convection–diffusion -nu u'' + beta u' = f (Dirichlet)
# Discretization: centered Laplacian + first-order upwind for advection
n = 600
nu = 1e-2  # diffusion
beta = 5   # advection (makes A non-symmetric)
h = 1 / (n + 1)
e = np.ones(n)
L = sp.spdiags([-e, 2 * e, -e], [-1, 0, 1], n, n, format='csr') / h**2  # -u''
U = sp.spdiags([-e, e], [-1, 0], n, n, format='csr') / h  # u' ~ (u_i - u_{i-1})/h
A = -nu * L + beta * U  # non-symmetric

b = np.random.randn(n)
x0 = np.zeros(n)
tol = 1e-8
maxit = 1000

# --- Full GMRES (built-in) on Ax=b
res_gm = []
def callback_gmres(rk):
    res_gm.append(np.linalg.norm(rk))

res_gm.append(np.linalg.norm(b - A @ x0))
x_gmres, flag_gm = gmres(A, b, x0=x0, tol=tol, maxiter=maxit, callback=callback_gmres)
iter_gm = len(res_gm) - 1
rel_gm = res_gm[-1] / res_gm[0]

# --- CG on normal equations (matrix-free): A^T A x = A^T b
AtA_op = LinearOperator((n, n), matvec=lambda x: A.T @ (A @ x), rmatvec=lambda x: A.T @ (A @ x))
rhs = A.T @ b

res_cg = []
def callback_cg(xk):
    res_cg.append(np.linalg.norm(rhs - AtA_op @ xk))

res_cg.append(np.linalg.norm(rhs - AtA_op @ x0))
x_cgne, flag_cg = cg(AtA_op, rhs, x0=x0, tol=tol, maxiter=maxit, callback=callback_cg)
iter_cg = len(res_cg) - 1
rel_cg = res_cg[-1] / res_cg[0] if res_cg[0] != 0 else 0.0

# --- Plot residual histories (2-norm residual; GMRES returns it directly)
plt.figure()
plt.semilogy(np.arange(len(res_gm)), res_gm, 'o-', label='GMRES on A')
plt.semilogy(np.arange(len(res_cg)), res_cg, 'x-', label='CG on A^T A')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||r_k||_2')
plt.title('GMRES vs. CG on normal equations (convection–diffusion)')
plt.legend(loc='lower left')

print('--- GMRES vs. CGNE ---')
print(f'GMRES: flag={flag_gm}, iters={iter_gm}, relres={rel_gm:.2e}')
print(f'CG(AtA): flag={flag_cg}, iters={iter_cg}, relres={rel_cg:.2e}')

