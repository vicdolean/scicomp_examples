import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt

# 1D Poisson: -u'' = f on (0,1), u(0)=u(1)=0
n = 400
h = 1 / (n + 1)
e = np.ones(n)
A = sp.spdiags([-e, 2 * e, -e], [-1, 0, 1], n, n, format='csr') / h**2

# Right-hand side and exact solution for test (sine mode)
xgrid = np.arange(1, n + 1) * h
u_exact = np.sin(np.pi * xgrid)
f = (np.pi**2) * u_exact
b = f

# CG (no preconditioner)
tol = 1e-8
maxit = 1000
x0 = np.zeros(n)

res_cg = []
def callback_cg(xk):
    res_cg.append(np.linalg.norm(b - A @ xk))

res_cg.append(np.linalg.norm(b - A @ x0))
_, _ = cg(A, b, x0=x0, tol=tol, maxiter=maxit, callback=callback_cg)

# PCG with Jacobi (diagonal) preconditioner
M_diag = A.diagonal()
M_inv = lambda r: r / M_diag
M = LinearOperator((n, n), matvec=M_inv)

res_pcg = []
def callback_pcg(xk):
    res_pcg.append(np.linalg.norm(b - A @ xk))

res_pcg.append(np.linalg.norm(b - A @ x0))
_, _ = cg(A, b, x0=x0, tol=tol, maxiter=maxit, M=M, callback=callback_pcg)

# Plot residual norms
plt.semilogy(np.arange(len(res_cg)), res_cg, 'o-', label='CG')
plt.semilogy(np.arange(len(res_pcg)), res_pcg, '*-', label='PCG (Jacobi)')
plt.grid(True)
plt.xlabel('Iteration k')
plt.ylabel('||r_k||_2')
plt.legend(loc='southwest')
plt.title('CG vs PCG (Jacobi) on 1D Laplacian')
plt.show()