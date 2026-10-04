import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, cg, LinearOperator
from scipy.sparse.linalg import norm as spla_norm
import matplotlib.pyplot as plt

n = 600
nu = 1e-2  # diffusion
beta = 5   # advection (makes A non-symmetric)
h = 1 / (n + 1)
e = np.ones(n)
L = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format='csr') / h**2  # discrete -u''
U = sp.diags([-e[:-1], e], [-1, 0], format='csr') / h                    # upwind u' ~ (u_i - u_{i-1})/h
A = nu * L + beta * U  # discrete -nu u'' + beta u': non-symmetric

rng = np.random.default_rng(0)   # fixed seed: reproducible numbers
b = rng.standard_normal(n)
x0 = np.zeros(n)
tol = 1e-8
maxit = 2000
print(f'||A - A^T||_F / ||A||_F = {spla_norm(A - A.T) / spla_norm(A):.2f}')

res_gm = [1.0]   # relative residual norms, ||r_0||/||r_0|| = 1
x_gmres, flag_gm = gmres(A, b, x0=x0, rtol=tol, restart=n, maxiter=1,
                         callback=lambda rk: res_gm.append(rk), callback_type='pr_norm')
iter_gm = len(res_gm) - 1
rel_gm = np.linalg.norm(b - A @ x_gmres) / np.linalg.norm(b)

AtA_op = LinearOperator((n, n), matvec=lambda x: A.T @ (A @ x))
rhs = A.T @ b

res_cg = [1.0]   # relative residual of the ORIGINAL system, ||b - A x_k|| / ||b||
def callback_cg(xk):
    res_cg.append(np.linalg.norm(b - A @ xk) / np.linalg.norm(b))

x_cgne, flag_cg = cg(AtA_op, rhs, x0=x0, rtol=tol, maxiter=maxit, callback=callback_cg)
iter_cg = len(res_cg) - 1
rel_cg = res_cg[-1]

plt.figure()
plt.semilogy(np.arange(len(res_gm)), res_gm, '-', label='GMRES on A')
plt.semilogy(np.arange(len(res_cg)), res_cg, '-', label='CG on A^T A')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||b - A x_k||_2 / ||b||_2')
plt.title('GMRES vs. CG on normal equations (convection–diffusion)')
plt.legend(loc='lower left')
plt.show()

print('--- GMRES vs. CGNE ---')
print(f'GMRES:   flag={flag_gm}, iters={iter_gm}, relres={rel_gm:.2e}')
print(f'CG(AtA): flag={flag_cg}, iters={iter_cg}, relres={rel_cg:.2e}')
