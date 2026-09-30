import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def steepest_descent(A, b, x0, tol, maxit, x_true=None):
    x = x0.copy()
    r = b - A @ x
    reshist = [np.linalg.norm(r)]
    errhist = []                      # energy norm ||x_k - x*||_A, if x_true is given
    if x_true is not None:
        e = x - x_true; errhist.append(np.sqrt(e @ (A @ e)))
    k = 0
    while reshist[-1] > tol and k < maxit:
        Ar = A @ r
        alpha = (r.T @ r) / (r.T @ Ar)  # exact line-search step
        x = x + alpha * r
        r = r - alpha * Ar
        reshist.append(np.linalg.norm(r))
        if x_true is not None:
            e = x - x_true; errhist.append(np.sqrt(e @ (A @ e)))
        k += 1
    return x, np.array(reshist), np.array(errhist)

def cg_spd(A, b, x0, tol, maxit, x_true=None):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rho = r.T @ r
    reshist = [np.sqrt(rho)]
    errhist = []
    if x_true is not None:
        e = x - x_true; errhist.append(np.sqrt(e @ (A @ e)))
    k = 0
    while reshist[-1] > tol and k < maxit:
        Ap = A @ p
        alpha = rho / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rho_new = r.T @ r
        beta = rho_new / rho
        p = r + beta * p
        rho = rho_new
        reshist.append(np.sqrt(rho))
        if x_true is not None:
            e = x - x_true; errhist.append(np.sqrt(e @ (A @ e)))
        k += 1
    return x, np.array(reshist), np.array(errhist)

n = 400                                           # Problem 4(d): n = 100
e = np.ones(n)
A = sp.diags([-e, 2 * e, -e], [-1, 0, 1], shape=(n, n), format='csr')  # 1D Poisson (Dirichlet)
np.random.seed(0)
b = np.random.randn(n)                            # generic RHS
x0 = np.zeros(n)
x_true = np.linalg.solve(A.toarray(), b)          # reference solution (small n only)

tol = 1e-8
maxit = 2000                                      # Problem 4(d): maxit = 30000

x_sd, res_sd, err_sd = steepest_descent(A, b, x0, tol, maxit, x_true)
x_cg, res_cg, err_cg = cg_spd(A, b, x0, tol, maxit, x_true)

plt.figure()
plt.semilogy(res_sd, 'o-', label='Steepest Descent')
plt.semilogy(res_cg, 'x-', label='CG')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||r_k||_2')
plt.title('Steepest Descent vs. Conjugate Gradient on SPD system')
plt.legend()
plt.show()

print(f"Steepest Descent iterations: {len(res_sd)-1}")
print(f"Conjugate Gradient iterations: {len(res_cg)-1}")

rel_sd = err_sd / err_sd[0]
rel_cg = err_cg / err_cg[0]

plt.figure()
plt.semilogy(rel_sd, label='Steepest Descent')
plt.semilogy(rel_cg, label='CG')
plt.axhline(1e-6, color='gray', ls='--', lw=0.8)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||e_k||_A / ||e_0||_A')
plt.title('Energy-norm error')
plt.legend()
plt.show()

lam = np.linalg.eigvalsh(A.toarray())
kappa = lam[-1] / lam[0]
first = lambda rel: next((k for k, v in enumerate(rel) if v <= 1e-6), None)
print(f"kappa(A) = {kappa:.1f}")
print(f"SD : estimate {kappa/2*np.log(1e6):.0f}, measured {first(rel_sd)} (None = not reached within maxit)")
print(f"CG : estimate {np.sqrt(kappa)/2*np.log(2e6):.0f}, finite termination n = {n}, measured {first(rel_cg)}")
print(f"CG residual increased in {np.sum(np.diff(res_cg) > 0)} of {len(res_cg)-1} steps")
