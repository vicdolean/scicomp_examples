import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# --- Helpers
def steepest_descent(A, b, x0, tol, maxit):
    x = x0.copy()
    r = b - A @ x
    reshist = [np.linalg.norm(r)]
    k = 0
    while reshist[-1] > tol and k < maxit:
        Ar = A @ r
        alpha = (r.T @ r) / (r.T @ Ar)  # exact line-search step
        x = x + alpha * r
        r = r - alpha * Ar
        reshist.append(np.linalg.norm(r))
        k += 1
    return x, np.array(reshist)

def cg_spd(A, b, x0, tol, maxit):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rho = r.T @ r
    reshist = [np.sqrt(rho)]
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
        k += 1
    return x, np.array(reshist)

# --- Main script
# Example 1 - Steepest Descent vs. CG on SPD (1D Poisson)
n = 400
e = np.ones(n)
A = sp.diags([-e, 2 * e, -e], [-1, 0, 1], n, n, format='csr') # 1D Poisson (Dirichlet)
b = np.random.randn(n)                            # generic RHS
x0 = np.zeros(n)

tol = 1e-8
maxit = 2000

# --- Steepest Descent (exact line search)
x_sd, res_sd = steepest_descent(A, b, x0, tol, maxit)

# --- Conjugate Gradient
x_cg, res_cg = cg_spd(A, b, x0, tol, maxit)

# --- Plot residual histories
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