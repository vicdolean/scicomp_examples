import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

# Compare Jacobi, Gauss-Seidel, and SOR on 1D Poisson system
n = 19          # matches Problem 5 of Chapter 3 and the week-3 solutions
h = 1 / (n + 1)
A_sparse = (1 / h**2) * sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
A = A_sparse.toarray()
b = np.ones(n)

# Iteration parameters
maxit = 2000    # Jacobi needs ~1487 and GS ~744 to reach 1e-8; see Problem 5(b)
tol = 1e-8
x0 = np.zeros(n)
norm_b = np.linalg.norm(b)

# --- Jacobi
D = np.diag(np.diag(A))
L = -np.tril(A, -1)
U = -np.triu(A, 1)
invD = np.linalg.inv(D)
GJ = invD @ (L + U)
cJ = invD @ b
x = x0.copy()
resJ = [np.linalg.norm(b - A @ x0) / norm_b]
for k in range(maxit):
    x = GJ @ x + cJ
    res = np.linalg.norm(b - A @ x) / norm_b
    resJ.append(res)
    if res < tol:
        break

# --- Gauss-Seidel
M_gs = D - L
x = x0.copy()
resGS = [np.linalg.norm(b - A @ x0) / norm_b]
for k in range(maxit):
    # Solve M*x_new = U*x + b using forward substitution
    x = solve_triangular(M_gs, U @ x + b, lower=True)
    res = np.linalg.norm(b - A @ x) / norm_b
    resGS.append(res)
    if res < tol:
        break

# --- SOR
omega = 2 / (1 + np.sin(np.pi / (n + 1)))  # optimal parameter
M_sor = (1/omega) * D - L
N_sor = (1 - omega)/omega * D + U
x = x0.copy()
resSOR = [np.linalg.norm(b - A @ x0) / norm_b]
for k in range(maxit):
    # Solve M_sor*x_new = N_sor*x + b using forward substitution
    x = solve_triangular(M_sor, N_sor @ x + b, lower=True)
    res = np.linalg.norm(b - A @ x) / norm_b
    resSOR.append(res)
    if res < tol:
        break

# --- Summary: does the run reproduce the table of Problem 5(b)? ---
rho_pred = {'Jacobi': np.cos(np.pi / (n + 1)),
            'Gauss-Seidel': np.cos(np.pi / (n + 1))**2,
            'SOR': omega - 1}
print(f"{'method':<14}{'iters to 1e-8':>14}{'observed factor':>18}{'predicted rho':>16}")
for name, res in (('Jacobi', resJ), ('Gauss-Seidel', resGS), ('SOR', resSOR)):
    conv = res[-1] < tol
    obs = res[-1] / res[-2]
    print(f"{name:<14}{(len(res) - 1 if conv else -1):>14}{obs:>18.6f}{rho_pred[name]:>16.6f}")
print("(-1 means the iteration hit maxit without reaching the tolerance)")

# --- Plot
plt.figure()
plt.semilogy(resJ, '-o', label='Jacobi')
plt.semilogy(resGS, '-s', label='Gauss-Seidel')
plt.semilogy(resSOR, '-^', label=f'SOR (optimal ω≈{omega:.2f})')
plt.xlabel('Iteration')
plt.ylabel('Relative residual')
plt.legend()
plt.grid(True)
plt.title(f'Convergence comparison (n={n})')
plt.show()