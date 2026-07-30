import numpy as np
from scipy.linalg import hilbert, solve_triangular
import matplotlib.pyplot as plt

# %% Numerical Linear Algebra Demo
# Illustrates norms, conditioning, stability, and floating-point effects.

print("%% 1. Vector norms")
x = np.array([1, -2, 3])
print(f'Vector norms for x = {x.T}:\n')
print(f'  ||x||_1  = {np.linalg.norm(x, 1):.4f}')
print(f'  ||x||_2  = {np.linalg.norm(x, 2):.4f}')
print(f'  ||x||_inf= {np.linalg.norm(x, np.inf):.4f}\n')

print("%% 2. Condition numbers")
I = np.eye(5)
D = np.diag(np.logspace(0, 4, 5))  # diag(1,10,100,1000,10000)
H = hilbert(5)

print('Condition numbers (2-norm):\n')
print(f'  Identity:        {np.linalg.cond(I):.2e}')
print(f'  Diagonal(1..1e4):{np.linalg.cond(D):.2e}')
print(f'  Hilbert(5x5):    {np.linalg.cond(H):.2e}\n')

print("%% 3. Catastrophic cancellation")
a = 1.0000000001
b = 1.0
exact = a - b
computed = np.float32(a) - np.float32(b)  # force rounding
print('Catastrophic cancellation:\n')
print(f'  Exact difference   = {exact:.10e}')
print(f'  Stored in single   = {computed:.10e}\n')

print("%% 4. Effect of conditioning on solving systems")
A1 = np.array([[1, 2], [3, 4]])
A2 = np.array([[1, 1], [1, 1.0001]])
b_sys = np.array([5, 11])

x1 = np.linalg.solve(A1, b_sys)
x2 = np.linalg.solve(A2, b_sys)

print('Effect of conditioning:\n')
print(f'  Solution with A1 (cond={np.linalg.cond(A1):.2e}): x = {x1.T}')
print(f'  Solution with A2 (cond={np.linalg.cond(A2):.2e}): x = {x2.T}\n')

print("%% 5. Floating-point summation stability")
n_sum = int(1e6)
vals = np.concatenate(([1.0], np.full(n_sum, 1e-8)))
print('Summation order matters:\n')
sum_forward = np.sum(vals)
sum_backward = np.sum(vals[::-1]) # large-to-small
print(f'  Sum small-to-large (approx): {sum_forward:.15e}')
print(f'  Sum large-to-small:          {sum_backward:.15e}\n')


print("%% 6. Fixed point iterations for SOAR correlation function")
n_soar = 100
L_soar = 0.4
a_soar = 1

# construct SOAR matrix (vectorized)
i, l = np.mgrid[0:n_soar, 0:n_soar]
theta_j = (2 * np.pi / n_soar) * np.abs(i - l)
dist = np.abs(2 * a_soar * np.sin(theta_j / 2)) / L_soar
C = (1 + dist) * np.exp(-dist)

# Iteration parameters
maxit = 5000
tol = 1e-5
x0 = np.zeros(n_soar)
b_soar = np.ones(n_soar)
norm_b = np.linalg.norm(b_soar)

# --- Jacobi
D = np.diag(np.diag(C))
L = -np.tril(C, -1)
U = -np.triu(C, 1)
invD = np.diag(1 / np.diag(D))
GJ = invD @ (L + U)
cJ = invD @ b_soar
x = x0.copy()
resJ = []
for k in range(maxit):
    x = GJ @ x + cJ
    res = np.linalg.norm(b_soar - C @ x) / norm_b
    resJ.append(res)
    if res < tol:
        break

# --- Gauss-Seidel
M = D - L
x = x0.copy()
resGS = []
for k in range(maxit):
    # Forward substitution to solve M*x_new = U*x + b
    x_new = solve_triangular(M, U @ x + b_soar, lower=True)
    x = x_new
    res = np.linalg.norm(b_soar - C @ x) / norm_b
    resGS.append(res)
    if res < tol:
        break

# --- SOR
omega = 2 / (1 + np.sin(np.pi / (n_soar + 1)))  # optimal parameter
M_sor = (1/omega) * D - L
N_sor = (1-omega)/omega * D + U
x = x0.copy()
resSOR = []
for k in range(maxit):
    # Forward substitution: M_sor * x_new = N_sor * x + b
    rhs = N_sor @ x + b_soar
    x_new = solve_triangular(M_sor, rhs, lower=True)
    x = x_new
    res = np.linalg.norm(b_soar - C @ x) / norm_b
    resSOR.append(res)
    if res < tol:
        break

plt.figure()
plt.semilogy(resGS, 'b-', label='Gauss-Seidel')
plt.semilogy(resSOR, 'r-', label='SOR')
plt.semilogy(resJ, 'k-', label='Jacobi')
plt.xlabel('Iteration')
plt.ylabel('Relative residual norm')
plt.title('Convergence of iterative methods for SOAR matrix')
plt.legend()
plt.grid(True)
plt.show()