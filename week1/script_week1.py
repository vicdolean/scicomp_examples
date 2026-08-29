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
# Not the subtraction that is inaccurate: the answer is so small that the error the
# inputs already carry is the same size as it. Problem 5.
a = 1.0000000001
b = 1.0
exact  = 1e-10                                 # true difference
double = a - b
single = np.float32(a) - np.float32(b)         # force rounding to ~7 digits

print('Catastrophic cancellation:\n')
print(f'  true a - b        = {exact:.10e}')
print(f'  double precision  = {double:.10e}   rel. err = {abs(double - exact) / exact:.2e}')
print(f'  single precision  = {single:.10e}   rel. err = {abs(single - exact) / exact:.2e}\n')

print("%% 3b. Unit roundoff, the gap eps, and the concept check")
import math
from decimal import Decimal

print('A', 0.1 + 0.2 == 0.3, ' ', Decimal(0.1 + 0.2), 'vs', Decimal(0.3))
print('B', 1.0 + 2**-53 == 1.0, ' ', (1.0 + 2**-53).hex())
print('C', 2**-53 == 0.0, ' ', float(2**-53).hex())
print('D', (1.0 + 1e16) - 1e16 == 1.0 + (1e16 - 1e16),
      ' ', (1.0 + 1e16) - 1e16, 'vs', 1.0 + (1e16 - 1e16))
print()
print(f'  gap at 1 (numpy eps)      = {np.finfo(float).eps}  = 2**-52')
print(f'  unit roundoff u           = {2**-53}  = 2**-53')
print(f'  smallest e with 1+e != 1  = {math.nextafter(2**-53, 1.0)}  = 2**-53 + 2**-105')
print(f'  gap at 1e16               = {math.ulp(1e16)}   <- the +1 falls off the end')

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
# Harmonic series in single precision, summed in both orders (Problem 6 of Chapter 1).
# np.cumsum accumulates sequentially, so its last entry is the naive running sum;
# np.sum would use pairwise summation and hide the effect.
N = 10**7
terms = np.float32(1.0) / np.arange(1, N + 1, dtype=np.float32)
large_first = np.cumsum(terms, dtype=np.float32)[-1]            # 1, 1/2, 1/3, ...
small_first = np.cumsum(terms[::-1], dtype=np.float32)[-1]      # ..., 1/3, 1/2, 1
reference = np.cumsum(1.0 / np.arange(1, N + 1, dtype=np.float64))[-1]
print('Summation order matters (harmonic series, N = 1e7):\n')
print(f'  single precision, large terms first : {large_first:.7f}')
print(f'  single precision, small terms first : {small_first:.7f}')
print(f'  double precision (reference)        : {reference:.7f}\n')


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
# Numerically search for the omega that minimizes the SOR iteration matrix's
# spectral radius on THIS matrix C (the Poisson closed-form formula doesn't apply:
# Young's theorem needs a consistently ordered matrix with real Jacobi eigenvalues,
# and here Jacobi does not even converge).
def sor_spectral_radius(w):
    M_w = (1 / w) * D - L
    N_w = (1 - w) / w * D + U
    Gw = np.linalg.solve(M_w, N_w)
    return np.max(np.abs(np.linalg.eigvals(Gw)))

omega_grid = np.linspace(0.1, 1.99, 100)
rhos = [sor_spectral_radius(w) for w in omega_grid]
omega = omega_grid[np.argmin(rhos)]
print(f'Numerically chosen omega = {omega:.4f} (spectral radius = {min(rhos):.5f})')
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