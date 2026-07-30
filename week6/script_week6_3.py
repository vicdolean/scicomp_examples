import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
import pandas as pd

# --- Helper: vanilla Arnoldi (modified Gram-Schmidt)
def simple_arnoldi(A, v1, m):
    n = len(v1)
    V = np.zeros((n, m + 1), dtype=np.complex128)
    H = np.zeros((m + 1, m), dtype=np.complex128)
    V[:, 0] = v1 / np.linalg.norm(v1)
    for j in range(m):
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i].conj(), w)
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-14:
            V[:, j + 1] = np.zeros(n)
            break
        V[:, j + 1] = w / H[j + 1, j]
    return V, H

# Example 3 — Arnoldi & Ritz values on a companion matrix: (z-1)^3
# Shows stagnation of GMRES for two steps, then finite termination at k=3
A = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, -3, 3]], dtype=float)
b = np.array([1, 0, 0], dtype=float)
x0 = np.zeros(3)
tol = 1e-14
maxit = 3

# --- GMRES residuals
resvec = []
def callback_gmres(rk):
    resvec.append(np.linalg.norm(rk))

# Initial residual for plot
resvec.append(np.linalg.norm(b - A @ x0))
x, flag = gmres(A, b, x0=x0, tol=tol, maxiter=maxit, callback=callback_gmres)
relres = resvec[-1] / resvec[0]
iter_num = len(resvec) - 1

plt.semilogy(np.arange(len(resvec)), resvec, 'o-')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||r_k||_2')
plt.title('GMRES on companion matrix: flat, flat, then zero at k=3')
plt.xticks(np.arange(maxit + 1))
plt.show()

print(f'GMRES: flag={flag}, iters={iter_num}, relres={relres:.2e}')

# --- Small Arnoldi to extract Ritz values and a posteriori residuals
m = 3
v1 = b / np.linalg.norm(b)
V, H = simple_arnoldi(A, v1, m)
Hk = H[:m, :m]
hk1k = H[m, m - 1]

eigvals, Y = np.linalg.eig(Hk)
theta = eigvals  # Ritz values

ritz_res = np.abs(hk1k) * np.abs(Y[-1, :])

results_df = pd.DataFrame({
    'RitzValue': theta,
    'ResidualBound': ritz_res
})
print(results_df)