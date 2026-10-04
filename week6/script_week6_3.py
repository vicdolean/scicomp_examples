import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
import pandas as pd

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

A = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, -3, 3]], dtype=float)
b = np.array([1, 0, 0], dtype=float)
x0 = np.zeros(3)
tol = 1e-14
maxit = 3

resvec = []
def callback_gmres(rk):
    resvec.append(np.linalg.norm(rk))

# Initial residual for plot
resvec.append(np.linalg.norm(b - A @ x0))
x, flag = gmres(A, b, x0=x0, rtol=tol, maxiter=maxit, callback=callback_gmres)
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

A3 = np.array([[1, 2, 0],
               [0, 1, 2],
               [0, 0, 1]], dtype=float)
b3 = np.array([0, 0, 1], dtype=float)
V3, H3 = simple_arnoldi(A3, b3, 2)
print('V_3 =\n', V3.real)
print('underline H_2 =\n', H3.real)
for k in (1, 2):
    Hk_ = H3[:k + 1, :k].real
    rhs_ = np.zeros(k + 1); rhs_[0] = 1.0
    y, *_ = np.linalg.lstsq(Hk_, rhs_, rcond=None)
    print(f'k={k}: x_k = {(V3[:, :k].real @ y).round(4)},  ||r_k|| = {np.linalg.norm(rhs_ - Hk_ @ y):.6f}')
print('2/sqrt(5) =', 2 / np.sqrt(5), '  4/sqrt(21) =', 4 / np.sqrt(21))
print('x* =', np.linalg.solve(A3, b3))
x = np.array([1, -np.sqrt(2), 1]) / 2
print('Rayleigh quotient x^T A x =', x @ A3 @ x, '(< 0, so 0 is in F(A))')
