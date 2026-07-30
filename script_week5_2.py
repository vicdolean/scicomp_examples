import numpy as np
import matplotlib.pyplot as plt

# %% Example 2 — PageRank by the power method (3-node toy + residual check)
# Graph:
# 1 -> 2, 2 -> 3, 3 -> 1 and 2
P = np.array([[0, 0, 0.5],
              [1, 0, 0.5],
              [0, 1, 0]])  # column-stochastic
alpha = 0.85
n = P.shape[0]
G = alpha * P + (1 - alpha) / n * np.ones((n, n))  # Google matrix

x = np.ones(n) / n  # initial rank vector
tol = 1e-12
maxit = 200
res = []

for k in range(maxit):
    x_new = G @ x  # power step (no need to renormalize; G is stochastic)
    residual = np.linalg.norm(x_new - x, 1)
    res.append(residual)
    x = x_new
    if residual < tol:
        break

print('PageRank vector (l1-normalized):')
x = x / np.sum(x)
print(x)
print(f'Iterations: {len(res)}, ||x_{{k+1}}-x_k||_1 = {res[-1]:.2e}')

plt.figure()
plt.semilogy(np.arange(1, len(res) + 1), res, 'o-')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||x_{k+1}-x_k||_1')
plt.title('Power method for PageRank (3-node toy)')
plt.show()

# (Optional) Compare to eig
eigvals, eigvecs = np.linalg.eig(G)
idx = np.argmax(np.real(eigvals)) # Find eigenvector for eigenvalue near 1
x_eig = np.real(eigvecs[:, idx])
x_eig = x_eig / np.sum(x_eig)

print(f'||x_power - x_eig||_1 = {np.linalg.norm(x - x_eig, 1):.2e}')