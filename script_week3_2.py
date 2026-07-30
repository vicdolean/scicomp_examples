import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Part 1: Fiedler vector and Jacobi iteration

# Laplacian of two triangles connected by an edge
A = np.array([[0, 1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 0],
              [1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1],
              [0, 0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 0]])
d = A.sum(axis=1)
D = np.diag(d)
L = D - A

# Eigen-decomposition
Lambda, V = np.linalg.eig(L)
idx = np.argsort(Lambda)
eigs_sorted = Lambda[idx]
V = V[:, idx]

# Fiedler vector (eigenvector for second-smallest eigenvalue)
fiedler = V[:, 1]

print('Fiedler eigenvalue:')
print(eigs_sorted[1])
print('\nFiedler vector:')
print(fiedler)

# Jacobi iteration matrix
GJ = np.linalg.solve(D, A)

# Start from random vector
np.random.seed(0)
x0 = np.random.randn(6)
x0 = x0 / np.linalg.norm(x0)

# Perform iterations
numit = 20
X = np.zeros((6, numit + 1))
X[:, 0] = x0

for k in range(numit):
    X[:, k + 1] = GJ @ X[:, k]
    # Project out the constant vector component and normalize
    X[:, k + 1] = X[:, k + 1] - np.mean(X[:, k + 1])
    X[:, k + 1] = X[:, k + 1] / np.linalg.norm(X[:, k + 1])

# Compare with Fiedler vector
plt.figure(figsize=(12, 8))
plot_iters = [0, 1, 2, 5, 10, 20]
for i, k in enumerate(plot_iters):
    plt.subplot(2, 3, i + 1)
    plt.stem(X[:, k])
    plt.title(f'Jacobi iteration k={k}')
    plt.ylim([-1, 1])
plt.suptitle('Convergence of Jacobi Iteration to Fiedler Vector')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure()
plt.stem(fiedler)
plt.title('Fiedler vector')
plt.ylim([-1, 1])

# Part 2: Comparing eigenvalues of Poisson problem matrix with those of P_n
n = 10
# matrix for Laplacian
Lap = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()

# matrix for P_n
path = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
path[0, 0] = 1
path[n - 1, n - 1] = 1

Lapeig = np.linalg.eigvalsh(Lap)
patheig = np.linalg.eigvalsh(path)

plt.figure()
plt.plot(np.sort(Lapeig), 'bx', markersize=8, label='Eigenvalues of Laplacian')
plt.plot(np.sort(patheig), 'ro', markerfacecolor='none', markersize=8, label='Eigenvalues of Path Matrix')
plt.xlabel('Index (sorted)')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalue Comparison: 1D Laplacian vs. Path Graph')
plt.legend()
plt.grid(True)
plt.show()