import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Part 1: the two Fiedler vectors, and the deflated lazy walk

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

# (a) COMBINATORIAL Fiedler vector:  L v = lambda v
lam_c, V_c = np.linalg.eigh(L)
fiedler_comb = V_c[:, 1]

# (b) NORMALIZED (random-walk) Fiedler direction:  L v = lambda D v, equivalently the
#     second eigenvector of S = D^{-1/2} A D^{-1/2} pulled back by D^{-1/2}.
#     THIS is what the deflated Jacobi iteration below converges to -- not (a).
S = np.diag(d ** -0.5) @ A @ np.diag(d ** -0.5)
mu, V_s = np.linalg.eigh(S)
order = np.argsort(mu)[::-1]
mu = mu[order]
fiedler_norm = np.diag(d ** -0.5) @ V_s[:, order[1]]
fiedler_norm /= np.linalg.norm(fiedler_norm)
if fiedler_comb @ fiedler_norm < 0:
    fiedler_norm = -fiedler_norm

print(f'combinatorial Fiedler value  lambda_2 = {lam_c[1]:.6f}')
print(f'combinatorial Fiedler vector = {np.round(fiedler_comb, 4)}')
print(f'normalized    Fiedler vector = {np.round(fiedler_norm, 4)}')
cos = min(abs(fiedler_comb @ fiedler_norm), 1.0)
print(f'angle between them = {np.degrees(np.arccos(cos)):.2f} degrees '
      '-- close on this graph, but not the same vector\n')

# --- deflated iteration ------------------------------------------------------
P = np.linalg.solve(D, A)            # random-walk matrix = undamped Jacobi
P_half = 0.5 * (np.eye(6) + P)       # lazy walk = damped Jacobi, omega = 1/2
one = np.ones(6)


def deflated(M, numit, x0):
    """Project out the constant mode in <.,.>_D, THEN normalise in ||.||_D."""
    X = np.zeros((6, numit + 1))
    x = x0 / np.sqrt(x0 @ D @ x0)
    X[:, 0] = x
    for k in range(numit):
        xh = M @ x
        xh = xh - (xh @ D @ one) / (one @ D @ one) * one
        x = xh / np.sqrt(xh @ D @ xh)
        X[:, k + 1] = x
    return X


rng = np.random.default_rng(0)
x0 = rng.standard_normal(6)
numit = 20
X_lazy = deflated(P_half, numit, x0)
X_plain = deflated(P, numit, x0)


def align(x):
    return abs(x @ D @ fiedler_norm) / np.sqrt((x @ D @ x) * (fiedler_norm @ D @ fiedler_norm))


print(f'after {numit} iterations, alignment with the normalized Fiedler direction:')
print(f'  lazy walk  (1/2)(I+P) : {align(X_lazy[:, -1]):.6f}')
print(f'  plain walk P          : {align(X_plain[:, -1]):.6f}')
print(f'predicted rate (1+mu_3)/(1+mu_2) = {(1 + mu[2]) / (1 + mu[1]):.4f}')
print(f'  mu_2 = {mu[1]:.4f}, mu_3 = {mu[2]:.4f}, mu_n = {mu[-1]:.4f}')
print('  mu_2 > |mu_n| here, so plain P happens to work on THIS graph;')
print('  script_week3_3, section 5, shows graphs where it does not.')

# --- plots -------------------------------------------------------------------
plt.figure(figsize=(12, 8))
plot_iters = [0, 1, 2, 5, 10, 20]
for i, k in enumerate(plot_iters):
    plt.subplot(2, 3, i + 1)
    plt.stem(X_lazy[:, k] / np.linalg.norm(X_lazy[:, k]))
    plt.title(f'deflated lazy walk, k={k}')
    plt.ylim([-1, 1])
plt.suptitle('Convergence of the deflated lazy walk to the normalized Fiedler direction')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure()
plt.plot(fiedler_comb, 'o-', label=r'combinatorial: $Lv=\lambda v$')
plt.plot(fiedler_norm, 's--', label=r'normalized: $Lv=\lambda Dv$')
plt.title('The two Fiedler vectors of the same graph')
plt.xlabel('vertex')
plt.ylim([-1, 1])
plt.legend()
plt.grid(True)

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