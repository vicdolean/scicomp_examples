import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)      # fixed seed: the numbers quoted in the notes and slides are reproducible

def lanczos(matvec, q1, m, deflate=None):
    n = q1.size
    Q = np.zeros((n, m + 1))
    alpha, beta = np.zeros(m), np.zeros(m)
    Q[:, 0] = q1 / np.linalg.norm(q1)
    q_prev, b_prev = np.zeros(n), 0.0
    for j in range(m):
        w = matvec(Q[:, j]) - b_prev * q_prev
        if deflate is not None:                       # keep the iteration orthogonal to a known eigenvector
            w -= (deflate @ w) / (deflate @ deflate) * deflate
        alpha[j] = Q[:, j] @ w
        w -= alpha[j] * Q[:, j]
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-12:                           # invariant subspace found: Lanczos stops exactly
            m = j + 1
            break
        q_prev, b_prev = Q[:, j], beta[j]
        Q[:, j + 1] = w / beta[j]
    T = np.diag(alpha[:m]) + np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
    return Q[:, :m], T

n = 200
e = np.ones(n)
A = sp.diags([-e, 2 * e, -e], [-1, 0, 1], shape=(n, n), format='csr')
lam = np.linalg.eigvalsh(A.toarray())                 # check only

q1 = rng.standard_normal(n)
m_max = 40
plt.figure(figsize=(7, 4))
for m in range(1, m_max + 1):
    _, T = lanczos(lambda v: A @ v, q1, m)
    theta = np.linalg.eigvalsh(T)
    plt.plot(np.full(theta.size, m), theta, 'k.', ms=3)
plt.axhline(lam[0], color='C0', lw=0.8, label=r'$\lambda_{\min}$')
plt.axhline(lam[-1], color='C3', lw=0.8, label=r'$\lambda_{\max}$')
plt.xlabel('Lanczos step m'); plt.ylabel('Ritz values')
plt.title('Ritz values of the 1D Poisson matrix, n = 200')
plt.legend(); plt.show()

for m in (10, 20, 40):
    _, T = lanczos(lambda v: A @ v, q1, m)
    theta = np.linalg.eigvalsh(T)
    print(f"m = {m:2d}: theta_min = {theta[0]:.5f} (lambda_min = {lam[0]:.5f}),  "
          f"theta_max = {theta[-1]:.5f} (lambda_max = {lam[-1]:.5f}),  "
          f"kappa estimate = {theta[-1] / theta[0]:9.1f} (true {lam[-1] / lam[0]:.1f})")

def cg_scalars(A, b, m):
    # plain CG from x0 = 0, returning the step lengths alpha_k and the coefficients beta_k
    r = b.copy(); p = r.copy(); rho = r @ r
    a_cg, b_cg = [], []
    for k in range(m):
        Ap = A @ p
        a = rho / (p @ Ap)
        r = r - a * Ap
        rho_new = r @ r
        a_cg.append(a); b_cg.append(rho_new / rho)
        p = r + (rho_new / rho) * p
        rho = rho_new
    return np.array(a_cg), np.array(b_cg)

def T_from_cg(a_cg, b_cg):
    m = a_cg.size
    diag = np.array([1 / a_cg[0]] + [1 / a_cg[j] + b_cg[j - 1] / a_cg[j - 1] for j in range(1, m)])
    off = np.sqrt(b_cg[:m - 1]) / a_cg[:m - 1]
    return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

b = rng.standard_normal(n)
m = 15
_, T_lanczos = lanczos(lambda v: A @ v, b, m)
T_cg = T_from_cg(*cg_scalars(A, b, m))
print(f"max |T_lanczos - T_cg| = {np.max(np.abs(T_lanczos - T_cg)):.2e}")
theta = np.linalg.eigvalsh(T_cg)
print(f"after {m} CG steps: lambda_min ~ {theta[0]:.4f}, lambda_max ~ {theta[-1]:.4f}, "
      f"kappa ~ {theta[-1] / theta[0]:.0f}   (true kappa = {lam[-1] / lam[0]:.0f})")

Adj = np.array([[0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0]], dtype=float)
L = np.diag(Adj.sum(axis=1)) - Adj
ones = np.ones(6)

q1 = rng.standard_normal(6)
q1 -= (ones @ q1) / 6 * ones                           # start orthogonal to the constant vector
for m in (1, 2, 3):
    Q, T = lanczos(lambda v: L @ v, q1, m, deflate=ones)
    print(f"m = {m}: Ritz values", np.round(np.linalg.eigvalsh(T), 4))

theta, Y = np.linalg.eigh(T)
v = Q @ Y[:, 0]
v *= np.sign(v[0])
print("Ritz vector for theta_1 =", np.round(theta[0], 4), ":", np.round(v, 4))
print("clusters:", np.where(v > 0)[0] + 1, "|", np.where(v < 0)[0] + 1)
print("check with a dense solver, spectrum of L:", np.round(np.linalg.eigvalsh(L), 4))

N, d = 1500, 400
X = rng.standard_normal((N, d))
w_true = np.zeros(d)
support = rng.choice(d, size=d // 20, replace=False)
w_true[support] = rng.standard_normal(support.size)
y = X @ w_true + 0.1 * rng.standard_normal(N)

def ridge_operator(lam_reg):
    return LinearOperator((d, d), matvec=lambda v: X.T @ (X @ v) + lam_reg * v, dtype=float)

lambda_reg = 1e-1
A_op = ridge_operator(lambda_reg)
rhs = X.T @ y

iters = [0]
def count(_):
    iters[0] += 1

w_cg, info = cg(A_op, rhs, rtol=1e-8, maxiter=500, callback=count)   # SciPy >= 1.12: rtol (not tol)
w_dir = np.linalg.solve(X.T @ X + lambda_reg * np.eye(d), rhs)       # check only (d is small here)
print(f"CG info = {info}, iterations = {iters[0]}, "
      f"||w_cg - w_dir|| / ||w_dir|| = {np.linalg.norm(w_cg - w_dir) / np.linalg.norm(w_dir):.2e}")

U, _ = np.linalg.qr(rng.standard_normal((N, d)))
V, _ = np.linalg.qr(rng.standard_normal((d, d)))
sig = np.logspace(0, -3, d)
Xc = U @ np.diag(sig) @ V.T
rhs_c = Xc.T @ rng.standard_normal(N)

print(" lambda     kappa(A_lambda)   CG iterations (rtol 1e-8)")
for lam_reg in (1e-6, 1e-4, 1e-2, 1e0):
    op = LinearOperator((d, d), matvec=lambda v, l=lam_reg: Xc.T @ (Xc @ v) + l * v, dtype=float)
    iters = [0]
    cg(op, rhs_c, rtol=1e-8, maxiter=5000, callback=count)
    kappa = (sig[0]**2 + lam_reg) / (sig[-1]**2 + lam_reg)
    print(f"{lam_reg:8.0e}   {kappa:14.3e}   {iters[0]:8d}")

Ns, ds = 200, 100
Us, _ = np.linalg.qr(rng.standard_normal((Ns, ds)))
Vs, _ = np.linalg.qr(rng.standard_normal((ds, ds)))
sig_s = np.logspace(0, -4, ds)
Xs = Us @ np.diag(sig_s) @ Vs.T
x_true = Vs @ sig_s                                   # smooth solution: coefficients decay like sigma_i
y_exact = Xs @ x_true
noise = rng.standard_normal(Ns)
delta = 1e-2 * np.linalg.norm(y_exact)
y_noisy = y_exact + delta * noise / np.linalg.norm(noise)

AtA, Atb = Xs.T @ Xs, Xs.T @ y_noisy
x = np.zeros(ds); r = Atb.copy(); p = r.copy()
err, res = [np.linalg.norm(x - x_true)], [np.linalg.norm(y_noisy - Xs @ x)]
for k in range(300):
    Ap = AtA @ p
    a = (r @ r) / (p @ Ap)
    x = x + a * p
    r_new = r - a * Ap
    p = r_new + (r_new @ r_new) / (r @ r) * p
    r = r_new
    err.append(np.linalg.norm(x - x_true)); res.append(np.linalg.norm(y_noisy - Xs @ x))
err = np.array(err) / np.linalg.norm(x_true)

k_best = int(np.argmin(err))
k_dp = next(k for k, v in enumerate(res) if v <= 1.1 * delta)
x_ls = np.linalg.lstsq(Xs, y_noisy, rcond=None)[0]

plt.figure(figsize=(7, 4))
plt.semilogy(err, label='CG on the normal equations')
plt.axvline(k_best, color='gray', ls='--', lw=0.8, label=f'best k = {k_best}')
plt.axvline(k_dp, color='C3', ls=':', lw=1.2, label=f'discrepancy principle, k = {k_dp}')
plt.xlabel('CG iteration k'); plt.ylabel('relative error ||x_k - x_true|| / ||x_true||')
plt.title('Semi-convergence'); plt.legend(); plt.show()

print(f"best iterate: k = {k_best}, relative error {err[k_best]:.3f}")
print(f"discrepancy principle: k = {k_dp}, relative error {err[k_dp]:.3f}")
print(f"after 300 iterations: relative error {err[-1]:.3f}")
print(f"least-squares solution (no regularisation): relative error "
      f"{np.linalg.norm(x_ls - x_true) / np.linalg.norm(x_true):.3f}")
