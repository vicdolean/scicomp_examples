import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

u = lambda x: np.sin(np.pi * x)
x0 = 0.5
exact = -np.pi**2                      # u''(0.5) for u = sin(pi x)

D_xx = lambda h: (u(x0 + h) - 2 * u(x0) + u(x0 - h)) / h**2

print(f"exact u''(0.5) = {exact:.8f}\n")
print(f"{'h':>8} {'D_xx^h u(0.5)':>16} {'error':>12} {'ratio':>8}")
prev = None
for h in [0.25, 0.125, 0.0625]:
    approx = D_xx(h)
    err = abs(approx - exact)
    ratio = "---" if prev is None else f"{prev / err:.3f}"
    print(f"{h:>8} {approx:>16.8f} {err:>12.4e} {ratio:>8}")
    prev = err

# The leading term of the error, h^2/12 * u''''(xi), predicts the h = 0.25 entry:
predicted = 0.25**2 / 12 * np.pi**4     # u'''' (0.5) = pi^4
print(f"\npredicted error at h = 0.25: {predicted:.4f}   observed: {abs(D_xx(0.25) - exact):.4f}")

eps = np.finfo(float).eps
hs = np.logspace(-1, -9, 200)
err = np.array([abs(D_xx(h) - exact) for h in hs])

trunc = hs**2 / 12 * np.pi**4           # ||u''''||_inf = pi^4
round_ = 4 * eps / hs**2                # ||u||_inf = 1

i = err.argmin()
print(f"measured minimum error {err[i]:.3e} at h = {hs[i]:.3e}")
print(f"eps_mach^(1/4)        = {eps**0.25:.3e}")
print(f"\nerror at h = 1e-2: {abs(D_xx(1e-2) - exact):.2e}")
print(f"error at h = 1e-6: {abs(D_xx(1e-6) - exact):.2e}")
print(f"error at h = 1e-8: {abs(D_xx(1e-8) - exact):.2e}   <- no correct digits left")

fig, ax = plt.subplots(figsize=(7, 4.6))

ax.loglog(hs, err, lw=2, label="measured total error")
ax.loglog(hs, trunc, "--", color="0.45", lw=1.4, label=r"truncation  $h^2\|u^{(4)}\|_\infty/12$")
ax.loglog(hs, round_, ":", color="0.45", lw=1.4, label=r"rounding  $4\varepsilon_{\rm mach}/h^2$")

ax.plot(hs[i], err[i], "o", ms=9, color="C3", zorder=5)
ax.annotate(f"best $h\\approx{hs[i]:.0e}$, error $\\approx{err[i]:.0e}$",
            xy=(hs[i], err[i]), xytext=(-14, -30), textcoords="offset points",
            ha="right", fontsize=9, color="C3")

ax.set_xlabel("$h$")
ax.set_ylabel("absolute error in $u''(0.5)$")
ax.set_title("Shrinking $h$ helps, until it doesn't")
ax.invert_xaxis()                        # h decreasing to the right, as in the story
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper center", fontsize=9)
plt.tight_layout()
plt.show()

# The balance point with the constants kept, against the measured minimum. Same order,
# different constant: the 4*eps/h^2 term is a bound, not the error actually committed.
h_star = (48 * eps * 1.0 / np.pi**4)**0.25
print(f"balanced h* = {h_star:.3e}    measured argmin = {hs[i]:.3e}    eps^(1/4) = {eps**0.25:.3e}")

def poisson_1d(n):
    # A = tridiag(-1,2,-1)/h^2 (sparse, CSR), together with the grid spacing h.
    h = 1 / (n + 1)
    A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr") / h**2
    return A, h


def solve_bvp(n, f, alpha=0.0, beta=0.0):
    # Solve -u'' = f on (0,1) with Dirichlet data; return interior grid and solution.
    A, h = poisson_1d(n)
    x = np.linspace(0, 1, n + 2)[1:-1]
    g = f(x).astype(float)
    g[0] += alpha / h**2
    g[-1] += beta / h**2
    return x, spla.spsolve(A.tocsc(), g)


# Problem 2: f = 1, n = 4, homogeneous Dirichlet. Exact solution u(x) = x(1-x)/2.
x, uh = solve_bvp(4, lambda x: np.ones_like(x))
print("computed :", np.array2string(uh, precision=6))
print("exact    :", np.array2string(x * (1 - x) / 2, precision=6))
print("max error:", f"{np.max(np.abs(uh - x * (1 - x) / 2)):.2e}")

# u(x) = sin(pi x)  =>  -u'' = pi^2 sin(pi x). Now u'''' is not zero and we should see O(h^2).
f = lambda x: np.pi**2 * np.sin(np.pi * x)
u_exact = lambda x: np.sin(np.pi * x)

print(f"{'n':>6} {'h':>10} {'max error':>12} {'ratio':>8}")
prev = None
for n in [9, 19, 39, 79, 159]:
    x, uh = solve_bvp(n, f)
    e = np.max(np.abs(uh - u_exact(x)))
    ratio = "---" if prev is None else f"{prev / e:.3f}"
    print(f"{n:>6} {1/(n+1):>10.5f} {e:>12.4e} {ratio:>8}")
    prev = e
