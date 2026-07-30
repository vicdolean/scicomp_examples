import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg, LinearOperator
import matplotlib.pyplot as plt

# Domain: indices 0..n-1; overlap nodes are i=alpha:beta (inclusive)
n = 127
h = 1 / (n + 1)
e = np.ones(n)
A = sp.spdiags([-e, 2 * e, -e], [-1, 0, 1], n, n, format='csr') / h**2

# Split indices: [0..iL-1] and [iR-1..n-1], with overlap [iR-1..iL-1]
iL = 80
iR = 50  # choose so that iR <= iL (overlap width ~ iL-iR+1)
I1 = np.arange(iL)        # left subdomain dofs
I2 = np.arange(iR - 1, n) # right subdomain dofs
Ao1 = A[I1, :][:, I1]
Ao2 = A[I2, :][:, I2]

# Partition of unity (half weights on overlap)
w1 = np.zeros(n)
w2 = np.zeros(n)
w1[I1] = 1
w2[I2] = 1
ov = np.intersect1d(I1, I2)
w1[ov] = 0.5
w2[ov] = 0.5

# Apply M^{-1} v = R1^T D1 A1^{-1} R1 v + R2^T D2 A2^{-1} R2 v
def apply_Minv(v):
    sol1 = spsolve(Ao1, v[I1])
    sol2 = spsolve(Ao2, v[I2])
    v1_ext = np.zeros(n)
    v2_ext = np.zeros(n)
    v1_ext[I1] = sol1
    v2_ext[I2] = sol2
    return w1 * v1_ext + w2 * v2_ext

# Build a function-handle preconditioner for pcg
Mfun = LinearOperator((n, n), matvec=apply_Minv)

# Test problem
xgrid = np.arange(1, n + 1) * h
uex = np.sin(2 * np.pi * xgrid)
f = (2 * np.pi)**2 * uex
b = f

# (a) Stand-alone additive Schwarz iteration u^{k+1} = u^k + M^{-1}(b-Au^k)
u = np.zeros(n)
rk = b - A @ u
resA = np.linalg.norm(rk)
res_sch = [resA]
itA = 0
while resA > 1e-8 and itA < 200:
    u = u + apply_Minv(rk)
    rk = b - A @ u
    resA = np.linalg.norm(rk)
    res_sch.append(resA)
    itA += 1

# (b) PCG with the same Schwarz preconditioner
tol = 1e-8
maxit = 500
x0 = np.zeros(n)
res_pcg = []
def callback_pcg(xk):
    res_pcg.append(np.linalg.norm(b - A @ xk))

res_pcg.append(np.linalg.norm(b - A @ x0))
_, _ = cg(A, b, x0=x0, tol=tol, maxiter=maxit, M=Mfun, callback=callback_pcg)

# Compare residuals
plt.figure()
plt.semilogy(np.arange(len(res_sch)), res_sch, 'o-', label='Additive Schwarz (stand-alone)')
plt.semilogy(np.arange(len(res_pcg)), res_pcg, '*-', label='PCG + Schwarz')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||r_k||_2')
plt.legend(loc='southwest')
plt.title('Schwarz as iterative method vs as PCG preconditioner')
plt.show()