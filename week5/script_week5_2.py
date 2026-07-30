import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator

# Ridge regression with CG (matrix-free)
np.random.seed(0)
n = 1500    # samples
d = 400     # features
X = np.random.randn(n, d)

# Use scipy.sparse.random for sprandn equivalent
w_true_sparse = sp.random(d, 1, density=0.05, data_rvs=np.random.randn)
w_true = w_true_sparse.toarray().flatten()

y = X @ w_true + 0.1 * np.random.randn(n)

lambda_reg = 1e-1

# Hessian-vector product as a LinearOperator
def hessian_vec_prod(v):
    return X.T @ (X @ v) + lambda_reg * v

A_op = LinearOperator((d, d), matvec=hessian_vec_prod, rmatvec=hessian_vec_prod)
b = X.T @ y

tol = 1e-8
maxit = 500

# To get the iteration count, we can use a callback.
iter_count = [0]
def callback(xk):
    iter_count[0] += 1

x_cg, info = cg(A_op, b, tol=tol, maxiter=maxit, callback=callback)

final_res_norm = np.linalg.norm(b - A_op @ x_cg)
b_norm = np.linalg.norm(b)
relres = final_res_norm / b_norm if b_norm > 0 else 0.0

print(f'CG info={info}, relres={relres:.2e}, iters={iter_count[0]}')

# Compare to direct solve (for moderate d, just for reference)
x_dir = np.linalg.solve(X.T @ X + lambda_reg * np.eye(d), b)

print(f'||x_cg - x_dir||/||x_dir|| = {np.linalg.norm(x_cg - x_dir)/np.linalg.norm(x_dir):.2e}')
