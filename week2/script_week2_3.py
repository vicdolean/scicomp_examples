import numpy as np
import matplotlib.pyplot as plt

def soar(N, L, a=1.0):
    # SOAR correlation matrix for N equally spaced points on a circle of radius a.
    k = np.arange(N)
    theta = 2 * np.pi * np.subtract.outer(k, k) / N
    r = np.abs(2 * a * np.sin(theta / 2))
    return (1 + r / L) * np.exp(-r / L)


C = soar(100, 0.4)

print(f"symmetric?           {np.allclose(C, C.T)}")
print(f"unit diagonal?       {np.allclose(np.diag(C), 1.0)}")
print(f"circulant?           {np.allclose(C[1], np.roll(C[0], 1))}")

# For a circulant matrix the eigenvalues are the DFT of the first row.
lam_fft = np.sort(np.real(np.fft.fft(C[0])))
lam_eig = np.sort(np.linalg.eigvalsh(C))
print(f"\nmax |FFT eigenvalues - eigh eigenvalues| = {np.max(np.abs(lam_fft - lam_eig)):.2e}")
print(f"smallest eigenvalue  {lam_eig[0]:.4e}   largest {lam_eig[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

axes[0].plot(C[0][:51], lw=2)
axes[0].axhline(0.2, color="C3", ls="--", lw=1.3, label=r"threshold $\tau=0.2$")
axes[0].set_xlabel("grid points from the diagonal")
axes[0].set_ylabel("correlation")
axes[0].set_title("First row of $C$  ($N=100$, $L=0.4$)")
axes[0].grid(True, alpha=0.25)
axes[0].legend(fontsize=9)

im = axes[1].imshow(C, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("$C$ — dense, banded-looking, but never zero")
fig.colorbar(im, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.show()

print(f"{'N':>6} {'L':>6} {'lambda_min':>13} {'kappa_2(C)':>13}")
for N, L in [(100, 0.1), (100, 0.4), (200, 0.4)]:
    w = np.linalg.eigvalsh(soar(N, L))
    print(f"{N:>6} {L:>6} {w[0]:>13.2e} {w[-1]/w[0]:>13.2e}")

N, L = 100, 0.4
C = soar(N, L)
k = np.arange(N)
dist = np.abs(np.subtract.outer(k, k))
dist = np.minimum(dist, N - dist)          # circular distance

print(f"{'tau':>6} {'nnz per row':>13} {'circ. half-bw':>15} {'lambda_min':>14} {'SPD?':>7}")
spectra = {}
for tau in [0.0, 0.05, 0.2, 0.5]:
    Ct = np.where(C < tau, 0.0, C)
    w = np.linalg.eigvalsh(Ct)
    spectra[tau] = w
    nnz = int((Ct[0] != 0).sum())
    hb = int(dist[Ct != 0].max())
    print(f"{tau:>6} {nnz:>13} {hb:>15} {w[0]:>+14.3e} {str(w[0] > 0):>7}")

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for tau, w in spectra.items():
    ax.plot(np.sort(w), lw=2, label=rf"$\tau={tau}$")

ax.axhline(0, color="0.3", lw=1.2)
ax.axhspan(min(w.min() for w in spectra.values()) * 1.15, 0, color="C3", alpha=0.07)
ax.text(0.97, 0.06, "not positive definite", color="C3", fontsize=9,
        ha="right", transform=ax.transAxes)

ax.set_xlabel("eigenvalue index (sorted)")
ax.set_ylabel("eigenvalue")
ax.set_title("Thresholding pushes eigenvalues below zero")
ax.set_ylim(-2.4, 4.0)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

def gaspari_cohn(r, c):
    # Gaspari-Cohn (1999) compactly supported correlation: zero for r > 2c.
    x = np.abs(r) / c
    out = np.zeros_like(x)

    m = x <= 1
    z = x[m]
    out[m] = 1 - (5 / 3) * z**2 + (5 / 8) * z**3 + (1 / 2) * z**4 - (1 / 4) * z**5

    m = (x > 1) & (x <= 2)
    z = x[m]
    out[m] = (4 - 5 * z + (5 / 3) * z**2 + (5 / 8) * z**3
              - (1 / 2) * z**4 + (1 / 12) * z**5 - 2 / (3 * z))
    return out


k = np.arange(N)
theta = 2 * np.pi * np.subtract.outer(k, k) / N
r = np.abs(2 * np.sin(theta / 2))          # a = 1, same chord distance as in soar()

print(f"{'method':>22} {'nnz per row':>13} {'lambda_min':>14} {'kappa_2':>12} {'SPD?':>7}")

w = np.linalg.eigvalsh(C)
print(f"{'C (no truncation)':>22} {N:>13} {w[0]:>+14.3e} {w[-1]/w[0]:>12.2e} {str(w[0] > 0):>7}")

Ct = np.where(C < 0.2, 0.0, C)
w = np.linalg.eigvalsh(Ct)
print(f"{'thresholded, tau=0.2':>22} {int((Ct[0] != 0).sum()):>13} {w[0]:>+14.3e} "
      f"{'---':>12} {str(w[0] > 0):>7}")

for c in [0.35, 0.5]:
    T = gaspari_cohn(r, c)
    # The premise of the Schur-product argument: the taper is itself positive definite.
    assert np.linalg.eigvalsh(T)[0] > 0, "taper is not PD"

    Ctap = C * T
    w = np.linalg.eigvalsh(Ctap)
    nnz = int((Ctap[0] != 0).sum())
    kap = f"{w[-1]/w[0]:12.2e}" if w[0] > 0 else f"{'---':>12}"
    print(f"{'tapered, c=' + str(c):>22} {nnz:>13} {w[0]:>+14.3e} {kap} {str(w[0] > 0):>7}")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

T = gaspari_cohn(r, 0.35)
axes[0].plot(C[0][:51], lw=2, label="$C$ (SOAR)")
axes[0].plot(T[0][:51], lw=2, label="$T$ (Gaspari-Cohn, $c=0.35$)")
axes[0].plot((C * T)[0][:51], lw=2, label=r"$C\circ T$ (tapered)")
axes[0].plot(np.where(C[0][:51] < 0.2, 0.0, C[0][:51]), lw=2, ls="--",
             label=r"thresholded, $\tau=0.2$")
axes[0].axhline(0, color="0.3", lw=1.0)
axes[0].set_xlabel("grid points from the diagonal")
axes[0].set_ylabel("correlation")
axes[0].set_title("First row: four ways")
axes[0].grid(True, alpha=0.25)
axes[0].legend(fontsize=8)

# Zoom on the BOTTOM of the spectrum: the largest eigenvalues are around 25 and would
# squash the only part that matters, which is whether the curves cross zero.
axes[1].plot(np.sort(np.linalg.eigvalsh(C)), lw=2, label="$C$")
axes[1].plot(np.sort(np.linalg.eigvalsh(np.where(C < 0.2, 0.0, C))), lw=2, ls="--",
             label=r"thresholded, $\tau=0.2$")
axes[1].plot(np.sort(np.linalg.eigvalsh(C * T)), lw=2, label=r"tapered, $c=0.35$")
axes[1].axhline(0, color="0.3", lw=1.2)
axes[1].axhspan(-0.75, 0, color="C3", alpha=0.07)
axes[1].set_ylim(-0.75, 1.2)
axes[1].set_xlabel("eigenvalue index (sorted)")
axes[1].set_ylabel("eigenvalue")
axes[1].set_title("Bottom of the spectrum (the top is off-scale)")
axes[1].grid(True, alpha=0.25)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
