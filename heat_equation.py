#!/usr/bin/env python3
"""
Reduced-order integration of the 1-D heat equation with a POD basis.

Author: (your name)
"""

import numpy as np
import scipy.linalg as la
import scipy.integrate as itg
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Parameters you may tweak
# ------------------------------------------------------------------
alpha     = 0.01   # thermal diffusivity
nx        = 1000   # spatial grid points
nt_snap   = 60     # number of snapshots used to build the POD basis
t_end     = 0.5    # final simulation time
m         = 2      # POD rank
ode_method = "RK45"  # or "Radau" / "BDF" for stiff implicit
make_plots = True
# ------------------------------------------------------------------

# Spatial grid (uniform, Dirichlet at x=0 and x=1 => interior points only)
x = np.linspace(0.0, 1.0, nx, endpoint=False)
dx = x[1] - x[0]

# Initial condition and exact solution generator -------------------
def u0(x):
    return np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)

def heat_exact(x, t, alpha):
    return (np.exp(-np.pi**2 * alpha * t) * np.sin(np.pi * x) +
            0.5 * np.exp(-4*np.pi**2 * alpha * t) * np.sin(2*np.pi * x))

# Snapshots --------------------------------------------------------
times_snap = np.linspace(0.0, t_end, nt_snap)
S = np.column_stack([heat_exact(x, t, alpha) for t in times_snap])

# POD via thin SVD --------------------------------------------------
U, s, VT = la.svd(S, full_matrices=False)
U_m = U[:, :m]               # POD spatial modes
Sigma_m = np.diag(s[:m])
print(f"Energy captured (m={m}): "
      f"{np.sum(s[:m]**2)/np.sum(s**2):.6f}")

# Discrete Laplacian L (central FD, Dirichlet BCs) -----------------
e = np.ones(nx)
L = (np.diag(-2*e) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1)) / dx**2
# Impose Dirichlet by zeroing first/last rows (already zero because u=0 outside)
# If you wanted ghost-point BCs, modify L accordingly.

# Reduced matrices -------------------------------------------------
M = U_m.T @ U_m                              # mass matrix
K = U_m.T @ (L @ U_m)                        # stiffness / Laplacian matrix

# Right-hand side for ODE:  M ȧ = α K a  -> ȧ = α M⁻¹K a
A = alpha * la.solve(M, K)                  # pre-compute once

# Initial reduced coefficients -------------------------------------
a0 = U_m.T @ u0(x)

# Integrate reduced ODE --------------------------------------------
def rhs(t, a):
    return A @ a

sol = itg.solve_ivp(rhs,
                    t_span=(0.0, t_end),
                    y0=a0,
                    t_eval=times_snap,      # same output times as snapshots
                    method=ode_method,
                    rtol=1e-8, atol=1e-10)

a_t = sol.y                                 # shape (m, nt_snap)
S_rom = U_m @ a_t                           # reconstructed field

# Error diagnostics -------------------------------------------------
err_rms = la.norm(S - S_rom) / np.sqrt(nx * nt_snap)
print(f"Global RMS error (ROM vs exact): {err_rms:.4e}")

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
if make_plots:
    # Pick middle snapshot for comparison
    j = nt_snap // 2
    plt.figure()
    plt.plot(x, S[:, j], label="Exact")
    plt.plot(x, S_rom[:, j], "--", label=f"POD-ROM (m={m})")
    plt.xlabel("x"); plt.ylabel("u")
    plt.title(f"Snapshot at t = {times_snap[j]:.3f}")
    plt.legend(); plt.tight_layout()

    # Singular values
    plt.figure()
    plt.semilogy(np.arange(1, len(s)+1), s, "o-")
    plt.xlabel("Mode"); plt.ylabel("σ")
    plt.title("Singular value spectrum"); plt.grid(True); plt.tight_layout()

    # Coefficient trajectories
    plt.figure()
    for k in range(m):
        plt.plot(sol.t, a_t[k, :], label=f"a[{k}]")
    plt.xlabel("t"); plt.ylabel("Coefficient value")
    plt.title("Reduced ODE coefficients"); plt.legend(); plt.tight_layout()

    plt.show()