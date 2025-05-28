#%%
#!/usr/bin/env python3
"""
heat_fd_cn.py
Solve the 1-D heat equation on [0,1] with homogeneous Dirichlet BCs
using second-order finite differences in space and the Crank–Nicolson
time-marching scheme.

Author: (your name) – 2025-05-19
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp
import scipy

#%%
#autoreload modules when code is run
%load_ext autoreload
%autoreload 2

#%%
# ────────────────────────────── Parameters ──────────────────────────────
alpha   = 0.01     # thermal diffusivity
nx      = 200      # number of interior grid points (total = nx+2 incl. BCs)
dx      = 1.0/(nx+1)
dt      = 2e-4     # time step size   (Crank–Nicolson is unconditionally
                   #                   stable, but accuracy still matters)
t_end   = 1.0      # final time
plot_interval = 100    # how many steps between plots (set None to skip)
# ────────────────────────────────────────────────────────────────────────

# (1) Grid (interior points only; BC nodes are implicit)
x = np.linspace(dx, 1.0-dx, nx)

# (3) Build the discrete Laplacian L (tri-diagonal, Dirichlet BCs)
main_diag = -2.0*np.ones(nx)
off_diag  = 1.0*np.ones(nx-1)
L = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc') / dx**2


#%% Using solve_ivp

def rhs(t, y, L= L):
    """Right‑hand side for the semi‑discrete heat equation:  y' = α L y."""
    return alpha * (L @ y)

# Integration times
n_steps = 100
t_eval = np.linspace(0.0, t_end, n_steps + 1)
# Re-create the initial condition vector (leaving the CN result untouched)
u0_ivp = np.sin(np.pi*x) + 0.5*np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.3*np.sin(8*np.pi*x)
# Integrate with an explicit adaptive Runge–Kutta method
sol_ivp = scipy.integrate.solve_ivp(rhs,
                              (0.0, t_end),
                              u0_ivp,
                              t_eval=t_eval,
                              method="RK45",
                              rtol=1e-6,
                              atol=1e-9)

u_ivp = sol_ivp.y
u_ivp_final = u_ivp[:, -1]

# Plot comparison
plt.figure()
plt.plot(x, u0_ivp, "--", label = "Initial condition")
plt.plot(x, u_ivp_final, label="solve_ivp (RK45)")
plt.xlabel("x"); plt.ylabel("u")
plt.title("Heat equation: solve_ivp vs exact")
plt.legend(); plt.tight_layout()
plt.show()




# %%

from utils import compute_pod_basis, solve_pod_system

# Get the V matrix via SVD
rank = 5
V, s = compute_pod_basis(u_ivp, rank= rank)

#%%
u0_ivp_recon = V@V.T@u0_ivp

plt.figure()

plt.plot(x, u0_ivp, label = "Initial condition")
plt.plot(x, u0_ivp_recon, label = "Recontructred initial condition")
plt.legend()
plt.show()

#%%
A = alpha * L.toarray()

sol_pod, sol_recon = solve_pod_system(A, V, u0_ivp, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)

small_rank = 3
sol_pod_small, sol_recon_small = solve_pod_system(A, V[:,:small_rank], u0_ivp, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)

rel_err_pod = np.linalg.norm(u_ivp[:, -1] - sol_recon[:, -1]) / np.linalg.norm(u_ivp[:, -1])
print(f"POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod:.3e}")

rel_err_pod_small = np.linalg.norm(u_ivp[:, -1] - sol_recon_small[:, -1]) / np.linalg.norm(u_ivp[:, -1])
print(f"Small POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod_small:.3e}")
# sol_pod, sol_recon = solve_pod_system(A, V, u0_ivp, t_span = (0.0, t_end),t_eval=t_eval,
#                      method="RK45", rtol=1e-8, atol=1e-10)


# rel_err = np.linalg.norm(sol_recon[:, -1] - u_ivp[:, -1]) / np.linalg.norm( u_ivp[:, -1])
# print(f"Relative L2 error at t = {t_end:.2f}  →  {rel_err:.3e}")


#%%
# Plot comparison
plt.figure()
plt.plot(x,u0_ivp, "--", label = "initial condition")
plt.plot(x, u_ivp_final, label="solve_ivp (RK45)")
plt.plot(x, sol_recon[:, -1], label=f"POD solution, rank {rank}, {rel_err_pod:.2e}")
plt.plot(x, sol_recon_small[:, -1], label=f"POD solution rank {small_rank}, {rel_err_pod_small:.2e}")
plt.xlabel("x"); plt.ylabel("u")
plt.title("Heat equation")
plt.legend(); plt.tight_layout()
#plt.show()

#%% Initial condition = solution at time 1.0

u0_ivp = u_ivp_final#-5*np.sin(np.pi*x) - 0.5*np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.2*np.sin(8*np.pi*x) -1*np.sin(16*np.pi*x)
# Integrate with an explicit adaptive Runge–Kutta method
sol_ivp = scipy.integrate.solve_ivp(rhs,
                              (0.0, t_end),
                              u0_ivp,
                              t_eval=t_eval,
                              method="RK45",
                              rtol=1e-6,
                              atol=1e-9)

u_ivp = sol_ivp.y
u_ivp_final = u_ivp[:, -1]

# %%

sol_pod, sol_recon = solve_pod_system(A, V, u0_ivp, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)

small_rank = 3
sol_pod_small, sol_recon_small = solve_pod_system(A, V[:,:small_rank], u0_ivp, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)

rel_err_pod = np.linalg.norm(u_ivp[:, -1] - sol_recon[:, -1]) / np.linalg.norm(u_ivp[:, -1])
print(f"POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod:.3e}")

rel_err_pod_small = np.linalg.norm(u_ivp[:, -1] - sol_recon_small[:, -1]) / np.linalg.norm(u_ivp[:, -1])
print(f"Small POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod_small:.3e}")
# %%
# Plot comparison
plt.figure()
plt.plot(x,u0_ivp, "--", label = "initial condition")
plt.plot(x, u_ivp_final, label=f"solve_ivp (RK45)")
plt.plot(x, sol_recon[:, -1], label=f"POD solution rank {rank}, {rel_err_pod:.2e}")
plt.plot(x, sol_recon_small[:, -1], label=f"POD solution rank {small_rank}, {rel_err_pod_small:.2e}")

plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Heat equation at time {t_end + 1.0}")
plt.legend(); plt.tight_layout()
#%% New initial condition

u0_ivp = -5*np.sin(np.pi*x) - 0.5*np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.2*np.sin(8*np.pi*x) -1*np.sin(16*np.pi*x)
# Integrate with an explicit adaptive Runge–Kutta method
sol_ivp = scipy.integrate.solve_ivp(rhs,
                              (0.0, t_end),
                              u0_ivp,
                              t_eval=t_eval,
                              method="RK45",
                              rtol=1e-6,
                              atol=1e-9)

u_ivp = sol_ivp.y
u_ivp_final = u_ivp[:, -1]




# %%

sol_pod, sol_recon = solve_pod_system(A, V, u0_ivp, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)

small_rank = 3
sol_pod_small, sol_recon_small = solve_pod_system(A, V[:,:small_rank], u0_ivp, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)

rel_err_pod = np.linalg.norm(u_ivp[:, -1] - sol_recon[:, -1]) / np.linalg.norm(u_ivp[:, -1])
print(f"POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod:.3e}")

rel_err_pod_small = np.linalg.norm(u_ivp[:, -1] - sol_recon_small[:, -1]) / np.linalg.norm(u_ivp[:, -1])
print(f"POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod_small:.3e}")
# %%


# Plot comparison
plt.figure()
plt.plot(x,u0_ivp, "--", label = "initial condition")
plt.plot(x, u_ivp_final, label=f"solve_ivp (RK45)")
plt.plot(x, sol_recon[:, -1], label=f"POD solution ({rank}), {rel_err_pod:.2e}")
plt.plot(x, sol_recon_small[:, -1], label=f"POD solution {small_rank}, {rel_err_pod_small:.2e}")

plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Heat equation with a new initial condition")
plt.legend(); plt.tight_layout()
# %%
