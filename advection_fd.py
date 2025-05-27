#%%
#!/usr/bin/env python3
"""
advection_fd.py
---------------
Finite–difference (FD) solver for the 1‑D linear advection equation

    u_t + c u_x = 0,     x ∈ [0, 1),  t ≥ 0,

with periodic boundary conditions and an arbitrary smooth initial
condition.  Space is discretised with a first‑order upwind stencil,
and time integration is performed by SciPy’s `solve_ivp`.

The script:

1. builds the semi‑discrete operator  A = −c D₁  (sparse);
2. wraps it in a RHS function for `solve_ivp`;
3. integrates the system up to `t_end`;
4. compares the numerical solution against the exact solution
   (a pure translation of the initial condition); and
5. plots both.

Author: ChatGPT (May 2025)
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#%%
# ---------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------
c        = 1.0       # advection speed  (positive ⇒ shifts right)
nx       = 2048        # number of grid points
dx       = 1.0 / nx
CFL      = 0.8         # Courant number  (≤1 for explicit upwind)
#dt       = CFL * dx / c
t_end    = 0.25      # final time (one full wrap if c=1)
method   = "RK45"      # integrator in solve_ivp
rtol     = 1e-9
atol     = 1e-11
#%%
# ---------------------------------------------------------------------
# Spatial grid and initial condition
# ---------------------------------------------------------------------
x = np.linspace(0.0, 1.0, nx, endpoint=False)

def initial_condition(x):
    """Example IC: sharp Gaussian pulse + smooth sine."""
    return np.sin(2 * np.pi * x) + np.exp(-300*(x-0.25)**2)

u0 = initial_condition(x)

#%%
# ---------------------------------------------------------------------
# Semi‑discrete operator  A = −c D₁  (first‑order upwind, periodic)
# ---------------------------------------------------------------------
ones = np.ones(nx)
# Upwind derivative: (u_i - u_{i-1}) / dx
D1 = sp.diags([-ones,  ones], [0, -1], shape=(nx, nx), format="csc") / dx
D1 = D1.tolil()
D1[0, -1] = 1.0 / dx     # periodic wrap–around term
D1 = D1.tocsc()

A = (c) * D1            # generator of ODE  du/dt = A u

#%%
# ---------------------------------------------------------------------
# Right‑hand side for solve_ivp
# ---------------------------------------------------------------------
def rhs(t, u):
    """Compute du/dt = A u."""
    return A @ u

# Integration times (uniform for plotting convenience)
steps = 100#int(np.ceil(t_end / dt))
t_eval = np.linspace(0.0, t_end, steps + 1)
#%%
# ---------------------------------------------------------------------
# Integrate with solve_ivp
# ---------------------------------------------------------------------
sol = integrate.solve_ivp(rhs, (0.0, t_end), u0,
                          t_eval=t_eval, method=method,
                          rtol=rtol, atol=atol)

u_num = sol.y                 # shape (nx, len(t_eval))

#%%
# ---------------------------------------------------------------------
# Exact solution = shift of initial condition
# ---------------------------------------------------------------------
u_exact_final = initial_condition((x - c * t_end))

# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(x, u0, label="t = 0")
plt.plot(x, u_num[:, -1], "--",  label="numerical (t = t_end)")
plt.plot(x, u_exact_final, ":",  label="exact shift")
plt.xlabel("x")
plt.ylabel("u")
plt.title("1‑D linear advection: FD + solve_ivp")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Print error metrics
# ---------------------------------------------------------------------
rel_err = np.linalg.norm(u_num[:, -1] - u_exact_final) / np.linalg.norm(u_exact_final)
print(f"Relative L2 error at t = {t_end:.2f}  →  {rel_err:.3e}")
#%%

from utils import compute_pod_basis, solve_pod_system

# Get the V matrix via SVD
rank = 10
V, s = compute_pod_basis(u_num, rank= rank)

#%%
u0_ivp_recon = V@V.T@u0

plt.figure()
plt.plot(x, u0, label = "Initial condition")
plt.plot(x, u0_ivp_recon, label = "Recontructred initial condition")
plt.legend()
plt.show()



#%%
sol_pod, sol_recon = solve_pod_system(A, V, u0, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)



rel_err_pod = np.linalg.norm(u_num[:, -1] - sol_recon[:, -1]) / np.linalg.norm(u_num[:, -1])
print(f"Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod:.3e}")


#%%
# Plot comparison
plt.figure()
plt.plot(x,u0, "--", label = "initial condition")
plt.plot(x, u_exact_final, '--', label="Exact solution")
plt.plot(x, u_num[:, -1], label=f"solve_ivp (RK45), {rel_err:.2e}")
plt.plot(x, sol_recon[:, -1], label=f"POD solution ({rank}), {rel_err_pod:.2e}")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Advection equation on the training data")
plt.legend(); plt.tight_layout()
#plt.show()



#%% new initial condition
def initial_condition(x):
    """Example IC: sharp Gaussian pulse + smooth sine."""
    return np.sin(2 * np.pi * x) + np.exp(-300*(x-0.25)**2)

u0 = initial_condition(x)               # shape (nx, len(t_eval))
t_end    = 0.25
steps = 100#int(np.ceil(t_end / dt))
t_eval = np.linspace(0.0, t_end, steps + 1)

#%%
sol = integrate.solve_ivp(rhs, (0.0, t_end), u0,
                          t_eval=t_eval, method=method,
                          rtol=rtol, atol=atol)
u_num = sol.y  
u_exact_final = initial_condition((x - c * t_end))

#%%

sol_pod, sol_recon = solve_pod_system(A, V, u0, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)

small_rank = 3
sol_pod_small, sol_recon_small = solve_pod_system(A, V[:,:small_rank], u0, t_span = (0.0, t_end),t_eval=t_eval,
                     method="RK45", rtol=1e-8, atol=1e-10)



rel_err = np.linalg.norm(u_num[:, -1] - u_exact_final) / np.linalg.norm(u_exact_final)
print(f"Relative L2 error at t = {t_end:.2f}  →  {rel_err:.3e}")

rel_err_pod = np.linalg.norm(u_num[:, -1] - sol_recon[:, -1]) / np.linalg.norm(u_num[:, -1])
print(f"POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod:.3e}")

rel_err_pod_small = np.linalg.norm(u_num[:, -1] - sol_recon_small[:, -1]) / np.linalg.norm(u_num[:, -1])
print(f"POD Relative L2 error at t = {t_end:.2f}  →  {rel_err_pod_small:.3e}")

#%%
# Plot comparison
plt.figure()
plt.plot(x,u0, "--", label = "initial condition")
plt.plot(x, u_exact_final, '--', label="Exact solution")
plt.plot(x, u_num[:, -1], label=f"solve_ivp (RK45), {rel_err:.2e}")
plt.plot(x, sol_recon[:, -1], label=f"POD solution ({rank}), {rel_err_pod:.2e}")
plt.plot(x, sol_recon_small[:, -1], label=f"POD solution ({small_rank}), {rel_err_pod_small:.2e}")

plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Adection equation with a new initial condition")
plt.legend(); plt.tight_layout()
# %%
