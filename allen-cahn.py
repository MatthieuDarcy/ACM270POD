#%%
#!/usr/bin/env python3
"""
allen_cahn_fd.py
----------------
1-D Allen–Cahn equation with periodic BCs, solved by
 • second-order finite differences in space
 • SciPy solve_ivp (adaptive RK45 by default) in time
"""

# ───────────────────────── Imports ──────────────────────────
import numpy as np
import scipy.sparse as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#%%
#autoreload modules when code is run
%load_ext autoreload
%autoreload 2

#%%
# ───────────────────────── Parameters ───────────────────────
eps2          = 1e-3   # ε²  (interface width parameter)
nx            = 200     # grid points (periodic, incl. x=0)
dx            = 1.0 / nx
t_end         = 1.0     # final time
n_steps       = 200     # number of output times
method        = "RK45"  # "Radau" or "BDF" are safer for stiff ε≪1
rtol, atol    = 1e-6, 1e-9
plot_interval = 100     # None → skip live plot
# ────────────────────────────────────────────────────────────

# (1) Grid (periodic: x in [0,1-dx])
x = np.linspace(0.0, 1.0 - dx, nx)

#%%

# (2) Discrete Laplacian L (periodic tri-diagonal + wrap)
main_diag = -2.0 * np.ones(nx)
off_diag  =  1.0 * np.ones(nx - 1)
L = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1],
             shape=(nx, nx), format='csc') / dx**2
# periodic wrap-around terms
L = L.tolil()
L[0,  -1] = 1.0 / dx**2
L[-1, 0]  = 1.0 / dx**2
L = L.tocsc()

#%%
# (3) Initial condition (small perturbation of ±1 phases)
u0 = np.cos(2 * np.pi * x) + 0.5*np.cos(4 * np.pi * x)  + np.sin(2 * np.pi * x) + 0.3*np.sin(4 * np.pi * x)#+ 0.01 * np.random.randn(nx)

# (4) RHS for solve_ivp: u' = ε² L u + u − u³

def f_non_linear(y):
    return -y**3

A = eps2 * L  + np.eye(L.shape[0])
def f_linear(y, A = A):
    y = np.squeeze(y)
    return A@y

def rhs(t, y):
    return f_linear(y)  + f_non_linear(y)
# (5) Integration times
t_eval = np.linspace(0.0, t_end, n_steps + 1)
t_span =  (0.0, t_end)

# (6) Integrate
sol = integrate.solve_ivp(rhs, t_span, u0,
                          t_eval=t_eval, method=method,
                          rtol=rtol, atol=atol)

u_sol = sol.y                # shape (nx, n_steps+1)

# (7) Plot initial & final profiles
plt.figure(figsize=(6,4))
plt.plot(x, u0,   "--"   ,          label="Initial condition")
plt.plot(x, u_sol[:, -1], label=f"RK45")
plt.xlabel("x"); plt.ylabel("u")
plt.title("1-D Allen–Cahn t = {t_end}")
plt.legend(); plt.tight_layout()
plt.show()
#%% Use the Emprical interpolation method

# First compute the usual POD basis

from utils import compute_pod_basis
from utils_deim import compute_deim_basis

# Get the V matrix via SVD
rank = 5
V, s = compute_pod_basis(u_sol, rank= rank)
Nsnapshots = f_non_linear(u_sol)
m =5
U_deim, p = compute_deim_basis(Nsnapshots, m)

#%% Solve the POD system

from utils_deim import solve_pod_deim_system
sol_pod = solve_pod_deim_system(A, V, U_deim, p, f_non_linear, u0,
                          t_span, t_eval=t_eval,
                          method="RK45", rtol=1e-8, atol=1e-10)

u_sol_pod = V@sol_pod.y

rel_err = np.linalg.norm(u_sol_pod[:, -1] - u_sol[:, -1]) / np.linalg.norm( u_sol[:, -1])
print(f"Relative L2 error at t = {t_end:.2f}  →  {rel_err:.3e}")

# %%
plt.figure(figsize=(6,4))
plt.plot(x, u0,   "--"   ,          label="Initial condition")
plt.plot(x, u_sol[:, -1], label=f"RK45")
plt.plot(x, u_sol_pod[:, -1], label=f"POD {rank}, {m}, Error {rel_err:.3e}")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"1-D Allen–Cahn t = {t_end}")
plt.legend(); plt.tight_layout()
plt.show()

#%% Starting from the last solution

u0 = u_sol[:, -1]
t_span_future = (t_end, t_end + 1.0)
t_eval_future = t_eval+1
sol = integrate.solve_ivp(rhs, t_span_future, u0,
                          t_eval=t_eval_future, method=method,
                          rtol=rtol, atol=atol)
u_sol = sol.y                # shape (nx, n_steps+1)

sol_pod = solve_pod_deim_system(A, V, U_deim, p, f_non_linear, u0,
                          t_span_future, t_eval=t_eval_future,
                          method="RK45", rtol=1e-8, atol=1e-10)

u_sol_pod = V@sol_pod.y

rel_err = np.linalg.norm(u_sol_pod[:, -1] - u_sol[:, -1]) / np.linalg.norm( u_sol[:, -1])
print(f"Relative L2 error at t = {t_end:.2f}  →  {rel_err:.2e}")

# %%

plt.figure(figsize=(6,4))
plt.plot(x, u0,   "--"   ,          label="Initial condition")
plt.plot(x, u_sol[:, -1], label=f"RK45")
plt.plot(x, u_sol_pod[:, -1], label=f"POD {rank}, {m}, Error {rel_err:.3e}")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"1-D Allen–Cahn t = {t_end + 1.0}")
plt.legend(); plt.tight_layout()
plt.show()


# %% Starting from a different initial condtion

def initial_condition(x):
    fourier = np.cos(2 * np.pi * x) + 0.5*np.cos(4 * np.pi * x)  + np.sin(2 * np.pi * x) - 0.3*np.sin(4 * np.pi * x) +  - 0.1*np.sin(16* np.pi * x)#1.25*np.cos(2 * np.pi * x) - 0.5*np.cos(4 * np.pi * x)  + 0.1*np.sin(2 * np.pi * x) -0.5*np.sin(4 * np.pi * x) #+ 0.5*np.sin(8*np.pi*x)
    return fourier
u0 = initial_condition(x)#+ 0.01 * np.random.randn(nx) + 

sol = integrate.solve_ivp(rhs, t_span, u0,
                          t_eval=t_eval, method=method,
                          rtol=rtol, atol=atol)
u_sol = sol.y                # shape (nx, n_steps+1)

sol_pod = solve_pod_deim_system(A, V, U_deim, p, f_non_linear, u0,
                          t_span, t_eval=t_eval,
                          method="RK45", rtol=1e-8, atol=1e-10)

u_sol_pod = V@sol_pod.y

rel_err = np.linalg.norm(u_sol_pod[:, -1] - u_sol[:, -1]) / np.linalg.norm( u_sol[:, -1])
print(f"Relative L2 error at t = {t_end:.2f}  →  {rel_err:.3e}")

# %%

plt.figure(figsize=(6,4))
plt.plot(x, u0,   "--"   ,          label="Initial condition")
plt.plot(x, u_sol[:, -1], label=f"RK45")
plt.plot(x, u_sol_pod[:, -1], label=f"POD {rank}, {m}, Error {rel_err:.2e}")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"1-D Allen–Cahn t = {t_end}")
plt.legend(); plt.tight_layout()
plt.show()

# %%
