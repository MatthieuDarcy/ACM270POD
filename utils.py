

import numpy as np
import scipy.linalg as la
import scipy.integrate as itg

# ------------------------------------------------------------------
# 1.  Compute a POD basis from a snapshot matrix
# ------------------------------------------------------------------
def compute_pod_basis(Y, rank=None, energy_thresh=None):
    """
    Parameters
    ----------
    Y : (n, k) ndarray
        Snapshot matrix.  Each column is a full-order state vector.
    rank : int, optional
        Number of POD modes to keep (m).  If None, choose rank large
        enough to satisfy `energy_thresh` (or keep all if that is None).
    energy_thresh : float in (0,1), optional
        Keep the smallest rank whose cumulative energy >= this value.
        Ignored if `rank` is supplied.

    Returns
    -------
    V : (n, m) ndarray
        Column-orthonormal POD basis (left singular vectors).
    s : (m,) ndarray
        Singular values associated with the kept modes.
    """
    # Thin SVD  –  Y = U Σ Vᵀ         (U = left singular vectors)
    U, s, _ = la.svd(Y, full_matrices=False)

    if rank is None:
        if energy_thresh is None:
            rank = len(s)            # keep everything
        else:                        # pick smallest m reaching thresh
            cum_energy = np.cumsum(s**2) / np.sum(s**2)
            rank = np.searchsorted(cum_energy, energy_thresh) + 1
    V = U[:, :rank].copy()
    return V, s[:rank]


# ------------------------------------------------------------------
# 2.  Integrate the reduced POD ODE   ẏ = Vᵀ A V y
# ------------------------------------------------------------------
def solve_pod_system(A, V, x0, t_span, t_eval=None,
                     method="RK45", rtol=1e-8, atol=1e-10):
    """
    Parameters
    ----------
    A : (n, n) array_like or LinearOperator
        Full-order system matrix (from the semi-discretised PDE).
    V : (n, m) ndarray
        POD basis (columns orthonormal).
    x0 : (n,) ndarray
        Full-order initial condition in original coordinates.
    t_span : (t0, tf) tuple
        Integration interval for the reduced system.
    t_eval : 1-D array, optional
        Times at which to save the solution (passed to solve_ivp).
    method, rtol, atol : passed straight to `scipy.integrate.solve_ivp`.

    Returns
    -------
    sol : OdeResult
        SciPy object containing times `sol.t`, reduced states `sol.y`
        (shape m×len(t)), and — for convenience — the reconstructed
        high-dimensional solution in `sol.u_full` (n×len(t)).
    """
    # Reduced matrices
    A_r = V.T @ (A @ V)        # (m, m)
    y0  = V.T @ x0             # reduced initial coefficients

    # Right–hand side  ẏ = A_r y
    def rhs(t, y):             # A_r is constant ⇒ affine ODE
        return A_r @ y

    sol = itg.solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                        method=method, rtol=rtol, atol=atol)

    # Reconstruct full-state snapshots on the fly
    #sol.u_full = V @ sol.y     # shape (n, len(t))
    return sol, V @ sol.y