import numpy as np
import scipy.linalg as la
import scipy.integrate as itg

# -------------------------------------------------------------
# 1.  Build DEIM basis  U  and interpolation indices  p
# -------------------------------------------------------------
def compute_deim_basis(N_snap, r):
    """
    Parameters
    ----------
    N_snap : (n, k) ndarray
        Snapshot matrix of the nonlinear term (each column = N(u) at a time).
    r : int
        Desired DEIM rank (# interpolation points).

    Returns
    -------
    U : (n, r) ndarray
        Column-orthonormal DEIM basis (left singular vectors).
    p : (r,) ndarray of int
        Greedy-selected interpolation indices (0-based).
    """
    U, _, _ = la.svd(N_snap, full_matrices=False)
    U = U[:, :r].copy()                 # keep first r modes

    # Classic DEIM (Chaturantabut & Sorensen 2010) greedy selection
    p = np.zeros(r, dtype=int)
    # 1st index = location of largest magnitude in first mode
    p[0] = np.argmax(np.abs(U[:, 0]))

    P = np.zeros((N_snap.shape[0], 1))  
    P[p[0], 0] = 1.0
    #print(P)
    #print(P, P.shape)
    for j in range(1, r):
        u = U[:, j]
        # Solve least-squares for coefficients on current basis
        c = la.lstsq(U[:, :j].T @ P, U[:, :j].T @ u)[0]
        r_vec = u - U[:, :j] @ c        # residual


        #new_idx = np.argmax(np.abs(r_vec))
        # We use a mask to exclude already selected indices 
        mask = np.ones_like(r_vec, dtype=bool)
        mask[p[:j]] = False           # exclude already-selected indices
        new_idx = np.argmax(np.abs(r_vec) * mask)

        #print(new_idx)
        p[j] = new_idx
        #print(new_idx)
        # augment P
        P = np.hstack([P, np.eye(N_snap.shape[0])[:, [p[j]]]])
        #print(P)

    return U, p


# -------------------------------------------------------------
# 2.  Integrate POD-DEIM reduced system
# -------------------------------------------------------------
def solve_pod_deim_system(A, V, U, p, N_func, x0,
                          t_span, t_eval=None,
                          method="RK45", rtol=1e-8, atol=1e-10):
    """
    Parameters
    ----------
    A : (n,n) ndarray or LinearOperator
        Linear part of full system  du/dt = A u + N(u).
    V : (n,m) ndarray
        POD basis for the state.
    U : (n,r) ndarray
        DEIM basis for the nonlinear snapshots.
    p : (r,) ndarray of int
        Interpolation indices chosen by DEIM.
    N_func : callable(u) -> (n,) ndarray
        Python function that returns N(u) at any full state u.
    x0 : (n,) ndarray
        Initial full-order state.
    t_span, t_eval, method, rtol, atol : as in solve_ivp.

    Returns
    -------
    sol : OdeResult
        Same fields as solve_pod_system, plus full reconstruction.
    """
    # Pre-compute reduced linear matrix
    A_r = V.T @ (A @ V)                    # (m,m)

    # Pre-compute DEIM projection matrices
    #   P  – selection matrix (n×r with unit columns at indices p)
    #   invPU = (Pᵀ U)^{-1}
    n = V.shape[0]
    r = len(p)
    P = np.zeros((n, r))
    P[p, np.arange(r)] = 1.0

    invPU = la.inv(P.T @ U)                # (r,r)

    # Reduced initial state
    y0 = V.T @ x0

    W = P.T@V

    # RHS in reduced coordinates
    def rhs(t, y):
        y = np.squeeze(y)
        u = V @ y                          # lift to full space
        #N_u = N_func(u)                    # nonlinear term (n,)
        N_u= N_func(W@y)                    # avoiding the depdence on n
        # DEIM approximation of Vᵀ N(u)
        #beta = invPU @ (P.T @ N_u)         # coefficients (r,)
        beta = invPU @ (N_u)         # coefficients (r,)
        g_y = V.T @ (U @ beta)             # projected nonlinear term (m,)
        return A_r @ y + g_y

    sol = itg.solve_ivp(rhs, t_span, y0,
                        t_eval=t_eval,
                        method=method, rtol=rtol, atol=atol)
    sol.u_full = V @ sol.y                 # reconstruct snapshots
    return sol