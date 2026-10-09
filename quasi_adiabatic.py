"""Stand-alone quasi-adiabatic (QA) transonic wind.

Physics
-------
Steady spherical wind in s = ln r with Y = dln(rho)/ds, Z = dln(T)/ds and
v = Mdot/(4 pi r^2 rho):

    momentum:   (v^2 - f2) Y = f1 Z + G M/r - 2 v^2
    energy:     dL/ds = Mdot (f1 Y - f3 Z)
    diffusion:  L = -A Z
    mass:       dM/ds = 4 pi r^3 rho

Integrating L directly is exponentially stiff in a trapped flow. When the sonic
point is optically thick (tau v/c >> 1) diffusion is slow compared with
advection, so the flow is nearly adiabatic. Write Z = (f1/f3) Y + Q and expand
in the small heat-exchange term Q:

    L0 = -A (f1/f3) Y                        (leading-order luminosity)
    Q  = -(1/(Mdot f3)) dL0/ds               (first-order non-adiabaticity)
    (v^2 - cs2_ad) Y = f1 Q + G M/r - 2 v^2,  cs2_ad = f2 + f1^2/f3

L is now algebraic, so the stiff mode is gone. The critical point is the
*adiabatic* sonic point v_c^2 = cs2_ad with regularity G M_c/r_c = 2 cs2 - f1 Q_c,
which fixes r_c and hence Mdot = 4 pi r_c^2 rho_c v_c for given (rho_c, T_c, M_c).

The expansion parameter  eta = |Q / ((f1/f3) Y)|  grows outward; the residual
error is O(eta^2). Integration stops at eta = eta_stop, where the bridge hands
over to the exact FLD equations.

Run ``python quasi_adiabatic.py`` for an example.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import helper_func as hf

H_STEP = 2e-4   # ln r step used to differentiate L0 when computing Q


# ---------------------------------------------------------------------------
# Local equations
# ---------------------------------------------------------------------------
def adiabatic_Y(s, lnrho, lnT, M, Mdot, Y_c, tol=1e-9):
    """Q = 0 density slope (GM/r - 2v^2)/(v^2 - cs2); uses Y_c at the 0/0 point."""
    r, rho, T = np.exp(s), np.exp(lnrho), np.exp(lnT)
    cs2 = hf.get_cs2(rho, T)
    v2 = (Mdot/(4*np.pi*r**2*rho))**2
    D = v2 - cs2
    if abs(D) <= tol*(v2 + cs2):
        return Y_c
    return (hf.G*M/r - 2*v2)/D


def get_Q(s, lnrho, lnT, M, Mdot, s_c, Y_c, kappa_const, h=H_STEP, n=4):
    """Q = -(1/(Mdot f3)) dL0/ds, with dL0/ds taken along the adiabatic flow.

    Using the adiabatic flow for the derivative is an O(Q^2) error. Within 3h of
    the critical point a one-sided stencil (pointing away from it) is used so
    the stencil never straddles the 0/0 point.
    """
    def march(ds):
        # n Euler sub-steps along the adiabatic flow
        ss, uu, tt, MM = s, lnrho, lnT, M
        for _ in range(n):
            rho, T = np.exp(uu), np.exp(tt)
            f1, _, f3 = hf.get_f(rho, T)
            Y = adiabatic_Y(ss, uu, tt, MM, Mdot, Y_c)
            MM = MM + (ds/n)*4*np.pi*rho*np.exp(ss)**3
            uu, tt, ss = uu + (ds/n)*Y, tt + (ds/n)*(f1/f3)*Y, ss + ds/n
        return ss, uu, tt, MM

    def L0(ss, uu, tt, MM):
        rho, T = np.exp(uu), np.exp(tt)
        f1, _, f3 = hf.get_f(rho, T)
        Y = adiabatic_Y(ss, uu, tt, MM, Mdot, Y_c)
        return -hf.get_A(np.exp(ss), rho, T, kappa_const)*(f1/f3)*Y

    if abs(s - s_c) < 3*h:
        sgn = 1.0 if s >= s_c else -1.0
        dL0ds = sgn*(-3*L0(s, lnrho, lnT, M) + 4*L0(*march(sgn*h))
                     - L0(*march(2*sgn*h)))/(2*h)
    else:
        dL0ds = (L0(*march(h)) - L0(*march(-h)))/(2*h)
    f3 = hf.get_f(np.exp(lnrho), np.exp(lnT))[2]
    return -dL0ds/(Mdot*f3)


def rhs(s, y, Mdot, s_c, Y_c, Q_c, kappa_const):
    """d/ds of y = [ln rho, ln T, M] for the first-order QA system."""
    lnrho, lnT, M = y
    r, rho, T = np.exp(s), np.exp(lnrho), np.exp(lnT)
    f1, f2, f3 = hf.get_f(rho, T)
    cs2 = f2 + f1*f1/f3
    v2 = (Mdot/(4*np.pi*r**2*rho))**2

    # The finite-difference Q is ill-conditioned inside its one-sided stencil
    # (|s - s_c| < 3h). The physical Q is smooth there, so hold Q_c in that core
    # and blend (C1 smoothstep) back to the raw estimate by |s - s_c| = 5h.
    Q = get_Q(s, lnrho, lnT, M, Mdot, s_c, Y_c, kappa_const)
    dist, core, band = abs(s - s_c), 3*H_STEP, 5*H_STEP
    if dist < band:
        w = np.clip((dist - core)/(band - core), 0.0, 1.0)
        w = w*w*(3 - 2*w)
        Q = (1 - w)*Q_c + w*Q

    N = f1*Q + hf.G*M/r - 2*v2
    D = v2 - cs2
    Y = Y_c if abs(D) <= 1e-9*(v2 + cs2) else N/D
    return np.array([Y, (f1/f3)*Y + Q, 4*np.pi*rho*r**3])


def get_eta(s, y, Mdot, s_c, Y_c, Q_c, kappa_const):
    """Expansion parameter eta = |Q / ((f1/f3) Y)| = |L - L0|/L0."""
    Y, Z, _ = rhs(s, y, Mdot, s_c, Y_c, Q_c, kappa_const)
    f1, _, f3 = hf.get_f(np.exp(y[0]), np.exp(y[1]))
    return abs((Z - (f1/f3)*Y)/((f1/f3)*Y))


# ---------------------------------------------------------------------------
# Critical (sonic) point
# ---------------------------------------------------------------------------
def critical_point(rho_c, T_c, M_c, Q_c=0.0):
    """r_c, v_c, Mdot from v_c^2 = cs2 and G M_c/r_c = 2 cs2 - f1 Q_c."""
    f1, f2, f3 = hf.get_f(rho_c, T_c)
    cs2 = f2 + f1*f1/f3
    denom = 2*cs2 - f1*Q_c
    if denom <= 0:
        raise ValueError("2 cs^2 - f1 Q_c <= 0: sonic point is leaky; QA does not apply")
    r_c = hf.G*M_c/denom
    v_c = np.sqrt(cs2)
    return r_c, v_c, 4*np.pi*rho_c*r_c**2*v_c


def critical_slope(r_c, rho_c, T_c, M_c, Mdot, Q_c, d=1e-6):
    """Y_c = dln(rho)/ds at the sonic point from L'Hopital: Y D' = N'.

    N'(Y) and D'(Y) are centred differences along the trial slope. Y D' - N' is
    quadratic in Y; of its two roots keep the one with D' > 0 (outward
    subsonic -> supersonic crossing).
    """
    s_c, u_c, t_c = np.log(r_c), np.log(rho_c), np.log(T_c)
    f1c, _, f3c = hf.get_f(rho_c, T_c)
    dMds = 4*np.pi*rho_c*r_c**3

    def ND(s, u, t, M):
        r, rho, T = np.exp(s), np.exp(u), np.exp(t)
        f1, f2, f3 = hf.get_f(rho, T)
        v2 = (Mdot/(4*np.pi*r**2*rho))**2
        return f1*Q_c + hf.G*M/r - 2*v2, v2 - (f2 + f1*f1/f3)

    def derivs(Y):
        Z = (f1c/f3c)*Y + Q_c
        Np, Dp = ND(s_c + d, u_c + Y*d, t_c + Z*d, M_c + dMds*d)
        Nm, Dm = ND(s_c - d, u_c - Y*d, t_c - Z*d, M_c - dMds*d)
        return (Np - Nm)/(2*d), (Dp - Dm)/(2*d)

    def F(Y):
        Np, Dp = derivs(Y)
        return Y*Dp - Np

    fm, f0, fp = F(-1.0), F(0.0), F(1.0)
    roots = np.roots([0.5*(fm - 2*f0 + fp), 0.5*(fp - fm), f0])
    real = [z.real for z in roots if abs(z.imag) <= 1e-8*max(1.0, abs(z.real))]
    wind = [Y for Y in real if derivs(Y)[1] > 0]
    if len(wind) != 1:
        raise RuntimeError(f"cannot select a unique outward sonic branch; roots={roots}")
    return wind[0]


def solve_critical_point(rho_c, T_c, M_c, kappa_const=0.0, n_iter=50, rtol=1e-6, atol=1e-12):
    """Self-consistent sonic point: iterate Q_c -> (r_c, Mdot, Y_c) -> Q_c.

    Returns a dict with r_c, v_c, Mdot, Q_c, Y_c, s_c and two regime checks:
      w    = 4 pi rho_c r_c^3 / M_c   (envelope mass per e-fold / M_c)
      trap = tau_c v_c / c with tau_c = kappa rho r/|Y| (QA needs trap >> 1)
    """
    Q_c = 0.0
    for _ in range(n_iter):
        r_c, v_c, Mdot = critical_point(rho_c, T_c, M_c, Q_c)
        Y_c = critical_slope(r_c, rho_c, T_c, M_c, Mdot, Q_c)
        s_c = np.log(r_c)
        Q_new = get_Q(s_c, np.log(rho_c), np.log(T_c), M_c, Mdot, s_c, Y_c, kappa_const)
        done = abs(Q_new - Q_c) <= atol + rtol*max(abs(Q_new), abs(Q_c))
        Q_c = Q_new
        if done:
            break
    else:
        raise RuntimeError(f"critical Q iteration did not converge in {n_iter} steps")

    r_c, v_c, Mdot = critical_point(rho_c, T_c, M_c, Q_c)
    Y_c = critical_slope(r_c, rho_c, T_c, M_c, Mdot, Q_c)
    tau_c = hf.get_kappa(rho_c, T_c, kappa_const)*rho_c*abs(r_c/Y_c)
    return dict(r_c=r_c, v_c=v_c, Mdot=Mdot, Q_c=Q_c, Y_c=Y_c, s_c=np.log(r_c),
                w=4*np.pi*rho_c*r_c**3/M_c, trap=float(tau_c*v_c/hf.c))


# ---------------------------------------------------------------------------
# Full QA wind
# ---------------------------------------------------------------------------
def solve_qa_wind(rho_c, T_c, M_c, kappa_const=0.0, eta_stop=0.3, r_in=None, r_out=None,
                  delta=1e-4, rtol=1e-10, atol=1e-12):
    """Integrate the QA wind outward and inward from its sonic point.

    Parameters (cgs)
    ----------------
    rho_c, T_c, M_c : density, temperature and enclosed mass at the sonic point
    kappa_const     : opacity floor [cm^2/g]
    eta_stop        : outward leg stops when eta reaches this value
    r_in            : inner radius (default 10 Rsun); skipped if r_in >= r_c
    r_out           : outer backstop radius (default 1e4 Rsun)
    delta           : launch offset in ln r from the 0/0 sonic point

    Returns a DataFrame (r, M, rho, T, x, v, kappa, L, eta) sorted in r. The
    sonic-point data are stored in ``df.attrs``.
    """
    r_in = 10*hf.Rsun if r_in is None else r_in
    r_out = 1e4*hf.Rsun if r_out is None else r_out
    if not np.isfinite(r_in) or r_in <= 0:
        raise ValueError("r_in must be positive and finite")

    crit = solve_critical_point(rho_c, T_c, M_c, kappa_const)
    r_c, Mdot, s_c, Y_c, Q_c = (crit[k] for k in ("r_c", "Mdot", "s_c", "Y_c", "Q_c"))
    f1, _, f3 = hf.get_f(rho_c, T_c)
    y_c = np.array([np.log(rho_c), np.log(T_c), M_c])
    dyds_c = np.array([Y_c, (f1/f3)*Y_c + Q_c, 4*np.pi*rho_c*r_c**3])

    fun = lambda s, y: rhs(s, y, Mdot, s_c, Y_c, Q_c, kappa_const)
    opts = dict(method="Radau", rtol=rtol, atol=np.array([atol, atol, atol*M_c]))

    def eta_event(s, y):
        return get_eta(s, y, Mdot, s_c, Y_c, Q_c, kappa_const) - eta_stop
    eta_event.terminal, eta_event.direction = True, 1

    out = solve_ivp(fun, (s_c + delta, np.log(r_out)), y_c + delta*dyds_c,
                    events=eta_event, **opts)
    if not out.success:
        raise RuntimeError(f"outward QA integration failed: {out.message}")
    if out.status == 0:
        warnings.warn(f"eta never reached {eta_stop} before r_out; outer edge set by r_out")

    inw = None
    if r_in < r_c:
        inw = solve_ivp(fun, (s_c - delta, np.log(r_in)), y_c - delta*dyds_c, **opts)
        if not inw.success or not np.isclose(np.exp(inw.t[-1]), r_in, rtol=2e-10):
            raise RuntimeError(f"inward QA integration did not reach r_in: {inw.message}")

    # Stitch inward leg + exact sonic point + outward leg.
    t_in, y_in = (inw.t, inw.y) if inw is not None else (np.empty(0), np.empty((3, 0)))
    s = np.concatenate((t_in, [s_c], out.t))
    y = np.concatenate((y_in, y_c[:, None], out.y), axis=1)
    order = np.argsort(s)
    s, y = s[order], y[:, order]
    r, rho, T, M = np.exp(s), np.exp(y[0]), np.exp(y[1]), y[2]

    # L and eta from the same RHS the integrator used (no grid differencing).
    L, eta = np.empty_like(r), np.empty_like(r)
    for i in range(len(r)):
        Y, Z, _ = fun(s[i], y[:, i])
        fi1, _, fi3 = hf.get_f(rho[i], T[i])
        L[i] = -hf.get_A(r[i], rho[i], T[i], kappa_const)*Z
        eta[i] = abs((Z - (fi1/fi3)*Y)/((fi1/fi3)*Y))

    df = pd.DataFrame({"r": r, "M": M, "rho": rho, "T": T, "x": hf.saha_x(rho, T),
                       "v": Mdot/(4*np.pi*r**2*rho),
                       "kappa": hf.get_kappa(rho, T, kappa_const), "L": L, "eta": eta})
    df.attrs.update(crit)
    df.attrs.update(eta_stop=eta_stop, stopped_on_eta=bool(out.status == 1), r_in=float(r_in),
                    inward_success=inw is not None, inward_integration_skipped=inw is None,
                    kappa_const=float(kappa_const))
    return df


if __name__ == "__main__":
    # Example: compact "Model B" sonic point.
    qa = solve_qa_wind(rho_c=1e-7, T_c=10**4.2, M_c=0.1*hf.Msun, kappa_const=1e-2, eta_stop=0.01)
    a = qa.attrs
    print(f"r_c        = {a['r_c']/hf.Rsun:.4f} Rsun")
    print(f"Mdot       = {a['Mdot']*hf.year/hf.Msun:.6f} Msun/yr")
    print(f"tau v/c    = {a['trap']:.1f}   (QA valid if >> 1)")
    print(f"r(eta_stop)= {qa['r'].iloc[-1]/hf.Rsun:.3f} Rsun")
