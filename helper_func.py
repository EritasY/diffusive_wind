"""Shared microphysics for the QA and FLD wind solvers (cgs units throughout).

Pure-hydrogen gas in LTE with a one-temperature radiation field E = a T^4:

* Saha ionisation fraction x(rho, T) and its logarithmic derivatives;
* total EOS (gas + ionisation + radiation) and the coefficients f1, f2, f3
  that appear in the wind equations;
* the range-free analytic Rosseland opacity;
* diffusion coefficient, Levermore-Pomraning flux limiter, optical depth,
  gravitational potential and Bernoulli function.

Both ``quasi_adiabatic.py`` and ``qa_fld_bridge.py`` import this file, so the
two zones always use identical physics.
"""

from collections import namedtuple

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import cumulative_trapezoid

# ---------------------------------------------------------------------------
# Constants (cgs)
# ---------------------------------------------------------------------------
me = const.m_e.cgs.value
mp = const.m_p.cgs.value
kB = const.k_B.cgs.value
h = const.h.cgs.value
c = const.c.cgs.value
G = const.G.cgs.value
sigma_sb = const.sigma_sb.cgs.value
a_rad = 4.0*sigma_sb/c
chi_H = 13.6*u.eV.to(u.erg)          # hydrogen ionisation potential
Rsun = const.R_sun.cgs.value
Msun = const.M_sun.cgs.value
Lsun = const.L_sun.cgs.value
year = u.yr.to(u.s)


# ---------------------------------------------------------------------------
# Saha ionisation
# ---------------------------------------------------------------------------
def saha_state(rho, T):
    """Return x, dx/dln(rho), dx/dln(T) for pure hydrogen in LTE.

    Saha: x^2/(1-x) = A, with
        A = (m_p/rho) (2 pi m_e k T/h^2)^{3/2} exp(-chi_H/kT).
    The root x = 2/(1+sqrt(1+4/A)) is evaluated in log space so it stays
    finite for both A -> 0 (neutral) and A -> inf (fully ionised).
    """
    rho, T = np.broadcast_arrays(np.asarray(rho, float), np.asarray(T, float))
    if np.any(~np.isfinite(rho) | ~np.isfinite(T) | (rho <= 0) | (T <= 0)):
        raise ValueError("rho and T must be positive and finite")
    lnA = (1.5*np.log(2*np.pi*me*kB*T) - 3*np.log(h) - chi_H/(kB*T)
           + np.log(mp/rho))
    lnx = np.log(2.0) - np.logaddexp(0.0, 0.5*np.logaddexp(0.0, np.log(4.0)-lnA))
    x = np.exp(lnx)
    # dx/dlnA = x(1-x)/(2-x); dlnA/dlnrho = -1, dlnA/dlnT = 3/2 + chi/kT.
    one_minus_x = np.minimum(1.0, np.exp(np.minimum(0.0, 2*lnx - lnA)))
    dx_dlnA = x*one_minus_x/(2.0 - x)
    return x, -dx_dlnA, dx_dlnA*(1.5 + chi_H/(kB*T))


def saha_x(rho, T):
    """Ionisation fraction only."""
    return saha_state(rho, T)[0]


# ---------------------------------------------------------------------------
# Equation of state: gas + ionisation + LTE radiation
# ---------------------------------------------------------------------------
EOSState = namedtuple("EOSState", "x P e h f1 f2 f3 cs2_ad")


def eos_state(rho, T):
    """Total EOS and the thermodynamic coefficients of the wind equations.

        f1 = (T/rho)(dP/dT)_rho,   f2 = (dP/drho)_T,   f3 = T(de/dT)_rho,

    so that  dP = rho(f2 dln(rho) + f1 dln(T))  and the first law reads
    T ds = f3 dln(T) - f1 dln(rho).  The adiabatic sound speed is
    cs2_ad = f2 + f1^2/f3.
    """
    rho, T = np.broadcast_arrays(np.asarray(rho, float), np.asarray(T, float))
    x, x_rho, x_T = saha_state(rho, T)
    RT = kB*T/mp
    aT4 = a_rad*T**4
    P = rho*RT*(1 + x) + aT4/3
    e = 1.5*RT*(1 + x) + aT4/rho + chi_H*x/mp
    f1 = RT*(1 + x + x_T) + 4*aT4/(3*rho)
    f2 = RT*(1 + x + x_rho)
    f3 = 1.5*RT*(1 + x) + (1.5*RT + chi_H/mp)*x_T + 4*aT4/rho
    return EOSState(x, P, e, e + P/rho, f1, f2, f3, f2 + f1*f1/f3)


def get_f(rho, T):
    s = eos_state(rho, T)
    return s.f1, s.f2, s.f3


def get_cs2(rho, T):
    return eos_state(rho, T).cs2_ad


def enthalpy(rho, T):
    return eos_state(rho, T).h


# ---------------------------------------------------------------------------
# Opacity and radiative transport
# ---------------------------------------------------------------------------
def get_kappa(rho, T, kappa_const=0.0):
    """Range-free Rosseland opacity [cm^2/g]:

        kappa = 0.4 x  +  8e22 rho T^{-7/2} x^2  +  kappa_const
              (e-scattering) (Kramers-like, ~x^2)   (floor)

    ``kappa_const`` stands in for molecules/dust/lines in neutral gas. It is not
    fixed by the formula and strongly controls the neutral outer atmosphere.
    """
    rho, T = np.broadcast_arrays(np.asarray(rho, float), np.asarray(T, float))
    x = saha_x(rho, T)
    return 0.4*x + 8e22*rho*T**-3.5*x**2 + kappa_const


def get_A(r, rho, T, kappa_const=0.0):
    """Diffusion coefficient in  L = -A dln(T)/dln(r):

        A = 16 pi a c r T^4 / (3 kappa rho).
    """
    r, rho, T = (np.asarray(v, float) for v in (r, rho, T))
    return 16*np.pi*a_rad*c*r*T**4/(3*get_kappa(rho, T, kappa_const)*rho)


def flux_factor(L, r, T):
    """q = F/(cE) = L / (4 pi r^2 c a T^4)."""
    return L/(4*np.pi*r**2*c*a_rad*T**4)


def lp_limiter(R):
    """Levermore-Pomraning flux limiter lambda(R) = (2+R)/(6+3R+R^2).

    R = |grad E|/(kappa rho E). lambda -> 1/3 (diffusion) for R << 1 and
    q = lambda R -> 1 (free streaming) for R >> 1.
    """
    R = np.asarray(R, float)
    if np.any(~np.isfinite(R) | (R < 0)):
        raise ValueError("R must be finite and non-negative")
    return (2 + R)/(6 + 3*R + R*R)


def lp_inverse(q):
    """Invert q = lambda(R) R for 0 <= q < 1 (rationalised quadratic root)."""
    q = np.asarray(q, float)
    if np.any(~np.isfinite(q) | (q < 0) | (q >= 1)):
        raise ValueError("flux factor must lie in [0, 1)")
    b = 2 - 3*q
    return 12*q/(np.sqrt(b*b + 24*q*(1 - q)) + b)


def get_tau(r, rho, kappa, tau_outer=0.0):
    """Optical depth tau(r) = tau_outer + int_r^{r_out} kappa rho dr (trapezoid)."""
    r, rho, kappa = (np.asarray(v, float) for v in (r, rho, kappa))
    if np.any(np.diff(r) <= 0):
        raise ValueError("r must increase strictly")
    dtau = 0.5*(kappa[1:]*rho[1:] + kappa[:-1]*rho[:-1])*np.diff(r)
    return tau_outer + np.append(np.cumsum(dtau[::-1])[::-1], 0.0)


# ---------------------------------------------------------------------------
# Gravity and energy
# ---------------------------------------------------------------------------
def potential_from_reference(r, M, phi0):
    """Phi(r) with Phi(r[0]) = phi0 and dPhi/dln(r) = G M(r)/r."""
    r, M = np.asarray(r, float), np.asarray(M, float)
    return phi0 + np.concatenate(([0.0], cumulative_trapezoid(G*M/r, np.log(r))))


def bernoulli(v, rho, T, phi):
    """B = v^2/2 + h + Phi. Total energy flux is Edot = Mdot B + L."""
    return 0.5*np.asarray(v, float)**2 + enthalpy(rho, T) + np.asarray(phi, float)
