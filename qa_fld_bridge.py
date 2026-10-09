"""Matched quasi-adiabatic (QA) -> grey flux-limited-diffusion (FLD) wind.

Strategy
--------
1. QA zone (``quasi_adiabatic.py``): cross the trapped adiabatic sonic point,
   which fixes Mdot. Stop at r_match, where eta = eta_match.
2. Energy anchor: fix one total energy flux at the inner base,
       Edot = Mdot B(r_in) + L_core,     B = v^2/2 + h + Phi,
   and use L_E(r) = Edot - Mdot B(r) as the energy-consistent QA luminosity.
3. FLD zone: solve the exact one-temperature equations from r_match to a *free*
   outer radius. States (ln rho, ln T, L, M, tau) + one parameter (ln r span)
   need six boundary conditions:
       seam (r_match):  rho = rho_QA,  M = M_QA,  Mdot B + L = Edot
       outer surface:   tau = tau_ph,  q = F/(cE) = q_ph,  dL/ds = 0
   The seam T and L are therefore selected by the BVP rather than copied from
   QA; this removes the exponentially growing luminosity mode. Mdot is
   inherited from the QA sonic point and is not re-selected.
4. Continuation: start from a short thick-diffusion seed and raise the outer q
   geometrically to q_ph. If kappa_const is below the bootstrap floor, then
   lower it in <= 5% steps, recomputing QA each time (kappa also enters Q).

FLD closure (Levermore-Pomraning), with s = ln r:
    q = L/(4 pi r^2 c a T^4) = lambda(R) R,     Z = -sgn(L) kappa rho r R/4,
    (v^2 - f2) Y = f1 Z + G M/r - 2 v^2,         dL/ds = Mdot (f1 Y - f3 Z).

Caveat: the exterior keeps E = a T^4 and the isotropic LTE radiation enthalpy,
so the q = 1/4, tau = 2/3 surface is a grey one-temperature closure, not a
production photosphere (``production_radiation_transport = False``).

Run ``python qa_fld_bridge.py`` for an example.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_bvp, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

import helper_func as hf
from quasi_adiabatic import solve_qa_wind


@dataclass
class FLDStep:
    """One converged FLD exterior (one point on the continuation path)."""
    frame: pd.DataFrame
    solution: object        # scipy solve_bvp result; seeds the next step
    diagnostics: dict


@dataclass
class BridgeSolution:
    profile: pd.DataFrame       # stitched QA + FLD profile (column "zone")
    diagnostics: dict           # acceptance flags and key numbers
    qa: pd.DataFrame            # raw QA profile (L = transport luminosity)
    qa_energy: pd.DataFrame     # QA with energy-anchored L; transport L kept as L_qad
    seed: pd.DataFrame          # short thick-diffusion seed
    fld: FLDStep                # final FLD exterior
    continuation: pd.DataFrame  # one row per continuation step
    kappa_const: float          # opacity floor actually solved


@dataclass
class Overlap:
    frame: pd.DataFrame     # QA and inward-FLD fields on a common grid
    summary: pd.DataFrame   # relative differences per field
    diagnostics: dict


# ---------------------------------------------------------------------------
# Energy anchor and thick seed
# ---------------------------------------------------------------------------
def anchor_energy(qa, L_core):
    """Attach Phi, B and the energy-consistent luminosity to a QA profile.

    Phi(r_in) = -G M/r at the base. The QA transport luminosity is kept as
    ``L_qad``; ``L`` becomes L_E = Edot - Mdot B. Using L_E stops a small relative
    error in the large Mdot B from becoming an O(1) error in the small L.
    """
    out = qa.copy()
    r, rho, T, M, v = (out[k].to_numpy(float) for k in ("r", "rho", "T", "M", "v"))
    Mdot = float(qa.attrs["Mdot"])
    phi = hf.potential_from_reference(r, M, -hf.G*M[0]/r[0])
    B = hf.bernoulli(v, rho, T, phi)
    Edot = Mdot*B[0] + L_core
    out["Phi"], out["B"], out["L_qad"] = phi, B, out["L"].to_numpy(float)
    out["L"] = Edot - Mdot*B
    out["Edot"] = Mdot*B + out["L"].to_numpy(float)
    out.attrs.update(qa.attrs)
    out.attrs.update(L_core=float(L_core), Edot_target=float(Edot),
                     seam_luminosity_correction=float(out["L"].iloc[-1]/out["L_qad"].iloc[-1] - 1))
    return out


def solve_thick_seed(qa_e, r_outer, kappa_const, n_mesh=180, tol=3e-6, max_nodes=80_000):
    """Short pure-diffusion BVP (L = -A Z, no limiter) just outside the seam.

    ODEs for (ln rho, ln T, L, M); BCs: rho, M, Edot at the seam and dL/ds = 0 at
    r_outer. Used only as the initial guess for the first FLD solve.
    """
    seam = qa_e.iloc[-1]
    r0, rho0, T0, M0, phi0, L0 = (float(seam[k]) for k in ("r", "rho", "T", "M", "Phi", "L"))
    Mdot, Edot = float(qa_e.attrs["Mdot"]), float(qa_e.attrs["Edot_target"])
    Ls, Ms = max(abs(Edot), abs(L0), hf.Lsun), M0      # scales for O(1) unknowns
    s0, s1 = np.log(r0), np.log(r_outer)
    span = s1 - s0

    def local(s, y):
        r = np.exp(s)
        rho = np.exp(np.clip(y[0], np.log(1e-30), np.log(1e2)))
        T = np.exp(np.clip(y[1], np.log(100.0), np.log(1e8)))
        L, M = y[2]*Ls, y[3]*Ms
        f1, f2, f3 = hf.get_f(rho, T)
        Z = -L/hf.get_A(r, rho, T, kappa_const)
        v2 = (Mdot/(4*np.pi*r**2*rho))**2
        Y = (f1*Z + hf.G*M/r - 2*v2)/(v2 - f2)
        dyds = np.array((Y, Z, Mdot*(f1*Y - f3*Z)/Ls, 4*np.pi*rho*r**3/Ms))
        return dyds, rho, T, v2

    def ode(x, y):
        return span*local(s0 + span*x, y)[0]

    def bc(ya, yb):
        _, rho, T, v2 = local(s0, ya)
        E = Mdot*(0.5*v2 + hf.enthalpy(rho, T) + phi0) + ya[2]*Ls
        return np.array((ya[0] - np.log(rho0), ya[3] - 1.0, (E - Edot)/Ls, local(s1, yb)[0][2]))

    x = np.linspace(0, 1, n_mesh)
    r = np.exp(s0 + span*x)
    guess = np.vstack((np.log(rho0*(r/r0)**-2), np.log(T0*(r/r0)**-0.25),
                       np.full_like(r, L0/Ls), (M0 + 4*np.pi*rho0*r0**2*(r - r0))/Ms))
    sol = solve_bvp(ode, bc, x, guess, tol=tol, max_nodes=max_nodes)
    if sol.status != 0:
        raise RuntimeError(f"thick seed solve failed: {sol.message}")
    xo = np.linspace(0, 1, max(n_mesh, len(sol.x)))
    y = sol.sol(xo)
    return pd.DataFrame({"r": np.exp(s0 + span*xo), "rho": np.exp(y[0]), "T": np.exp(y[1]),
                         "L": y[2]*Ls, "M": y[3]*Ms})


# ---------------------------------------------------------------------------
# Exact FLD equations
# ---------------------------------------------------------------------------
def _fld_equations(s, state, *, Mdot, Ls, Ms, kappa_const, protect):
    """One-temperature FLD equations in s = ln r.

    state = [ln rho, ln T, L/Ls, M/Ms, tau]. Returns (d state/ds, fields dict).
    protect=True (BVP): clip trial states and guard the v^2 = f2 denominator so
    unconverged Newton iterates cannot crash. protect=False (IVP): such states
    are unphysical, so raise instead.
    """
    state = np.asarray(state, float)
    if protect:
        lnrho = np.clip(state[0], np.log(1e-30), np.log(1e2))
        lnT = np.clip(state[1], np.log(100.0), np.log(1e8))
    else:
        if np.any(~np.isfinite(state)):
            raise RuntimeError("FLD integration produced a non-finite state")
        lnrho, lnT = state[0], state[1]
    r, rho, T = np.exp(s), np.exp(lnrho), np.exp(lnT)
    L, M = state[2]*Ls, state[3]*Ms
    f1, f2, f3 = hf.get_f(rho, T)
    kappa = hf.get_kappa(rho, T, kappa_const)
    q = hf.flux_factor(L, r, T)

    if protect:
        q_used = np.minimum(np.abs(q), 1 - 1e-9)
    else:
        if not np.all((L > 0) & (M > 0) & (rho > 0) & (T > 0) & (np.abs(q) < 1)):
            raise RuntimeError("FLD integration left the positive, causal (|q|<1) domain")
        q_used = np.abs(q)
    R = hf.lp_inverse(q_used)
    Z = -np.sign(L)*kappa*rho*r*R/4

    v2 = (Mdot/(4*np.pi*r**2*rho))**2
    D = v2 - f2                         # vanishes at the *diffusive* sonic point
    tiny = 1e-12*(v2 + np.abs(f2))
    if protect:
        D = np.where(np.abs(D) < tiny, np.sign(D + 1e-300)*tiny, D)
    elif np.any(~np.isfinite(D) | (D <= tiny)):
        raise RuntimeError("FLD integration reached the diffusive sonic point v^2 = f2")
    Y = (f1*Z + hf.G*M/r - 2*v2)/D
    Z_ad = (f1/f3)*Y
    dLds = Mdot*(f1*Y - f3*Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.abs((Z - Z_ad)/Z_ad)   # same eta as QA, from gradients
    dyds = np.array((Y, Z, dLds/Ls, 4*np.pi*rho*r**3/Ms, -kappa*rho*r))
    fields = dict(r=r, rho=rho, T=T, L=L, M=M, kappa=kappa, q=q, R=R, Y=Y, Z=Z,
                  eta=eta, D=v2 - f2, f2=f2, dLds=dLds, v2=v2)
    return dyds, fields


def _qa_seam_slopes(seam, Mdot, kappa_const):
    """Y, Z, eta implied at the seam by the QA transport luminosity L_qad."""
    r, rho, T, M = (float(seam[k]) for k in ("r", "rho", "T", "M"))
    f1, f2, f3 = (float(f) for f in hf.get_f(rho, T))
    v2 = (Mdot/(4*np.pi*r**2*rho))**2
    Z = -float(seam["L_qad"])/float(hf.get_A(r, rho, T, kappa_const))
    Y = (f1*Z + hf.G*M/r - 2*v2)/(v2 - f2)
    return dict(Y=Y, Z=Z, eta=abs((Z - (f1/f3)*Y)/((f1/f3)*Y)))


def _quality_flags(d, eta_match, q_target, tol):
    """Physical/numerical checks that stop continuation jumping branches."""
    T_lim, L_lim = max(eta_match, 5e-3), max(2*eta_match, 2e-2)
    return {
        "flux_target_ok": abs(d["outer_flux_factor"]/q_target - 1) <= max(10*tol, 2e-4),
        "seam_temperature_ok": abs(d["seam_temperature_shift"]) <= T_lim,
        "seam_luminosity_ok": abs(d["seam_luminosity_vs_transport"]) <= L_lim,
        "seam_slopes_ok": max(abs(d["seam_Y_relative_mismatch"]),
                              abs(d["seam_Z_relative_mismatch"])) <= L_lim,
        "thick_seam_ok": d["tau_match"] >= 10 and abs(d["seam_flux_factor"]) <= 0.03,
        "positive_luminosity": d["minimum_luminosity"] > 0,
        "diffusive_zone_supersonic": d["minimum_D_over_f2"] > 0,     # v^2 > f2 throughout
        "physical_flux_cone": d["maximum_abs_flux_factor"] < 1,
        "energy_conserved": d["energy_relative_drift"] <= max(5*tol, 1e-4),
        "boundary_residual_ok": d["boundary_residual_max"] <= max(20*tol, 1e-4),
        "collocation_residual_ok": d["rms_residual_max"] <= max(2*tol, 1e-4),
    }


# ---------------------------------------------------------------------------
# FLD exterior (free-boundary BVP)
# ---------------------------------------------------------------------------
def solve_fld_exterior(qa_e, seed, q_target, kappa_const, *, previous=None, tau_surface=2/3,
                       eta_match=0.01, n_mesh=180, tol=2e-4, max_nodes=80_000):
    """Solve the FLD exterior from the QA seam to a free outer surface.

    Unknowns: 5 states on x in [0, 1] (s = s_match + span*x) and p = ln(span).
    ``previous`` (an FLDStep) is used as the initial guess when continuing.
    """
    if not 0 < q_target < 1:
        raise ValueError("q_target must lie in (0, 1)")
    seam = qa_e.iloc[-1]
    s_m = np.log(float(seam["r"]))
    Mdot, Edot = float(qa_e.attrs["Mdot"]), float(qa_e.attrs["Edot_target"])
    phi_m, rho_m, M_m = float(seam["Phi"]), float(seam["rho"]), float(seam["M"])
    Ls, Ms = max(abs(Edot), hf.Lsun), M_m

    if previous is None:
        # Seed profile on [s_m, ln r_edge]; tau integrated inward from tau_surface.
        sd = seed.sort_values("r")
        sd_s = np.log(sd["r"].to_numpy(float))
        span0 = sd_s[-1] - s_m
        x = np.linspace(0, 1, n_mesh)
        s = s_m + span0*x
        at = lambda col: np.interp(s, sd_s, sd[col].to_numpy(float))
        rho = np.exp(np.interp(s, sd_s, np.log(sd["rho"].to_numpy(float))))
        T = np.exp(np.interp(s, sd_s, np.log(sd["T"].to_numpy(float))))
        tau = hf.get_tau(np.exp(s), rho, hf.get_kappa(rho, T, kappa_const), tau_surface)
        guess = np.vstack((np.log(rho), np.log(T), at("L")/Ls, at("M")/Ms, tau))
        p0 = np.array([np.log(span0)])
    else:
        x = np.linspace(0, 1, max(n_mesh, len(previous.solution.x)))
        guess, p0 = previous.solution.sol(x), previous.solution.p.copy()

    def local(s, y):
        return _fld_equations(s, y, Mdot=Mdot, Ls=Ls, Ms=Ms, kappa_const=kappa_const, protect=True)

    def ode(x, y, p):
        span = np.exp(np.clip(p[0], -8, 8))
        return span*local(s_m + span*x, y)[0]

    def bc(ya, yb, p):
        span = np.exp(np.clip(p[0], -8, 8))
        _, a = local(s_m, ya)
        _, b = local(s_m + span, yb)
        E = Mdot*(0.5*a["v2"] + hf.enthalpy(a["rho"], a["T"]) + phi_m) + a["L"]
        return np.array((ya[0] - np.log(rho_m),        # seam: rho
                         ya[3] - M_m/Ms,               # seam: M
                         (E - Edot)/Ls,                # seam: total energy flux
                         yb[4] - tau_surface,          # outer: tau
                         b["q"]/q_target - 1,          # outer: flux factor
                         b["dLds"]/Ls))                # outer: L has levelled off

    sol = solve_bvp(ode, bc, x, guess, p=p0, tol=tol, max_nodes=max_nodes)
    if sol.status != 0:
        raise RuntimeError(f"FLD solve failed: {sol.message}")

    # Evaluate on a fine output grid.
    span = float(np.exp(sol.p[0]))
    xo = np.linspace(0, 1, max(500, len(sol.x)))
    y = sol.sol(xo)
    _, f = local(s_m + span*xo, y)
    r, rho, T, L, M, q = f["r"], f["rho"], f["T"], f["L"], f["M"], f["q"]
    v = np.sqrt(f["v2"])
    lam = hf.lp_limiter(f["R"])
    phi = hf.potential_from_reference(r, M, phi_m)
    B = hf.bernoulli(v, rho, T, phi)
    frame = pd.DataFrame({
        "r": r, "M": M, "rho": rho, "T": T, "x": hf.saha_x(rho, T), "v": v,
        "kappa": f["kappa"], "L": L, "Phi": phi, "B": B, "Edot": Mdot*B + L, "tau": y[4],
        "flux_factor": q, "R": f["R"], "flux_limiter": lam, "eddington_factor": lam + q**2,
        "Y": f["Y"], "Z": f["Z"], "dLds": f["dLds"],
    })
    # Advective / diffusive radiation-flux ratio, (1 + chi) v / (c q).
    frame["radiation_trapping_ratio"] = (1 + lam + q**2)*v/(hf.c*np.maximum(np.abs(q), 1e-300))

    qa = _qa_seam_slopes(seam, Mdot, kappa_const)
    L_qad = float(seam["L_qad"])
    d = {
        "r_match": r[0], "r_outer": r[-1],
        "outer_flux_factor": q[-1], "surface_tau": y[4, -1], "tau_match": y[4, 0],
        "outer_temperature": T[-1], "outer_density": rho[-1], "outer_luminosity": L[-1],
        "seam_flux_factor": q[0],
        "seam_temperature_shift": T[0]/float(seam["T"]) - 1,         # FLD vs QA
        "seam_luminosity_vs_transport": L[0]/L_qad - 1,              # vs QA L_qad
        "seam_luminosity_vs_energy": L[0]/float(seam["L"]) - 1,      # vs QA L_E
        "seam_Y_relative_mismatch": f["Y"][0]/qa["Y"] - 1,
        "seam_Z_relative_mismatch": f["Z"][0]/qa["Z"] - 1,
        "eta_match_reconstructed": qa["eta"],
        "seam_scale_trapping_v_tau_over_c":
            f["kappa"][0]*rho[0]*r[0]/max(abs(f["Y"][0]), 1e-300)*v[0]/hf.c,
        "minimum_D_over_f2": np.min(f["D"]/f["f2"]),
        "minimum_luminosity": np.min(L),
        "maximum_abs_flux_factor": np.max(np.abs(q)),
        "energy_relative_drift": np.max(np.abs(Mdot*B + L - Edot))/Ls,
        "flux_factor_monotonic": bool(np.all(np.diff(q) >= -2e-6*max(abs(q[-1]), 1e-10))),
        "nodes": len(sol.x),
        "rms_residual_max": np.max(sol.rms_residuals),
        "boundary_residual_max": np.max(np.abs(bc(sol.y[:, 0], sol.y[:, -1], sol.p))),
    }
    d = {k: (float(v) if isinstance(v, (np.floating, np.ndarray)) else v) for k, v in d.items()}
    flags = _quality_flags(d, eta_match, q_target, tol)
    d.update(flags)
    d["step_accepted"] = bool(all(flags.values()))
    frame.attrs.update(d)
    return FLDStep(frame=frame, solution=sol, diagnostics=d)


# ---------------------------------------------------------------------------
# Full bridge with continuation
# ---------------------------------------------------------------------------
def _prepare_inner(*, rho_c, T_c, M_c, r_in, L_core, kappa_const, eta_match, n_mesh, tol, max_nodes):
    """QA solution, energy anchor and thick seed at one opacity floor."""
    qa = solve_qa_wind(rho_c, T_c, M_c, kappa_const=kappa_const, eta_stop=eta_match, r_in=r_in)
    qa_e = anchor_energy(qa, L_core)
    seed = solve_thick_seed(qa_e, 1.05*float(qa_e["r"].iloc[-1]), kappa_const,
                            n_mesh=max(100, n_mesh), tol=min(3e-6, tol), max_nodes=max_nodes)
    edge = seed.iloc[-1]
    return qa, qa_e, seed, float(hf.flux_factor(edge["L"], edge["r"], edge["T"]))


def _flux_targets(q0, q1, growth):
    """Geometric sequence q0, q0*growth, ... capped at q1."""
    if not 1 < growth <= 1.5:
        raise ValueError("flux_growth must lie in (1, 1.5]")
    targets = [q0]
    while targets[-1] < q1:
        targets.append(min(q1, targets[-1]*growth))
    return targets


def _floor_targets(k0, k1):
    """Opacity floors from k0 down to k1 in steps of at most 5%."""
    if k1 >= k0:
        return []
    if k1 == 0:   # zero cannot be reached geometrically: approach, then one last step
        return _floor_targets(k0, max(k0*1e-3, 1e-8)) + [0.0]
    n = max(1, int(np.ceil(abs(np.log(k1/k0))/abs(np.log(0.95)))))
    return list(np.geomspace(k0, k1, n + 1)[1:])


def _fld_step(records, label, qa_e, seed, q_target, kappa_const, previous, kw):
    """Try one continuation step. Returns (step or None, failure message or None)."""
    try:
        step = solve_fld_exterior(qa_e, seed, q_target, kappa_const, previous=previous, **kw)
    except Exception as exc:
        return None, f"{label} step failed at q={q_target:g}, kappa_const={kappa_const:g}: {exc}"
    records.append({**step.diagnostics, "flux_factor_target": q_target,
                    "kappa_const": kappa_const, "continuation_parameter": label})
    if not step.diagnostics["step_accepted"]:
        bad = [k for k, v in step.diagnostics.items() if k.endswith("_ok") and v is False]
        return step, f"{label} step left the controlled branch: " + ", ".join(bad)
    return step, None


def _combine_profile(qa_e, fld_frame, kappa_const, tau_surface):
    """Join QA (r < r_match) and FLD (r >= r_match) and add derived columns.

    In the QA zone ``L`` is the transport luminosity L_qad; the energy-anchored
    value is kept as ``L_energy_constraint`` (use it for energy audits).
    """
    inner = qa_e.iloc[:-1].copy()            # seam row comes from the FLD side
    inner["L_energy_constraint"], inner["L"] = inner["L"], inner["L_qad"]
    inner["zone"] = "quasi_adiabatic"
    outer = fld_frame.copy()
    outer["L_energy_constraint"] = outer["L"]
    outer["zone"] = "grey_fld"
    p = pd.concat((inner, outer), ignore_index=True, sort=False)

    r, rho, T, L, M, v = (p[k].to_numpy(float) for k in ("r", "rho", "T", "L", "M", "v"))
    kappa = hf.get_kappa(rho, T, kappa_const)
    q = hf.flux_factor(L, r, T)
    R = hf.lp_inverse(np.clip(np.abs(q), 0, 1 - 1e-12))
    lam = hf.lp_limiter(R)
    p["kappa"] = kappa
    p["c_s_ad"] = np.sqrt(hf.get_cs2(rho, T))   # QA critical speed
    p["v_esc"] = np.sqrt(2*hf.G*M/r)            # local, enclosed-mass escape speed
    p["flux_factor"], p["R"], p["flux_limiter"] = q, R, lam
    p["eddington_factor"] = lam + q**2
    p["radiation_trapping_ratio"] = (1 + lam + q**2)*v/(hf.c*np.maximum(np.abs(q), 1e-300))
    p["tau"] = hf.get_tau(r, rho, kappa, tau_outer=tau_surface)
    return p


def solve_bridge(rho_c=1e-7, T_c=10**4.2, M_c=0.1*hf.Msun, *, r_in=10*hf.Rsun, L_core=hf.Lsun,
                 kappa_const=1e-3, bootstrap_kappa_const=1.5e-3, eta_match=0.01,
                 q_ph=0.25, tau_ph=2/3, flux_growth=1.2, tol=2e-4, n_mesh=180,
                 max_nodes=80_000, strict=True):
    """Build the QA -> FLD wind (cgs inputs).

    rho_c, T_c, M_c        sonic-point density, temperature, enclosed mass
    r_in                   inner base radius of the QA zone
    L_core                 luminosity at r_in; anchors Edot = Mdot B(r_in) + L_core
    kappa_const            opacity floor [cm^2/g] (strongly affects the neutral exterior)
    bootstrap_kappa_const  floor used for the first flux-factor continuation
    eta_match              QA/FLD hand-over; QA error there is O(eta_match^2)
    q_ph, tau_ph           outer surface: F/(cE) = q_ph and tau = tau_ph
    flux_growth            ratio between successive outer-q targets
    tol, n_mesh, max_nodes solve_bvp settings
    strict                 raise (True) or warn (False) if not accepted
    """
    if not (np.isfinite(kappa_const) and kappa_const >= 0):
        raise ValueError("kappa_const must be finite and non-negative")
    k0 = max(float(kappa_const), float(bootstrap_kappa_const))
    inner_kw = dict(rho_c=rho_c, T_c=T_c, M_c=M_c, r_in=r_in, L_core=L_core,
                    eta_match=eta_match, n_mesh=n_mesh, tol=tol, max_nodes=max_nodes)
    fld_kw = dict(tau_surface=tau_ph, eta_match=eta_match, n_mesh=n_mesh, tol=tol,
                  max_nodes=max_nodes)

    # Stage 1: raise the outer flux factor from the thick seed value to q_ph.
    inner = _prepare_inner(kappa_const=k0, **inner_kw)
    qa_e, seed, q0 = inner[1], inner[2], inner[3]
    if not 0 < q0 < q_ph:
        raise RuntimeError("the thick seed does not lie below the requested q_ph")
    records, prev, failure = [], None, None
    for q in _flux_targets(q0, q_ph, flux_growth):
        step, failure = _fld_step(records, "flux_factor", qa_e, seed, q, k0, prev, fld_kw)
        prev = step if step is not None else prev
        if failure:
            break
    if prev is None:
        raise RuntimeError(f"FLD continuation failed at its first step: {failure}")

    # Stage 2: lower the opacity floor (direct continuation at low floor hits a fold).
    k_solved = k0
    if failure is None:
        for k in _floor_targets(k0, float(kappa_const)):
            trial = _prepare_inner(kappa_const=k, **inner_kw)
            step, failure = _fld_step(records, "kappa_const", trial[1], trial[2], q_ph, k, prev, fld_kw)
            if failure:
                break
            prev, inner, k_solved = step, trial, k

    qa, qa_e, seed, _ = inner
    profile = _combine_profile(qa_e, prev.frame, k_solved, tau_ph)
    a, d = qa.attrs, dict(prev.diagnostics)
    reached = bool(failure is None
                   and np.isclose(k_solved, kappa_const, rtol=1e-10, atol=1e-14)
                   and abs(d["outer_flux_factor"]/q_ph - 1) <= max(10*tol, 2e-4))
    d.update(
        photosphere_reached=reached, continuation_failure=failure,
        kappa_const=k_solved, kappa_const_requested=float(kappa_const), bootstrap_kappa_const=k0,
        eta_match_requested=float(eta_match),
        photosphere_flux_factor_target=float(q_ph), photosphere_tau_target=float(tau_ph),
        r_critical=float(a["r_c"]), v_critical=float(a["v_c"]), Mdot=float(a["Mdot"]),
        sonic_trapping_v_tau_over_c=float(a["trap"]),
        r_inner=float(profile["r"].iloc[0]), r_inner_requested=float(r_in),
        inward_integration_skipped=bool(a["inward_integration_skipped"]),
        inward_solution_reached=bool(a["inward_success"] and np.isclose(
            profile["r"].iloc[0], r_in, rtol=2e-8, atol=0)),
        production_radiation_transport=False,   # one-temperature grey closure (see top)
    )
    # QA valid (trapped sonic point) and QA/FLD overlap consistent.
    d["controlled_QA_diffusion_bridge"] = bool(
        d["sonic_trapping_v_tau_over_c"] >= 10 and d["thick_seam_ok"]
        and d["seam_temperature_ok"] and d["seam_luminosity_ok"] and d["seam_slopes_ok"]
        and d["diffusive_zone_supersonic"] and d["energy_conserved"])
    d["accepted"] = bool(
        reached and d["step_accepted"] and d["flux_factor_monotonic"]
        and d["controlled_QA_diffusion_bridge"]
        and (d["inward_solution_reached"] or d["inward_integration_skipped"]))
    profile.attrs.update(d)

    result = BridgeSolution(profile=profile, diagnostics=d, qa=qa, qa_energy=qa_e, seed=seed,
                            fld=prev, continuation=pd.DataFrame.from_records(records),
                            kappa_const=k_solved)
    if not d["accepted"]:
        msg = "bridge did not reach the requested surface on the controlled branch"
        if strict:
            raise RuntimeError(msg + (f": {failure}" if failure else ""))
        warnings.warn(msg, RuntimeWarning)
    return result


# ---------------------------------------------------------------------------
# Optional diagnostic: finite QA/FLD overlap
# ---------------------------------------------------------------------------
def _qa_branch(qa_e, r_match):
    """Outer part of the QA profile (r <= r_match) on which eta rises monotonically."""
    qa = qa_e.sort_values("r").drop_duplicates("r", keep="last")
    qa = qa[qa["r"] <= r_match*(1 + 1e-10)].reset_index(drop=True)
    eta = qa["eta"].to_numpy(float)
    # Walk inward from the seam while eta keeps decreasing (skips the tiny
    # non-monotone patch next to the sonic point).
    i = len(qa) - 1
    while i > 0 and eta[i-1] <= eta[i] + 1e-8*max(eta[i], eta[i-1], 1e-12):
        i -= 1
    branch = qa.iloc[i:].reset_index(drop=True)
    if len(branch) < 3:
        raise ValueError("could not isolate a monotone QA overlap branch")
    return branch


def _radius_at_eta(branch, eta_target):
    lnr, eta = np.log(branch["r"].to_numpy(float)), branch["eta"].to_numpy(float)
    if not eta[0] <= eta_target <= eta[-1]:
        raise ValueError(f"QA branch eta range [{eta[0]:.3g}, {eta[-1]:.3g}] misses {eta_target:g}")
    f = PchipInterpolator(lnr, eta)
    return float(np.exp(brentq(lambda s: float(f(s)) - eta_target, lnr[0], lnr[-1],
                               xtol=2e-13, rtol=2e-13)))


def check_overlap(sol, eta_inner=0.005, n_points=600, rtol=2e-10, atol=2e-12, max_dlnr=2e-3):
    """Integrate the exact FLD equations *inward* from the seam and compare with QA.

    Stops where eta_FLD = |(Z - (f1/f3)Y)/((f1/f3)Y)| falls to eta_inner.
    QA and FLD should agree to O(eta^2) over this overlap. The stitched wind
    itself is not changed. Luminosity is compared with the QA transport L_qad.
    """
    qa_e, fld, k = sol.qa_energy, sol.fld, sol.kappa_const
    seam = qa_e.iloc[-1]
    r_m = float(fld.frame["r"].iloc[0])
    Mdot, Edot = float(qa_e.attrs["Mdot"]), float(qa_e.attrs["Edot_target"])
    Ls, Ms = max(abs(Edot), hf.Lsun), float(seam["M"])
    branch = _qa_branch(qa_e, r_m)
    # Integrate toward the QA radius where eta = eta_inner/2 so the event is bracketed.
    r_guard = (_radius_at_eta(branch, 0.5*eta_inner) if branch["eta"].iloc[0] <= 0.5*eta_inner
               else float(branch["r"].iloc[0]))

    def local(s, y):
        return _fld_equations(s, y, Mdot=Mdot, Ls=Ls, Ms=Ms, kappa_const=k, protect=False)

    y0 = np.asarray(fld.solution.sol(0.0), float)
    if not eta_inner < float(local(np.log(r_m), y0)[1]["eta"]):
        raise ValueError("eta_inner must be below the FLD eta at the seam")

    def event(s, y):
        return float(local(s, y)[1]["eta"]) - eta_inner
    event.terminal, event.direction = True, -1

    ivp = solve_ivp(lambda s, y: local(s, y)[0], (np.log(r_m), np.log(r_guard)), y0,
                    method="DOP853", events=event, dense_output=True,
                    rtol=rtol, atol=atol, max_step=max_dlnr)
    if not ivp.success or len(ivp.t_events[0]) != 1:
        raise RuntimeError(f"inward FLD integration did not reach eta={eta_inner}: {ivp.message}")

    s = np.linspace(float(ivp.t_events[0][0]), np.log(r_m), int(n_points))
    y = ivp.sol(s)
    _, f = local(s, y)
    r = f["r"]
    qa_s = np.log(branch["r"].to_numpy(float))

    def qa_at(col, log=False):
        vals = branch[col].to_numpy(float)
        out = PchipInterpolator(qa_s, np.log(vals) if log else vals)(s)
        return np.exp(out) if log else out

    rho_qa, T_qa = qa_at("rho", log=True), qa_at("T", log=True)
    v_fld = np.sqrt(f["v2"])
    # Phi anchored at the common seam, so the FLD energy drift is an independent check.
    phi = cumulative_trapezoid(hf.G*f["M"]/r, s, initial=0.0)
    phi = float(seam["Phi"]) + phi - phi[-1]
    frame = pd.DataFrame({
        "r": r,
        "rho_QA": rho_qa, "rho_FLD": f["rho"],
        "T_QA": T_qa, "T_FLD": f["T"],
        "x_QA": hf.saha_x(rho_qa, T_qa), "x_FLD": hf.saha_x(f["rho"], f["T"]),
        "kappa_QA": hf.get_kappa(rho_qa, T_qa, k), "kappa_FLD": f["kappa"],
        "L_QA": qa_at("L_qad"), "L_FLD": f["L"], "L_energy_QA": qa_at("L"),
        "v_QA": Mdot/(4*np.pi*r**2*rho_qa), "v_FLD": v_fld,
        "M_QA": qa_at("M", log=True), "M_FLD": f["M"],
        "eta_QA": qa_at("eta"), "eta_FLD": f["eta"],
        "tau_FLD": y[4], "flux_factor_FLD": f["q"], "D_over_f2_FLD": f["D"]/f["f2"],
        "Edot_FLD": Mdot*hf.bernoulli(v_fld, f["rho"], f["T"], phi) + f["L"],
    })

    rows = []
    for field in ("rho", "T", "x", "kappa", "L", "v"):
        rel = frame[f"{field}_FLD"]/frame[f"{field}_QA"] - 1
        frame[f"{field}_relative_difference"] = rel
        rows.append(dict(field=field, max_abs_relative_difference=np.max(np.abs(rel)),
                         inner_relative_difference=rel.iloc[0],
                         seam_relative_difference=rel.iloc[-1]))
    summary = pd.DataFrame(rows).set_index("field")

    def monotonic(a):
        return bool(np.all(np.diff(a) >= -1e-8*np.max(a)))
    eta_f, eta_q = frame["eta_FLD"].to_numpy(), frame["eta_QA"].to_numpy()
    d = dict(
        r_inner=float(r[0]), r_match=r_m, radial_width=r_m - float(r[0]),
        eta_FLD_inner=eta_f[0], eta_QA_inner=eta_q[0], eta_FLD_match=eta_f[-1], eta_QA_match=eta_q[-1],
        eta_monotonic=monotonic(eta_f) and monotonic(eta_q),
        minimum_D_over_f2=float(frame["D_over_f2_FLD"].min()),
        maximum_abs_flux_factor=float(np.max(np.abs(f["q"]))),
        minimum_tau=float(np.min(y[4])),
        energy_relative_drift=float(np.max(np.abs(frame["Edot_FLD"] - Edot))/Ls),
        **{f"max_{k_}_relative_difference": float(v)
           for k_, v in summary["max_abs_relative_difference"].items()},
    )
    d["accepted"] = bool(
        d["eta_monotonic"] and d["minimum_D_over_f2"] > 0 and d["maximum_abs_flux_factor"] < 1
        and d["minimum_tau"] > 10
        and summary.loc[["rho", "T", "x", "kappa", "L"], "max_abs_relative_difference"].max() < 0.02)
    return Overlap(frame=frame, summary=summary, diagnostics=d)


if __name__ == "__main__":
    # Example: compact "Model B" (0.1 Msun core, trapped H-recombination wind).
    sol = solve_bridge(rho_c=1e-7, T_c=10**4.2, M_c=0.1*hf.Msun, r_in=10*hf.Rsun,
                       L_core=hf.Lsun, kappa_const=1e-2)
    d = sol.diagnostics
    print(pd.Series({
        "accepted": d["accepted"],
        "Mdot [Msun/yr]": d["Mdot"]*hf.year/hf.Msun,
        "r_c [Rsun]": d["r_critical"]/hf.Rsun,
        "sonic tau v/c": d["sonic_trapping_v_tau_over_c"],
        "r_match [Rsun]": d["r_match"]/hf.Rsun,
        "tau_match": d["tau_match"],
        "r_ph [Rsun]": d["r_outer"]/hf.Rsun,
        "T_ph [K]": d["outer_temperature"],
        "L_ph [Lsun]": d["outer_luminosity"]/hf.Lsun,
        "seam dT/T": d["seam_temperature_shift"],
        "seam dL/L": d["seam_luminosity_vs_transport"],
    }).to_string())
