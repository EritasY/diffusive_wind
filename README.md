# QA–FLD wind bridge

Steady, spherical, radiation-trapped wind from a stellar core, solved in two matched zones:

1. **Quasi-adiabatic (QA) zone**: crosses the trapped adiabatic sonic point and fixes $\dot M$.
2. **Grey flux-limited-diffusion (FLD) zone**: exact one-temperature radiation-hydrodynamics from the hand-over radius to a free outer surface ($\tau = 2/3$, $F/cE = 1/4$).

Pure-hydrogen LTE gas (Saha ionisation + radiation), analytic Rosseland opacity, cgs units throughout.

## Files

| File | Contents |
|---|---|
| `helper_func.py` | Shared physics: constants, Saha, EOS ($f_1,f_2,f_3,c_s^2$), opacity, diffusion coefficient, LP flux limiter, $\tau$, $\Phi$, Bernoulli |
| `quasi_adiabatic.py` | Stand-alone QA wind: sonic point, $\dot M$, inward/outward integration to $\eta_{\rm stop}$ |
| `qa_fld_bridge.py` | Energy anchor, FLD free-boundary BVP, continuation, stitched profile, overlap check |

## Install

```bash
pip install -r requirements.txt     # numpy, scipy, pandas, astropy, matplotlib
```

## Quick start

Run from inside this folder (each run takes a few seconds):

```bash
python quasi_adiabatic.py   # QA zone only
python qa_fld_bridge.py     # full QA + FLD wind
```

The bridge example ("Model B": $M_c = 0.1\,M_\odot$, $\rho_c = 10^{-7}$, $T_c = 10^{4.2}$ K, $\kappa_0 = 10^{-2}$) should print

```
accepted                 True
Mdot [Msun/yr]        0.17351
r_c [Rsun]          32.356539
sonic tau v/c        61.06667
r_match [Rsun]     221.385481
tau_match         2074.317676
r_ph [Rsun]       5376.926606
T_ph [K]           830.633593
L_ph [Lsun]       12399.37676
...
```

## Example in Python

```python
import matplotlib.pyplot as plt
import helper_func as hf
from qa_fld_bridge import solve_bridge, check_overlap

sol = solve_bridge(rho_c=1e-7, T_c=10**4.2, M_c=0.1*hf.Msun,   # sonic-point state
                   r_in=10*hf.Rsun, L_core=hf.Lsun,              # inner base
                   kappa_const=1e-2)                             # opacity floor [cm^2/g]
p, d = sol.profile, sol.diagnostics
print(d["accepted"], d["Mdot"]*hf.year/hf.Msun, d["r_outer"]/hf.Rsun)

r = p["r"]/hf.Rsun
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
ax[0].loglog(r, p["rho"]); ax[0].set_ylabel(r"$\rho$ [g cm$^{-3}$]")
ax[1].loglog(r, p["T"]);   ax[1].set_ylabel("T [K]")
ax[2].loglog(r, p["v"]/1e5, label="v")
ax[2].loglog(r, p["c_s_ad"]/1e5, "--", label=r"$c_{s,\rm ad}$")
ax[2].set_ylabel("speed [km/s]"); ax[2].legend()
for a in ax:
    a.axvline(d["r_critical"]/hf.Rsun, color="gray", ls=":")    # sonic point
    a.axvline(d["r_match"]/hf.Rsun, color="gray", ls="--")      # QA -> FLD seam
    a.set_xlabel(r"$r/R_\odot$")
plt.tight_layout(); plt.show()

print(check_overlap(sol).summary)   # optional: QA vs inward-FLD agreement near the seam
```

QA alone:

```python
from quasi_adiabatic import solve_qa_wind
qa = solve_qa_wind(1e-7, 10**4.2, 0.1*hf.Msun, kappa_const=1e-2, eta_stop=0.01)
qa.attrs["Mdot"], qa.attrs["r_c"], qa.attrs["trap"]   # trap = tau v/c at the sonic point
```

Setting `kappa_const` below `bootstrap_kappa_const` (default $1.5\times10^{-3}$) switches on opacity-floor continuation (about 20 s). For example, `kappa_const=1e-3` gives $r_{\rm ph}\approx 574\,R_\odot$.

## Physics

With $s=\ln r$, $Y = d\ln\rho/ds$, $Z = d\ln T/ds$ and $v = \dot M/4\pi r^2\rho$:

$$
(v^2 - f_2)\,Y = f_1 Z + \frac{GM_r}{r} - 2v^2,\qquad
\frac{dL}{ds} = \dot M\,(f_1Y - f_3Z),\qquad
\frac{dM_r}{ds} = 4\pi r^3\rho,
$$

with $f_1 = (T/\rho)(\partial P/\partial T)_\rho$, $f_2 = (\partial P/\partial\rho)_T$, $f_3 = T(\partial e/\partial T)_\rho$ from the total (gas + ionisation + radiation) EOS, and

$$
\kappa_R = 0.4\,x + 8\times10^{22}\rho T^{-7/2}x^2 + \kappa_0 .
$$

**QA zone.** If the sonic point is trapped ($\tau v/c \gg 1$), write $Z = (f_1/f_3)Y + Q$ and expand in $Q$:

$$
L_0 = -A\frac{f_1}{f_3}Y,\quad Q = -\frac{1}{\dot M f_3}\frac{dL_0}{ds},\quad
(v^2 - c_s^2)\,Y = f_1Q + \frac{GM_r}{r} - 2v^2,
$$

where $A = 16\pi acrT^4/3\kappa\rho$ and $c_s^2 = f_2 + f_1^2/f_3$. Because $L$ is now algebraic, the stiff diffusion mode disappears. Regularity at $v_c = c_s$ fixes $r_c$ and therefore $\dot M = 4\pi r_c^2\rho_c v_c$. The error is $O(\eta^2)$ with $\eta = |Q/((f_1/f_3)Y)|$, and the zone ends where $\eta = \eta_{\rm match}$.

**Energy anchor.** $\dot E = \dot M B(r_{\rm in}) + L_{\rm core}$ with $B = v^2/2 + h + \Phi$ is held fixed through both zones.

**FLD zone.** Levermore–Pomraning closure: $q = L/(4\pi r^2 c a T^4) = \lambda(\mathcal R)\mathcal R$, $\lambda = (2+\mathcal R)/(6+3\mathcal R+\mathcal R^2)$, $Z = -\kappa\rho r\mathcal R/4$. The BVP has 5 ODEs $(\ln\rho,\ln T,L,M_r,\tau)$ and a free outer radius, so it needs 6 boundary conditions:

| Seam ($r_{\rm match}$) | Outer surface |
|---|---|
| $\rho = \rho_{\rm QA}$, $M_r = M_{\rm QA}$, $\dot M B + L = \dot E$ | $\tau = 2/3$, $q = 1/4$, $dL/ds = 0$ |

The seam $T$, the seam $L$ and $r_{\rm ph}$ are eigen-quantities selected by the BVP. They are not copied from QA, which suppresses the growing luminosity mode. The outer $q$ is raised to $1/4$ by continuation.

## Outputs

`sol.profile` (pandas DataFrame) contains `r, M, rho, T, x, v, c_s_ad, v_esc, kappa, L, tau, flux_factor, eddington_factor, zone, ...`. In the QA zone, `L` is the transport luminosity; use `L_energy_constraint` for energy audits.

Key flags in `sol.diagnostics`:

- `accepted`: surface reached on the controlled branch with all checks passed.
- `controlled_QA_diffusion_bridge`: trapped sonic point ($\tau v/c \ge 10$), thick seam, small seam corrections, $v^2 > f_2$ outside, energy conserved.
- `production_radiation_transport = False`: always false (see below).

## Limitations

- One temperature with $E = aT^4$ and isotropic LTE radiation enthalpy, so the $q = 1/4$ surface is a grey closure, not a true photosphere. A two-temperature model with a separate absorption opacity is the next step.
- QA requires a trapped sonic point. For $\tau v/c \lesssim 1$, solve a direct gas + radiation transonic problem instead.
- $\dot M$ is set by the QA sonic point and is not re-selected by the outer BVP.
- The outer structure depends strongly on $\kappa_0$, so always report it.
