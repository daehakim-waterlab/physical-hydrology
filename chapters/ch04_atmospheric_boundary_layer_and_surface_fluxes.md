---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 4. The atmospheric boundary layer and surface fluxes

```{admonition} Core ideas
:class: tip
1. The **atmospheric boundary layer** is the lowest kilometer or two of the atmosphere, where the surface is felt through turbulence. Its lowest tenth, the **surface layer**, has nearly constant fluxes with height, which makes simple similarity laws possible there.
2. In neutral conditions, dimensional analysis gives the **logarithmic profile**: wind speed, temperature and humidity vary with the logarithm of height, scaled by the friction velocity and the surface roughness.
3. **Monin–Obukhov similarity** adds one length scale, the **Obukhov length**, that measures the importance of buoyancy relative to shear. Universal functions of $z/L$ correct the profiles and fluxes for stability.
4. The bulk transfer equations and the **aerodynamic resistance** used in every evaporation formula are integrated forms of these profiles. Unstable conditions reduce the resistance; stable conditions increase it greatly.
5. The **surface energy budget** is the boundary condition that links the fluxes. Eddy-covariance measurements typically leave 10–30% of the available energy unaccounted for, the **closure problem**.
6. Over **heterogeneous surfaces**, each change in roughness or wetness creates an **internal boundary layer** that grows downwind. A flux measured at a given height represents an upwind **footprint**, not the ground beneath the instrument.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from myst_nb import glue
%config InlineBackend.figure_format = "retina"

c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical colors
ink, ink2, bg = "#0b0b0b", "#52514e", "#fcfcfb" # text, secondary text, background
plt.rcParams.update({
    "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
    "text.color": ink, "axes.labelcolor": ink2, "axes.edgecolor": ink2,
    "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
```

## 4.1 Structure of the boundary layer

The surface heats, cools, moistens and slows the air above it, and turbulence spreads these effects upward. The layer in which this happens within about an hour is the **atmospheric boundary layer** (ABL). Its depth changes through the day (Figure 4.1).

- **Daytime.** Over land on a sunny day, heating from below drives convection that mixes heat and vapor through a **mixed layer** 1–2 km deep, capped by an inversion through which the layer grows by entraining air from above (Figure 3.1).
- **Night.** After sunset the ground cools by radiation, and a shallow **stable boundary layer**, often only 100–300 m deep, forms near the surface. Turbulence there is weak and intermittent. Above it, the remains of the daytime mixed layer persist as a **residual layer**.

Close to the ground, in roughly the lowest 10% of the ABL, the **surface layer** has a special property: the vertical fluxes of momentum, heat and vapor change by less than about 10% with height, because the layer is thin compared with the distance over which storage and advection matter. With nearly constant fluxes, the profiles of wind, temperature and humidity depend only on those fluxes, the height and the surface. This is the basis of the similarity theory that follows. Directly above tall vegetation or buildings, the flow is disturbed by individual roughness elements in a **roughness sublayer** a few element heights deep, where the simple theory does not apply.

```{code-cell} python
:tags: [hide-input]
hours = np.linspace(0, 24, 241)
day = np.clip(np.sin(np.pi * (hours - 6) / 13), 0, None)
h_mix = 150 + 1450 * np.clip(np.sin(np.pi * np.clip(hours - 7, 0, 12) / 24) ** 1.3, 0, None) * (hours < 19)
h_mix = np.maximum(h_mix, 150)
h_stable = np.where((hours < 7) | (hours > 18.5), 200.0, np.nan)
h_resid = np.where((hours < 8) | (hours > 18.5), 1600.0, np.nan)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.fill_between(hours, 0, h_mix, where=(hours >= 7) & (hours <= 18.5), color=c2, alpha=0.20)
ax.plot(hours[(hours >= 7) & (hours <= 18.5)], h_mix[(hours >= 7) & (hours <= 18.5)], color=c2, lw=2.2)
ax.fill_between(hours, 0, h_stable, color=c1, alpha=0.22)
ax.fill_between(hours, h_stable, h_resid, color=ink2, alpha=0.08)
ax.text(12.5, 650, "convective\nmixed layer", ha="center", fontsize=10, color=ink)
ax.text(2.5, 80, "stable layer", ha="center", fontsize=9.5, color=c1)
ax.text(21.5, 80, "stable layer", ha="center", fontsize=9.5, color=c1)
ax.text(3.0, 900, "residual layer", ha="center", fontsize=9.5, color=ink2)
ax.text(21.3, 900, "residual\nlayer", ha="center", fontsize=9.5, color=ink2)
ax.text(12.5, 1700, "free atmosphere", ha="center", fontsize=9.5, color=ink2)
ax.set_xlim(0, 24); ax.set_ylim(0, 2000); ax.set_xticks(range(0, 25, 6))
ax.set_xlabel("Local time (h)"); ax.set_ylabel("Height (m)")
plt.show()
```

*Figure 4.1. Schematic diurnal evolution of the boundary layer over land on a clear day. Daytime heating drives a convective mixed layer that grows to about 1.6 km by mid-afternoon; at night a shallow stable layer forms near the ground beneath the residual layer left by the previous day.*

## 4.2 The logarithmic profile

### Dimensional analysis

In the neutral surface layer over a uniform surface, the only quantities on which the wind shear $\partial\overline{u}/\partial z$ can depend are the height $z$ and the momentum flux, expressed through the **friction velocity** $u_* = (\tau/\rho)^{1/2}$. The only combination with the dimensions of a shear is $u_*/z$, so

\begin{equation*}
\frac{\partial \overline{u}}{\partial z} = \frac{u_*}{\kappa\, z},
\tag{4.1}
\end{equation*}

where $\kappa \approx 0.40$ is the von Kármán constant, determined by experiment. Integrating from the height where the extrapolated wind vanishes,

\begin{equation*}
\overline{u}(z) = \frac{u_*}{\kappa}\,\ln\!\left(\frac{z - d_0}{z_{0m}}\right),
\tag{4.2}
\end{equation*}

where $z_{0m}$ is the **roughness length for momentum** and $d_0$ the **displacement height**, the level at which the canopy appears to absorb the momentum. For vegetation of height $h_c$, typical values are $d_0 \approx 0.67\,h_c$ and $z_{0m} \approx 0.1\,h_c$; for open water $z_{0m}$ is about 0.1–1 mm and depends on the waves.

### Scalars

The same reasoning applies to potential temperature and specific humidity, with scales $\theta_* = -\overline{w'\theta'}/u_*$ and $q_* = -\overline{w'q'}/u_*$:

\begin{equation*}
\overline{\theta}(z) - \theta_s = \frac{\theta_*}{\kappa}\ln\!\left(\frac{z - d_0}{z_{0h}}\right),
\qquad
\overline{q}(z) - q_s = \frac{q_*}{\kappa}\ln\!\left(\frac{z - d_0}{z_{0v}}\right),
\tag{4.3}
\end{equation*}

where $\theta_s$ and $q_s$ are surface values and $z_{0h}$, $z_{0v}$ are the **scalar roughness lengths**. Scalars are not transferred by pressure forces at the surface, as momentum is by form drag on roughness elements, so their roughness lengths are smaller than $z_{0m}$, often expressed as $\kappa B^{-1} = \ln(z_{0m}/z_{0h}) \approx 2$ for vegetation, and larger over bluff, sparse elements.

## 4.3 Monin–Obukhov similarity

### The Obukhov length

When the surface heats or cools the air, buoyancy adds or removes turbulent energy (Chapter 3), and the profiles depart from the logarithmic form. Monin and Obukhov (1954) argued that in the surface layer the only additional parameter is the buoyancy flux $(g/\overline{\theta}_v)\,\overline{w'\theta_v'}$. Combined with $u_*$, it defines one new length scale, the **Obukhov length**

\begin{equation*}
L = -\frac{u_*^3\,\overline{\theta}_v}{\kappa\, g\,\overline{w'\theta_v'}} ,
\tag{4.4}
\end{equation*}

roughly the height above which buoyancy dominates the production of turbulence. $L$ is negative in unstable conditions (upward heat flux), positive in stable conditions and infinite in neutral ones. The dimensionless height $\zeta = (z - d_0)/L$ is the **stability parameter**.

### Universal functions

Similarity then requires the dimensionless gradients to be universal functions of $\zeta$:

\begin{equation*}
\frac{\kappa (z - d_0)}{u_*}\frac{\partial \overline{u}}{\partial z} = \phi_m(\zeta),
\qquad
\frac{\kappa (z - d_0)}{\theta_*}\frac{\partial \overline{\theta}}{\partial z} = \phi_h(\zeta),
\tag{4.5}
\end{equation*}

with $\phi_v = \phi_h$ for vapor, to a good approximation. The functions are determined by field experiments; the widely used **Businger–Dyer** forms are

\begin{equation*}
\phi_m = (1 - 16\zeta)^{-1/4},\quad \phi_h = (1 - 16\zeta)^{-1/2} \quad (\zeta < 0);
\qquad
\phi_m = \phi_h = 1 + 5\zeta \quad (0 \le \zeta \le 1).
\tag{4.6}
\end{equation*}

Integrating gives profiles of the form

\begin{equation*}
\overline{u}(z) = \frac{u_*}{\kappa}\left[\ln\frac{z - d_0}{z_{0m}} - \psi_m(\zeta) + \psi_m\!\left(\frac{z_{0m}}{L}\right)\right],
\tag{4.7}
\end{equation*}

and similarly for $\theta$ and $q$ with $\psi_h$, where for unstable conditions (Paulson, 1970), with $x = (1 - 16\zeta)^{1/4}$,

\begin{equation*}
\psi_m = 2\ln\frac{1 + x}{2} + \ln\frac{1 + x^2}{2} - 2\arctan x + \frac{\pi}{2},
\qquad
\psi_h = 2\ln\frac{1 + x^2}{2},
\tag{4.8}
\end{equation*}

and $\psi_m = \psi_h = -5\zeta$ for stable conditions. Unstable stratification makes the profiles flatter above the lowest meters, because convection mixes the air efficiently; stable stratification makes them steeper (Figure 4.2).

```{code-cell} python
:tags: [hide-input]
kappa, g = 0.40, 9.81
def psi_m(zeta):
    zeta = np.asarray(zeta, float)
    x = (1 - 16 * np.minimum(zeta, 0)) ** 0.25
    unst = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
    return np.where(zeta < 0, unst, -5 * zeta)
def psi_h(zeta):
    zeta = np.asarray(zeta, float)
    x = (1 - 16 * np.minimum(zeta, 0)) ** 0.25
    return np.where(zeta < 0, 2 * np.log((1 + x ** 2) / 2), -5 * zeta)

z0m, ustar = 0.05, 0.35
zz = np.logspace(np.log10(0.2), np.log10(50), 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 4.2))
for L, col, lab in [(-15.0, c2, "unstable, L = −15 m"), (np.inf, ink, "neutral"), (40.0, c1, "stable, L = 40 m")]:
    zeta = zz / L if np.isfinite(L) else 0 * zz
    u = ustar / kappa * (np.log(zz / z0m) - psi_m(zeta) + (psi_m(z0m / L) if np.isfinite(L) else 0))
    a1.plot(u, zz, color=col, lw=2.2, label=lab)
a1.set_yscale("log"); a1.set_xlabel("Wind speed (m s⁻¹)"); a1.set_ylabel("Height (m)")
a1.set_title("(a) Profiles, $u_*$ = 0.35 m s⁻¹"); a1.legend(loc="lower right", fontsize=8.5)
zeta = np.linspace(-2, 1, 300)
phim = np.where(zeta < 0, (1 - 16 * np.minimum(zeta, 0)) ** -0.25, 1 + 5 * zeta)
phih = np.where(zeta < 0, (1 - 16 * np.minimum(zeta, 0)) ** -0.5, 1 + 5 * zeta)
a2.plot(zeta, phim, color=c1, lw=2.2, label="φ$_m$")
a2.plot(zeta, phih, color=c2, lw=2.2, ls="--", label="φ$_h$")
a2.axvline(0, color=ink2, lw=0.8)
a2.set_xlabel("ζ = z/L"); a2.set_title("(b) Universal functions"); a2.legend(loc="upper left", fontsize=9.5)
a2.set_ylim(0, 6)
plt.show()
```

*Figure 4.2. (a) Wind profiles over grassland ($z_{0m}$ = 5 cm) with the same friction velocity under three stabilities, on a logarithmic height axis where neutral profiles are straight lines. (b) The Businger–Dyer universal functions: gradients are reduced in unstable conditions (ζ < 0) and enhanced in stable ones.*

## 4.4 Bulk transfer and aerodynamic resistance

### From profiles to fluxes

Evaporation formulas need the flux, not the profile. Writing Equations (4.7) and its scalar counterpart between the surface and a measurement height $z$, and eliminating $u_*$, $\theta_*$ and $q_*$, gives the **bulk transfer** equations

\begin{equation*}
E = \rho\,C_E\,\overline{u}(z)\,\bigl(q_s - \overline{q}(z)\bigr),
\qquad
H = \rho\,c_p\,C_H\,\overline{u}(z)\,\bigl(\theta_s - \overline{\theta}(z)\bigr),
\tag{4.9}
\end{equation*}

with transfer coefficients

\begin{equation*}
C_E = \frac{\kappa^2}{\left[\ln\dfrac{z - d_0}{z_{0m}} - \psi_m(\zeta)\right]\left[\ln\dfrac{z - d_0}{z_{0v}} - \psi_h(\zeta)\right]} ,
\tag{4.10}
\end{equation*}

and $C_H$ the same with $z_{0h}$. The **aerodynamic resistance** is $r_a = 1/(C_E\,\overline{u})$ (s m⁻¹), so that $E = \rho\,(q_s - \overline{q})/r_a$. In neutral conditions this is the formula used in the FAO-56 Penman–Monteith equation (Chapter 7).

### Solving for stability

Because $L$ depends on the fluxes, and the fluxes on $L$, the equations are solved by iteration: start with neutral conditions, compute the fluxes, compute $L$, update $\psi$, and repeat until $L$ converges (Example 4.2). Figure 4.3 shows the result: a moderate surface-to-air temperature difference changes the aerodynamic resistance by a factor of two or more. In very stable conditions turbulence may collapse altogether, and the formulas are unreliable.

```{code-cell} python
:tags: [hide-input]
def bulk_fluxes(u, z, Ts, Ta, z0m, z0h, d0=0.0, rho=1.2, cp=1005.0, n_iter=60):
    """Sensible heat flux, friction velocity, Obukhov length and r_a (for heat) by iteration."""
    L = np.inf
    for _ in range(n_iter):
        zeta = (z - d0) / L if np.isfinite(L) else 0.0
        zeta = float(np.clip(zeta, -5, 1))
        us = kappa * u / (np.log((z - d0) / z0m) - psi_m(zeta) + (psi_m(z0m / L) if np.isfinite(L) else 0))
        ra = (np.log((z - d0) / z0h) - psi_h(zeta)) / (kappa * us)
        H = rho * cp * (Ts - Ta) / ra
        wT = H / (rho * cp)
        L_new = -us ** 3 * (Ta + 273.15) / (kappa * g * wT) if abs(wT) > 1e-9 else np.inf
        if np.isfinite(L) and np.isfinite(L_new) and abs(L_new - L) < 1e-4 * abs(L): L = L_new; break
        L = L_new
    return H, us, L, ra

dT = np.linspace(-4, 10, 71)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for u, col in [(1.5, c2), (3.0, c1), (6.0, c3)]:
    ra = [bulk_fluxes(u, 2.0, 20.0 + d, 20.0, 0.012, 0.0012)[3] for d in dT]
    ra0 = bulk_fluxes(u, 2.0, 20.0001, 20.0, 0.012, 0.0012)[3]
    ax.plot(dT, np.array(ra) / ra0, color=col, lw=2.2, label=f"u(2 m) = {u} m s⁻¹")
ax.axhline(1, color=ink2, lw=0.8, ls=":"); ax.axvline(0, color=ink2, lw=0.8)
ax.set_xlabel("Surface minus air temperature (K)"); ax.set_ylabel("$r_a$ / $r_a$(neutral)")
ax.set_yscale("log"); ax.set_ylim(0.4, 5); ax.legend(loc="upper right", fontsize=9.5)
ax.set_yticks([0.5, 1, 2, 3, 4], ["0.5", "1", "2", "3", "4"]); ax.minorticks_off()
plt.show()
```

*Figure 4.3. Aerodynamic resistance for heat over short grass ($z_{0m}$ = 1.2 cm, $z_{0h}$ = 1.2 mm, measurements at 2 m), relative to its neutral value, as a function of the surface-to-air temperature difference. A warm surface (unstable air) lowers the resistance, most strongly in light winds; a cold surface raises it steeply.*

## 4.5 The surface energy budget as a boundary condition

### The budget and its terms

The fluxes of Section 4.4 cannot be chosen freely: at the surface they must satisfy the energy budget

\begin{equation*}
R_n - G = H + \lambda E ,
\tag{4.11}
\end{equation*}

where storage in a thin surface layer, photosynthesis and advection are neglected. This budget closes the problem of evaporation. In the bulk formulas, $q_s$ and $\theta_s$ are linked to the surface temperature, which adjusts until the energy budget is satisfied, as the combination equations of Chapter 7 exploit.

### Ground heat flux

The soil conducts heat according to the diffusion equation $\partial T/\partial t = \kappa_s\,\partial^2 T/\partial z^2$, where $\kappa_s$ is the thermal diffusivity (about 0.3–0.8 × 10⁻⁶ m² s⁻¹ for moist soils). If the surface temperature varies sinusoidally with amplitude $A_0$ and angular frequency $\omega$ (daily or yearly), the solution is a damped wave:

\begin{equation*}
T(z, t) = \overline{T} + A_0\,e^{-z/D_T}\sin\!\left(\omega t - \frac{z}{D_T}\right),
\qquad
D_T = \sqrt{\frac{2\kappa_s}{\omega}} ,
\tag{4.12}
\end{equation*}

where the **damping depth** $D_T$ is about 10 cm for the daily cycle and 2 m for the annual cycle. The ground heat flux at the surface, $G = -k_s\,\partial T/\partial z|_0$, leads the surface temperature by an eighth of a cycle (three hours for the daily wave), and its daily mean is close to zero. Over a day it matters for the timing of fluxes; over months it matters only in the seasonal storage and release of heat by deep water bodies and soils.

### The closure problem

When all four terms of Equation (4.11) are measured independently, $H + \lambda E$ from eddy covariance is typically 10–30% smaller than $R_n - G$ (Foken, 2008). Proposed causes include:

- **Large, slow eddies and secondary circulations** that a single tower and a 30-minute average do not capture (Chapter 3);
- **storage of heat** in the canopy and the soil above the heat-flux plate;
- **Footprint mismatch**: the radiometer, the soil sensors and the eddy-covariance system see different areas.

When eddy-covariance evaporation is used to test models, the residual is often distributed in proportion to the measured Bowen ratio, which assumes that the missing energy is partitioned like the measured fluxes. Whether that is right remains an open question.

## 4.6 Heterogeneous surfaces

### Internal boundary layers

Real landscapes are patchworks of fields, forests, towns and water. When air crosses a boundary between two surfaces, the change propagates upward through a new **internal boundary layer** whose depth grows with distance $x$ from the edge, roughly as

\begin{equation*}
\delta(x) \approx 0.75\, z_{0}\left(\frac{x}{z_{0}}\right)^{0.8},
\tag{4.13}
\end{equation*}

where $z_0$ is the roughness length of the downwind surface (Elliott, 1958). Only within the lowest 10% or so of this layer are the profiles in equilibrium with the new surface. A rule of thumb follows: a flux measured at height $z$ represents the local surface only if the **fetch**, the uniform upwind distance, is at least about 100 times $z$.

### Footprints

Turned around, the same idea defines the **footprint** of a measurement: the upwind area that contributes to the flux measured at a given height. For neutral conditions, a simple analytical model (Schuepp et al., 1990) gives the cumulative fraction of the measured flux that comes from within a distance $x$ upwind as

\begin{equation*}
\mathcal{F}(x) = \exp\!\left[-\frac{\overline{u}\,(z - d_0)}{\kappa\,u_*\,x}\right].
\tag{4.14}
\end{equation*}

The contributions peak at a distance $x_{\max} = \overline{u}(z - d_0)/(2\kappa u_*)$ and then decline slowly, so a substantial part of the flux comes from far upwind (Figure 4.4). Measurements higher above the surface see a larger, more distant area; unstable conditions shrink the footprint and stable conditions stretch it. At scales larger than a few kilometers, the boundary layer responds to the landscape as a whole, and fluxes over a patchwork can be represented by effective parameters above a **blending height** where the influence of individual patches has been mixed away.

```{code-cell} python
:tags: [hide-input]
x = np.linspace(1, 1500, 3000)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for zm, col in [(2.0, c3), (5.0, c1), (10.0, c2)]:
    d0, z0 = 0.08, 0.015
    us = 0.3; ubar = us / kappa * np.log((zm - d0) / z0)
    a = ubar * (zm - d0) / (kappa * us)
    F = np.exp(-a / x)
    ax.plot(x, F, color=col, lw=2.2, label=f"z = {zm:g} m")
    glue(f"fp80_{int(zm)}", float(-a / np.log(0.8)), display=False)
ax.axhline(0.8, color=ink2, lw=0.8, ls=":")
ax.set_xlabel("Upwind distance x (m)"); ax.set_ylabel("Cumulative fraction of flux")
ax.set_xlim(0, 1500); ax.set_ylim(0, 1); ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 4.4. Cumulative flux footprint over short grass in neutral conditions for three measurement heights. Eighty percent of the flux measured at 2 m comes from within {glue:text}`fp80_2:.0f` m upwind, but at 10 m the same fraction requires {glue:text}`fp80_10:.0f` m.*

## 4.7 Worked examples

**Example 4.1 — Fitting a logarithmic profile.** Wind speeds measured over a grass field in near-neutral conditions are 2.55, 3.05, 3.42 and 3.74 m s⁻¹ at 0.5, 1, 2 and 4 m. Assuming $d_0$ = 0.07 m, find $u_*$ and $z_{0m}$.

```{code-cell} python
zm = np.array([0.5, 1.0, 2.0, 4.0]); um = np.array([2.55, 3.05, 3.42, 3.74]); d0 = 0.07
slope, intercept = np.polyfit(np.log(zm - d0), um, 1)
us = kappa * slope; z0 = np.exp(-intercept / slope)
print(f"u* = {us:.3f} m/s, z0m = {z0 * 100:.2f} cm, shear stress = {1.2 * us ** 2:.3f} N/m²")
print("residuals (m/s):", np.round(um - (slope * np.log(zm - d0) + intercept), 3))
```

**Example 4.2 — Stability and the heat flux.** At 2 m above short grass the wind speed is 2.0 m s⁻¹ and the air is 24 °C, while the surface radiometric temperature is 31 °C. Taking $z_{0m}$ = 1.2 cm and $z_{0h}$ = 1.2 mm, compute the sensible heat flux with and without the stability correction, and the Obukhov length.

```{code-cell} python
H, us, L, ra = bulk_fluxes(2.0, 2.0, 31.0, 24.0, 0.012, 0.0012)
ra_n = np.log(2.0 / 0.0012) * np.log(2.0 / 0.012) / (kappa ** 2 * 2.0)
H_n = 1.2 * 1005 * 7.0 / ra_n
print(f"Neutral:  r_a = {ra_n:.0f} s/m, H = {H_n:.0f} W/m²")
print(f"With MOST: r_a = {ra:.0f} s/m, H = {H:.0f} W/m², u* = {us:.3f} m/s, L = {L:.1f} m, z/L = {2 / L:.2f}")
```

Ignoring stability underestimates the heat flux by about a quarter on this light-wind afternoon.

**Example 4.3 — How deep does the daily heat wave reach?** A moist loam has a thermal diffusivity of 0.6 × 10⁻⁶ m² s⁻¹ and a thermal conductivity of 1.2 W m⁻¹ K⁻¹. The surface temperature varies daily with an amplitude of 10 K. Find the damping depth, the amplitude at 20 cm, the time lag at 20 cm, and the amplitude of the surface ground heat flux.

```{code-cell} python
kap_s, k_s, A0 = 0.6e-6, 1.2, 10.0
w_day = 2 * np.pi / 86400
D_T = np.sqrt(2 * kap_s / w_day)
G_amp = k_s * A0 * np.sqrt(2) / D_T                 # |dT/dz| at the surface has amplitude A0·√2/D_T
print(f"D_T = {D_T * 100:.1f} cm; amplitude at 20 cm = {A0 * np.exp(-0.2 / D_T):.2f} K; "
      f"lag at 20 cm = {0.2 / D_T / w_day / 3600:.1f} h; G amplitude = {G_amp:.0f} W/m²")
```

**Example 4.4 — Fetch requirement.** An eddy-covariance system will be mounted at 3 m over an irrigated field 250 m long in the direction of the prevailing wind. Using Equation (4.14) with $u_*$ = 0.3 m s⁻¹, $z_{0m}$ = 1.5 cm and $d_0$ = 8 cm, what fraction of the measured flux comes from the field?

```{code-cell} python
zm, d0, z0, us = 3.0, 0.08, 0.015, 0.3
ubar = us / kappa * np.log((zm - d0) / z0)
F250 = np.exp(-ubar * (zm - d0) / (kappa * us * 250.0))
print(f"u(3 m) = {ubar:.2f} m/s; fraction from within 250 m = {F250:.2f}; "
      f"peak contribution at {ubar * (zm - d0) / (2 * kappa * us):.0f} m")
```

About a third of the measured flux comes from beyond the field. If the surrounding land is dry, the measured evaporation underestimates that of the irrigated crop.

## 4.8 Lab (planned)

A Colab notebook will fit logarithmic and Monin–Obukhov profiles to tower measurements of wind, temperature and humidity at several heights from an open flux-network site, estimate $u_*$, $z_{0m}$ and $L$, and compare the bulk-transfer heat flux with the eddy-covariance measurement.

## 4.9 Common mistakes

- **Using the logarithmic profile outside neutral conditions.** In daytime over land, stability corrections change fluxes by tens of percent (Example 4.2).
- **Setting $z_{0h} = z_{0m}$.** Heat and vapor are transferred less efficiently than momentum near the surface; ignoring this overestimates the sensible heat flux.
- **Applying surface-layer theory inside the roughness sublayer** of forests or cities.
- **Assuming the ground heat flux is negligible at hourly scales.** It can reach 100 W m⁻² at midday over bare soil, though it nearly cancels over a day.
- **Ignoring the footprint.** A tower near a field edge measures a mixture of surfaces.

## 4.10 Problems

```{code-cell} python
:tags: [remove-cell]
glue("s1_us", float(0.4 * 4.0 / np.log(10 / 0.1)), display=False)
glue("s2_L", float(-(0.3 ** 3) * 300 / (0.4 * 9.81 * (150 / (1.2 * 1005)))), display=False)
glue("s3", float(np.sqrt(2 * 0.5e-6 / (2 * np.pi / (365.25 * 86400)))), display=False)
glue("s4", 0.75 * 0.5 * (300 / 0.5) ** 0.8, display=False)
ub5 = 0.25 / 0.4 * np.log((4 - 0.1) / 0.02)
glue("s5", float(ub5 * (4 - 0.1) / (2 * 0.4 * 0.25)), display=False)
```

1. Over a surface with $z_{0m}$ = 0.1 m and $d_0$ = 0, the wind at 10 m is 4.0 m s⁻¹ in neutral conditions. Find $u_*$.
2. With $u_*$ = 0.3 m s⁻¹, a sensible heat flux of 150 W m⁻² (air density 1.2 kg m⁻³, $\overline{\theta}_v$ ≈ 300 K), find the Obukhov length. Is a measurement at 2 m in the dynamic (shear-dominated) or convective regime?
3. A dry sandy soil has a thermal diffusivity of 0.5 × 10⁻⁶ m² s⁻¹. Find the damping depth of the annual temperature wave.
4. Wind blows from a meadow onto a forest ($z_0$ = 0.5 m). Estimate the depth of the internal boundary layer 300 m downwind of the forest edge.
5. For a measurement at 4 m over a surface with $z_{0m}$ = 2 cm, $d_0$ = 0.1 m and $u_*$ = 0.25 m s⁻¹, at what distance upwind is the footprint contribution largest?

```{admonition} Answers
:class: dropdown
1. $u_*$ = {glue:text}`s1_us:.3f` m s⁻¹.
2. $L$ = {glue:text}`s2_L:.1f` m; $z/L$ ≈ −0.12, near the dynamic regime but already noticeably unstable.
3. About {glue:text}`s3:.1f` m.
4. About {glue:text}`s4:.0f` m.
5. About {glue:text}`s5:.0f` m.
```

## 4.11 Open questions

- **Universal functions in complex conditions.** Monin–Obukhov theory assumes horizontal homogeneity and stationarity. How should it be modified over slopes, tall canopies and cities, and in very stable nights?
- **The energy imbalance.** Is the missing 10–30% carried mainly by large-scale circulations, and how should evaporation measurements be corrected?
- **Effective parameters for heterogeneous landscapes.** What roughness and resistance should a model grid cell of several kilometers use over a mosaic of surfaces?

## 4.12 Key papers

- Businger, J. A., Wyngaard, J. C., Izumi, Y., and Bradley, E. F. (1971). Flux-profile relationships in the atmospheric surface layer. *Journal of the Atmospheric Sciences*, 28, 181–189.
- Dyer, A. J. (1974). A review of flux-profile relationships. *Boundary-Layer Meteorology*, 7, 363–372.
- Foken, T. (2008). The energy balance closure problem: an overview. *Ecological Applications*, 18, 1351–1367.
- Monin, A. S., and Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Geofizicheskogo Instituta, Akademiya Nauk SSSR*, 24(151), 163–187 (in Russian).
- Paulson, C. A. (1970). The mathematical representation of wind speed and temperature profiles in the unstable atmospheric surface layer. *Journal of Applied Meteorology*, 9, 857–861.
- Schuepp, P. H., Leclerc, M. Y., MacPherson, J. I., and Desjardins, R. L. (1990). Footprint prediction of scalar fluxes from analytical solutions of the diffusion equation. *Boundary-Layer Meteorology*, 50, 355–373.

## 4.13 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The sections on the atmospheric boundary layer, similarity, the energy-budget constraint and disturbed boundary layers.
- Brutsaert, W. (1982). *Evaporation into the Atmosphere*. Reidel.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on principles of surface–atmosphere exchange.
- Garratt, J. R. (1992). *The Atmospheric Boundary Layer*. Cambridge University Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
