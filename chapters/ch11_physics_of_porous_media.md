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

# 11. Physics of porous media

```{admonition} Core ideas
:class: tip
1. A soil is described at the scale of a **representative elementary volume** by its porosity, its water content and the distribution of its pore sizes.
2. Capillarity links pore size to suction (Young–Laplace). The **water retention curve** is therefore a picture of the pore-size distribution. Brooks–Corey and van Genuchten functions describe it with three or four parameters, and it is **hysteretic**.
3. **Darcy's law** is the laminar limit of flow through pores; a bundle-of-tubes argument gives the **Kozeny–Carman** permeability. Hydraulic conductivity depends on the fluid as well as the medium, through its viscosity.
4. Unsaturated conductivity falls by orders of magnitude as the soil dries. **Mualem's** model predicts it from the retention curve.
5. Combining Darcy's law with conservation of mass gives the **Richards equation**. Solved numerically, it shows why a draining soil appears to stop at **field capacity**: drainage does not stop, it slows down as the conductivity collapses.
6. Soil hydraulic properties are measured in the laboratory and the field, or estimated from texture with **pedotransfer functions**; their spatial variability is large.
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
g, rho_w = 9.81, 1000.0
```

## 11.1 Describing porous materials

A soil or aquifer material consists of solid particles, the pores between them, and the water and air that fill the pores. Its basic descriptors are:

- the **bulk density** $\rho_b$, the dry mass per unit total volume, and the **particle density** $\rho_s$ (about 2650 kg m⁻³ for mineral soils), which together give the **porosity** $n = 1 - \rho_b/\rho_s$;
- the **volumetric water content** $\theta$, between a residual value $\theta_r$ and the saturated value $\theta_s \approx n$; and the **effective saturation** $S_e = (\theta - \theta_r)/(\theta_s - \theta_r)$;
- the **texture**, the proportions of sand, silt and clay, and the **structure**, the arrangement of particles into aggregates, which creates larger pores between them;
- the **pore-size distribution**, which, more than the porosity itself, controls how the medium holds and transmits water.

All these quantities are defined as averages over a volume large enough to contain many pores, the **representative elementary volume** (Chapter 1). Below that scale, the medium is either solid or void; above it, its properties may vary gradually in space. Field soils often violate the ideal: **macropores** made by roots, worms and cracks can carry water past the matrix much faster than the averaged equations predict (Chapter 14).

## 11.2 Capillarity and the water retention curve

### Young–Laplace

Water wets mineral surfaces, and in a pore the air–water interface curves. The pressure jump across a curved interface is given by the **Young–Laplace** equation. For a cylindrical pore of radius $r$ and contact angle $\beta$, expressed as a pressure head,

\begin{equation*}
\psi = -\frac{2\sigma\cos\beta}{\rho_w g\, r},
\tag{11.1}
\end{equation*}

where $\sigma$ = 0.073 N m⁻¹ is the surface tension of water. A pore of radius 15 µm holds water against a suction of about 1 m; one of 0.1 µm against about 150 m, the conventional wilting point. As a soil dries, the largest pores empty first, so the water retention curve $\theta(\psi)$ is in effect the **cumulative pore-size distribution** read through Equation (11.1).

### Retention functions

Two families of functions are standard.

\begin{equation*}
\text{Brooks–Corey: } S_e = \begin{cases} \left(\dfrac{\psi_b}{\psi}\right)^{\lambda} & \psi \le \psi_b \\ 1 & \psi > \psi_b \end{cases}
\qquad
\text{van Genuchten: } S_e = \bigl[1 + (\alpha|\psi|)^{n_v}\bigr]^{-m},\ m = 1 - \tfrac{1}{n_v},
\tag{11.2}
\end{equation*}

where $\psi_b$ (negative) is the **air-entry** or bubbling pressure head, the suction at which the largest pores begin to drain, and $\lambda$ the pore-size distribution index: large for uniform sands, small for clays with a wide range of pore sizes. The van Genuchten function is smooth through saturation, which helps numerical models; Brooks–Corey has a sharp air-entry point, which is closer to the behavior of coarse, uniform media. For large suctions the two are nearly equivalent, with $\lambda \approx m n_v = n_v - 1$.

### Hysteresis

The retention curve depends on the history of wetting and drying. At the same suction, a draining soil holds more water than a wetting one. Two mechanisms contribute: pores with narrow necks (**ink-bottle effect**) empty only when the suction exceeds that of the neck, but fill only when it falls below that of the wider body; and the contact angle is larger on an advancing than on a receding interface. Air trapped during wetting also keeps the field-saturated water content below the porosity, typically at 80–90% of it. Hysteresis is often neglected in hydrological models, but it affects redistribution after rain and the water left in the soil between storms.

Figure 11.1 fits both functions to synthetic laboratory data, a typical first step in characterizing a soil.

```{code-cell} python
:tags: [hide-input]
from scipy.optimize import curve_fit
rng = np.random.default_rng(11)
def vg(psi, tr, ts, a, n):
    return tr + (ts - tr) / (1 + (a * np.abs(psi)) ** n) ** (1 - 1 / n)
def bc(psi, tr, ts, psib, lam):
    psi = np.abs(psi); return np.where(psi > psib, tr + (ts - tr) * (psib / psi) ** lam, ts)
true = dict(tr=0.07, ts=0.43, a=3.6, n=1.56)
psi_obs = -np.array([0.01, 0.05, 0.1, 0.2, 0.33, 0.5, 1.0, 2.0, 3.3, 5.0, 10, 30, 100, 150])
th_obs = vg(psi_obs, **true) + 0.006 * rng.standard_normal(len(psi_obs))
p_vg, _ = curve_fit(vg, psi_obs, th_obs, p0=(0.05, 0.4, 2, 1.5), bounds=([0, 0.3, 0.1, 1.05], [0.2, 0.6, 50, 5]))
p_bc, _ = curve_fit(bc, psi_obs, th_obs, p0=(0.05, 0.43, 0.2, 0.5), bounds=([0, 0.3, 0.01, 0.05], [0.2, 0.6, 5, 5]))
ps = -np.logspace(-2.5, 2.5, 400)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogy(th_obs, -psi_obs, "o", color=ink, ms=5, label="Laboratory data (synthetic)")
ax.semilogy(vg(ps, *p_vg), -ps, color=c1, lw=2.2, label=f"van Genuchten: α = {p_vg[2]:.2f} m⁻¹, n = {p_vg[3]:.2f}")
ax.semilogy(bc(ps, *p_bc), -ps, color=c2, lw=2.0, ls="--", label=f"Brooks–Corey: |ψ_b| = {p_bc[2]:.2f} m, λ = {p_bc[3]:.2f}")
ax.set_xlabel("Volumetric water content θ"); ax.set_ylabel("Suction |ψ| (m)")
ax.legend(loc="upper right", fontsize=9)
plt.show()
glue("vg_n", float(p_vg[3]), display=False); glue("bc_lam", float(p_bc[3]), display=False)
```

*Figure 11.1. Van Genuchten and Brooks–Corey functions fitted to synthetic retention data for a loam. The fitted van Genuchten $n_v$ is {glue:text}`vg_n:.2f` and the Brooks–Corey $\lambda$ is {glue:text}`bc_lam:.2f`, close to $n_v - 1$. The two curves differ mainly near saturation, where Brooks–Corey has a sharp air-entry point.*

## 11.3 Darcy's law

### From pores to Darcy

Water moves through pores slowly, at Reynolds numbers based on grain size well below 1 in most soils and aquifers, so inertia is negligible and the flow in each pore is laminar. In a tube of radius $r$, laminar flow follows the Hagen–Poiseuille law: the mean velocity is $r^2/(8\mu)$ times the pressure gradient (Chapter 5 of the [Hydraulics](https://hydroclimatol.com/hydraulics/chapters/ch05_pipe_flow_and_pipe_systems.html) book). Averaging over many tortuous pores gives **Darcy's law** (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#groundwater-from-darcy-to-theis)):

\begin{equation*}
\mathbf{q} = -\frac{k}{\mu}\,\nabla p^* = -K\,\nabla H, \qquad K = \frac{k\,\rho_w g}{\mu},
\tag{11.3}
\end{equation*}

where $\mathbf{q}$ is the Darcy flux (volume per unit total area per unit time), $H = \psi + z$ the hydraulic head, $k$ the **intrinsic permeability** (m²), a property of the medium alone, and $K$ the **hydraulic conductivity**, which also depends on the fluid. Because water's viscosity falls by about 40% between 5 °C and 25 °C, the same soil conducts water more than 1.5 times faster in summer than in winter (Example 11.2).

A bundle-of-tubes argument with a hydraulic radius proportional to $d\,n/(1 - n)$ for grains of diameter $d$ gives the **Kozeny–Carman** relation

\begin{equation*}
k = \frac{d^2}{180}\,\frac{n^3}{(1 - n)^2},
\tag{11.4}
\end{equation*}

which explains why permeability scales with the square of grain size and is so sensitive to porosity. It works for uniform sands and fails for clays and structured soils, whose flow is controlled by a few large pores.

Darcy's law breaks down at high flow rates, near wells in coarse gravel and in karst conduits, where inertia adds a term proportional to the square of the velocity (the Forchheimer correction).

### Unsaturated flow

In an unsaturated medium the same law holds with a conductivity that depends on the water content, $K(\psi)$ or $K(\theta)$, as proposed by Buckingham (1907):

\begin{equation*}
\mathbf{q} = -K(\psi)\,\nabla(\psi + z) .
\tag{11.5}
\end{equation*}

As the soil dries, water occupies only the smaller pores. The cross-section available for flow shrinks, and the flow in each remaining pore is slower, since it scales with the square of the radius. Conductivity therefore falls far faster than water content.

## 11.4 Conductivity functions

Measuring $K(\psi)$ directly is slow and difficult, so it is usually predicted from the retention curve by treating the medium as a bundle of tubes with the pore-size distribution implied by $\theta(\psi)$. **Mualem's** (1976) model gives the relative conductivity

\begin{equation*}
K_r(S_e) = S_e^{\,l}\left[\frac{\int_0^{S_e} \psi^{-1}\,dS}{\int_0^{1} \psi^{-1}\,dS}\right]^2 ,
\tag{11.6}
\end{equation*}

with a tortuosity–connectivity exponent $l \approx 0.5$. With the van Genuchten retention function the integrals have a closed form (van Genuchten, 1980), and with Brooks–Corey a power law:

\begin{equation*}
K_r = S_e^{1/2}\left[1 - \left(1 - S_e^{1/m}\right)^m\right]^2
\quad\text{(van Genuchten–Mualem)},
\qquad
K_r = S_e^{\,2.5 + 2/\lambda}
\quad\text{(Brooks–Corey–Mualem)} .
\tag{11.7}
\end{equation*}

Figure 11.2 shows the predicted conductivity for the two fits of Figure 11.1. Although the retention curves are nearly identical, the predicted conductivities differ by a factor of several over most of the range. Near saturation the van Genuchten–Mualem curve starts to fall at very small suctions, while Brooks–Corey holds $K = K_s$ up to the air-entry point. Conductivity near saturation is sensitive to the largest pores, which the retention data barely resolve. This is one reason why measured saturated or near-saturated conductivities should be used whenever available.

```{code-cell} python
:tags: [hide-input]
Ks = 25.0                                                          # mm/h, saturated conductivity
def Kr_vg(psi, tr, ts, a, n):
    m = 1 - 1 / n; Se = (1 + (a * np.abs(psi)) ** n) ** (-m)
    return Se ** 0.5 * (1 - (1 - Se ** (1 / m)) ** m) ** 2
def Kr_bc(psi, tr, ts, psib, lam):
    Se = np.where(np.abs(psi) > psib, (psib / np.abs(psi)) ** lam, 1.0)
    return Se ** (2.5 + 2 / lam)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.loglog(-ps, Ks * Kr_vg(ps, *p_vg), color=c1, lw=2.2, label="van Genuchten–Mualem")
ax.loglog(-ps, Ks * Kr_bc(ps, *p_bc), color=c2, lw=2.0, ls="--", label="Brooks–Corey–Mualem")
ax.set_xlabel("Suction |ψ| (m)"); ax.set_ylabel("K (mm h⁻¹)"); ax.set_ylim(1e-8, 50)
ax.legend(loc="lower left", fontsize=9.5)
plt.show()
r1 = Kr_bc(np.array([-0.3]), *p_bc)[0] / Kr_vg(np.array([-0.3]), *p_vg)[0]
glue("kr_ratio", float(r1), display=False)
```

*Figure 11.2. Unsaturated conductivity predicted with Mualem's model from the two retention fits of Figure 11.1, with $K_s$ = 25 mm h⁻¹. At a suction of 0.3 m the two predictions differ by a factor of {glue:text}`kr_ratio:.0f`, although the retention curves agree closely.*

## 11.5 Field equations

### The Richards equation

Conservation of mass in a volume of soil, $\partial\theta/\partial t = -\nabla\cdot\mathbf{q}$, combined with Equation (11.5) gives the **Richards equation** (Richards, 1931). In one vertical dimension, with $z$ positive upward,

\begin{equation*}
\frac{\partial \theta}{\partial t} = \frac{\partial}{\partial z}\left[K(\psi)\left(\frac{\partial \psi}{\partial z} + 1\right)\right].
\tag{11.8}
\end{equation*}

It can be written in three forms, each useful in different situations:

- the **mixed form** above, with $\theta$ and $\psi$ linked by the retention curve, which numerical schemes use to conserve mass;
- the **pressure-head form**, $C(\psi)\,\partial\psi/\partial t = \partial_z[K(\partial_z\psi + 1)]$ with the specific moisture capacity $C = d\theta/d\psi$, which also covers saturated zones;
- the **water-content form**, $\partial\theta/\partial t = \partial_z[D(\theta)\,\partial_z\theta] + \partial_z K(\theta)$, with the **soil water diffusivity** $D = K\,d\psi/d\theta$, which reveals the equation's character: nonlinear diffusion plus gravity-driven advection.

The water-content form leads to the similarity solutions for infiltration in Chapter 12. In saturated aquifers, $\theta$ is constant except for small elastic changes, and the equation becomes the groundwater flow equation of Chapter 13.

### Why field capacity

A classic numerical experiment explains a familiar concept. A deep soil column, saturated by rain, drains freely at its base, with no evaporation. The Richards equation is solved below with an implicit, mass-conservative scheme (modified Picard iteration; Celia et al., 1990). The column drains quickly at first; within a day or two the drainage rate has fallen by orders of magnitude, because the conductivity collapses as the large pores empty. The water content keeps falling, but ever more slowly, so that a few days after rain a measurement looks like an equilibrium (Figure 11.3). That apparent equilibrium is what agronomists call **field capacity**. It is not a soil constant: it depends on how long one waits, on the depth of the profile and on the layering below.

```{code-cell} python
:tags: [hide-input]
from scipy.linalg import solve_banded
soil = dict(tr=0.065, ts=0.41, a=7.5, n=1.89)                      # a sandy loam (illustrative)
Ks_ms = 1.2e-5                                                    # m/s (43 mm/h)
mm_ = 1 - 1 / soil["n"]
theta_f = lambda h: np.where(h < 0, soil["tr"] + (soil["ts"] - soil["tr"]) / (1 + (soil["a"] * np.abs(h)) ** soil["n"]) ** mm_, soil["ts"])
def K_f(h):
    Se = np.where(h < 0, (1 + (soil["a"] * np.abs(h)) ** soil["n"]) ** (-mm_), 1.0)
    return Ks_ms * Se ** 0.5 * (1 - (1 - Se ** (1 / mm_)) ** mm_) ** 2
def C_f(h, dh=1e-6):
    return np.where(h < 0, (theta_f(h + dh) - theta_f(h - dh)) / (2 * dh), 1e-9)

def richards_drainage(T_end, Lz=1.5, nz=100, dt0=30.0, save=(0, 0.1, 1, 2, 5, 20)):
    """Free drainage of an initially near-saturated column; z positive downward, modified Picard."""
    dz = Lz / nz; h = np.full(nz, -0.01); t, dt = 0.0, dt0
    saved = {0: theta_f(h).copy()}; next_save = [s * 86400 for s in save[1:]]
    times, rates = [], []
    while t < T_end - 1e-9:
        dt = min(dt, T_end - t, next_save[0] - t if next_save else T_end - t)
        th_old, hn = theta_f(h), h.copy()
        for it in range(60):
            K = K_f(hn); Kh = 0.5 * (K[1:] + K[:-1]); C = C_f(hn)
            diag = C / dt; rhs = -(theta_f(hn) - th_old) / dt + C * hn / dt
            up = np.zeros(nz); lo = np.zeros(nz)
            diag[1:] += Kh / dz ** 2; up[1:] = -Kh / dz ** 2; rhs[1:] += Kh / dz       # face above each cell
            diag[:-1] += Kh / dz ** 2; lo[:-1] = -Kh / dz ** 2; rhs[:-1] -= Kh / dz    # face below each cell
            rhs[-1] -= K[-1] / dz                                                        # free drainage at the base
            ab = np.zeros((3, nz)); ab[1] = diag; ab[0, 1:] = lo[:-1]; ab[2, :-1] = up[1:]
            h_new = solve_banded((1, 1), ab, rhs)
            done = np.max(np.abs(h_new - hn)) < 1e-6
            hn = h_new
            if done: break
        rates.append(K_f(hn)[-1] * 1000 * 86400); times.append(t + dt)
        h, t = hn, t + dt
        if next_save and abs(t - next_save[0]) < 1e-6:
            saved[next_save.pop(0) / 86400] = theta_f(h).copy()
        dt = min(dt * 1.25, 3600.0) if it < 6 else dt * 0.6
    return np.arange(nz) * dz + dz / 2, saved, np.array(times), np.array(rates)

zc, prof, tr_, qr_ = richards_drainage(20 * 86400)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 4.7))
for (day, th), col, ls in zip(prof.items(), [ink, c2, c1, c3, ink2, c1], ["-", "-", "-", "-", "-", "--"]):
    a1.plot(th, zc, color=col, ls=ls, lw=2.0 if day in (0, 20) else 1.4, label=f"{day:g} d")
a1.set_ylim(1.5, 0); a1.set_xlabel("θ"); a1.set_ylabel("Depth (m)"); fig.legend(loc="outside lower center", ncol=6, fontsize=9, handlelength=1.5, columnspacing=1.0)
a1.set_title("(a) Water content")
a2.loglog(tr_ / 86400, qr_, color=c1, lw=2.2)
a2.set_xlabel("Time (d)"); a2.set_ylabel("Drainage (mm d⁻¹)"); a2.set_title("(b) Drainage at 1.5 m")
plt.show()
dz = 1.5 / 100
glue("fc_q2", float(np.interp(2.0, tr_ / 86400, qr_)), display=False)
glue("fc_q20", float(qr_[-1]), display=False)
glue("fc_th2", float(prof[2][20]), display=False); glue("fc_th20", float(prof[20][20]), display=False)
```

*Figure 11.3. Free drainage of a 1.5 m sandy-loam column from near saturation, solved with the Richards equation. (a) Water-content profiles after 0 to 20 days. (b) The drainage rate at the base falls from hundreds of millimeters per day to {glue:text}`fc_q2:.1f` mm d⁻¹ after 2 days and {glue:text}`fc_q20:.2f` mm d⁻¹ after 20. Between days 2 and 20 the water content at 0.3 m depth falls from {glue:text}`fc_th2:.3f` to {glue:text}`fc_th20:.3f`, and it is still falling: drainage never stops, it only slows down.*

## 11.6 Measurement and estimation

**Water content** is measured by gravimetric sampling (the reference method), by electromagnetic sensors that exploit the high dielectric permittivity of water (time-domain reflectometry and capacitance probes), by neutron probes, and over large areas by cosmic-ray neutron sensors and microwave satellites. **Matric potential** is measured by tensiometers in the wet range (down to about −8 m) and by psychrometers or dielectric sensors in the dry range. The **retention curve** is measured on undisturbed cores with hanging columns and pressure plates; **saturated conductivity** with constant- or falling-head permeameters in the laboratory and with ring infiltrometers, tension infiltrometers or borehole permeameters in the field.

These measurements are slow and expensive, and soils vary greatly in space: saturated conductivity commonly has a coefficient of variation of 50–150% within a single field. In practice, hydraulic parameters are often estimated from more easily measured properties (texture, bulk density, organic matter) with **pedotransfer functions**, regression or machine-learning models fitted to large databases (for example Saxton and Rawls, 2006; Schaap et al., 2001). Their predictions are useful first estimates, but their uncertainty, especially for $K_s$, is about an order of magnitude.

## 11.7 Worked examples

**Example 11.1 — Pore sizes from suction.** What is the largest pore radius still filled with water at suctions of 0.1 m, 3.4 m (field capacity) and 153 m (wilting point), for a contact angle of zero at 20 °C?

```{code-cell} python
sigma = 0.0728
for s in (0.1, 3.4, 153.0):
    print(f"|ψ| = {s:6.1f} m -> r = {2 * sigma / (rho_w * g * s) * 1e6:8.3f} µm")
```

**Example 11.2 — Permeability, conductivity and temperature.** Estimate the permeability of a uniform sand with grain diameter 0.3 mm and porosity 0.38 with Kozeny–Carman, and its hydraulic conductivity at 5 °C and 25 °C (viscosities 1.52 × 10⁻³ and 0.89 × 10⁻³ Pa s).

```{code-cell} python
d, nporo = 0.3e-3, 0.38
k = d ** 2 / 180 * nporo ** 3 / (1 - nporo) ** 2
for T, mu in [(5, 1.52e-3), (25, 0.89e-3)]:
    K = k * rho_w * g / mu
    print(f"T = {T:2d} °C: K = {K * 1000 * 3600:.0f} mm/h ({K:.2e} m/s)")
print(f"k = {k:.2e} m²; ratio K(25)/K(5) = {1.52 / 0.89:.2f}")
```

**Example 11.3 — Diffusivity from the retention and conductivity functions.** For the van Genuchten–Mualem fit of Figure 11.1 with $K_s$ = 25 mm h⁻¹, compute the soil water diffusivity $D = K\,d\psi/d\theta$ at $S_e$ = 0.3, 0.6 and 0.9.

```{code-cell} python
tr_f, ts_f, a_f, n_f = p_vg
m_f = 1 - 1 / n_f
for Se in (0.3, 0.6, 0.9):
    psi = -((Se ** (-1 / m_f) - 1) ** (1 / n_f)) / a_f
    dpsi = 1e-5
    dth = (vg(psi + dpsi, *p_vg) - vg(psi - dpsi, *p_vg)) / (2 * dpsi)
    K = Ks / 1000 / 3600 * Kr_vg(np.array([psi]), *p_vg)[0]
    print(f"Se = {Se}: ψ = {psi:7.3f} m, D = {K / dth:.2e} m²/s")
```

The diffusivity rises by orders of magnitude toward saturation, which is why wetting fronts in dry soil are sharp.

## 11.8 Common mistakes

- **Treating porosity as the controlling property.** Clays have higher porosity than sands but far lower conductivity; the pore-size distribution matters.
- **Mixing permeability and conductivity.** $k$ (m²) belongs to the medium; $K$ (m s⁻¹) also depends on the fluid's viscosity and density.
- **Extrapolating fitted retention functions to the conductivity near saturation.** Use a measured near-saturated conductivity.
- **Calling field capacity a constant.** It is the slow tail of drainage and depends on time, depth and layering.
- **Ignoring macropores and hysteresis** when matrix-based parameters fail to reproduce observed fast flow or water left after drying.

## 11.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("z1", float(2 * 0.0728 / (rho_w * g * 10e-6)), display=False)
glue("z2", 0.5e-3 ** 2 / 180 * 0.35 ** 3 / (1 - 0.35) ** 2, display=False)
glue("z3", float(1 - 1450 / 2650), display=False)
glue("z4", float(0.5 ** (2.5 + 2 / 0.4)), display=False)
glue("z5", float(1.0e-11 * rho_w * g / 1.0e-3 * 1000 * 3600), display=False)
```

1. What suction is needed to empty a pore of radius 10 µm?
2. Estimate the Kozeny–Carman permeability of a sand with 0.5 mm grains and porosity 0.35.
3. A soil has a bulk density of 1450 kg m⁻³. Estimate its porosity.
4. With Brooks–Corey–Mualem and λ = 0.4, what is the relative conductivity at an effective saturation of 0.5?
5. A fine sand has an intrinsic permeability of 10⁻¹¹ m². What is its hydraulic conductivity in mm h⁻¹ at 20 °C ($\mu$ = 1.0 × 10⁻³ Pa s)?

```{admonition} Answers
:class: dropdown
1. About {glue:text}`z1:.2f` m.
2. $k$ = {glue:text}`z2:.2e` m².
3. $n$ = {glue:text}`z3:.2f`.
4. $K_r$ = {glue:text}`z4:.4f`.
5. About {glue:text}`z5:.0f` mm h⁻¹.
```

## 11.10 Open questions

- **Upscaling hydraulic properties.** What effective $K(\psi)$ and $\theta(\psi)$ should a model use for a field or a grid cell with heterogeneous soils and macropores?
- **Dual-domain behavior.** How should preferential flow in macropores be coupled to slow matrix flow, and how can its parameters be measured?
- **Beyond the Richards equation.** Gravity-driven fingering, air entrapment and non-equilibrium between water content and suction are not captured by the standard equation.

## 11.11 Key papers

- Brooks, R. H., and Corey, A. T. (1964). *Hydraulic Properties of Porous Media*. Hydrology Paper 3, Colorado State University.
- Celia, M. A., Bouloutas, E. T., and Zarba, R. L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. *Water Resources Research*, 26, 1483–1496.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. *Water Resources Research*, 12, 513–522.
- Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. *Physics*, 1, 318–333.
- Saxton, K. E., and Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. *Soil Science Society of America Journal*, 70, 1569–1578.
- Schaap, M. G., Leij, F. J., and van Genuchten, M. Th. (2001). Rosetta: a computer program for estimating soil hydraulic parameters with hierarchical pedotransfer functions. *Journal of Hydrology*, 251, 163–176.
- van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44, 892–898.

## 11.12 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The chapter on fluid mechanics of flow in porous materials.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on principles of subsurface flow.
- Hillel, D. (1998). *Environmental Soil Physics*. Academic Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
