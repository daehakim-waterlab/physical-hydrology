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

# 5. Precipitation

```{admonition} Core ideas
:class: tip
1. Cloud droplets form on small soluble particles at supersaturations of a fraction of a percent. The **Köhler curve** explains why: curvature raises, and dissolved salt lowers, the vapor pressure needed for a droplet to grow.
2. Growth by vapor diffusion alone is far too slow to make raindrops in the lifetime of a cloud. **Collision–coalescence** in warm clouds and the **ice-crystal process** in cold clouds bridge the gap of a factor of a million in mass.
3. The **drop-size distribution** links what a radar measures (reflectivity, proportional to $D^6$) to what a hydrologist wants (rain rate, proportional to about $D^{3.7}$). The empirical **Z–R relation** follows from it, and so does its uncertainty.
4. Precipitation is produced by lifting: in fronts, in convective cells, over mountains, in tropical cyclones and in **atmospheric rivers**. A simple **upslope model** captures the first-order pattern of orographic rain and the rain shadow.
5. Rainfall fields are correlated in space over distances that shrink with the duration of accumulation. **Kriging** uses that correlation to interpolate gauge data optimally and to quantify the error of the estimate.
6. All gauges undercatch, most strongly for snow in wind. Heavy short-duration rainfall tends to increase with temperature at about the **Clausius–Clapeyron rate** or faster.
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

## 5.1 Cloud microphysics

### Nucleation: the Köhler curve

A cloud forms when rising air cools to saturation (Chapter 3). But a pure water droplet cannot form directly from vapor at the saturation vapor pressure of a flat surface. Molecules at a curved surface are less tightly bound, so the equilibrium vapor pressure over a droplet of radius $r$ exceeds that over a flat surface (the **Kelvin effect**). A droplet a few molecules across would need a supersaturation of hundreds of percent to survive. In the real atmosphere droplets form instead on **cloud condensation nuclei**, aerosol particles of sea salt, sulfate and organic matter. Dissolved solute lowers the equilibrium vapor pressure (**Raoult's law**), opposing the curvature effect. The combination is the **Köhler curve**:

\begin{equation*}
S_{eq}(r) - 1 \approx \frac{a}{r} - \frac{b}{r^3},
\qquad a = \frac{2\sigma_w}{\rho_w R_v T} \approx \frac{3.3\times10^{-7}}{T}\ \text{m},
\qquad b = \frac{3\, i\, m_s M_w}{4\pi \rho_w M_s},
\tag{5.1}
\end{equation*}

where $S_{eq}$ is the equilibrium saturation ratio $e/e^*$, $\sigma_w$ the surface tension of water, $m_s$ and $M_s$ the mass and molar mass of the dissolved solute, $M_w$ the molar mass of water and $i$ the number of ions per molecule (2 for sodium chloride). The curve has a maximum at the **critical radius** $r_c = \sqrt{3b/a}$, where the **critical supersaturation** is $S_c - 1 = \sqrt{4a^3/27b}$ (Figure 5.1). If the ambient supersaturation exceeds this value, the droplet is **activated** and grows freely; otherwise it stays as a small haze droplet. Larger nuclei activate at lower supersaturation, which is why clouds in clean maritime air, with few but large nuclei, have fewer and larger droplets than clouds in polluted continental air.

```{code-cell} python
:tags: [hide-input]
T_k = 293.15
a_k = 3.3e-7 / T_k
Mw, Ms, i_vh, rho_w = 0.018015, 0.05844, 2.0, 1000.0          # NaCl
r = np.logspace(-8, -5, 600)                                    # m
fig, ax = plt.subplots(figsize=(6.4, 4.0))
crit = {}
for ms, col in [(1e-19, c2), (1e-18, c1), (1e-17, c3)]:
    b_k = 3 * i_vh * ms * Mw / (4 * np.pi * rho_w * Ms)
    S = 100 * (a_k / r - b_k / r ** 3)
    ok = S > -1.0
    ax.semilogx(r[ok] * 1e6, S[ok], color=col, lw=2.2, label=f"NaCl, {ms:.0e} kg")
    rc, Sc = np.sqrt(3 * b_k / a_k), np.sqrt(4 * a_k ** 3 / (27 * b_k))
    ax.plot(rc * 1e6, Sc * 100, "o", color=col, ms=6)
    crit[ms] = (rc, Sc)
ax.axhline(0, color=ink2, lw=0.8)
ax.set_xlabel("Droplet radius (µm)"); ax.set_ylabel("Equilibrium supersaturation (%)")
ax.set_xlim(0.01, 10); ax.set_ylim(-0.5, 0.6); ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("kc_r1", float(crit[1e-19][0] * 1e6), display=False); glue("kc_s1", float(crit[1e-19][1] * 100), display=False)
glue("kc_r3", float(crit[1e-17][0] * 1e6), display=False); glue("kc_s3", float(crit[1e-17][1] * 100), display=False)
```

*Figure 5.1. Köhler curves for droplets formed on sodium chloride nuclei of three masses at 20 °C. The dots mark the critical radius and supersaturation. The smallest nucleus must reach {glue:text}`kc_s1:.2f`% supersaturation (at a radius of {glue:text}`kc_r1:.2f` µm) to activate, the largest only {glue:text}`kc_s3:.3f`% (at {glue:text}`kc_r3:.1f` µm).*

### Growth by diffusion

Once activated, a droplet grows by diffusion of vapor toward its surface, while the latent heat released must be conducted away. Both processes limit the rate, and together they give

\begin{equation*}
r\,\frac{dr}{dt} = G\,(S - 1)
\quad\Longrightarrow\quad
r^2(t) = r_0^2 + 2\,G\,(S - 1)\,t ,
\tag{5.2}
\end{equation*}

where $G \approx 10^{-10}$ m² s⁻¹ at temperatures near 10 °C. Because the growth rate falls as $1/r$, diffusion makes droplets of 10–20 µm within minutes, but a 1 mm raindrop would take days at typical supersaturations of 0.1–0.5% (Example 5.2). Clouds last minutes to hours, so another process must be at work.

### Collision–coalescence and the ice-crystal process

In **warm clouds**, a few droplets that are slightly larger than the rest fall faster, sweep up smaller droplets in their path and grow by **coalescence**. Because the collection rate increases with the drop's cross-section and fall speed, growth accelerates. In **cold clouds** with temperatures below 0 °C, ice crystals and supercooled droplets coexist. The saturation vapor pressure over ice is lower than over water, so air that is just saturated with respect to the droplets is about 10% supersaturated with respect to ice at −10 °C. The crystals grow rapidly at the droplets' expense (the **Wegener–Bergeron–Findeisen** process), then aggregate into snowflakes or collect droplets as rime. Most precipitation outside the tropics begins as ice, even when it reaches the ground as rain.

### Fall speed

A falling drop accelerates until drag balances gravity. Small droplets obey **Stokes' law**, $v_t = 2 g r^2 (\rho_w - \rho_a)/(9\mu)$, so a 10 µm droplet falls at about 1 cm s⁻¹. Larger drops are deformed by the airflow, and their terminal velocity levels off near 9 m s⁻¹; an empirical fit for raindrops of diameter $D$ (mm) is $v_t \approx 9.65 - 10.3\,e^{-0.6 D}$ m s⁻¹ (Atlas et al., 1973). Raindrops larger than about 5 mm are unstable and break up.

## 5.2 The drop-size distribution and radar

A volume of rain contains drops of many sizes. Marshall and Palmer (1948) found that the number of drops per unit volume and per unit diameter interval decreases exponentially with diameter,

\begin{equation*}
N(D) = N_0\, e^{-\Lambda D}, \qquad N_0 = 8000\ \text{m}^{-3}\,\text{mm}^{-1}, \qquad \Lambda = 4.1\,R^{-0.21}\ \text{mm}^{-1},
\tag{5.3}
\end{equation*}

with $R$ the rain rate in mm h⁻¹: heavier rain has relatively more large drops. Two moments of this distribution matter. A weather radar measures the **reflectivity factor**, $Z = \int N(D)\,D^6\,dD$ (mm⁶ m⁻³), because small particles scatter microwaves in proportion to the square of their volume. The rain rate is the volume flux of water,

\begin{equation*}
R = \frac{\pi}{6}\int N(D)\,D^3\,v_t(D)\,dD .
\tag{5.4}
\end{equation*}

Since $Z$ weights large drops much more heavily than $R$ does, the two are related only through the shape of the distribution. Computing both moments for Equation (5.3) gives a power law $Z = a R^b$ (Example 5.3). Real distributions vary from storm to storm, between convective and stratiform rain, and with height, so no single Z–R relation is correct; a 50% error in radar rain rate from the Z–R conversion alone is common. Dual-polarization radars, which measure the flattening of large drops, reduce this uncertainty.

## 5.3 Precipitation systems

### Lifting mechanisms

As in Chapter 3, precipitation requires air to rise and cool, and the mechanism of lifting shapes the rainfall:

- **Frontal and cyclonic systems** lift air over hundreds of kilometers at a few centimeters per second and give widespread, moderate rain lasting hours to days.
- **Convective storms** lift air in updrafts of 1–10 m s⁻¹ or more, over a few kilometers, and give intense rain lasting tens of minutes. They may organize into lines and **mesoscale convective systems** that last many hours.
- **Orographic lifting** forces air over mountains; it enhances both frontal and convective rain on windward slopes.
- **Tropical cyclones** combine deep convection with an organized inflow that converges moisture from a wide area of warm ocean.

### Atmospheric rivers and the monsoon

Much of the moisture transported poleward outside the tropics moves in narrow, long filaments of strong vapor flux, a few hundred kilometers wide and thousands long, called **atmospheric rivers** (Zhu and Newell, 1998). They are usually identified by an integrated vapor transport above about 250 kg m⁻¹ s⁻¹. When an atmospheric river meets a mountain range, the orographic lift can produce extreme multi-day precipitation. In East Asia, the moisture plume along the summer monsoon front (Chapter 2) plays a similar role, and intense rainfall in Korea and Japan is often associated with such plumes feeding embedded convective systems.

### An upslope model of orographic precipitation

A simple model captures the main pattern of orographic rain. Air flowing at speed $U$ over terrain $h(x)$ is lifted at $w = U\,dh/dx$. If the air is saturated and rises along a moist adiabat, the rate at which vapor condenses in the column above a point is approximately proportional to the lifting rate and to the saturation vapor density near the surface, $\rho_{vs}$. A fraction $\epsilon$ of the condensate falls out, giving the **upslope precipitation**

\begin{equation*}
P(x) = \epsilon\, \rho_{vs}\, U \frac{dh}{dx} \quad \text{for } \frac{dh}{dx} > 0, \qquad P = 0 \text{ otherwise}.
\tag{5.5}
\end{equation*}

Rain then falls where the slope is steepest, upwind of the crest, and stops on the lee side. In reality the condensate takes 10–30 minutes to grow into precipitation and fall, so the pattern is carried downwind by a distance $U\tau_f$, and part of the precipitation spills over the crest. Figure 5.2 shows the pattern with and without this delay. More complete linear models (Smith and Barstad, 2004) add the airflow dynamics and the growth time of hydrometeors explicitly.

```{code-cell} python
:tags: [hide-input]
x = np.linspace(-60e3, 60e3, 1201)                         # m
h = 1500.0 * np.exp(-(x / 15e3) ** 2)                       # Gaussian ridge, m
U, eps_p = 12.0, 0.4
es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
rho_vs = es(15.0) * 1000 / (461.5 * (15 + 273.15))          # kg/m³ at 15 °C
P_up = eps_p * rho_vs * U * np.clip(np.gradient(h, x), 0, None) * 3600     # mm/h
tau_f = 1200.0                                               # s, time for condensate to fall out
shift = U * tau_f
P_drift = np.interp(x - shift, x, P_up, left=0, right=0)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
a1.plot(x / 1e3, P_up, color=c1, lw=2.2, label="Upslope model")
a1.plot(x / 1e3, P_drift, color=c2, lw=2.2, ls="--", label=f"Carried downwind for {tau_f / 60:.0f} min")
a1.set_ylabel("P (mm h⁻¹)"); a1.legend(loc="upper left", fontsize=9.5); a1.set_ylim(0, None)
a1.annotate("", xy=(-35, a1.get_ylim()[1] * 0.40), xytext=(-55, a1.get_ylim()[1] * 0.40),
            arrowprops=dict(arrowstyle="-|>", color=ink2)); a1.text(-55, a1.get_ylim()[1] * 0.46, f"wind {U:.0f} m s⁻¹", fontsize=9, color=ink2)
a2.fill_between(x / 1e3, 0, h, color=ink2, alpha=0.25); a2.plot(x / 1e3, h, color=ink, lw=1.4)
a2.set_ylabel("Elevation (m)"); a2.set_xlabel("Distance (km)"); a2.set_xlim(-60, 60)
plt.show()
crest = np.abs(x).argmin()
glue("up_max", float(P_up.max()), display=False)
glue("up_lee", float(P_drift[x > 0].sum() / P_drift.sum() * 100), display=False)
```

*Figure 5.2. Precipitation over a 1500 m Gaussian ridge from the upslope model (air at 15 °C, wind 12 m s⁻¹, 40% precipitation efficiency). The upslope model gives a maximum of {glue:text}`up_max:.1f` mm h⁻¹ halfway up the windward slope and nothing on the lee. Allowing 20 minutes for the condensate to fall carries the pattern downwind, so that {glue:text}`up_lee:.0f`% of the precipitation falls beyond the crest.*

## 5.4 Space–time structure and areal estimation

### Spatial correlation

Rainfall at two gauges is correlated, and the correlation decreases with their separation $d$. For a given accumulation period it is often well described by an exponential function,

\begin{equation*}
\rho(d) = \exp\!\left(-\frac{d}{\lambda}\right),
\tag{5.6}
\end{equation*}

with a **correlation length** $\lambda$ that grows with the duration of accumulation: a few kilometers for 10-minute rainfall in convective storms, tens of kilometers for daily totals, and hundreds for monthly totals. This is the statistical counterpart of the areal reduction factors used in design (see the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch11_design_rainfall.html) book): over short durations, the rainfall at a point is a poor guide to the areal mean.

### Kriging

Interpolation methods such as inverse distance weighting or Thiessen polygons choose weights by geometry alone. **Kriging** chooses them to minimize the expected squared error, given a model of the spatial correlation. The usual tool is the **semivariogram**,

\begin{equation*}
\gamma(d) = \tfrac{1}{2}\,\mathrm{E}\bigl[(P(\mathbf{x}) - P(\mathbf{x} + \mathbf{d}))^2\bigr] = \sigma^2\,\bigl[1 - \rho(d)\bigr] ,
\tag{5.7}
\end{equation*}

estimated from pairs of gauges and fitted with a model such as Equation (5.6). **Ordinary kriging** estimates the value at an unsampled point as $\hat{P}_0 = \sum_i \lambda_i P_i$, with weights that sum to one and minimize the error variance. They are the solution of the linear system

\begin{equation*}
\sum_j \lambda_j\,\gamma(d_{ij}) + \mu = \gamma(d_{i0}) \quad (i = 1, \dots, n), \qquad \sum_j \lambda_j = 1 ,
\tag{5.8}
\end{equation*}

where $\mu$ is a Lagrange multiplier. The method also gives the **kriging variance**, $\sigma_K^2 = \sum_i \lambda_i \gamma(d_{i0}) + \mu$, a map of the interpolation error that is small near gauges and large in data gaps. Kriging with external drift, or regression kriging, adds covariates such as elevation or radar estimates. **Merging radar and gauges** in this way combines the radar's spatial pattern with the gauges' accuracy, and is now standard in operational precipitation products.

Figure 5.3 compares kriging with inverse distance weighting on a synthetic rain field whose true values are known everywhere.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(55)
ng = 36; Lx = 60.0                                         # 36 × 36 grid over 60 km
gx = np.linspace(0, Lx, ng); GX, GY = np.meshgrid(gx, gx)
pts = np.c_[GX.ravel(), GY.ravel()]
lam_c, sig_ln = 15.0, 0.6                                  # correlation length (km), log-std
dd = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
Cmat = np.exp(-dd / lam_c) + 1e-8 * np.eye(len(pts))
field = 30.0 * np.exp(sig_ln * (np.linalg.cholesky(Cmat) @ rng.standard_normal(len(pts))) - sig_ln ** 2 / 2)
field_map = field.reshape(ng, ng)                          # daily rain, mm

n_g = 25
gid = rng.choice(len(pts), n_g, replace=False)
gxy, gval = pts[gid], field[gid]

def variogram(d, sill, lam):
    return sill * (1 - np.exp(-d / lam))
def ordinary_kriging(xy, z, targets, sill, lam):
    n = len(z)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    K = np.ones((n + 1, n + 1)); K[:n, :n] = variogram(D, sill, lam); K[n, n] = 0
    d0 = np.sqrt(((xy[:, None, :] - targets[None, :, :]) ** 2).sum(-1))
    rhs = np.ones((n + 1, len(targets))); rhs[:n] = variogram(d0, sill, lam)
    sol = np.linalg.solve(K, rhs)
    est = sol[:n].T @ z
    var = (sol[:n] * rhs[:n]).sum(0) + sol[n]
    return est, var
def idw(xy, z, targets, p=2):
    d0 = np.sqrt(((xy[:, None, :] - targets[None, :, :]) ** 2).sum(-1)) + 1e-9
    w = 1 / d0 ** p
    return (w * z[:, None]).sum(0) / w.sum(0)

sill = np.var(gval)
P_ok, V_ok = ordinary_kriging(gxy, gval, pts, sill, lam_c)
P_idw = idw(gxy, gval, pts)
rmse = lambda est: np.sqrt(np.mean((est - field) ** 2))
fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))
vmax = field.max()
for ax, data, title in [(axes[0, 0], field, "(a) True field"), (axes[0, 1], P_ok, "(b) Ordinary kriging"),
                        (axes[1, 0], P_idw, "(c) Inverse distance"), (axes[1, 1], np.sqrt(np.clip(V_ok, 0, None)), "(d) Kriging std. error")]:
    im = ax.pcolormesh(GX, GY, data.reshape(ng, ng), cmap="Blues" if "error" not in title else "Oranges",
                       vmin=0, vmax=vmax if "error" not in title else None, shading="auto")
    ax.plot(gxy[:, 0], gxy[:, 1], "k.", ms=4)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.75, label="mm")
plt.show()
glue("rmse_ok", float(rmse(P_ok)), display=False); glue("rmse_idw", float(rmse(P_idw)), display=False)
glue("mean_true", float(field.mean()), display=False); glue("mean_ok", float(P_ok.mean()), display=False)
glue("mean_g", float(gval.mean()), display=False)
```

*Figure 5.3. A synthetic daily rain field (lognormal, correlation length 15 km) sampled by 25 gauges (dots). Over the whole 60 km × 60 km area, the root-mean-square error is {glue:text}`rmse_ok:.1f` mm for ordinary kriging and {glue:text}`rmse_idw:.1f` mm for inverse distance weighting. The kriging standard error (d) is lowest at the gauges and highest in gaps between them. The true areal mean is {glue:text}`mean_true:.1f` mm; kriging gives {glue:text}`mean_ok:.1f` mm and the simple mean of the gauges {glue:text}`mean_g:.1f` mm.*

In this example kriging is only slightly more accurate than inverse distance weighting at individual points. Its main advantages are the error map, which shows where new gauges would be most useful, and weights that account for clustered gauges and for the spatial correlation when an areal mean is computed.

## 5.5 Measurement errors and extremes

### Gauge errors

Every gauge catches less precipitation than reaches the ground. The main systematic error is **wind-induced undercatch**: the gauge deflects the airflow, which speeds up over the orifice and carries hydrometeors past it. The loss is a few percent for moderate rain but can exceed 50% for light snow in strong wind; international intercomparisons have produced catch-ratio functions of wind speed for each gauge type, and wind shields (and gauges sunk in pits) reduce the loss. Wetting and evaporation losses add a fraction of a millimeter per event, and tipping-bucket gauges undercount at high intensities. In mountains, where precipitation is largest and gauges are fewest and most exposed, the combined bias of gauge-based precipitation can exceed 20%, which matters for every water balance in Chapter 1.

### Radar and satellite errors

Radar estimates suffer from errors in the Z–R relation (Section 5.2), from the **bright band** where melting snow reflects strongly, from **attenuation** in heavy rain at shorter wavelengths, from **beam blockage** by terrain, and from sampling high above the ground, where the precipitation may still evaporate or grow before it lands. Satellite products combine passive-microwave and infrared observations; they cover the oceans and ungauged regions but underestimate intense short rainfall and orographic enhancement.

### Extremes and temperature

Chapter 2 showed that the atmosphere's water vapor increases by about 7% per kelvin. Heavy rainfall over short durations draws on the vapor converging into individual storms, so its intensity can be expected to scale at about the same rate. Studies of station records relating extreme hourly rainfall to the temperature or dew point on the day of the event often find increases close to this rate, and in some climates up to twice it at higher temperatures, possibly because warmer convective storms are more vigorous (Lenderink and van Meijgaard, 2008). Daily extremes, which depend more on the large-scale circulation, often scale more weakly. This dependence of rainfall extremes on duration is one reason flood risk in small, fast-responding catchments is especially sensitive to warming (see the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch13_why_simple_models_work.html) book, Chapter 13).

## 5.6 Worked examples

**Example 5.1 — Activating a cloud droplet.** Find the critical radius and supersaturation for a droplet formed on an ammonium sulfate particle of mass 5 × 10⁻¹⁹ kg ($M_s$ = 0.132 kg mol⁻¹, $i$ = 3) at 10 °C.

```{code-cell} python
Tk = 283.15; a10 = 3.3e-7 / Tk
b10 = 3 * 3 * 5e-19 * Mw / (4 * np.pi * rho_w * 0.132)
print(f"r_c = {np.sqrt(3 * b10 / a10) * 1e6:.2f} µm, S_c - 1 = {np.sqrt(4 * a10 ** 3 / (27 * b10)) * 100:.3f}%")
```

**Example 5.2 — Diffusion is too slow.** With $G$ = 10⁻¹⁰ m² s⁻¹ and a supersaturation of 0.3%, how long does a droplet take to grow from 1 µm to 20 µm, and to 1 mm?

```{code-cell} python
G, s = 1e-10, 0.003
for r_end in (20e-6, 1e-3):
    t = (r_end ** 2 - 1e-6 ** 2) / (2 * G * s)
    print(f"to {r_end * 1e6:7.0f} µm: {t:9.0f} s = {t / 3600:7.2f} h = {t / 86400:6.1f} days")
```

**Example 5.3 — Deriving a Z–R relation.** Compute the reflectivity and rain rate of the Marshall–Palmer distribution (Equation 5.3) with the fall-speed fit of Section 5.1, for nominal rain rates from 1 to 100 mm h⁻¹, and fit $Z = a R^b$.

```{code-cell} python
Dg = np.linspace(0.05, 8, 4000)                                   # mm
vt = np.clip(9.65 - 10.3 * np.exp(-0.6 * Dg), 0, None)            # m/s
R_nom = np.array([1, 2, 5, 10, 20, 50, 100.0])
Zc, Rc = [], []
for R in R_nom:
    N = 8000 * np.exp(-4.1 * R ** -0.21 * Dg)                     # m⁻³ mm⁻¹
    Zc.append(np.trapezoid(N * Dg ** 6, Dg))                       # mm⁶ m⁻³
    Rc.append(np.trapezoid(N * vt * np.pi / 6 * Dg ** 3, Dg) * 1e-9 * 3.6e6)   # mm/h
b_zr, ln_a = np.polyfit(np.log(Rc), np.log(Zc), 1)
print("computed R (mm/h):", np.round(Rc, 1))
print(f"Z = {np.exp(ln_a):.0f} R^{b_zr:.2f}  (Marshall and Palmer's empirical fit: Z = 200 R^1.6)")
glue("zr_a", float(np.exp(ln_a)), display=False); glue("zr_b", float(b_zr), display=False)
```

The computed rain rates differ from the nominal ones by up to 20%, a reminder that the empirical distribution and the fall-speed fit come from different data sets.

**Example 5.4 — Orographic rain.** Using the upslope model, estimate the maximum precipitation rate on the windward slope of a ridge where the terrain rises at a slope of 0.08, for saturated air at 20 °C, a wind of 15 m s⁻¹ and a precipitation efficiency of 0.35.

```{code-cell} python
rho_vs20 = es(20.0) * 1000 / (461.5 * 293.15)
P_max = 0.35 * rho_vs20 * 15.0 * 0.08 * 3600
print(f"ρ_vs = {rho_vs20 * 1000:.1f} g/m³; P = {P_max:.0f} mm/h; over 12 h: {12 * P_max:.0f} mm")
```

Sustained over half a day, this gives several hundred millimeters, the order of the largest orographic storms in the Korean mountains during typhoons.

**Example 5.5 — Kriging weights.** Three gauges at (0, 0), (10, 0) and (0, 12) km recorded 42, 55 and 31 mm. With an exponential variogram (sill 120 mm², correlation length 15 km), find the ordinary-kriging estimate and its standard error at (4, 4) km, and compare with inverse distance weighting.

```{code-cell} python
xy3 = np.array([[0, 0], [10, 0], [0, 12.0]]); z3 = np.array([42, 55, 31.0]); t0 = np.array([[4.0, 4.0]])
est, var = ordinary_kriging(xy3, z3, t0, 120.0, 15.0)
n3 = 3
D3 = np.sqrt(((xy3[:, None] - xy3[None]) ** 2).sum(-1))
K = np.ones((4, 4)); K[:3, :3] = variogram(D3, 120, 15); K[3, 3] = 0
rhs = np.r_[variogram(np.sqrt(((xy3 - t0) ** 2).sum(1)), 120, 15), 1]
wts = np.linalg.solve(K, rhs)[:3]
print("kriging weights:", np.round(wts, 3), " sum =", wts.sum().round(3))
print(f"OK estimate = {est[0]:.1f} mm ± {np.sqrt(var[0]):.1f} mm; IDW = {idw(xy3, z3, t0)[0]:.1f} mm")
```

## 5.7 Common mistakes

- **Assuming droplets form at 100% relative humidity on any surface.** Activation needs a nucleus and a small supersaturation that depends on its size and chemistry.
- **Using one Z–R relation for all storms.** The drop-size distribution varies; calibrate against gauges, at least by storm type.
- **Interpolating short-duration rainfall with long-duration correlation lengths.** The correlation length shrinks with duration.
- **Treating gauge records in windy, snowy sites as true precipitation.** Correct for undercatch, or treat the water balance as uncertain.
- **Expecting all extremes to scale at 7% per kelvin.** Scaling depends on duration, season and storm type.

## 5.8 Problems

```{code-cell} python
:tags: [remove-cell]
b1 = 3 * 2 * 1e-18 * Mw / (4 * np.pi * rho_w * Ms); a1 = 3.3e-7 / 283.15
glue("t1", float(np.sqrt(4 * a1 ** 3 / (27 * b1)) * 100), display=False)
glue("t2", 2 * 9.81 * (25e-6) ** 2 * 1000 / (9 * 1.8e-5) * 100, display=False)
glue("t3", float((10 ** 4.5 / 200) ** (1 / 1.6)), display=False); glue("t3b", float((10 ** 4.5 / np.exp(ln_a)) ** (1 / b_zr)), display=False)
rv = es(10.0) * 1000 / (461.5 * 283.15)
glue("t4", float(0.3 * rv * 8 * 0.05 * 3600), display=False)
glue("t5", float(np.exp(-20 / 25)), display=False)
```

1. Find the critical supersaturation for a sodium chloride nucleus of 10⁻¹⁸ kg at 10 °C.
2. Using Stokes' law ($\mu$ = 1.8 × 10⁻⁵ Pa s), find the terminal velocity of a droplet of radius 25 µm, in cm s⁻¹.
3. A radar measures 45 dBZ. Estimate the rain rate with $Z = 200R^{1.6}$ and with the relation derived in Example 5.3.
4. Saturated air at 10 °C flows at 8 m s⁻¹ up a slope of 0.05. With a precipitation efficiency of 0.3, estimate the upslope precipitation rate.
5. If daily rainfall has an exponential correlation with a correlation length of 25 km, what is the correlation between two gauges 20 km apart?

```{admonition} Answers
:class: dropdown
1. About {glue:text}`t1:.3f`%.
2. About {glue:text}`t2:.1f` cm s⁻¹.
3. {glue:text}`t3:.1f` mm h⁻¹ with Marshall–Palmer; {glue:text}`t3b:.1f` mm h⁻¹ with the derived relation.
4. About {glue:text}`t4:.1f` mm h⁻¹.
5. {glue:text}`t5:.2f`.
```

## 5.9 Open questions

- **Aerosols and precipitation.** Does pollution, by increasing the number of nuclei, suppress drizzle, invigorate convection, or both? The effect on surface precipitation remains uncertain.
- **Scaling of extremes.** Why do some regions show super-Clausius–Clapeyron scaling of hourly extremes, and will it hold in a warmer climate?
- **Merging data sources.** How should gauges, radars and satellites be combined so that both the spatial pattern and the long-term totals are right, especially in mountains?

## 5.10 Key papers

- Atlas, D., Srivastava, R. C., and Sekhon, R. S. (1973). Doppler radar characteristics of precipitation at vertical incidence. *Reviews of Geophysics and Space Physics*, 11, 1–35.
- Köhler, H. (1936). The nucleus in and the growth of hygroscopic droplets. *Transactions of the Faraday Society*, 32, 1152–1161.
- Lenderink, G., and van Meijgaard, E. (2008). Increase in hourly precipitation extremes beyond expectations from temperature changes. *Nature Geoscience*, 1, 511–514.
- Marshall, J. S., and Palmer, W. McK. (1948). The distribution of raindrops with size. *Journal of Meteorology*, 5, 165–166.
- Smith, R. B., and Barstad, I. (2004). A linear theory of orographic precipitation. *Journal of the Atmospheric Sciences*, 61, 1377–1391.
- Zhu, Y., and Newell, R. E. (1998). A proposed algorithm for moisture fluxes from atmospheric rivers. *Monthly Weather Review*, 126, 725–735.

## 5.11 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The sections on the formation of precipitation, weather systems, distribution on the ground and the reliability of measurements.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on precipitation.
- Rogers, R. R., and Yau, M. K. (1989). *A Short Course in Cloud Physics* (3rd ed.). Pergamon.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
