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

# 7. Evaporation: physical theory

```{admonition} Core ideas
:class: tip
1. Evaporation is a transport problem: vapor must diffuse from the evaporating surface through a chain of **resistances**, stomatal, leaf boundary-layer and aerodynamic, to the free air. Heat follows a parallel path, minus the stomata.
2. **Mass-transfer** and **energy-budget** methods each need the surface temperature. **Combination** equations (Penman, Penman–Monteith) eliminate it by linearizing the saturation curve. The approximation is excellent in ordinary conditions and its error can be computed exactly.
3. The **decoupling coefficient** Ω measures how far a surface is from the air above it. Short, smooth crops (Ω → 1) evaporate at a rate set by **available energy**; tall, rough forests (Ω → 0) at a rate set by the **vapor pressure deficit** and their stomata.
4. In the limit of a saturated atmosphere, evaporation tends to the **equilibrium** rate. Real surfaces in humid regions evaporate about 1.26 times faster (**Priestley–Taylor**) because the boundary layer entrains dry air from above.
5. **Stomata** respond to light, vapor pressure deficit, temperature and soil water. Simple multiplicative (Jarvis-type) models capture the main responses, including the midday closure that limits transpiration on dry, hot days.
6. Deep lakes store heat in spring and release it in autumn, so their evaporation is **delayed by months** relative to the radiation that drives it.
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
es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))          # kPa, T in °C
slope = lambda T: 4098 * es(T) / (T + 237.3) ** 2                   # kPa/°C
rho_a, cp, lam, gam = 1.2, 1005.0, 2.45e6, 0.066                    # SI; γ in kPa/°C
```

## 7.1 Evaporation as transport

Evaporation needs three things: energy to change liquid into vapor, a gradient of vapor concentration, and turbulence to carry the vapor away (Chapter 3). At the scale of a leaf or a water surface, the transport is diffusion; above it, turbulent mixing. Both can be written in the same form, a flux equal to a concentration difference divided by a **resistance** (Chapter 4):

\begin{equation*}
E = \rho_a\,\frac{q^*(T_s) - q_a}{r_a + r_s},
\qquad
H = \rho_a\,c_p\,\frac{T_s - T_a}{r_a},
\tag{7.1}
\end{equation*}

where $q^*(T_s)$ is the saturation specific humidity at the temperature of the evaporating surface, $q_a$ and $T_a$ are the humidity and temperature at a reference height in the air, $r_a$ is the aerodynamic resistance from the surface to the reference height (including the thin quasi-laminar layer around leaves, sometimes written separately as $r_b$), and $r_s$ is the **surface resistance**: zero for open water and wet canopies, the resistance of the stomata (and of the soil surface) for vegetation. Heat leaves the surface without passing through the stomata, so $r_s$ appears only in the vapor flux. Writing specific humidity in terms of vapor pressure, $q \approx \varepsilon e/p$, the vapor flux becomes $\lambda E = (\rho_a c_p/\gamma)\,[e^*(T_s) - e_a]/(r_a + r_s)$, with the psychrometric constant $\gamma = c_p p/(\varepsilon\lambda)$.

## 7.2 Mass transfer and energy budget

Equation (7.1) for $E$ alone is a **mass-transfer** method. It is used for lakes, with the surface water temperature measured, and in the bulk form $E = \rho_a C_E\,\overline{u}\,(q_s - q_a)$ of Chapter 4. Its weakness for land surfaces is that $T_s$ is seldom measured, and a small error in $T_s$ produces a large error in $e^*(T_s)$ (about 6% per kelvin).

The **energy budget**, $R_n - G = \lambda E + H$, is the second route. With the **Bowen ratio** $\beta = H/\lambda E$ it gives $\lambda E = (R_n - G)/(1 + \beta)$, and with Equation (7.1) for a wet surface, $\beta = \gamma\,(T_s - T_a)/[e^*(T_s) - e_a]$. Again, the surface temperature is needed.

The two equations together, energy budget and transport, contain two unknowns, $E$ and $T_s$, and can be solved simultaneously. Solving them exactly requires iteration because $e^*(T_s)$ is nonlinear. Penman's insight was to linearize it.

## 7.3 Combination equations

### Penman's linearization

Approximate the saturation vapor pressure at the surface by a tangent at the air temperature,

\begin{equation*}
e^*(T_s) \approx e^*(T_a) + \Delta\,(T_s - T_a),
\tag{7.2}
\end{equation*}

where $\Delta = de^*/dT$ at $T_a$. Substituting into the flux equations and using the energy budget to eliminate $T_s$ gives, for a wet surface ($r_s = 0$), the Penman equation, and with a surface resistance the **Penman–Monteith** equation

\begin{equation*}
\lambda E = \frac{\Delta\,(R_n - G) + \rho_a c_p\,\delta e / r_a}{\Delta + \gamma\,(1 + r_s/r_a)} ,
\tag{7.3}
\end{equation*}

where $\delta e = e^*(T_a) - e_a$ is the vapor pressure deficit of the air. (The step-by-step derivation is in Chapter 4 of the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch04_evaporation_and_evapotranspiration.html) book.)

### How good is the linearization?

The only approximation in Equation (7.3), beyond the resistance model itself, is Equation (7.2). Its error grows with the square of $T_s - T_a$. Figure 7.1 compares the Penman estimate with the exact solution of the energy budget and transport equations, found by iteration for the surface temperature. For a wet surface the error is a few percent, except when the available energy is small and evaporation cools the surface well below the air. It grows to more than 10% for a dry canopy under strong radiation, whose surface becomes much warmer than the air. When the error matters, it can be removed by a second-order correction (Paw U and Gao, 1988) or by solving for $T_s$ directly, which is now easy.

```{code-cell} python
:tags: [hide-input]
from scipy.optimize import brentq

def exact_LE(Rn_G, Ta, de, ra, rs):
    """Solve the energy budget and transport equations exactly for Ts, return λE (W/m²) and Ts."""
    ea = es(Ta) - de
    f = lambda Ts: Rn_G - rho_a * cp * (Ts - Ta) / ra - rho_a * cp / gam * (es(Ts) - ea) / (ra + rs)
    Ts = brentq(f, Ta - 30, Ta + 40)
    return rho_a * cp / gam * (es(Ts) - ea) / (ra + rs), Ts

def pm_LE(Rn_G, Ta, de, ra, rs):
    return (slope(Ta) * Rn_G + rho_a * cp * de / ra) / (slope(Ta) + gam * (1 + rs / ra))

Ta, de, ra = 25.0, 1.5, 50.0
Rn_G = np.linspace(50, 700, 60)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for rs, col in [(0.0, c1), (70.0, c3), (300.0, c2)]:
    err = [100 * (pm_LE(R, Ta, de, ra, rs) / exact_LE(R, Ta, de, ra, rs)[0] - 1) for R in Rn_G]
    ax.plot(Rn_G, err, color=col, lw=2.2, label=f"$r_s$ = {rs:g} s m⁻¹")
ax.axhline(0, color=ink2, lw=0.8)
ax.set_xlabel("Available energy $R_n - G$ (W m⁻²)"); ax.set_ylabel("Error of linearized λE (%)")
ax.legend(loc="lower left", fontsize=9.5)
plt.show()
LEx, Tsx = exact_LE(600, Ta, de, ra, 300.0)
glue("lin_err", float(100 * (pm_LE(600, Ta, de, ra, 300.0) / LEx - 1)), display=False)
glue("lin_dT", float(Tsx - Ta), display=False)
```

*Figure 7.1. Error of the Penman–Monteith equation (linearized saturation curve) relative to the exact solution of the same energy-budget and transport equations, for air at 25 °C, a vapor pressure deficit of 1.5 kPa and $r_a$ = 50 s m⁻¹. For a dry canopy ($r_s$ = 300 s m⁻¹) with 600 W m⁻² of available energy, the surface is {glue:text}`lin_dT:.1f` K warmer than the air and the error is {glue:text}`lin_err:.1f`%.*

### Decoupling

Equation (7.3) can be rearranged into a weighted average of two limits (Jarvis and McNaughton, 1986):

\begin{equation*}
\lambda E = \Omega\,\lambda E_{\text{eq}} + (1 - \Omega)\,\lambda E_{\text{imp}},
\qquad
\Omega = \frac{\Delta + \gamma}{\Delta + \gamma\,(1 + r_s/r_a)},
\tag{7.4}
\end{equation*}

where $\lambda E_{\text{eq}} = \Delta(R_n - G)/(\Delta + \gamma)$ is the **equilibrium evaporation** (Section 7.4) and $\lambda E_{\text{imp}} = \rho_a c_p\,\delta e/(\gamma\, r_s)$ is the **imposed evaporation** that the air's vapor pressure deficit would drive through the stomata alone. The **decoupling coefficient** Ω ranges from 0 to 1. For short, smooth vegetation, $r_a$ is large compared with $r_s$, Ω approaches 1, and evaporation is controlled by the available energy: the surface is "decoupled" from the air above. For tall, rough forests, $r_a$ is small, Ω may be only 0.1–0.2, and evaporation is controlled by the vapor pressure deficit and the stomata: the canopy is well "coupled" to the atmosphere (Figure 7.2). This explains why stomatal control matters much more for forest transpiration than for grassland.

```{code-cell} python
:tags: [hide-input]
ra_g = np.logspace(0.5, 2.6, 200)                               # s/m
fig, ax = plt.subplots(figsize=(6.4, 3.8))
D20, g20 = slope(20.0), gam
for rs, col in [(50.0, c3), (150.0, c1), (400.0, c2)]:
    Om = (D20 + g20) / (D20 + g20 * (1 + rs / ra_g))
    ax.semilogx(ra_g, Om, color=col, lw=2.2, label=f"$r_s$ = {rs:g} s m⁻¹")
for ra_v, lab in [(8, "forest"), (40, "tall crop"), (110, "short grass")]:
    ax.axvline(ra_v, color=ink2, lw=0.7, ls=":"); ax.text(ra_v * 1.05, 0.03, lab, fontsize=9, color=ink2, rotation=90, va="bottom")
ax.set_xlabel("Aerodynamic resistance $r_a$ (s m⁻¹)"); ax.set_ylabel("Decoupling coefficient Ω")
ax.set_ylim(0, 1); ax.set_xlim(3, 400); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
```

*Figure 7.2. The decoupling coefficient at 20 °C as a function of the aerodynamic resistance for three surface resistances. Typical values of $r_a$ for a 2 m s⁻¹ wind are marked. Forests are tightly coupled to the atmosphere (small Ω); short grass is largely decoupled.*

## 7.4 Equilibrium evaporation and Priestley–Taylor

If moist air passes over a long, uniform wet surface, its vapor pressure deficit is gradually removed and the second term of the Penman equation vanishes, leaving the equilibrium evaporation $\lambda E_{\text{eq}}$. Observations over well-watered surfaces, however, consistently show evaporation larger than this, by a nearly constant factor (Priestley and Taylor, 1972):

\begin{equation*}
\lambda E_{PT} = \alpha_{PT}\,\frac{\Delta}{\Delta + \gamma}\,(R_n - G), \qquad \alpha_{PT} \approx 1.26 .
\tag{7.5}
\end{equation*}

Why is $\alpha_{PT} > 1$? The convective boundary layer is not a closed box. As it grows during the day it **entrains** warmer, drier air from the free atmosphere above the inversion (Chapter 4), which maintains a vapor pressure deficit near the surface even over a large wet region. Mixed-layer models that include entrainment reproduce values of α close to 1.26 for typical conditions (McNaughton and Spriggs, 1986). The coefficient is therefore a property of the coupled surface–boundary-layer system, not of the surface alone, and it varies: it is larger in dry, advective conditions and smaller over surfaces that restrict transpiration. Chapter 8 uses the Priestley–Taylor rate as the "wet-environment" evaporation in the complementary relationship.

## 7.5 Transpiration and stomatal control

### Stomatal conductance

Transpiration passes through stomata, pores whose aperture the plant adjusts to balance the uptake of carbon dioxide against the loss of water. The canopy (surface) conductance $g_s = 1/r_s$ responds to:

- **light**: stomata open with photosynthetically active radiation;
- **vapor pressure deficit**: they close as the air becomes drier;
- **temperature**: they have an optimum;
- **soil water**: they close as the root zone dries and the leaf water potential falls.

The widely used **Jarvis-type** model multiplies independent response functions, each between 0 and 1:

\begin{equation*}
g_s = g_{\max}\; f_1(R_s)\; f_2(\delta e)\; f_3(T_a)\; f_4(\theta),
\tag{7.6}
\end{equation*}

with forms such as $f_1 = R_s/(R_s + k_R)$ and $f_2 = \max(1 - k_D\,\delta e,\ \epsilon)$ (Jarvis, 1976; Stewart, 1988). More mechanistic models link stomatal conductance to photosynthesis (Ball–Berry type) or derive it from optimality: the plant maximizes carbon gain for a given water loss. Figure 7.3 simulates a sunny summer day for a forest with Jarvis-type stomatal control: transpiration peaks before noon, because stomata close as the afternoon vapor pressure deficit rises.

```{code-cell} python
:tags: [hide-input]
hr = np.linspace(5, 20, 151)
sun = np.clip(np.sin(np.pi * (hr - 6) / 13), 0, None)
Rs = 900 * sun                                                     # W/m²
Rn_G = 0.75 * Rs - 40 * (sun > 0)                                  # W/m², forest
Ta_d = 18 + 12 * np.clip(np.sin(np.pi * (hr - 8) / 14), -0.2, None)
de_d = es(Ta_d) - es(14.0)                                          # dew point stays at 14 °C
gmax, kR, kD = 0.012, 150.0, 0.25                                   # m/s, W/m², 1/kPa
gs = gmax * Rs / (Rs + kR) * np.maximum(1 - kD * de_d, 0.1)
rs = 1 / np.maximum(gs, 1e-5)
ra_f = 10.0
LE_j = np.clip(pm_LE(Rn_G, Ta_d, de_d, ra_f, rs), 0, None)
LE_fix = np.clip(pm_LE(Rn_G, Ta_d, de_d, ra_f, np.full(len(hr), 1 / (gmax * 0.4))), 0, None)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True)
a1.plot(hr, de_d, color=c2, lw=2.2, label="Vapor pressure deficit (kPa)")
a1b = a1.twinx(); a1b.plot(hr, gs * 1000, color=c3, lw=2.2, label="Canopy conductance")
a1b.set_ylabel("$g_s$ (mm s⁻¹)", color=c3); a1b.grid(False); a1b.spines["right"].set_visible(True)
a1.set_ylabel("δe (kPa)", color=c2); a1.set_title("(a) Atmospheric demand and stomatal response")
a2.plot(hr, LE_j, color=c1, lw=2.4, label="Jarvis-type stomata")
a2.plot(hr, LE_fix, color=ink2, lw=1.6, ls="--", label="Constant conductance")
a2.plot(hr, Rn_G, color=c2, lw=1.2, ls=":", label="$R_n - G$")
a2.set_xlabel("Hour"); a2.set_ylabel("λE (W m⁻²)"); a2.legend(loc="upper left", fontsize=9)
a2.set_title("(b) Transpiration"); a2.set_xlim(5, 20)
plt.show()
glue("tr_peak", float(hr[np.argmax(LE_j)]), display=False)
glue("tr_day_j", float(np.trapezoid(LE_j, hr * 3600) / lam), display=False)
glue("tr_day_f", float(np.trapezoid(LE_fix, hr * 3600) / lam), display=False)
```

*Figure 7.3. A synthetic sunny day over a forest ($r_a$ = 10 s m⁻¹). As the vapor pressure deficit climbs through the afternoon, Jarvis-type stomata close, and transpiration peaks at about {glue:text}`tr_peak:.0f` h, well before the maximum demand. The day's transpiration is {glue:text}`tr_day_j:.1f` mm, against {glue:text}`tr_day_f:.1f` mm with a constant conductance.*

### One source or two

The big-leaf model of Equation (7.3) treats the canopy and the soil beneath as a single surface. Sparse canopies, orchards and savannas violate this assumption: the soil can be dry while the plants transpire, or wet while they are dormant. **Two-source** models (Shuttleworth and Wallace, 1985) write separate Penman–Monteith-type equations for the canopy and the soil, coupled through a common air temperature and humidity within the canopy, and are the basis of the evaporation schemes in most land-surface models.

## 7.6 Evaporation from open water

For a lake, $r_s = 0$, the albedo is low (about 0.06), and the surface temperature can be measured, so both the Penman equation and the mass-transfer method apply. The term $G$ must be replaced by the change in **heat stored** in the water. A deep lake absorbs heat in spring and early summer, when radiation is strong and the water is colder than the air, and releases it in autumn and winter. Its evaporation is therefore shifted later in the year than the radiation that drives it: a deep lake can evaporate more in October than in June, at night as well as by day. Figure 7.4 shows the effect with a simple heat-balance model of a well-mixed water layer, driven by the same synthetic climate for a shallow and a deep lake.

```{code-cell} python
:tags: [hide-input]
days = np.arange(365 * 3); dd = days % 365
Ta_y = 14 + 12 * np.sin(2 * np.pi * (dd - 110) / 365)
Rs_y = (17 + 10 * np.sin(2 * np.pi * (dd - 80) / 365)) * 1e6 / 86400       # W/m²
sig, RH, u = 5.67e-8, 0.7, 3.0
ra_w = np.log(2 / 0.0002) ** 2 / (0.41 ** 2 * u)
def lake(depth):
    Tw, out = 10.0, []
    for k in range(len(days)):
        Ld = 0.75 * 1.085 * sig * (Ta_y[k] + 273.15) ** 4
        Rn = 0.94 * Rs_y[k] + Ld - 0.97 * sig * (Tw + 273.15) ** 4
        H = rho_a * cp * (Tw - Ta_y[k]) / ra_w
        LE = rho_a * cp / gam * (es(Tw) - RH * es(Ta_y[k])) / ra_w
        Tw += (Rn - H - LE) * 86400 / (1000 * 4186 * depth)
        out.append((max(LE, 0) / lam * 86400, Tw))
    return np.array(out[-365:])
fig, ax = plt.subplots(figsize=(6.4, 3.9))
peaks = {}
for depth, col in [(2.0, c2), (30.0, c1)]:
    r = lake(depth)
    sm = np.convolve(np.r_[r[-15:, 0], r[:, 0], r[:15, 0]], np.ones(31) / 31, "valid")
    ax.plot(np.arange(365), sm, color=col, lw=2.2, label=f"{depth:g} m deep: {r[:, 0].sum():.0f} mm/yr")
    peaks[depth] = np.argmax(sm)
ax.plot(np.arange(365), (Rs_y[:365] * 0.94 * 86400 / lam) * 0.55, color=ink2, lw=1.2, ls=":", label="Absorbed shortwave (scaled)")
ax.set_xticks([0, 90, 181, 273, 364], ["Jan", "Apr", "Jul", "Oct", "Dec"]); ax.set_xlim(0, 364); ax.set_ylim(0, 8.5)
ax.set_ylabel("Evaporation (mm d⁻¹)"); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("pk_shallow", int(peaks[2.0]), display=False); glue("pk_deep", int(peaks[30.0]), display=False)
```

*Figure 7.4. Evaporation from a shallow (2 m) and a deep (30 m) well-mixed lake under the same synthetic climate (31-day means). The shallow lake follows the radiation, peaking around day {glue:text}`pk_shallow`; the deep lake stores summer heat and peaks around day {glue:text}`pk_deep`, in early autumn. Their annual totals are similar.*

## 7.7 Worked examples

**Example 7.1 — Exact versus linearized.** For a forest on a dry day ($R_n - G$ = 450 W m⁻², $T_a$ = 28 °C, $\delta e$ = 2.2 kPa, $r_a$ = 12 s m⁻¹, $r_s$ = 250 s m⁻¹), compute λE with the Penman–Monteith equation and with the exact solution, and the surface temperature.

```{code-cell} python
LEe, Tse = exact_LE(450, 28.0, 2.2, 12.0, 250.0)
LEp = pm_LE(450, 28.0, 2.2, 12.0, 250.0)
print(f"P–M: {LEp:.0f} W/m²; exact: {LEe:.0f} W/m² ({100 * (LEp / LEe - 1):+.1f}%); Ts - Ta = {Tse - 28:.2f} K")
print(f"Bowen ratio = {(450 - LEe) / LEe:.2f}; E = {LEe / lam * 3600:.2f} mm/h")
```

**Example 7.2 — Coupled or decoupled?** Compute Ω, the equilibrium and the imposed evaporation for (a) grass with $r_a$ = 100 s m⁻¹ and $r_s$ = 70 s m⁻¹, and (b) a forest with $r_a$ = 8 s m⁻¹ and $r_s$ = 120 s m⁻¹, both at 20 °C with $R_n - G$ = 400 W m⁻² and $\delta e$ = 1.2 kPa.

```{code-cell} python
for name, ra_x, rs_x in [("grass", 100.0, 70.0), ("forest", 8.0, 120.0)]:
    Om = (slope(20) + gam) / (slope(20) + gam * (1 + rs_x / ra_x))
    LEeq = slope(20) * 400 / (slope(20) + gam)
    LEimp = rho_a * cp * 1.2 / (gam * rs_x)
    print(f"{name:6s}: Ω = {Om:.2f}, λE_eq = {LEeq:.0f}, λE_imp = {LEimp:.0f}, "
          f"λE = {Om * LEeq + (1 - Om) * LEimp:.0f} W/m² (P–M: {pm_LE(400, 20, 1.2, ra_x, rs_x):.0f})")
```

**Example 7.3 — Priestley–Taylor.** A well-watered crop at 22 °C has $R_n - G$ = 13.5 MJ m⁻² d⁻¹. Find the equilibrium and Priestley–Taylor evaporation in mm d⁻¹.

```{code-cell} python
w = slope(22) / (slope(22) + gam)
Eeq = w * 13.5e6 / lam
print(f"Δ/(Δ+γ) = {w:.3f}; E_eq = {Eeq:.2f} mm/d; E_PT = {1.26 * Eeq:.2f} mm/d")
```

**Example 7.4 — Effect of stomatal closure.** In Figure 7.3, by how much would daily transpiration change if the stomata were insensitive to vapor pressure deficit ($k_D$ = 0)?

```{code-cell} python
gs0 = gmax * Rs / (Rs + kR)
LE0 = np.clip(pm_LE(Rn_G, Ta_d, de_d, ra_f, 1 / np.maximum(gs0, 1e-5)), 0, None)
print(f"With VPD response: {np.trapezoid(LE_j, hr * 3600) / lam:.2f} mm; without: {np.trapezoid(LE0, hr * 3600) / lam:.2f} mm")
```

## 7.8 Common mistakes

- **Using $r_s$ in the sensible heat flux.** Heat leaves through $r_a$ only; the stomata act on vapor.
- **Applying Priestley–Taylor in dry, advective conditions.** The coefficient 1.26 is for humid environments; elsewhere it underestimates.
- **Ignoring heat storage in lakes.** Monthly lake evaporation from radiation alone can be wrong by more than a factor of two.
- **Using a single big leaf for sparse canopies.** Soil and canopy then evaporate under very different conditions.
- **Treating the linearization as the main error.** It is usually small; the uncertainty in $r_s$, $r_a$ and $R_n - G$ is far larger.

## 7.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("v1", float(pm_LE(350, 18.0, 0.8, 60.0, 0.0) / lam * 3600), display=False)
glue("v2", float((slope(15) + gam) / (slope(15) + gam * (1 + 100 / 20.0))), display=False)
glue("v3", float(1.26 * slope(30) / (slope(30) + gam) * 15e6 / lam), display=False)
glue("v4", float(rho_a * cp * 2.0 / (gam * 200.0)), display=False)
glue("v5", float(1000 * 4186 * 30 * 3.0 / lam), display=False)
```

1. Find the Penman (wet-surface) evaporation in mm h⁻¹ for $R_n - G$ = 350 W m⁻², $T_a$ = 18 °C, $\delta e$ = 0.8 kPa and $r_a$ = 60 s m⁻¹.
2. A canopy has $r_a$ = 20 s m⁻¹ and $r_s$ = 100 s m⁻¹ at 15 °C. Find Ω. Is its evaporation controlled mainly by radiation or by vapor pressure deficit?
3. Find the Priestley–Taylor evaporation in mm d⁻¹ for $R_n - G$ = 15 MJ m⁻² d⁻¹ at 30 °C.
4. Find the imposed evaporation (W m⁻²) for $\delta e$ = 2.0 kPa and $r_s$ = 200 s m⁻¹.
5. A 30 m deep lake cools by 3 °C in a month. If all the released heat went to evaporation, how many millimeters would it evaporate?

```{admonition} Answers
:class: dropdown
1. {glue:text}`v1:.2f` mm h⁻¹.
2. Ω = {glue:text}`v2:.2f`: closer to coupled; the deficit and the stomata matter more than radiation.
3. {glue:text}`v3:.1f` mm d⁻¹.
4. {glue:text}`v4:.0f` W m⁻².
5. About {glue:text}`v5:.0f` mm.
```

## 7.10 Open questions

- **What sets α in Priestley–Taylor?** The coefficient depends on boundary-layer dynamics and on the surface; whether it changes in a warmer climate is uncertain.
- **Stomatal behavior under rising CO₂.** Higher CO₂ tends to close stomata and reduce transpiration per leaf area, but plants may grow more leaves; the net effect on catchment evaporation is debated.
- **From leaves to landscapes.** How should stomatal models calibrated at the leaf scale be scaled to canopies and grid cells?

## 7.11 Key papers

- Jarvis, P. G. (1976). The interpretation of the variations in leaf water potential and stomatal conductance found in canopies in the field. *Philosophical Transactions of the Royal Society B*, 273, 593–610.
- Jarvis, P. G., and McNaughton, K. G. (1986). Stomatal control of transpiration: scaling up from leaf to region. *Advances in Ecological Research*, 15, 1–49.
- McNaughton, K. G., and Spriggs, T. W. (1986). A mixed-layer model for regional evaporation. *Boundary-Layer Meteorology*, 34, 243–262.
- Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205–234.
- Paw U, K. T., and Gao, W. (1988). Applications of solutions to non-linear energy budget equations. *Agricultural and Forest Meteorology*, 43, 121–145.
- Penman, H. L. (1948). Natural evaporation from open water, bare soil and grass. *Proceedings of the Royal Society of London A*, 193, 120–145.
- Priestley, C. H. B., and Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using large-scale parameters. *Monthly Weather Review*, 100, 81–92.
- Shuttleworth, W. J., and Wallace, J. S. (1985). Evaporation from sparse crops: an energy combination theory. *Quarterly Journal of the Royal Meteorological Society*, 111, 839–855.
- Stewart, J. B. (1988). Modelling surface conductance of pine forest. *Agricultural and Forest Meteorology*, 43, 19–35.

## 7.12 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The sections on evaporation mechanisms, mass transfer and energy-budget formulations.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on evapotranspiration.
- Monteith, J. L., and Unsworth, M. H. (2013). *Principles of Environmental Physics* (4th ed.). Academic Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
