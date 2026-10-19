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

# 8. Actual evaporation at landscape scale

```{admonition} Core ideas
:class: tip
1. "Potential evaporation" has several meanings: open-water Penman, reference-crop Penman–Monteith, wet-environment Priestley–Taylor. They differ by tens of percent, so the definition must always be stated.
2. Over large areas, potential evaporation is not independent of actual evaporation. As the land dries, the energy not used for evaporation heats and dries the air, which **raises** the apparent potential evaporation. This is Bouchet's **complementary relationship**.
3. The **advection–aridity** model and the **generalized complementary relationship** turn this idea into equations that estimate actual evaporation from routine weather data alone, without soil or vegetation information.
4. Over long periods, the catchment water balance gives $E = P - Q$. The **Budyko framework** relates the evaporative ratio $E/P$ to the aridity index $E_p/P$ through curves bounded by the water and energy limits; one-parameter forms (Fu, Mezentsev–Choudhury–Yang) describe catchment differences.
5. The complementary relationship and the Budyko framework describe the same coupling of water and energy at different time scales, and they can be used to constrain each other.
6. Pan evaporation declined over much of the world in the late twentieth century while the climate warmed. This **evaporation paradox** is explained by declining radiation and wind and, in water-limited regions, by the complementary relationship.
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
es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
slope = lambda T: 4098 * es(T) / (T + 237.3) ** 2
lam, gam = 2.45, 0.066                                            # MJ/kg, kPa/°C
```

## 8.1 Potential and actual evaporation

Chapter 7 derived the evaporation from a surface with given resistances under given weather. The rate at which the land actually evaporates, $E$, also depends on how much water is available to it. **Potential evaporation** $E_p$ was introduced to separate the two: the evaporation that would occur if water were not limiting. The idea is simple, but its definitions differ:

- **open-water Penman evaporation**, with a wind function calibrated over water;
- **reference-crop evaporation** $E_0$ (FAO-56), the Penman–Monteith rate for a defined grass surface with $r_s$ = 70 s m⁻¹;
- **wet-environment evaporation** $E_w = \alpha_{PT}\,E_{\text{eq}}$ from the Priestley–Taylor equation, the rate from a large, uniformly wet region;
- **pan evaporation**, measured directly but from a small, isolated water surface.

Under dry conditions these can differ by 30% or more, so every study must say which one it uses.

A common modeling approach then writes $E = \beta(\theta)\,E_p$, with a reduction factor that falls as the soil water content $\theta$ decreases (the FAO-56 stress coefficient of Chapter 4 in the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch04_evaporation_and_evapotranspiration.html) book is one example). This treats $E_p$ as an external forcing, independent of the land. At the scale of a field irrigated within a dry region, that is reasonable. At the scale of a region, it is not.

## 8.2 The complementary relationship

### Bouchet's hypothesis

Consider a large, homogeneous region that is initially wet, so that it evaporates at the wet-environment rate $E_w$, and $E = E_p = E_w$. Now let the region dry out at constant available energy. Its evaporation falls below $E_w$. The energy that is no longer used to evaporate water must go somewhere, and over a large region it goes into sensible heat: the boundary-layer air becomes warmer and drier. An evaporation pan, or a Penman calculation using the measured air temperature and humidity, then shows a **higher** potential evaporation. Bouchet (1963) proposed that the two changes are equal in magnitude:

\begin{equation*}
E_p - E_w = E_w - E \quad\Longleftrightarrow\quad E + E_p = 2E_w .
\tag{8.1}
\end{equation*}

Actual and apparent potential evaporation are thus **complementary**. As a region dries, their gap widens symmetrically around the wet-environment rate. The relationship turns the usual reasoning around. A high potential evaporation does not necessarily indicate a high evaporative demand to be met; in a water-limited region it is a **symptom** of low actual evaporation.

### The advection–aridity model

Brutsaert and Stricker (1979) made Equation (8.1) operational by taking the apparent potential evaporation from the Penman equation, with the air's measured drying power (the "advection" term), and the wet-environment rate from the Priestley–Taylor equation:

\begin{equation*}
E = 2E_w - E_{\text{pen}} = (2\alpha_{PT} - 1)\,\frac{\Delta}{\Delta + \gamma}\,\frac{R_n - G}{\lambda} - \frac{\gamma}{\Delta + \gamma}\,f(u)\,\delta e .
\tag{8.2}
\end{equation*}

The **advection–aridity** (AA) model needs only radiation, temperature, humidity and wind, which makes it attractive for regions without data on soils and vegetation. Its weaknesses are the assumed symmetry and the fact that it can produce negative evaporation in very dry conditions.

### The generalized complementary relationship

Writing the relationship in dimensionless form clarifies its structure. Let $x = E_w/E_p$, which equals 1 for a wet region and decreases as the region dries, and $y = E/E_p$. The symmetric relation (8.1) is the straight line $y = 2x - 1$. Brutsaert (2015) argued that a physically consistent relation must satisfy boundary conditions at both ends: at the wet end ($x = 1$), $y = 1$ and $dy/dx = 1$; at the dry end, $y = 0$ and $dy/dx = 0$ as $x \to 0$. The simplest polynomial that satisfies them is

\begin{equation*}
y = (2 - x)\,x^2 ,
\tag{8.3}
\end{equation*}

the **generalized complementary relationship** (GCR). Unlike the symmetric line, its slope is 1 at the wet end, so that actual evaporation at first barely falls as the apparent potential evaporation rises above the wet-environment rate, and it bends smoothly to zero at the dry end (Figure 8.1). The complementarity is therefore asymmetric: most of the early drying shows up as a rise in $E_p$ rather than a fall in $E$. Rescaled versions and more flexible forms have since been developed, including power-function complementary relationships that can be constrained with the steady-state Budyko equation (Kim et al., 2024) and that perform consistently in Earth system models under warming (Im et al., 2026). Calibrated versions have been used to estimate evaporation and drought from weather data alone, in the contiguous United States (Kim et al., 2019) and other regions.

```{code-cell} python
:tags: [hide-input]
x = np.linspace(0.01, 1, 300)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 7.0))
a1.plot(x, (2 - x) * x ** 2, color=c1, lw=2.4, label="Generalized CR: y = (2 − x)x²")
a1.plot(x, 2 * x - 1, color=c2, lw=2.0, ls="--", label="Symmetric CR (Bouchet, AA): y = 2x − 1")
a1.plot([0, 1], [0, 1], color=ink2, lw=0.8, ls=":")
a1.set_xlabel("x = $E_w / E_p$  (wetter →)"); a1.set_ylabel("y = $E / E_p$")
a1.set_xlim(0, 1); a1.set_ylim(-0.4, 1.05); a1.legend(loc="upper left", fontsize=9.5)
a1.set_title("(a) Dimensionless forms")
Ew = 5.0                                                     # mm/d, wet-environment rate
xx = np.linspace(0.3, 1, 200)
Ep = Ew / xx; E_gcr = Ep * (2 - xx) * xx ** 2; E_sym = 2 * Ew - Ep
wet = 1 - (1 - xx) / 0.7                                      # 1 = wet, 0 = driest shown
a2.plot(wet, Ep, color=c2, lw=2.4, label="Apparent potential $E_p$")
a2.plot(wet, E_gcr, color=c1, lw=2.4, label="Actual E (generalized CR)")
a2.plot(wet, E_sym, color=c1, lw=1.6, ls="--", label="Actual E (symmetric CR)")
a2.axhline(Ew, color=ink2, lw=1.2, ls=":", label="Wet-environment $E_w$")
a2.set_xlabel("Surface water availability (relative)"); a2.set_ylabel("mm d⁻¹")
a2.set_xlim(0, 1); a2.set_ylim(0, 17.5); a2.legend(loc="upper right", fontsize=9)
a2.set_title("(b) Complementary behavior at constant available energy")
plt.show()
```

*Figure 8.1. (a) The symmetric and the generalized complementary relationships in dimensionless form; the dotted line $y = x$ corresponds to $E = E_w$, and the symmetric line becomes negative for $x < 0.5$. (b) The same relationships drawn as rates against the water available to the land surface, at a fixed wet-environment evaporation of 5 mm d⁻¹. As the land dries, actual evaporation falls and apparent potential evaporation rises.*

### Where the complementary relationship holds

The relationship is a statement about large, homogeneous regions over periods long enough (days or longer) for the boundary layer to adjust to the land surface. It does not apply to an irrigated field in a dry region, whose evaporation does not change the air passing over it. It also weakens where advection of air from elsewhere (from the sea, for example) dominates the local feedback. Within its domain it has proved robust in many data sets. Its practical appeal is that the land-surface condition is inferred from the state of the air, which is routinely measured or available from reanalyses.

## 8.3 Water-budget estimates

The most direct estimate of actual evaporation over a catchment is the residual of its water balance (Chapter 1):

\begin{equation*}
E = P - Q - \frac{dS}{dt} - L .
\tag{8.4}
\end{equation*}

Over many years, the storage term vanishes and $E \approx P - Q$. Over shorter periods it does not, and satellite gravimetry (GRACE and its successor) now measures monthly changes in total water storage over regions larger than about 200,000 km², which allows monthly water-budget evaporation for large basins. The uncertainties add as in Equation (1.12): precipitation errors usually dominate, especially in mountains (Chapter 5).

Water budgets of smaller control volumes give evaporation at other scales:

- **soil columns**, from repeated soil-moisture profiles, minus drainage;
- **weighing lysimeters**, which give the most direct measurement but represent a few square meters;
- **lakes**, from inflow, outflow and level changes;
- **the atmosphere**, from the moisture convergence of Equation (1.11).

Each has its own scale and errors, and comparisons between them, and with eddy covariance (Chapter 3), are the main way to test evaporation models.

## 8.4 Long-term partition: the Budyko framework

### The limits and the curves

Over long periods the catchment balance reduces to $\bar{P} = \bar{E} + \bar{Q}$, and the evaporative ratio $\bar{E}/\bar{P}$ is bounded by the water limit ($\bar{E} \le \bar{P}$) and the energy limit ($\bar{E} \le \bar{E}_p$), as introduced in Chapter 2. Several curves have been proposed that approach both limits as functions of the aridity index $\phi = \bar{E}_p/\bar{P}$:

\begin{equation*}
\text{Schreiber: } \frac{\bar{E}}{\bar{P}} = 1 - e^{-\phi},
\qquad
\text{Ol'dekop: } \frac{\bar{E}}{\bar{P}} = \phi\tanh\frac{1}{\phi},
\tag{8.5}
\end{equation*}

and Budyko (1974) used their geometric mean. Parameter-free curves cannot describe differences between catchments in the same climate. Two one-parameter families are widely used:

\begin{equation*}
\text{Fu: } \frac{\bar{E}}{\bar{P}} = 1 + \phi - \bigl(1 + \phi^{\,w}\bigr)^{1/w},
\qquad
\text{Mezentsev–Choudhury–Yang: } \frac{\bar{E}}{\bar{P}} = \frac{\phi}{\bigl(1 + \phi^{\,n}\bigr)^{1/n}} .
\tag{8.6}
\end{equation*}

Both describe nearly the same family of curves, with $n \approx w - 0.72$ (Yang et al., 2008); Example 8.3 checks this numerically. Larger $w$ or $n$ means more evaporation for the same climate. The parameter correlates with vegetation cover, rooting depth and soil water storage, with the seasonality of precipitation relative to energy (catchments where rain falls in the cold season evaporate less of it) and with storm characteristics. Figure 8.2 compares the curves.

```{code-cell} python
:tags: [hide-input]
phi = np.linspace(0.02, 5, 400)
schreiber = 1 - np.exp(-phi)
oldekop = phi * np.tanh(1 / phi)
budyko = np.sqrt(schreiber * oldekop)
fu = lambda p, w: 1 + p - (1 + p ** w) ** (1 / w)
mcy = lambda p, n: p / (1 + p ** n) ** (1 / n)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot([0, 1, 5], [0, 1, 1], color=ink, lw=1.2, ls=":")
ax.plot(phi, schreiber, color=c3, lw=1.8, label="Schreiber")
ax.plot(phi, oldekop, color=c2, lw=1.8, label="Ol'dekop")
ax.plot(phi, budyko, color=ink, lw=2.2, label="Budyko")
ax.plot(phi, fu(phi, 2.6), color=c1, lw=2.2, ls="--", label="Fu, w = 2.6")
ax.plot(phi, mcy(phi, 1.9), color=c1, lw=1.4, ls=":", label="Mezentsev–Choudhury–Yang, n = 1.9")
ax.set_xlabel("Aridity index φ = $E_p/P$"); ax.set_ylabel("Evaporative ratio $E/P$")
ax.set_xlim(0, 5); ax.set_ylim(0, 1.05); ax.legend(loc="lower right", fontsize=9)
plt.show()
```

*Figure 8.2. Budyko-type curves of the long-term evaporative ratio against the aridity index, bounded by the energy limit (left dotted line) and the water limit (top dotted line). The parameter-free curves of Schreiber and Ol'dekop bracket Budyko's; Fu's curve with w = 2.6 and the Mezentsev–Choudhury–Yang curve with n = 1.9 nearly coincide with Budyko's.*

### Budyko and the complementary relationship

The two frameworks describe the same physics at different time scales. The complementary relationship relates actual to potential evaporation through the state of the air at daily to monthly scales; the Budyko framework relates long-term evaporation to precipitation and energy supply. If the complementary relationship holds, the long-term apparent potential evaporation in the Budyko aridity index is itself affected by the land's dryness, and the two must be consistent. Formal links between them have been derived. One route revisits two-parameter Budyko equations with the complementary principle so that the surface energy balance is respected (Kim and Chun, 2021). Another uses the Budyko equation to constrain the parameters of the complementary relationship for ungauged regions (Kim et al., 2022; Kim et al., 2024). These links matter for climate-change studies. Potential evaporation computed offline from projected warming rises strongly, and taken at face value suggests widespread drying, but part of that rise is the complementary response to drier land, not an independent driver (Chapter 2).

## 8.5 Evaporation climatology and trends

### The evaporation paradox

Records of pan evaporation from many countries declined over the second half of the twentieth century, while air temperatures rose (Peterson et al., 1995). This seemed to contradict the expectation that a warmer climate would evaporate more water. Two explanations, not mutually exclusive, have been offered:

- **Lower demand.** Solar radiation reaching the ground decreased in many regions ("dimming", linked to aerosols and cloud), and near-surface wind speeds declined, both of which reduce pan evaporation directly (Roderick and Farquhar, 2002).
- **Complementarity.** Where actual evaporation is limited by water, an *increase* in precipitation and actual evaporation cools and moistens the boundary layer, which, by the complementary relationship, *lowers* pan evaporation (Brutsaert and Parlange, 1998).

The lesson is that trends in potential evaporation alone say little about trends in actual evaporation or water availability; both must be interpreted through the coupling between land and air.

### Large-scale estimates

Global and regional maps of actual evaporation now come from several families of products:

- **Remote-sensing energy-balance models**, which use land-surface temperature to partition available energy;
- **Penman–Monteith-type models** with surface resistances scaled by vegetation indices (for example the MODIS product of Mu et al., 2011);
- **Priestley–Taylor-based models** with stress factors from soil moisture (for example GLEAM, Miralles et al., 2011);
- **Reanalyses**;
- **Complementary-relationship models** driven by meteorological data.

They differ by 10–20% even in annual means, and their disagreement is largest in transition zones between humid and arid climates, where evaporation is most sensitive to water supply.

## 8.6 Worked examples

**Example 8.1 — Advection–aridity evaporation.** On a summer day over a large, partly dry plain, $R_n - G$ = 14.0 MJ m⁻² d⁻¹, the air temperature is 26 °C, the vapor pressure deficit is 1.8 kPa and the wind speed at 2 m is 2.5 m s⁻¹. Using the Penman wind function $f(u) = 2.6(1 + 0.54u_2)$ mm d⁻¹ kPa⁻¹ and $\alpha_{PT}$ = 1.26, compute $E_{\text{pen}}$, $E_w$ and the AA estimate of $E$, and compare with the generalized CR.

```{code-cell} python
Rn_G, T, de, u2 = 14.0, 26.0, 1.8, 2.5
w_r = slope(T) / (slope(T) + gam)
E_eq = w_r * Rn_G / lam
E_pen = E_eq + (1 - w_r) * 2.6 * (1 + 0.54 * u2) * de
E_w = 1.26 * E_eq
E_aa = 2 * E_w - E_pen
xg = E_w / E_pen
E_g = E_pen * (2 - xg) * xg ** 2
print(f"E_eq = {E_eq:.2f}, E_w = {E_w:.2f}, E_pen = {E_pen:.2f} mm/d; x = {xg:.2f}")
print(f"AA: E = {E_aa:.2f} mm/d; generalized CR: E = {E_g:.2f} mm/d")
```

The two estimates differ by more than a third at this intermediate value of $x$, where the symmetric and asymmetric forms diverge most. In practice, $\alpha_{PT}$ and the wind function (and, in rescaled versions of the GCR, the bounds of $x$) are calibrated against water-budget or flux data, which reduces the difference.

**Example 8.2 — Inferring wetness from the air.** Two regions receive the same available energy and have the same wet-environment evaporation of 4.5 mm d⁻¹. Region A has an apparent potential evaporation of 5.0 mm d⁻¹, region B of 9.0 mm d⁻¹. Estimate their actual evaporation with the generalized CR.

```{code-cell} python
for name, Ep in [("A", 5.0), ("B", 9.0)]:
    xr = 4.5 / Ep
    print(f"Region {name}: x = {xr:.2f}, E = {Ep * (2 - xr) * xr ** 2:.2f} mm/d (E/E_w = {Ep * (2 - xr) * xr ** 2 / 4.5:.2f})")
```

The region with the higher potential evaporation evaporates less: its high potential rate is the signature of dry land.

**Example 8.3 — Two parameterizations of the Budyko curve.** For values of $w$ in Fu's equation from 1.5 to 4, find the $n$ in the Mezentsev–Choudhury–Yang equation that gives the closest curve over $0.1 \le \phi \le 5$, and check the approximation $n \approx w - 0.72$.

```{code-cell} python
from scipy.optimize import minimize_scalar
ph = np.linspace(0.1, 5, 300)
for w in (1.5, 2.0, 2.6, 3.0, 4.0):
    n_best = minimize_scalar(lambda n: np.sum((mcy(ph, n) - fu(ph, w)) ** 2), bounds=(0.5, 6), method="bounded").x
    err = np.abs(mcy(ph, n_best) - fu(ph, w)).max()
    print(f"w = {w:.1f}: best n = {n_best:.2f} (w - 0.72 = {w - 0.72:.2f}), max difference in E/P = {err:.3f}")
```

**Example 8.4 — A catchment on the Budyko plane.** A catchment receives 1250 mm a year with a wet-environment evaporation of 850 mm, and its mean runoff is 590 mm. Find its evaporative ratio, aridity index and Fu parameter, and its predicted runoff if precipitation falls by 10%, assuming $w$ stays constant.

```{code-cell} python
from scipy.optimize import brentq
P, Ep, Q = 1250.0, 850.0, 590.0
phi_c = Ep / P; EP = (P - Q) / P
w_c = brentq(lambda w: fu(phi_c, w) - EP, 1.01, 20)
Q_new = 0.9 * P * (1 - fu(Ep / (0.9 * P), w_c))
print(f"E/P = {EP:.3f}, φ = {phi_c:.2f}, w = {w_c:.2f}; runoff after -10% P: {Q_new:.0f} mm ({100 * (Q_new / Q - 1):+.0f}%)")
```

## 8.7 Lab (planned)

A Colab notebook will compute daily apparent potential evaporation, wet-environment evaporation and actual evaporation from the advection–aridity and generalized complementary relationships, using ERA5 reanalysis fields of radiation, temperature, dew point and wind. It will then aggregate them over catchments and compare them with long-term water-budget evaporation and with Budyko curves.

## 8.8 Common mistakes

- **Treating apparent potential evaporation as an external forcing over large regions.** It responds to the land's dryness.
- **Comparing studies that use different definitions of $E_p$.** Penman, FAO-56 and Priestley–Taylor rates can differ by 30% in dry climates.
- **Using the symmetric CR in very dry conditions.** It can predict negative evaporation; use a form with a proper dry-end boundary condition.
- **Reading a Budyko parameter fitted in one climate as transferable.** It reflects vegetation, soils and seasonality as well as the climate.
- **Inferring drying from rising potential evaporation alone.** Check precipitation and actual evaporation, or use a framework that couples them.

## 8.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("w1", 2 * 4.0 - 6.5, display=False)
xw = 4.0 / 6.5; glue("w2", 6.5 * (2 - xw) * xw ** 2, display=False)
glue("w3_s", float(1 - np.exp(-1.5)), display=False); glue("w3_o", float(1.5 * np.tanh(1 / 1.5)), display=False)
glue("w3_f", float(fu(1.5, 2.6)), display=False)
glue("w4", float(brentq(lambda w: fu(0.8, w) - 0.62, 1.01, 20)), display=False)
glue("w5", 1100 - 480 - 25 - 0.0, display=False)
```

1. A region has $E_w$ = 4.0 mm d⁻¹ and $E_p$ = 6.5 mm d⁻¹. Estimate $E$ with the symmetric complementary relationship.
2. Repeat Problem 1 with the generalized complementary relationship.
3. For an aridity index of 1.5, compute $E/P$ from the Schreiber, Ol'dekop and Fu ($w$ = 2.6) curves.
4. A catchment with an aridity index of 0.8 has $E/P$ = 0.62. Find its Fu parameter.
5. Over a year, a large basin receives 1100 mm of precipitation, discharges 480 mm and gains 25 mm of total water storage (from satellite gravimetry). Estimate its evaporation.

```{admonition} Answers
:class: dropdown
1. {glue:text}`w1:.1f` mm d⁻¹.
2. {glue:text}`w2:.2f` mm d⁻¹.
3. Schreiber {glue:text}`w3_s:.3f`; Ol'dekop {glue:text}`w3_o:.3f`; Fu {glue:text}`w3_f:.3f`.
4. $w$ = {glue:text}`w4:.2f`.
5. {glue:text}`w5:.0f` mm.
```

## 8.10 Open questions

- **The physical basis of the complementary relationship.** What sets the shape of the curve, and how does it depend on the boundary layer, the surface and advection?
- **Potential evaporation in a warming climate.** How much of the projected rise in potential evaporation is a feedback from drier land, and how should aridity indices be corrected for it?
- **What controls the Budyko parameter?** Can it be predicted from vegetation, soils and climate seasonality, and will it change as ecosystems adjust?

## 8.11 Key papers

- Bouchet, R. J. (1963). Évapotranspiration réelle et potentielle, signification climatique. *International Association of Scientific Hydrology Publication*, 62, 134–142.
- Brutsaert, W. (2015). A generalized complementary principle with physical constraints for land-surface evaporation. *Water Resources Research*, 51, 8087–8093.
- Brutsaert, W., and Parlange, M. B. (1998). Hydrologic cycle explains the evaporation paradox. *Nature*, 396, 30.
- Brutsaert, W., and Stricker, H. (1979). An advection-aridity approach to estimate actual regional evapotranspiration. *Water Resources Research*, 15, 443–450.
- Im, S., Choi, M., and Kim, D. (2026). Persistent performance of the power-function complementary relationship in Earth system models under a warming scenario. *Journal of Hydrology*, 677, 135813.
- Kim, D., and Chun, J. A. (2021). Revisiting a two-parameter Budyko equation with the complementary evaporation principle for proper consideration of surface energy balance. *Water Resources Research*, 57, e2021WR030838.
- Kim, D., Choi, M., and Chun, J. A. (2022). Linking the complementary evaporation relationship with the Budyko framework for ungauged areas in Australia. *Hydrology and Earth System Sciences*, 26, 5955–5969.
- Kim, D., Cho, E., and Chun, J. A. (2024). Constraining the power-function complementary relationship with a steady-state Budyko equation for predicting terrestrial evapotranspiration globally. *Agricultural and Forest Meteorology*, 344, 109808.
- Kim, D., Lee, W.-S., Kim, S. T., and Chun, J. A. (2019). Historical drought assessment over the contiguous United States using the generalized complementary principle of evapotranspiration. *Water Resources Research*, 55, 6244–6267.
- Miralles, D. G., et al. (2011). Global land-surface evaporation estimated from satellite-based observations. *Hydrology and Earth System Sciences*, 15, 453–469.
- Mu, Q., Zhao, M., and Running, S. W. (2011). Improvements to a MODIS global terrestrial evapotranspiration algorithm. *Remote Sensing of Environment*, 115, 1781–1800.
- Peterson, T. C., Golubev, V. S., and Groisman, P. Ya. (1995). Evaporation losing its strength. *Nature*, 377, 687–688.
- Roderick, M. L., and Farquhar, G. D. (2002). The cause of decreased pan evaporation over the past 50 years. *Science*, 298, 1410–1411.
- Yang, H., Yang, D., Lei, Z., and Sun, F. (2008). New analytical derivation of the mean annual water-energy balance equation. *Water Resources Research*, 44, W03410.

## 8.12 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The sections on complementarity, water-budget methods and evaporation climatology.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on evapotranspiration.
- Budyko, M. I. (1974). *Climate and Life*. Academic Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
