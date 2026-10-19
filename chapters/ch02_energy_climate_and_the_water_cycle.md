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

# 2. Energy, climate and the water cycle

```{admonition} Core ideas
:class: tip
1. The sun and the Earth radiate in two almost separate bands, **shortwave** and **longwave**. That separation is what allows the atmosphere to be nearly transparent to sunlight but nearly opaque to the Earth's own radiation: the **greenhouse effect**.
2. Evaporating the world's annual precipitation uses about **80 W m⁻²**, nearly a quarter of the solar energy reaching the top of the atmosphere. The global energy and water cycles are the same cycle seen from two sides.
3. The tropics gain more radiative energy than they lose and the poles lose more than they gain. The **poleward transport** of about 5 PW by the atmosphere and oceans organizes the general circulation, and with it the belts of wet and dry climates.
4. The ratio of the energy supply to the water supply, the **aridity index**, controls how precipitation is partitioned into evaporation and runoff. The same relation predicts that **runoff in dry regions is more sensitive** to a change in precipitation.
5. The **critical zone**, from the treetops to the bottom of the groundwater, is where water, energy, rock and life interact; its structure is itself a product of climate.
6. In a warmer climate, atmospheric water vapor rises by about 7% per degree, but global precipitation by only 1–3%, because it is limited by the energy available to evaporate water. The pattern of $P - E$, by contrast, scales with vapor, so **wet regions tend to get wetter and dry regions drier**.
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

## 2.1 Radiation laws

Every body above absolute zero emits electromagnetic radiation. A **black body**, which absorbs all radiation falling on it, emits at each wavelength $\lambda_w$ the maximum possible amount, given by **Planck's law**:

\begin{equation*}
E_\lambda(T) = \frac{2\pi h c^2}{\lambda_w^5}\,\frac{1}{\exp\!\left(\dfrac{h c}{\lambda_w k_B T}\right) - 1},
\tag{2.1}
\end{equation*}

where $E_\lambda$ is the emitted energy flux per unit wavelength (W m⁻² m⁻¹), $T$ the absolute temperature, $h$ Planck's constant, $c$ the speed of light and $k_B$ Boltzmann's constant. Two results follow from it.

- **Wien's displacement law.** The wavelength of maximum emission is inversely proportional to temperature, $\lambda_{\max} = b/T$ with $b = 2898$ µm K.
- **The Stefan–Boltzmann law.** Integrating over all wavelengths gives the total emission $\sigma T^4$, with $\sigma = 5.67\times10^{-8}$ W m⁻² K⁻⁴. A real surface emits $\varepsilon\sigma T^4$, where the emissivity $\varepsilon \le 1$.

The sun, with an effective surface temperature of about 5800 K, emits most strongly near 0.5 µm. The Earth, at about 255–300 K, emits most strongly near 10 µm. Figure 2.1 shows that the two spectra hardly overlap, which is why atmospheric scientists and hydrologists treat **shortwave** (solar, below about 4 µm) and **longwave** (terrestrial, above about 4 µm) radiation as separate fluxes. The calculation of the daily solar radiation reaching the top of the atmosphere at any latitude and date is given in Appendix D and was used in the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch02_water_and_energy_at_the_land_surface.html) book.

```{code-cell} python
:tags: [hide-input]
h, c, kB = 6.62607e-34, 2.99792e8, 1.380649e-23
def planck(lam_m, T):
    with np.errstate(over="ignore"):                      # exp overflows harmlessly at short wavelengths
        return 2 * np.pi * h * c ** 2 / lam_m ** 5 / np.expm1(h * c / (lam_m * kB * T))
lam = np.logspace(-1, 2, 600) * 1e-6                       # 0.1 to 100 µm
sun = planck(lam, 5800.0) * lam; earth = planck(lam, 288.0) * lam   # λ·Eλ: area under a log axis = energy
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.semilogx(lam * 1e6, sun / sun.max(), color=c2, lw=2.4, label="Sun, 5800 K")
ax.semilogx(lam * 1e6, earth / earth.max(), color=c1, lw=2.4, label="Earth's surface, 288 K")
ax.axvline(4.0, color=ink2, lw=0.9, ls="--"); ax.text(4.3, 0.6, "4 µm", color=ink2, fontsize=9.5)
ax.text(0.105, 0.85, "shortwave", color=c2, fontsize=10); ax.text(30, 0.85, "longwave", color=c1, fontsize=10)
ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Normalized λ·E$_λ$")
ax.set_xlim(0.1, 100); ax.set_ylim(0, 1.35); ax.legend(loc="upper center", fontsize=9.5, ncol=2)
plt.show()
lam_all = np.logspace(-2, 3, 20000) * 1e-6
frac_sun = np.trapezoid(planck(lam_all[lam_all < 4e-6], 5800.0), lam_all[lam_all < 4e-6]) / np.trapezoid(planck(lam_all, 5800.0), lam_all)
frac_earth = np.trapezoid(planck(lam_all[lam_all > 4e-6], 288.0), lam_all[lam_all > 4e-6]) / np.trapezoid(planck(lam_all, 288.0), lam_all)
glue("f_sun", float(frac_sun * 100), display=False); glue("f_earth", float(frac_earth * 100), display=False)
```

*Figure 2.1. Black-body emission spectra of the sun and the Earth's surface, each normalized to its maximum and plotted as λ·E$_λ$ so that equal areas under the curves on the logarithmic axis represent equal energy. Of the sun's emission, {glue:text}`f_sun:.0f`% lies below 4 µm; of the Earth's, {glue:text}`f_earth:.1f`% lies above it.*

## 2.2 The planetary energy budget

### Effective temperature

Averaged over the globe, the Earth intercepts solar radiation on a disk of area $\pi R^2$ but emits from a sphere of area $4\pi R^2$. With a solar constant $S_0 \approx 1361$ W m⁻² and a planetary albedo $\alpha_p \approx 0.30$, the balance between absorbed and emitted radiation,

\begin{equation*}
\frac{S_0}{4}\,(1 - \alpha_p) = \sigma T_e^4 ,
\tag{2.2}
\end{equation*}

gives an **effective emission temperature** $T_e \approx 255$ K, or −18 °C. The observed global mean surface temperature is about 15 °C. The 33 K difference is the greenhouse effect.

### A layered greenhouse

The simplest model that explains the difference treats the atmosphere as a single layer that is transparent to shortwave radiation and absorbs a fraction $\varepsilon_a$ of the longwave radiation from the surface, emitting $\varepsilon_a \sigma T_a^4$ both upward and downward. Energy balances for the top of the atmosphere, the layer and the surface give

\begin{equation*}
T_s = T_e\left(\frac{2}{2 - \varepsilon_a}\right)^{1/4} .
\tag{2.3}
\end{equation*}

A fully absorbing layer ($\varepsilon_a = 1$) raises the surface temperature by a factor $2^{1/4}$, to about 303 K. The observed 288 K corresponds to $\varepsilon_a \approx 0.78$ (Example 2.2). The model is crude: it ignores convection, which carries much of the surface energy upward as sensible and latent heat. But it captures the essential point that the surface is warmed by downward longwave radiation from the atmosphere, about 340 W m⁻² globally, more than the solar radiation it absorbs.

### Radiative forcing and sensitivity

Increasing the concentration of carbon dioxide reduces the longwave radiation leaving the top of the atmosphere. The resulting imbalance, the **radiative forcing**, is approximated by

\begin{equation*}
\Delta F = 5.35 \ln\!\left(\frac{C}{C_0}\right) \quad (\text{W m}^{-2}),
\tag{2.4}
\end{equation*}

so a doubling of CO₂ gives about 3.7 W m⁻² (Myhre et al., 1998). Without feedbacks, the warming needed to restore balance follows from differentiating the Stefan–Boltzmann law at $T_e$: $\Delta T = \Delta F / (4\sigma T_e^3) \approx 1.0$ K (Example 2.3); calculations that resolve the vertical temperature structure of the atmosphere give about 1.2 K. Feedbacks, above all the increase of water vapor with temperature (Section 2.6), roughly double to triple this value.

### Energy and water at the surface

At the Earth's surface, the absorbed shortwave and the net longwave radiation are balanced mostly by evaporation. Global estimates of the latent heat flux are about 80–88 W m⁻² (for example Trenberth et al., 2009). This number can be checked against the water cycle. Global precipitation of about 1 m per year must be evaporated first, which requires

\begin{equation*}
\lambda E = \lambda\,\rho_w\,\frac{1\ \text{m}}{1\ \text{yr}} = 2.45\times10^{6}\ \text{J kg}^{-1}\times 1000\ \text{kg m}^{-3}\times \frac{1\ \text{m}}{3.156\times10^{7}\ \text{s}} \approx 78\ \text{W m}^{-2}.
\tag{2.5}
\end{equation*}

The two estimates agree within their uncertainty. Evaporation is the largest single use of the energy absorbed by the surface, and the global water cycle is driven by, and limited by, the energy budget.

## 2.3 The general circulation

### Why the atmosphere moves

The planetary balance of Section 2.2 holds only for the globe as a whole. At low latitudes the sun is high and the absorbed shortwave radiation exceeds the outgoing longwave radiation; at high latitudes the reverse is true. The atmosphere and the oceans carry the surplus poleward. A simple **energy balance model** shows how much must be carried. It treats the surface temperature $T(\phi)$ as a function of latitude only, represents the outgoing longwave radiation by a linear fit $A + B\,T$ (with $T$ in °C), and represents the poleward transport as diffusion of heat. Writing $x = \sin\phi$, the steady balance is

\begin{equation*}
\frac{S_0}{4}\,s(x)\,\bigl[1 - \alpha(x)\bigr] - (A + B\,T) + \frac{d}{dx}\!\left[D\,(1 - x^2)\,\frac{dT}{dx}\right] = 0 ,
\tag{2.6}
\end{equation*}

where $s(x)$ describes the annual-mean distribution of solar radiation with latitude and $D$ is a diffusion coefficient (W m⁻² K⁻¹). The poleward energy transport across a latitude circle is $\mathcal{H} = -2\pi R^2 D (1 - x^2)\, dT/dx$. Figure 2.2 shows the solution.

```{code-cell} python
:tags: [hide-input]
def ebm(D, n=180, S0=1361.0, A=203.3, B=2.09):
    """Steady diffusive energy balance model on a grid in x = sin(latitude); finite volumes."""
    xf = np.linspace(-1, 1, n + 1); xc = 0.5 * (xf[1:] + xf[:-1]); dx = xf[1] - xf[0]
    s = 1 - 0.482 * 0.5 * (3 * xc ** 2 - 1)              # annual-mean insolation shape
    alb = 0.30 + 0.12 * xc ** 2                          # brighter toward the poles
    Q_abs = S0 / 4 * s * (1 - alb)
    w = D * (1 - xf[1:-1] ** 2) / dx ** 2
    M = np.diag(np.full(n, B))
    for k in range(n - 1):
        M[k, k] += w[k]; M[k + 1, k + 1] += w[k]; M[k, k + 1] -= w[k]; M[k + 1, k] -= w[k]
    T = np.linalg.solve(M, Q_abs - A)
    H = -2 * np.pi * 6.371e6 ** 2 * D * (1 - xf[1:-1] ** 2) * np.diff(T) / dx / 1e15   # PW
    return xc, T, xf[1:-1], H, Q_abs, A + B * T

xc, T, xh, H, Q_abs, OLR = ebm(0.55)
lat_c, lat_h = np.degrees(np.arcsin(xc)), np.degrees(np.arcsin(xh))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
a1.plot(lat_c, Q_abs, color=c2, lw=2.2, label="Absorbed shortwave")
a1.plot(lat_c, OLR, color=c1, lw=2.2, label="Outgoing longwave")
a1.fill_between(lat_c, Q_abs, OLR, where=Q_abs > OLR, color=c2, alpha=0.15)
a1.fill_between(lat_c, Q_abs, OLR, where=Q_abs < OLR, color=c1, alpha=0.15)
a1.set_ylabel("W m⁻²"); a1.legend(loc="lower center", fontsize=9.5); a1.set_title("(a) Radiation balance")
a2.plot(lat_h, H, color=ink, lw=2.2)
a2.axhline(0, color=ink2, lw=0.8)
a2.set_ylabel("Northward transport (PW)"); a2.set_title("(b) Poleward energy transport")
a2.set_xlabel("Latitude"); a2.set_xlim(-90, 90)
a2.set_xticks([-90, -60, -30, 0, 30, 60, 90], ["90° S", "60° S", "30° S", "0°", "30° N", "60° N", "90° N"])
plt.show()
glue("ebm_eq", float(T[len(T) // 2]), display=False); glue("ebm_pole", float(T[-1]), display=False)
glue("ebm_H", float(H.max()), display=False); glue("ebm_lat", float(lat_h[np.argmax(H)]), display=False)
```

*Figure 2.2. A diffusive energy balance model of the Earth. (a) Absorbed solar radiation exceeds outgoing longwave radiation between about 35° S and 35° N; poleward of these latitudes the reverse holds. (b) The implied poleward transport of energy peaks at {glue:text}`ebm_H:.1f` PW near {glue:text}`ebm_lat:.0f`° latitude. The modeled surface temperature is {glue:text}`ebm_eq:.0f` °C at the equator and {glue:text}`ebm_pole:.0f` °C at the poles.*

Observations show a peak transport of about 5–6 PW, carried by the atmosphere and the oceans together. In the atmosphere, part of it travels as **latent heat**: water evaporated in the subtropics and carried poleward and equatorward condenses elsewhere and releases its energy there. The water and energy cycles are again inseparable.

### Cells, belts and monsoons

The transport is organized into a few large circulations. In the **Hadley cells**, air rises near the equator in the **Intertropical Convergence Zone** (ITCZ), with deep convection and heavy rain, moves poleward aloft, and sinks near 20–30° latitude in the **subtropical highs**, where the descending air is warmed by compression and the great deserts lie. In the mid-latitudes, energy is carried by traveling cyclones and anticyclones embedded in the **westerlies**; their fronts bring most of the mid-latitude precipitation. The seasonal migration of the ITCZ with the sun, and the different heating of land and sea, produce the **monsoons**. In East Asia, the summer monsoon brings moist air from the Pacific and Indian Oceans along a quasi-stationary front, the Meiyu–Baiu–Changma front, that moves northward from May to July.

### Teleconnections

The circulation also varies from year to year in coherent patterns that link distant regions, called **teleconnections**. The best known is the **El Niño–Southern Oscillation** (ENSO): changes in sea-surface temperature in the tropical Pacific shift the regions of tropical convection and, through atmospheric waves, alter precipitation and temperature across much of the globe. Teleconnections are the main source of predictability of seasonal water resources, and they mean that hydrological anomalies in different basins are often correlated.

## 2.4 Aridity and the partition of precipitation

Over a long period, the precipitation on a land area is divided into evaporation and runoff, $\bar{P} = \bar{E} + \bar{Q}$ (Chapter 1). What controls the split? Evaporation needs both water and energy. Its upper limits are the water supply, $\bar{E} \le \bar{P}$, and the energy supply, $\bar{E} \le \bar{E}_p$, where $\bar{E}_p$ is the evaporation that the available energy could sustain (Chapters 7–8). The ratio

\begin{equation*}
\phi = \frac{\bar{E}_p}{\bar{P}}
\tag{2.7}
\end{equation*}

is the **aridity index**: $\phi < 1$ in humid climates, where energy is scarce relative to water, and $\phi > 1$ in arid ones. Budyko (1974) showed that long-term evaporation from large regions follows a curve $\bar{E}/\bar{P} = F(\phi)$ that approaches both limits: $F \to \phi$ as $\phi \to 0$ and $F \to 1$ as $\phi \to \infty$. Chapter 8 develops this framework in detail. Here we use it to answer a question about climate.

### Runoff elasticity

How much does runoff change when precipitation changes? The **precipitation elasticity of runoff** is

\begin{equation*}
\varepsilon_P = \frac{\partial \bar{Q}/\bar{Q}}{\partial \bar{P}/\bar{P}} ,
\tag{2.8}
\end{equation*}

the percentage change in runoff per percent change in precipitation. With $\bar{Q} = \bar{P}\,[1 - F(\phi)]$ and $\phi = \bar{E}_p/\bar{P}$, holding $\bar{E}_p$ fixed,

\begin{equation*}
\frac{\partial \bar{Q}}{\partial \bar{P}} = 1 - F(\phi) + \phi\,F'(\phi),
\qquad
\varepsilon_P = 1 + \frac{\phi\,F'(\phi)}{1 - F(\phi)} .
\tag{2.9}
\end{equation*}

Since $F' > 0$, the elasticity always exceeds 1: runoff changes proportionally more than precipitation. It grows with aridity, so **runoff in dry regions is far more sensitive** to a change in precipitation than in wet ones (Figure 2.3). Using Fu's form of the curve, $F(\phi) = 1 + \phi - (1 + \phi^w)^{1/w}$, the elasticity can be computed for any climate.

```{code-cell} python
:tags: [hide-input]
def fu(phi, w):
    return 1 + phi - (1 + phi ** w) ** (1 / w)
def fu_prime(phi, w):
    return 1 - phi ** (w - 1) * (1 + phi ** w) ** (1 / w - 1)
def elasticity(phi, w):
    return 1 + phi * fu_prime(phi, w) / (1 - fu(phi, w))

phi = np.linspace(0.1, 3.0, 300)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for w, col in [(1.8, c3), (2.6, c1), (3.5, c2)]:
    ax.plot(phi, elasticity(phi, w), color=col, lw=2.2, label=f"w = {w}")
ax.axhline(1, color=ink2, lw=0.8, ls=":")
ax.set_xlabel("Aridity index φ = $E_p/P$"); ax.set_ylabel("Precipitation elasticity of runoff $ε_P$")
ax.set_xlim(0.1, 3); ax.set_ylim(0, 6); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("el_wet", float(elasticity(0.5, 2.6)), display=False); glue("el_dry", float(elasticity(2.0, 2.6)), display=False)
```

*Figure 2.3. Precipitation elasticity of runoff predicted by Fu's form of the Budyko curve. With $w$ = 2.6, a humid catchment ($\phi$ = 0.5) has an elasticity of {glue:text}`el_wet:.1f`, while a dry one ($\phi$ = 2) has {glue:text}`el_dry:.1f`: a 10% drop in precipitation would reduce its runoff by about a quarter.*

The same reasoning gives the sensitivity of runoff to a change in potential evaporation, $\varepsilon_{E_p} = -\phi F'(\phi)/(1 - F(\phi)) = 1 - \varepsilon_P$, which is negative and also grows in magnitude with aridity.

## 2.5 The critical zone

The **critical zone** is the near-surface layer of the Earth in which rock, soil, water, air and living organisms interact: from the top of the vegetation canopy down to the base of the actively circulating groundwater. Almost all the processes in this book take place within it. Its structure is not fixed. Over thousands to millions of years, climate weathers rock into soil, water carves valleys and drainage networks, and vegetation adapts its rooting depth and leaf area to the water and energy available. The result is a landscape whose properties are correlated with the climate that shaped it:

- deep, weathered soils with large water storage in warm, humid regions;
- thin soils and sparse vegetation in arid ones;
- plants whose roots reach deeper where rainfall is seasonal.

This **coevolution** is one reason why catchment behavior is more regular than the variety of landscapes might suggest, and why parameters such as $w$ in Fu's equation correlate with climate and vegetation (Chapter 8). It also means that a catchment's response to a change in climate may itself change, as vegetation and soils adjust over decades.

## 2.6 The water cycle in a warmer climate

### Water vapor

The saturation vapor pressure increases with temperature by about 6–7% per kelvin near the Earth's surface, following the Clausius–Clapeyron relation (Chapter 3). Observations and climate models agree that relative humidity in the lower troposphere changes little as the climate warms, so the **atmospheric water vapor content rises at roughly the same rate**, about 7% per kelvin. Water vapor is itself the main greenhouse gas, and this increase is the largest positive feedback in the climate system.

### Precipitation

Global mean precipitation must equal global evaporation, and evaporation is limited by the energy that the surface and the atmosphere can lose. The atmosphere cools radiatively at a rate that changes only slowly with temperature, so global precipitation increases by only about **1–3% per kelvin**, much less than water vapor. The difference means that the atmosphere becomes more humid while its overturning circulation slows down.

### The pattern of $P - E$

The net flux of water into the land and out of the oceans, $P - E$, is set by the transport of vapor by the circulation (Equation 1.11). If the circulation changes little while the vapor it carries increases by a fraction $\alpha_{CC}\,\Delta T$ (with $\alpha_{CC} \approx 0.07$ K⁻¹), then the existing pattern of $P - E$ is simply amplified (Held and Soden, 2006):

\begin{equation*}
\Delta(P - E) \approx \alpha_{CC}\,\Delta T\,(P - E) .
\tag{2.10}
\end{equation*}

Regions where $P > E$ (the ITCZ, the mid-latitude storm tracks) tend to get wetter, and regions where $P < E$ (the subtropics) drier. "Wet gets wetter, dry gets drier" is a useful first approximation over the oceans. It is much less reliable over land, where $P - E$ is limited by the water available to evaporate and where changes in the circulation and in land cover matter. The intensity of heavy rainfall, which depends on the vapor available to individual storms, tends to follow the 7% per kelvin rate or more (Chapter 5).

## 2.7 Worked examples

**Example 2.1 — Where the spectra peak.** Use Wien's law to find the wavelength of maximum emission of the sun (5800 K), a land surface at 30 °C and a snow surface at −10 °C, and check the first against the maximum of Planck's law computed numerically.

```{code-cell} python
b = 2898.0                                        # µm K
for name, T in [("Sun", 5800.0), ("Land at 30 °C", 303.15), ("Snow at −10 °C", 263.15)]:
    print(f"{name:15s} λmax = {b / T:7.3f} µm")
lam = np.linspace(0.1, 3, 200_000) * 1e-6
print(f"Numerical maximum of Planck's law at 5800 K: {lam[np.argmax(planck(lam, 5800.0))] * 1e6:.3f} µm")
```

**Example 2.2 — A one-layer greenhouse.** Compute the effective temperature for $S_0$ = 1361 W m⁻² and $\alpha_p$ = 0.30, and find the longwave absorptivity of a single atmospheric layer that gives a surface temperature of 288 K.

```{code-cell} python
sigma, S0, ap = 5.670e-8, 1361.0, 0.30
Te = (S0 / 4 * (1 - ap) / sigma) ** 0.25
eps_a = 2 - 2 * (Te / 288.0) ** 4
print(f"Te = {Te:.1f} K ({Te - 273.15:.1f} °C); ε_a = {eps_a:.3f}; with ε_a = 1, Ts = {Te * 2 ** 0.25:.1f} K")
```

**Example 2.3 — No-feedback warming.** Estimate the radiative forcing and the no-feedback warming for an increase of CO₂ from 280 ppm to 420 ppm and to 560 ppm.

```{code-cell} python
for C in (420.0, 560.0):
    dF = 5.35 * np.log(C / 280.0)
    print(f"C = {C:.0f} ppm: ΔF = {dF:.2f} W/m², ΔT (no feedback) = {dF / (4 * sigma * Te ** 3):.2f} K")
```

**Example 2.4 — Energy for the water cycle.** Global mean precipitation is about 990 mm per year (500,000 km³ over 510 million km²). How much energy per unit area does evaporating it require, and what fraction is that of the 340 W m⁻² of solar radiation arriving at the top of the atmosphere?

```{code-cell} python
P_glob = 500e3 / 510e6 * 1e6                      # mm/yr
LE = 2.45e6 * P_glob / (365.25 * 86400)           # W/m² (1 mm = 1 kg/m²)
print(f"P = {P_glob:.0f} mm/yr -> λE = {LE:.0f} W/m², {LE / 340:.0%} of incoming solar radiation")
```

**Example 2.5 — Runoff sensitivity.** A catchment receives 1200 mm of precipitation a year and has a potential evaporation of 900 mm; another receives 600 mm with 1200 mm of potential evaporation. With $w$ = 2.6, estimate the runoff of each and the change in runoff if precipitation decreases by 10%, from the elasticity and exactly.

```{code-cell} python
for P, Ep in [(1200.0, 900.0), (600.0, 1200.0)]:
    ph = Ep / P
    Q = P * (1 - fu(ph, 2.6)); eps = elasticity(ph, 2.6)
    Q_new = 0.9 * P * (1 - fu(Ep / (0.9 * P), 2.6))
    print(f"P = {P:.0f}, Ep = {Ep:.0f}: φ = {ph:.2f}, Q = {Q:.0f} mm, ε_P = {eps:.2f}; "
          f"-10% P -> {-10 * eps:.0f}% (linear), {100 * (Q_new / Q - 1):.0f}% (exact)")
```

## 2.8 Common mistakes

- **Using the solar constant as the mean incoming radiation.** Averaged over the sphere it is $S_0/4 \approx 340$ W m⁻².
- **Forgetting the downward longwave radiation.** At the surface it exceeds the absorbed solar radiation; it is the reason the surface is warmer than $T_e$.
- **Assuming precipitation follows Clausius–Clapeyron.** Water vapor does; global precipitation is limited by energy and rises much more slowly. Heavy short-duration rainfall is a different matter.
- **Reading elasticity as a constant.** It depends on aridity and on catchment properties ($w$), and the linear estimate becomes inaccurate for large changes.
- **Applying "wet gets wetter" over land without checking water availability.** Over land, $P - E$ cannot be negative for long.

## 2.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("q1", 2898.0 / 6000.0, display=False)
Te_v = (2601.0 / 4 * (1 - 0.76) / sigma) ** 0.25
glue("q2", float(Te_v), display=False)
glue("q3", float(3.0 / (5.35 * np.log(2) / (4 * sigma * Te ** 3))), display=False)
glue("q4_e", float(elasticity(1.0, 2.6)), display=False); glue("q4_q", float(5 * elasticity(1.0, 2.6)), display=False)
glue("q5", 0.07 * 2.0 * 600, display=False)
```

1. Find the wavelength of maximum emission of a black body at 6000 K.
2. Venus receives 2601 W m⁻² from the sun and has a planetary albedo of about 0.76. Compute its effective temperature. (Its surface temperature is about 740 K; what does that say about its atmosphere?)
3. If the equilibrium warming for a doubling of CO₂ is 3.0 K, by what factor do feedbacks amplify the no-feedback response of Example 2.3?
4. A catchment has an aridity index of 1.0 and $w$ = 2.6. Find its precipitation elasticity of runoff, and the percentage change in runoff for a 5% increase in precipitation.
5. A tropical ocean region has a long-term $P - E$ of 600 mm per year. Use Equation (2.10) to estimate its change for 2 K of warming.

```{admonition} Answers
:class: dropdown
1. About {glue:text}`q1:.2f` µm.
2. About {glue:text}`q2:.0f` K. The surface is far warmer than the effective temperature: a very strong greenhouse effect.
3. About {glue:text}`q3:.1f` times.
4. $\varepsilon_P$ = {glue:text}`q4_e:.2f`; runoff increases by about {glue:text}`q4_q:.0f`%.
5. About +{glue:text}`q5:.0f` mm per year.
```

## 2.10 Open questions

- **How much will precipitation and runoff change?** The thermodynamic part (more vapor) is robust; the dynamic part (changes in circulation) is not, and it dominates regional projections over land.
- **Does the Budyko curve itself change?** If vegetation and soils coevolve with climate, the parameter $w$ may shift as the climate changes, altering the sensitivities of Figure 2.3.
- **Closing the surface energy budget.** Global estimates of surface fluxes still carry uncertainties of 10–20 W m⁻², comparable to the changes caused by greenhouse forcing.

## 2.11 Key papers

- Budyko, M. I. (1974). *Climate and Life*. Academic Press.
- Held, I. M., and Soden, B. J. (2006). Robust responses of the hydrological cycle to global warming. *Journal of Climate*, 19, 5686–5699.
- Myhre, G., Highwood, E. J., Shine, K. P., and Stordal, F. (1998). New estimates of radiative forcing due to well mixed greenhouse gases. *Geophysical Research Letters*, 25, 2715–2718.
- North, G. R., Cahalan, R. F., and Coakley, J. A. (1981). Energy balance climate models. *Reviews of Geophysics and Space Physics*, 19, 91–121.
- Sankarasubramanian, A., Vogel, R. M., and Limbrunner, J. F. (2001). Climate elasticity of streamflow in the United States. *Water Resources Research*, 37, 1771–1781.
- Trenberth, K. E., Fasullo, J. T., and Kiehl, J. (2009). Earth's global energy budget. *Bulletin of the American Meteorological Society*, 90, 311–324.

## 2.12 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on the global context.
- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The sections on the global water balance and on climate.
- Hartmann, D. L. (2016). *Global Physical Climatology* (2nd ed.). Elsevier.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
