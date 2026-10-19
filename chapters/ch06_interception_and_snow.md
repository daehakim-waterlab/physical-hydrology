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

# 6. Interception and snow

```{admonition} Core ideas
:class: tip
1. A vegetation canopy stores a millimeter or two of rain and returns it to the air by evaporation. Because a wet canopy has no stomatal resistance and a small aerodynamic resistance, **interception loss** from forests is typically 10–30% of rainfall, more than the canopy storage suggests.
2. A running water balance of the canopy store (**Rutter-type** models) and its analytical storm-by-storm solution (**Gash-type** models) both follow from conservation of mass with an evaporation rate proportional to the wetted fraction.
3. A snowpack must first be warmed to 0 °C (its **cold content**) and wetted to its liquid-holding capacity before it releases water. Melt is the residual of the snowpack **energy balance**: 1 W m⁻² sustained for a day melts 0.26 mm of water.
4. Net radiation dominates melt under clear skies, turbulent fluxes in warm, windy and humid weather, and rain adds heat and water at once. A high, fast-decaying **albedo** makes fresh snow remarkably resistant to melt.
5. **Temperature-index** models work because air temperature is correlated with the main energy inputs. Their **degree-day factor** is an effective parameter that can be derived from, and checked against, the energy balance.
6. Warming shifts precipitation from snow to rain and moves melt earlier, and the effect is largest where winter temperatures are near 0 °C.
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

## 6.1 Interception

### Partitioning rain in a canopy

Rain falling on vegetation is divided into three parts:

- **free throughfall**, which passes through gaps without touching the leaves, a fraction $p$ of the gross rainfall;
- water caught by the canopy, which either evaporates from the wet surfaces during and after the storm (**interception loss**), or drips to the ground (**canopy drainage**);
- **stemflow**, which runs down branches and trunks.

The canopy holds a limited amount of water, its **storage capacity** $S_c$, typically 0.5–2 mm for forests and less for crops and grasses, roughly proportional to the leaf area index. Interception loss is a true loss to the catchment: the water returns to the atmosphere without reaching the soil.

### Why interception loss is large

A storage of 1–2 mm per storm seems small, but interception loss from forests is often 10–30% of gross rainfall, and up to 40% in some coniferous forests in maritime climates. The reason is the high evaporation rate from a wet canopy. In the Penman–Monteith equation (Chapter 7), a wet canopy has no stomatal resistance, and a tall, rough forest has a very small aerodynamic resistance (Chapter 4). Wet-canopy evaporation then responds strongly to the vapor pressure deficit and can continue during rain, at night and in winter. Rates of 0.2–0.5 mm h⁻¹ during rain are common, and can exceed the available radiant energy, the deficit being supplied by sensible heat drawn from the air.

### A running water balance: Rutter-type models

Let $C$ be the water stored on the canopy. Its balance during rain of intensity $R$ is

\begin{equation*}
\frac{dC}{dt} = (1 - p)\,R - D(C) - E_w\,\frac{C}{S_c} ,
\tag{6.1}
\end{equation*}

where $E_w$ is the evaporation rate from a fully wet canopy and the factor $C/S_c$ represents the wetted fraction. Rutter et al. (1971) expressed the drainage as $D = D_s\,\exp[b\,(C - S_c)]$, which is small until the canopy is nearly full and grows steeply when $C$ exceeds $S_c$. The model is integrated forward in time with rainfall and an evaporation rate from the Penman–Monteith equation.

### An analytical storm model: Gash-type models

Gash (1979) obtained a simple storm-by-storm solution by idealizing each storm as rain at a constant mean intensity $\overline{R}$ with a constant mean wet-canopy evaporation rate $\overline{E}_w$, and by assuming that drainage is negligible until the canopy is saturated. Measuring time by the cumulative gross rainfall $P = \overline{R}\,t$, Equation (6.1) during wetting becomes

\begin{equation*}
\frac{dC}{dP} = (1 - p) - \frac{\overline{E}_w}{\overline{R}}\,\frac{C}{S_c} ,
\end{equation*}

whose solution reaches $C = S_c$ when the gross rainfall equals

\begin{equation*}
P'_G = -\frac{\overline{R}\,S_c}{\overline{E}_w}\,\ln\!\left[1 - \frac{\overline{E}_w}{(1 - p)\,\overline{R}}\right].
\tag{6.2}
\end{equation*}

For a storm smaller than $P'_G$, the canopy never saturates and all the water it catches eventually evaporates: $I = (1 - p)\,P$. For a larger storm, the interception loss is the sum of three parts: the water caught while wetting up, less what remains stored at saturation; evaporation at the rate $\overline{E}_w$ while the canopy is saturated; and evaporation of the full store $S_c$ after the rain stops. These add up to

\begin{equation*}
I = (1 - p)\,P'_G + \frac{\overline{E}_w}{\overline{R}}\,\bigl(P - P'_G\bigr) \qquad (P \ge P'_G).
\tag{6.3}
\end{equation*}

Gash's original model also separates stemflow and trunk storage, and sparse-canopy versions scale the evaporation by canopy cover. The key parameters, $S_c$, $p$ and the ratio $\overline{E}_w/\overline{R}$, can be estimated from throughfall measurements: plotting throughfall against gross rainfall for many storms gives $p$ from the slope for small storms and $S_c$ and $\overline{E}_w/\overline{R}$ from the large ones. Figure 6.1 compares the analytical solution with a Rutter-type simulation.

```{code-cell} python
:tags: [hide-input]
def rutter(R, Ew, dt, Sc=1.2, p=0.25, Ds=0.002 * 60, b=3.7):
    """Canopy water balance (mm) with hourly-or-finer steps; R and Ew in mm/h, dt in h. Returns loss and drip."""
    C, loss, drip = 0.0, 0.0, 0.0
    for r, e in zip(R, Ew):
        C += (1 - p) * r * dt
        ev = min(e * min(C / Sc, 1.0) * dt, C); C -= ev; loss += ev
        d = min(Ds * np.exp(b * (C - Sc)) * dt, C) if C > 0 else 0.0
        C -= d; drip += d
    return loss, drip

Sc, p_free, Rbar, Ewbar = 1.2, 0.25, 2.5, 0.25
dt = 1 / 12                                                       # 5-minute steps
P_storm = np.linspace(0.5, 60, 80)
I_sim, I_gash = [], []
PpG = -Rbar * Sc / Ewbar * np.log(1 - Ewbar / ((1 - p_free) * Rbar))
for P in P_storm:
    n_rain = int(np.ceil(P / Rbar / dt)); R = np.zeros(n_rain + int(24 / dt)); R[:n_rain] = P / (n_rain * dt)
    Ew = np.full(len(R), Ewbar)
    I_sim.append(rutter(R, Ew, dt, Sc, p_free)[0])
    I_gash.append((1 - p_free) * P if P < PpG else (1 - p_free) * PpG + Ewbar / Rbar * (P - PpG))
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(P_storm, I_sim, color=c1, lw=2.4, label="Rutter-type simulation")
ax.plot(P_storm, I_gash, color=c2, lw=2.0, ls="--", label="Analytical (Gash-type)")
ax.axvline(PpG, color=ink2, lw=0.8, ls=":"); ax.text(PpG + 0.8, 0.5, "$P'_G$", color=ink2)
ax.set_xlabel("Gross storm rainfall P (mm)"); ax.set_ylabel("Interception loss I (mm)")
ax.set_xlim(0, 60); ax.set_ylim(0, None); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("PpG", float(PpG), display=False)
glue("I20", float(np.interp(20, P_storm, I_sim)), display=False)
```

*Figure 6.1. Interception loss per storm for a forest canopy ($S_c$ = 1.2 mm, free throughfall fraction 0.25) with rain at 2.5 mm h⁻¹ and wet-canopy evaporation of 0.25 mm h⁻¹. The canopy saturates after {glue:text}`PpG:.1f` mm of rain; beyond that, the loss grows by 10% of the additional rain. A 20 mm storm loses {glue:text}`I20:.1f` mm to interception.*

## 6.2 The snowpack

### Properties

A snowpack is a porous medium of ice grains, air and, when it is melting, liquid water. Its water content is expressed as the **snow water equivalent**, $\text{SWE} = h_s\,\rho_s/\rho_w$. Fresh snow has a density of 50–150 kg m⁻³. The pack densifies with time by compaction under its own weight, by the rounding of grains through vapor diffusion (**metamorphism**), and by melt and refreezing, reaching 300–500 kg m⁻³ in spring.

### Cold content and ripening

A snowpack below 0 °C must be warmed before it can melt. The energy required is its **cold content**,

\begin{equation*}
U_c = c_i\,\rho_w\,\text{SWE}\,(T_m - \overline{T}_s),
\tag{6.4}
\end{equation*}

where $c_i$ = 2100 J kg⁻¹ K⁻¹ is the specific heat of ice, $T_m$ = 0 °C and $\overline{T}_s$ the mean pack temperature. Cold content is small compared with the energy needed to melt the pack. A 300 mm pack at −5 °C has a cold content of 3.2 MJ m⁻², while melting it takes 100 MJ m⁻². But it controls the timing of the first runoff. Melt water from the surface percolates into a cold pack and refreezes, releasing its latent heat and warming the pack from within. Once the pack is isothermal at 0 °C, it retains liquid water up to a **holding capacity** of a few percent of its mass before water drains from its base. This sequence of warming, ripening and output means that a snowpack can absorb days of melt or rain without producing runoff, and then release water rapidly.

## 6.3 The energy balance of melting snow

The energy available for melt per unit area is the residual of the snowpack energy balance:

\begin{equation*}
Q_m = R_n + H + \lambda_s E + G + Q_R - \frac{dU}{dt},
\tag{6.5}
\end{equation*}

where $R_n$ is net radiation, $H$ and $\lambda_s E$ the sensible and latent heat fluxes from the air (positive toward the pack; $\lambda_s$ is the latent heat of sublimation, 2.83 MJ kg⁻¹, when the surface is frozen), $G$ the ground heat flux, $Q_R$ the heat brought by rain and $dU/dt$ the change in internal energy (cold content). When $Q_m > 0$ and the pack is ripe, the melt rate is

\begin{equation*}
M = \frac{Q_m}{\rho_w\,\lambda_f}, \qquad \lambda_f = 0.334\ \text{MJ kg}^{-1},
\tag{6.6}
\end{equation*}

so that 1 W m⁻² sustained for a day melts 0.26 mm of water. The terms behave differently from those over snow-free ground.

- **Shortwave radiation** is largely reflected. Fresh snow has an albedo of 0.8–0.9, which decays to 0.5–0.6 as the snow ages and becomes dirty and wet; the decay is one of the strongest positive feedbacks of melt.
- **Longwave radiation** is usually a net loss because the surface cannot be warmer than 0 °C, while clouds and a warm, humid atmosphere increase the downward flux.
- **Turbulent fluxes** are limited by the stable stratification that forms over a cold surface, but in warm, windy and humid weather they can dominate. Condensation on the pack is especially effective: condensing 1 mm releases enough latent heat to melt about 7.5 mm.
- **Rain** brings sensible heat ($Q_R = \rho_w c_w P_r T_r$), which is modest, but rain on snow often comes with warm, humid, windy air and dense cloud. The turbulent and longwave fluxes that come with it, together with the water it adds to an already ripe pack, make **rain-on-snow** events a major cause of winter floods.

Figure 6.2 shows an hourly energy-balance simulation of a 300 mm snowpack through a synthetic spring.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(6)
nd = 60; nh = nd * 24
t = np.arange(nh) / 24; hod = np.arange(nh) % 24
Ta = -6 + 0.22 * t + 5 * np.sin(2 * np.pi * (hod - 9) / 24) + np.repeat(1.5 * rng.standard_normal(nd), 24)
sun = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None)
cloud = np.repeat(np.clip(0.4 + 0.3 * rng.standard_normal(nd), 0, 1), 24)
Rs = 800 * sun * (1 - 0.65 * cloud) * (0.9 + 0.004 * t)                  # W/m²
RH = np.repeat(np.clip(0.7 + 0.15 * rng.standard_normal(nd), 0.3, 0.98), 24)
u2 = np.repeat(np.clip(2.5 + 1.0 * rng.standard_normal(nd), 0.5, None), 24)
Pr = np.zeros(nh)
for d in rng.choice(nd, 8, replace=False):
    Pr[d * 24 + 10: d * 24 + 16] = rng.uniform(0.5, 2.5)                   # mm/h

def snowpack(Ta, Rs, RH, u2, Pr, cloud, swe0=300.0, z0=0.001, hold=0.05):
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    esi = lambda T: 0.6108 * np.exp(21.87 * T / (T + 265.5))               # over ice
    sig, rho_a, cp, Lf, Ls, p_air = 5.67e-8, 1.25, 1005.0, 3.34e5, 2.834e6, 90.0
    ra = np.log(2 / z0) ** 2 / (0.4 ** 2 * u2)                              # neutral; stability ignored
    swe, cc, liq, age = swe0, 0.0, 0.0, 10.0
    rec = {k: np.zeros(len(Ta)) for k in ("swe", "out", "Rn", "H", "LE", "QR", "cc")}
    for k in range(len(Ta)):
        rain = Pr[k] if Ta[k] > 1 else 0.0
        snow = Pr[k] - rain; swe += snow
        age = 0.0 if snow > 0.5 else age + 1 / 24
        alb = 0.5 + 0.35 * np.exp(-age / 6)
        Ts = min(0.0, Ta[k]) if (cc > 0 or Ta[k] < 0) else 0.0
        Rn = (1 - alb) * Rs[k] + (0.7 + 0.25 * cloud[k]) * sig * (Ta[k] + 273.15) ** 4 - 0.98 * sig * (Ts + 273.15) ** 4
        H = rho_a * cp * (Ta[k] - Ts) / ra[k]
        e0 = esi(Ts) if Ts < 0 else es(0.0)
        LE = Ls * rho_a * 0.622 / p_air * (RH[k] * es(Ta[k]) - e0) / ra[k]
        QR = 4186.0 * rain * max(Ta[k], 0.0) / 3600                       # W/m² (1 mm = 1 kg/m²)
        E = (Rn + H + LE + QR + 2.0) * 3600                               # J/m² this hour (+2 W/m² from the ground)
        if E < 0:
            cc -= E
        else:
            use = min(E, cc); cc -= use; E -= use
            m = min(E / Lf, swe); swe -= m; liq += m                        # mm (kg/m²)
        liq += rain
        o = max(liq - hold * swe, 0.0); liq -= o
        for key, val in (("swe", swe + liq), ("out", o), ("Rn", Rn), ("H", H), ("LE", LE), ("QR", QR), ("cc", cc)):
            rec[key][k] = val
    return rec

res = snowpack(Ta, Rs, RH, u2, Pr, cloud)
days = np.arange(nd)
daily = lambda x: x.reshape(nd, 24).mean(axis=1)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
a1.plot(t, res["swe"], color=c1, lw=2.2, label="Snow water equivalent")
a1.bar(days + 0.5, res["out"].reshape(nd, 24).sum(1) * 5, width=0.9, color=c3, alpha=0.6, label="Daily outflow × 5")
a1.set_ylabel("mm"); a1.legend(loc="upper right", fontsize=9.5); a1.set_title("(a) Snowpack and outflow")
for key, col, lab in [("Rn", c2, "Net radiation"), ("H", c1, "Sensible heat"), ("LE", c3, "Latent heat"), ("QR", ink2, "Rain heat")]:
    a2.plot(days + 0.5, daily(res[key]), color=col, lw=1.8, label=lab)
a2.axhline(0, color=ink2, lw=0.8)
a2.set_ylabel("Daily mean flux (W m⁻²)"); a2.set_xlabel("Day"); a2.set_title("(b) Energy inputs")
a2.legend(loc="upper left", fontsize=9, ncol=2); a2.set_xlim(0, nd)
plt.show()
gone = np.argmax(res["swe"] < 1.0) / 24 if (res["swe"] < 1.0).any() else np.nan
melt_days = res["swe"] > 1.0
share_Rn = daily(res["Rn"])[daily(res["Rn"]) > -999][:int(gone)].clip(0).sum()
tot_pos = sum(daily(res[k])[:int(gone)].clip(0).sum() for k in ("Rn", "H", "LE", "QR"))
glue("snow_gone", float(gone), display=False)
glue("snow_ccmax", float(res["cc"].max() / 1e6), display=False)
glue("snow_rn_share", float(share_Rn / tot_pos * 100), display=False)
```

*Figure 6.2. Hourly energy-balance simulation of a 300 mm snowpack through a synthetic spring with warming temperatures, eight rain or snow events and a decaying albedo. The pack disappears on day {glue:text}`snow_gone:.0f`. Net radiation supplies {glue:text}`snow_rn_share:.0f`% of the positive daily energy inputs; turbulent fluxes grow in importance as the air warms. The cold content never exceeds {glue:text}`snow_ccmax:.1f` MJ m⁻² and is refilled each night.*

## 6.4 Temperature-index models

The data needed for Equation (6.5) are rarely available. Air temperature usually is, and the **degree-day** method uses it alone (see Chapter 3 of the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch03_precipitation.html) book for its basic use):

\begin{equation*}
M = C_m\,(T_a - T_b)^+ ,
\tag{6.7}
\end{equation*}

where $(\cdot)^+$ denotes the positive part, $T_b$ a base temperature near 0 °C and $C_m$ the degree-day factor (mm °C⁻¹ d⁻¹).

### Why it works

Linearize the main terms of Equation (6.5) about $T_a = 0$ °C, with the surface at 0 °C:

- **Sensible heat:** $H = \rho c_p T_a / r_a$ is proportional to $T_a$.
- **Longwave:** incoming longwave radiation rises by about 4–5 W m⁻² per kelvin.
- **Latent heat:** at fixed relative humidity, the vapor pressure, and hence condensation on the pack, rises with temperature.
- **Shortwave:** absorbed shortwave radiation is correlated with temperature through the seasonal cycle and cloudiness.

Collecting these terms gives $Q_m \approx a + b\,T_a$, and therefore Equation (6.7) with $C_m = 86400\, b/(\rho_w \lambda_f)$ and $T_b = -a/b$.

The degree-day factor is therefore an **effective parameter** (Chapter 1). It lumps the wind speed, the albedo, the humidity and the radiation regime of a place and season into one number. It increases through the melt season as the albedo falls and the sun rises, and it is larger on south-facing slopes than north-facing ones. **Radiation-augmented** forms such as $M = (C_m + a_r R_s)(T_a - T_b)^+$ (Hock, 1999) restore part of the lost physics with a measured or computed shortwave input. Example 6.4 derives a degree-day factor from the simulation of Figure 6.2.

## 6.5 Snow in a warming climate

Warming affects snow hydrology in three ways.

- **Less precipitation falls as snow.** The fraction of precipitation falling as snow drops steeply around a near-surface air temperature of 0–2 °C (Figure 6.3), so regions whose winters hover near freezing lose a large share of their snow for each degree of warming, while cold continental regions lose little.
- **Melt starts earlier and proceeds more slowly.** Earlier melt happens when the sun is lower, so it releases water more gradually, and the spring runoff peak moves earlier.
- **Storage is lost.** Snowpacks act as seasonal reservoirs that release water in late spring and summer. Their decline shifts runoff from the dry season to the wet season (Barnett et al., 2005), with consequences for water supply and reservoir operation in snow-fed basins.

```{code-cell} python
:tags: [hide-input]
Tg = np.linspace(-6, 8, 300)
f_snow = lambda T, T50=1.0, s=0.9: 1 / (1 + np.exp((T - T50) / s))
rng = np.random.default_rng(14)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(Tg, f_snow(Tg), color=c1, lw=2.4)
for Tw, col, lab in [(-4.0, c3, "cold site, mean −4 °C"), (0.5, c2, "marginal site, mean +0.5 °C")]:
    Td = Tw + 3.0 * rng.standard_normal(20000)
    f0, f2 = f_snow(Td).mean(), f_snow(Td + 2).mean()
    ax.axvline(Tw, color=col, lw=1.4, ls="--")
    ax.text(Tw + 0.15, 0.55 if Tw < 0 else 0.25, f"{lab}\nsnow fraction {f0:.0%} → {f2:.0%}\nwith +2 °C", fontsize=8.5, color=col)
    glue(f"sf_{'cold' if Tw < 0 else 'marg'}", float((f0 - f2) / f0 * 100), display=False)
ax.set_xlabel("Daily mean air temperature (°C)"); ax.set_ylabel("Fraction falling as snow")
ax.set_xlim(-6, 8); ax.set_ylim(0, 1.05)
plt.show()
```

*Figure 6.3. A logistic relation between air temperature and the fraction of precipitation falling as snow (50% at 1 °C), applied to two sites with day-to-day temperature variability of 3 °C. A warming of 2 °C reduces the snow share of winter precipitation by {glue:text}`sf_cold:.0f`% (relative) at the cold site but by {glue:text}`sf_marg:.0f`% at the marginal site.*

## 6.6 Worked examples

**Example 6.1 — Storm interception.** For a forest with $S_c$ = 1.6 mm and $p$ = 0.2, a storm delivers 35 mm at a mean intensity of 3 mm h⁻¹ with a mean wet-canopy evaporation rate of 0.3 mm h⁻¹. Find the rainfall needed to saturate the canopy and the interception loss.

```{code-cell} python
Sc1, p1, R1, E1, P1 = 1.6, 0.2, 3.0, 0.3, 35.0
PpG1 = -R1 * Sc1 / E1 * np.log(1 - E1 / ((1 - p1) * R1))
I1 = (1 - p1) * PpG1 + E1 / R1 * (P1 - PpG1)
print(f"P'_G = {PpG1:.2f} mm; I = {I1:.2f} mm = {I1 / P1:.0%} of the storm")
```

**Example 6.2 — Cold content.** A 420 mm snowpack has a mean temperature of −6 °C. Find its cold content and the depth of surface melt water that must refreeze to warm it to 0 °C.

```{code-cell} python
swe, Tm = 420.0, -6.0
Uc = 2100 * swe * (0 - Tm)                       # J/m² (1 mm SWE = 1 kg/m²)
print(f"Cold content = {Uc / 1e6:.2f} MJ/m²; refreezing needed = {Uc / 3.34e5:.1f} mm of melt water")
```

**Example 6.3 — A melt day from the energy balance.** On a spring day, a ripe snowpack receives a daily mean of 210 W m⁻² of shortwave radiation with an albedo of 0.62, a net longwave loss of 35 W m⁻², sensible heat of 18 W m⁻² and latent heat of −6 W m⁻² (sublimation). Find the daily melt.

```{code-cell} python
Qm = (1 - 0.62) * 210 - 35 + 18 - 6
print(f"Q_m = {Qm:.1f} W/m² -> melt = {Qm * 86400 / 3.34e5:.1f} mm/d")
```

**Example 6.4 — A degree-day factor from the energy balance.** Fit Equation (6.7) with $T_b$ = 0 °C to the daily outflow and mean air temperature of the simulation in Figure 6.2, for days with snow on the ground and positive temperature.

```{code-cell} python
d_out = res["out"].reshape(nd, 24).sum(1); d_T = Ta.reshape(nd, 24).mean(1)
sel = (d_T > 0) & (res["swe"].reshape(nd, 24)[:, -1] > 1)
Cm = (d_out[sel] * d_T[sel]).sum() / (d_T[sel] ** 2).sum()
r = np.corrcoef(d_out[sel], Cm * d_T[sel])[0, 1]
print(f"{sel.sum()} melt days: C_m = {Cm:.2f} mm/°C/d, correlation with simulated outflow r = {r:.2f}")
```

The degree-day factor falls in the usual range of 2–6 mm °C⁻¹ d⁻¹. The modest correlation shows what the index cannot capture: daily outflow also depends on radiation, rain and the liquid stored in the pack.

**Example 6.5 — Rain on snow.** Rain of 40 mm at 4 °C falls in a day on a ripe snowpack, with warm, windy, humid air supplying a sensible heat flux of 60 W m⁻² and a condensation (latent) flux of 45 W m⁻². Net radiation is close to zero under the cloud. Find the melt and the total water released.

```{code-cell} python
QR = 4186 * 40 * 4 / 86400                       # W/m², daily mean
Qm = QR + 60 + 45
melt = Qm * 86400 / 3.34e5
print(f"Rain heat = {QR:.1f} W/m², total {Qm:.0f} W/m² -> melt {melt:.1f} mm; water released ≈ {40 + melt:.0f} mm")
```

The rain's own heat melts only about 2 mm; most of the melt comes from the turbulent fluxes that accompany the warm, moist air.

## 6.7 Common mistakes

- **Equating interception loss with canopy storage.** Evaporation during and after rain makes the loss several times the storage for large storms.
- **Neglecting interception in forest water balances.** It can be a quarter of the annual evapotranspiration.
- **Melting snow from net radiation alone with a fixed albedo.** The albedo decay and the turbulent fluxes change melt rates by a factor of two or more.
- **Ignoring cold content and holding capacity.** Without them, models release water too early in spring.
- **Transferring a degree-day factor between regions or seasons.** It is an effective parameter tied to local energy conditions.

## 6.8 Problems

```{code-cell} python
:tags: [remove-cell]
PpG_p = -2.0 * 1.0 / 0.2 * np.log(1 - 0.2 / (0.7 * 2.0))
glue("u1_P", float(PpG_p), display=False); glue("u1_I", float(0.7 * PpG_p + 0.2 / 2.0 * (25 - PpG_p)), display=False)
glue("u2", 2100 * 250 * 8 / 1e6, display=False)
glue("u3", 120 * 0.26, display=False)
glue("u4", 2.834e6 / 3.34e5, display=False)
glue("u5", 4.5 * 3.34e5 / 86400, display=False)
```

1. A canopy has $S_c$ = 1.0 mm and $p$ = 0.3, and during a storm $\overline{R}$ = 2.0 mm h⁻¹ and $\overline{E}_w$ = 0.2 mm h⁻¹. Find $P'_G$ and the interception loss for a 25 mm storm.
2. What is the cold content of a 250 mm snowpack at a mean temperature of −8 °C, in MJ m⁻²?
3. A snowpack receives a steady net energy input of 120 W m⁻². How much does it melt per day?
4. How many millimeters of snow can be melted by the latent heat released when 1 mm of vapor condenses onto a snowpack at 0 °C? (Use the latent heat of sublimation for deposition.)
5. A degree-day model is fitted with $C_m$ = 4.5 mm °C⁻¹ d⁻¹. What energy flux (W m⁻²) per degree of air temperature does this imply?

```{admonition} Answers
:class: dropdown
1. $P'_G$ = {glue:text}`u1_P:.2f` mm; $I$ = {glue:text}`u1_I:.2f` mm.
2. {glue:text}`u2:.1f` MJ m⁻².
3. About {glue:text}`u3:.0f` mm d⁻¹.
4. About {glue:text}`u4:.1f` mm.
5. About {glue:text}`u5:.1f` W m⁻² per °C.
```

## 6.9 Open questions

- **Interception in a changing climate.** How will changes in rainfall intensity and in the frequency of small storms alter interception loss, which is largest for frequent small events?
- **Snow in forests.** Canopies intercept snow, shade the pack and emit longwave radiation; the net effect of forest cover on snow accumulation and melt varies with climate and is the subject of active research.
- **Mountain snow from space.** Estimating SWE, not just snow cover, over mountain ranges from satellites remains unsolved.

## 6.10 Key papers

- Barnett, T. P., Adam, J. C., and Lettenmaier, D. P. (2005). Potential impacts of a warming climate on water availability in snow-dominated regions. *Nature*, 438, 303–309.
- Gash, J. H. C. (1979). An analytical model of rainfall interception by forests. *Quarterly Journal of the Royal Meteorological Society*, 105, 43–55.
- Hock, R. (1999). A distributed temperature-index ice- and snowmelt model including potential direct solar radiation. *Journal of Glaciology*, 45, 101–111.
- Hock, R. (2003). Temperature index melt modelling in mountain areas. *Journal of Hydrology*, 282, 104–115.
- Rutter, A. J., Kershaw, K. A., Robins, P. C., and Morton, A. J. (1971). A predictive model of rainfall interception in forests, 1. Derivation of the model from observations in a plantation of Corsican pine. *Agricultural Meteorology*, 9, 367–384.

## 6.11 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The sections on interception and snowmelt.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on snow and snowmelt.
- DeWalle, D. R., and Rango, A. (2008). *Principles of Snow Hydrology*. Cambridge University Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
