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

# Appendix B. Properties of water and moist air

This appendix gives the physical properties used in the book, as functions that can be evaluated in code, with tables at common temperatures. The formulas are standard approximations. Over the range of natural surface temperatures (0–40 °C) they are accurate to better than 1%, except the viscosity formula, which is about 2% low near 0 °C.

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this appendix.
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
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
def table(header, rows, fmt):
    """Render a small HTML table from rows of numbers."""
    h = "".join(f"<th>{x}</th>" for x in header)
    b = "".join("<tr>" + "".join(f"<td>{f.format(v)}</td>" for f, v in zip(fmt, r)) + "</tr>" for r in rows)
    display(HTML(f'<table class="table"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'))
```

## B.1 Liquid water and ice

The properties of liquid water that appear most often are its density (Chapters 9 and 11), viscosity (Darcy's law and laminar flow, Chapters 9 and 11), surface tension (capillarity, Chapter 11) and latent heats (Chapters 2, 6 and 7). In code:

```{code-cell} python
def rho_water(T):
    """Density of liquid water (kg m-3), T in °C; Thiesen-type formula."""
    return 1000 * (1 - (T + 288.9414) / (508929.2 * (T + 68.12963)) * (T - 3.9863) ** 2)
def mu_water(T):
    """Dynamic viscosity (Pa s), T in °C; Vogel-type equation."""
    return 2.414e-5 * 10 ** (247.8 / (T + 273.15 - 140))
def sigma_water(T):
    """Surface tension against air (N m-1), T in °C; IAPWS form."""
    tau = 1 - (T + 273.15) / 647.096
    return 0.2358 * tau ** 1.256 * (1 - 0.625 * tau)
def lambda_v(T):
    """Latent heat of vaporization (MJ kg-1), T in °C."""
    return 2.501 - 0.002361 * T
```

```{code-cell} python
:tags: [hide-input]
Ts = np.arange(0, 41, 5.0)
table(["T (°C)", "ρ (kg m⁻³)", "μ (10⁻³ Pa s)", "ν (10⁻⁶ m² s⁻¹)", "σ (N m⁻¹)", "λ (MJ kg⁻¹)"],
      [(T, rho_water(T), mu_water(T) * 1e3, mu_water(T) / rho_water(T) * 1e6, sigma_water(T), lambda_v(T)) for T in Ts],
      ["{:.0f}", "{:.2f}", "{:.3f}", "{:.3f}", "{:.4f}", "{:.3f}"])
```

*Table B.1. Properties of liquid water at atmospheric pressure.*

Other constants:

- specific heat of liquid water, $c_w$ = 4.19 kJ kg⁻¹ K⁻¹; of ice, $c_i$ = 2.1 kJ kg⁻¹ K⁻¹;
- latent heat of fusion, $\lambda_f$ = 0.334 MJ kg⁻¹; latent heat of sublimation, $\lambda_s = \lambda_v + \lambda_f \approx$ 2.835 MJ kg⁻¹ at 0 °C;
- density of ice, 917 kg m⁻³; of fresh snow, typically 50–150 kg m⁻³; of old snow, 300–500 kg m⁻³ (Chapter 6);
- thermal conductivity of water, 0.57–0.62 W m⁻¹ K⁻¹; of ice, 2.2 W m⁻¹ K⁻¹;
- the energy needed to evaporate 1 mm of water from 1 m² is $\rho_w\lambda_v \times 10^{-3}$ m ≈ 2.45 MJ, so 1 mm d⁻¹ corresponds to about 28.4 W m⁻² (Appendix C).

## B.2 Moist air

### Gas constants and humidity

Air is treated as a mixture of ideal gases: dry air, with gas constant $R_d$ = 287.05 J kg⁻¹ K⁻¹, and water vapor, with $R_v$ = 461.5 J kg⁻¹ K⁻¹. Their ratio $\varepsilon = R_d/R_v$ = 0.622 appears in all humidity conversions. With total pressure $p$ and vapor pressure $e$:

\begin{equation*}
q = \frac{\varepsilon e}{p - (1 - \varepsilon)e}, \qquad
r = \frac{\varepsilon e}{p - e}, \qquad
\rho_v = \frac{e}{R_v T}, \qquad
\text{RH} = \frac{e}{e^*(T)} ,
\tag{B.1}
\end{equation*}

for the specific humidity, mixing ratio, absolute humidity (vapor density) and relative humidity. The **virtual temperature** $T_v = T(1 + 0.61q)$ is the temperature dry air would need to have the same density as the moist air, $\rho_a = p/(R_d T_v)$; moist air is lighter than dry air at the same temperature and pressure.

### Saturation vapor pressure

The saturation vapor pressure over a flat surface of pure water follows from the Clausius–Clapeyron equation. This book uses the Tetens-type form over water, and an analogous form over ice:

\begin{equation*}
e^*_w = 0.6108\,\exp\left(\frac{17.27\,T}{T + 237.3}\right), \qquad
e^*_i = 0.6108\,\exp\left(\frac{21.875\,T}{T + 265.5}\right),
\tag{B.2}
\end{equation*}

in kPa with $T$ in °C. The slope of the saturation curve over water is $\Delta = de^*_w/dT = 4098\,e^*_w/(T + 237.3)^2$ kPa K⁻¹, and the dew point, the temperature at which $e^*_w = e$, is $T_d = 237.3\,\ell/(17.27 - \ell)$ with $\ell = \ln(e/0.6108)$.

```{code-cell} python
:tags: [hide-input]
es_w = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
es_i = lambda T: 0.6108 * np.exp(21.875 * T / (T + 265.5))
slope = lambda T: 4098 * es_w(T) / (T + 237.3) ** 2
T = np.linspace(-20, 40, 300)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.6))
a1.plot(T, es_w(T), color=c1, lw=2.2, label="Over water")
a1.plot(T[T <= 0], es_i(T[T <= 0]), color=c2, lw=2.0, ls="--", label="Over ice")
a1.set_xlabel("T (°C)"); a1.set_ylabel("e* (kPa)"); a1.set_title("(a) Saturation vapor pressure"); a1.legend(loc="upper left", fontsize=9)
a2.plot(T, slope(T), color=c3, lw=2.2)
a2.set_xlabel("T (°C)"); a2.set_ylabel("Δ (kPa K⁻¹)"); a2.set_title("(b) Slope")
plt.show()
```

*Figure B.1. Saturation vapor pressure over water and ice, Equation (B.2), and the slope of the curve over water. The saturation vapor pressure roughly doubles for every 10 K of warming, about 6–7% per kelvin near 20 °C.*

### Other properties of air

- **Specific heat** of dry air at constant pressure, $c_p$ = 1005 J kg⁻¹ K⁻¹ (moist air: about $1005(1 + 0.84q)$).
- **Psychrometric constant**, $\gamma = c_p\,p/(\varepsilon\lambda_v)$, about 0.066 kPa K⁻¹ at sea level and 20 °C.
- **Atmospheric pressure** at elevation $z$ (m), for a standard atmosphere: $p \approx 101.3\,[(293 - 0.0065z)/293]^{5.26}$ kPa.
- **Kinematic viscosity** of air, about 1.5 × 10⁻⁵ m² s⁻¹ at 20 °C; **thermal diffusivity**, about 2.2 × 10⁻⁵ m² s⁻¹; **molecular diffusivity of water vapor**, about 2.5 × 10⁻⁵ m² s⁻¹. All increase by roughly 0.6–0.7% per kelvin.
- **Density** of dry air at sea level: 1.29 kg m⁻³ at 0 °C, 1.20 kg m⁻³ at 20 °C.
- **von Kármán constant**, $k$ = 0.40 (Chapter 4); **Stefan–Boltzmann constant**, $\sigma$ = 5.67 × 10⁻⁸ W m⁻² K⁻⁴; **solar constant**, about 1361 W m⁻² (Chapter 2).

```{code-cell} python
:tags: [hide-input]
def gamma_psy(T, p=101.3): return 1.005e-3 * p / (0.622 * lambda_v(T))
def rho_air(T, p=101.3, RH=0.0):
    e = RH * es_w(T); q = 0.622 * e / (p - 0.378 * e)
    return p * 1e3 / (287.05 * (T + 273.15) * (1 + 0.61 * q))
Ts = np.arange(-10, 41, 5.0)
table(["T (°C)", "e* (kPa)", "Δ (kPa K⁻¹)", "γ (kPa K⁻¹)", "Δ/(Δ + γ)", "ρₐ, dry (kg m⁻³)"],
      [(T, es_w(T), slope(T), gamma_psy(T), slope(T) / (slope(T) + gamma_psy(T)), rho_air(T)) for T in Ts],
      ["{:.0f}", "{:.3f}", "{:.4f}", "{:.4f}", "{:.3f}", "{:.3f}"])
```

*Table B.2. Properties of air at a pressure of 101.3 kPa: saturation vapor pressure over water, its slope, the psychrometric constant, the Priestley–Taylor weighting factor (Chapter 7) and the density of dry air.*

## B.3 Further reading

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56.
- Brutsaert, W. (1982). *Evaporation into the Atmosphere*. D. Reidel.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The appendix on water as a substance.
- Wallace, J. M., and Hobbs, P. V. (2006). *Atmospheric Science: An Introductory Survey* (2nd ed.). Academic Press.

---

*Figures and tables in this appendix are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
