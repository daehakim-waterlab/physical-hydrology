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

# Appendix D. Solar geometry and clear-sky radiation

The solar radiation reaching the top of the atmosphere depends only on astronomy: the Earth–Sun distance, the solar declination, the latitude and the time of day. It sets the upper limit of the energy available for evaporation and snowmelt (Chapters 2, 6 and 7), and it is the reference against which measured radiation is compared to estimate cloudiness. This appendix gives the standard formulas, adequate for hydrological work at daily and hourly time steps.

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this appendix.
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

## D.1 Earth–Sun geometry

**Distance.** The Earth's orbit is slightly elliptical. The flux at the top of the atmosphere varies with the inverse square of the Earth–Sun distance, by about ±3.3% over the year, highest in early January. For day of year $J$,

\begin{equation*}
d_r = 1 + 0.033\cos\left(\frac{2\pi J}{365}\right)
\tag{D.1}
\end{equation*}

is the ratio of the flux to its value at the mean distance.

**Declination.** The angle between the Sun's rays and the equatorial plane, the **solar declination** $\delta$, varies between +23.45° at the June solstice and −23.45° at the December solstice. A simple approximation, in radians, is

\begin{equation*}
\delta = 0.409\,\sin\left(\frac{2\pi J}{365} - 1.39\right) .
\tag{D.2}
\end{equation*}

**Zenith angle.** At latitude $\varphi$ and **hour angle** $\omega$ (zero at solar noon, changing by 15° per hour, negative in the morning), the cosine of the solar zenith angle $\theta_z$ is

\begin{equation*}
\cos\theta_z = \sin\varphi\,\sin\delta + \cos\varphi\,\cos\delta\,\cos\omega .
\tag{D.3}
\end{equation*}

The Sun rises and sets when $\cos\theta_z = 0$, at the **sunset hour angle** $\omega_s = \arccos(-\tan\varphi\,\tan\delta)$, and the day length is $N = 24\omega_s/\pi$ hours. When $|\tan\varphi\tan\delta| > 1$, poleward of the polar circles, the Sun does not set (polar day, $\omega_s = \pi$) or does not rise (polar night, $\omega_s = 0$).

Solar time differs from clock time because of the time zone (4 minutes per degree of longitude from the zone's standard meridian) and the **equation of time**, which ranges over about ±16 minutes during the year. Both must be accounted for when comparing hourly calculations with measurements.

## D.2 Extraterrestrial radiation

The flux on a horizontal surface at the top of the atmosphere is $G_{sc}\,d_r\cos\theta_z$, where $G_{sc} \approx$ 1361 W m⁻² is the solar constant. Integrating Equation (D.3) over the hours of daylight, from $-\omega_s$ to $\omega_s$, gives the daily total:

\begin{equation*}
R_a = \frac{86400}{\pi}\,G_{sc}\,d_r\left[\omega_s\sin\varphi\sin\delta + \cos\varphi\cos\delta\sin\omega_s\right],
\tag{D.4}
\end{equation*}

in J m⁻² d⁻¹ with $\omega_s$ in radians. The factor $86400/\pi$ converts the integral over the hour angle (radians) into seconds, since the Earth turns through $2\pi$ radians in 86,400 s.

```{code-cell} python
def solar(J, lat_deg):
    """Declination (rad), sunset hour angle (rad), day length (h) and daily extraterrestrial radiation (MJ m-2 d-1)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    Ra = 86400 / np.pi * 1361e-6 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return dec, ws, 24 * ws / np.pi, Ra
```

```{code-cell} python
:tags: [hide-input]
J = np.arange(1, 366)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 4.1))
for lat, col, ls in zip([0, 20, 35, 50, 70], [ink2, c3, c1, c2, ink], ["-", "-", "-", "-", "--"]):
    _, _, N, Ra = solar(J, lat)
    a1.plot(J, Ra, color=col, lw=2.0, ls=ls, label=f"{lat}° N")
    a2.plot(J, N, color=col, lw=2.0, ls=ls)
a1.set_xlabel("Day of year"); a1.set_ylabel("$R_a$ (MJ m⁻² d⁻¹)"); a1.set_title("(a) Extraterrestrial radiation")
fig.legend(loc="outside lower center", ncol=5, fontsize=9, handlelength=1.6, columnspacing=1.0)
a2.set_xlabel("Day of year"); a2.set_ylabel("Day length (h)"); a2.set_title("(b) Day length")
plt.show()
_, _, _, Ra35 = solar(np.array([172, 355]), 35)
glue("ra_jun", float(Ra35[0]), display=False); glue("ra_dec", float(Ra35[1]), display=False)
```

*Figure D.1. Daily extraterrestrial radiation on a horizontal surface and day length over the year, Equations (D.2)–(D.4). At 35° N, $R_a$ ranges from {glue:text}`ra_dec:.1f` MJ m⁻² d⁻¹ at the winter solstice to {glue:text}`ra_jun:.1f` MJ m⁻² d⁻¹ at the summer solstice. In June, high latitudes receive more radiation per day than the equator, because of their long days.*

## D.3 Radiation at the surface

### Clear-sky radiation

On its way through the atmosphere, solar radiation is scattered and absorbed by gases, aerosols and clouds. Under clear skies, a simple estimate of the radiation reaching the surface is

\begin{equation*}
R_{so} = (0.75 + 2\times10^{-5}z)\,R_a ,
\tag{D.5}
\end{equation*}

with the elevation $z$ in meters, as recommended for evaporation calculations by Allen et al. (1998). The clear-sky transmissivity of about 0.75 at sea level increases with elevation because there is less air above. More detailed models account for water vapor, aerosols and the path length through the atmosphere, which depends on the zenith angle.

### Cloudiness and measured radiation

The ratio of measured global radiation $R_s$ to $R_a$, the **clearness index**, or to $R_{so}$, the **relative shortwave radiation**, summarizes the effect of clouds. Where radiation is not measured, the **Ångström–Prescott** relation estimates it from the relative sunshine duration $n/N$,

\begin{equation*}
R_s = \left(a_s + b_s\,\frac{n}{N}\right) R_a ,
\tag{D.6}
\end{equation*}

with typical coefficients $a_s \approx$ 0.25 and $b_s \approx$ 0.50, which should be calibrated locally where possible. The ratio $R_s/R_{so}$ is also used to estimate the cloud effect on net longwave radiation (Chapter 2).

### Sloping surfaces

On a slope, the zenith angle in Equation (D.3) is replaced by the angle between the Sun's rays and the normal to the surface, which depends on the slope's inclination and aspect. South-facing slopes in the Northern Hemisphere receive more radiation than flat ground, especially in winter, and north-facing slopes less; the difference affects snowmelt (Chapter 6), evaporation and vegetation. Shading by surrounding terrain reduces direct radiation further in mountains.

```{code-cell} python
:tags: [hide-input]
lat = 35.0; phi = np.radians(lat)
hours = np.linspace(4, 20, 400); w = np.radians(15 * (hours - 12))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for J0, lab, col in [(172, "June solstice", c2), (80, "March equinox", c3), (355, "December solstice", c1)]:
    dec = 0.409 * np.sin(2 * np.pi * J0 / 365 - 1.39); dr = 1 + 0.033 * np.cos(2 * np.pi * J0 / 365)
    cz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(w)
    ax.plot(hours, 1361 * dr * np.maximum(cz, 0), color=col, lw=2.2, label=lab)
ax.set_xlabel("Solar time (h)"); ax.set_ylabel("Flux on a horizontal surface (W m⁻²)")
ax.set_xlim(4, 20); ax.legend(loc="upper left", fontsize=9)
plt.show()
```

*Figure D.2. Diurnal course of the extraterrestrial flux on a horizontal surface at 35° N, Equation (D.3), at the solstices and the March equinox.*

## D.4 Worked example

**Example D.1 — Daily radiation.** Compute the extraterrestrial and clear-sky radiation at latitude 37° N and elevation 300 m on 15 May (day 135), and the corresponding evaporation equivalent of the clear-sky radiation (Appendix C).

```{code-cell} python
dec, ws, N, Ra = solar(135, 37.0)
Rso = (0.75 + 2e-5 * 300) * Ra
print(f"declination = {np.degrees(dec):.1f}°, day length = {N:.1f} h")
print(f"Ra = {Ra:.1f} MJ/m2/d, Rso = {Rso:.1f} MJ/m2/d = {Rso / 2.45:.1f} mm/d of evaporation equivalent")
```

## D.5 Further reading

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56.
- Duffie, J. A., and Beckman, W. A. (2013). *Solar Engineering of Thermal Processes* (4th ed.). Wiley.
- Iqbal, M. (1983). *An Introduction to Solar Radiation*. Academic Press.

---

*Figures in this appendix are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
