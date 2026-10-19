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

# 3. Water vapor and the lower atmosphere

```{admonition} Core ideas
:class: tip
1. Moist air is a mixture of ideal gases. Water vapor is lighter than dry air, so **moist air is less dense** than dry air at the same temperature and pressure. The **virtual temperature** accounts for this in every buoyancy calculation.
2. Pressure falls with height almost exactly hydrostatically. A rising parcel cools by expansion, and the **potential temperature** removes that effect: in a well-mixed layer it is constant with height.
3. The **static stability** of the air is set by the vertical gradient of the virtual potential temperature. It decides whether turbulence is suppressed (stable) or amplified (unstable) by buoyancy.
4. Near the ground, vapor, heat and momentum are carried almost entirely by **turbulent eddies**. **Reynolds averaging** separates mean and fluctuating parts and shows that the vertical flux of any quantity is the covariance of its fluctuations with the vertical wind.
5. **Eddy covariance** measures that covariance directly from fast measurements of wind and humidity. Coordinate rotation, an averaging period long enough to capture the large eddies, and density corrections are essential for a correct flux.
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

## 3.1 Moist air

### A mixture of ideal gases

The lower atmosphere is a mixture of dry air (mostly nitrogen and oxygen, molar mass $M_d$ = 28.97 g mol⁻¹) and water vapor ($M_v$ = 18.02 g mol⁻¹). Each component obeys the ideal-gas law with its own partial pressure, and Dalton's law of partial pressures adds them: $p = p_d + e$, where $e$ is the vapor pressure. With the specific gas constants $R_d = 287.0$ J kg⁻¹ K⁻¹ and $R_v = 461.5$ J kg⁻¹ K⁻¹, whose ratio is $\varepsilon = R_d/R_v = M_v/M_d = 0.622$, the density of moist air is

\begin{equation*}
\rho = \frac{p_d}{R_d T} + \frac{e}{R_v T} = \frac{p}{R_d T}\left[1 - (1 - \varepsilon)\frac{e}{p}\right] .
\tag{3.1}
\end{equation*}

Because the bracket is less than 1, **moist air is lighter than dry air** at the same pressure and temperature. It is convenient to keep the dry-air gas law and absorb the effect into a fictitious temperature, the **virtual temperature**

\begin{equation*}
T_v = \frac{T}{1 - (1 - \varepsilon)\,e/p} \approx T\,(1 + 0.61\,q) ,
\qquad
\rho = \frac{p}{R_d\,T_v} ,
\tag{3.2}
\end{equation*}

where $q$ is the specific humidity. A parcel of humid air is buoyant relative to drier air at the same temperature; in the tropics this vapor effect on density is comparable to a temperature difference of a few kelvin.

### Measures of humidity

The **specific humidity** $q$ is the mass of vapor per unit mass of moist air and the **mixing ratio** $r$ the mass of vapor per unit mass of dry air:

\begin{equation*}
q = \frac{\varepsilon\, e}{p - (1 - \varepsilon)\,e}, \qquad r = \frac{\varepsilon\, e}{p - e} .
\tag{3.3}
\end{equation*}

Both are conserved in a parcel that neither gains nor loses water, which makes them the natural variables for transport. The saturation vapor pressure $e^*(T)$ follows from the Clausius–Clapeyron equation; this book uses the form $e^* = 0.6108 \exp[17.27 T/(T + 237.3)]$ kPa ($T$ in °C) for saturation over water, whose derivation and derivative are given in Chapter 2 of the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch02_water_and_energy_at_the_land_surface.html) book. Over ice the saturation vapor pressure is lower, which drives the growth of ice crystals in mixed clouds (Chapter 5).

## 3.2 Hydrostatics and static stability

### The hydrostatic atmosphere

For large-scale motions, vertical accelerations are tiny compared with gravity, and pressure obeys the hydrostatic equation $\partial p/\partial z = -\rho g$. With Equation (3.2), $\partial \ln p/\partial z = -g/(R_d T_v)$. For an isothermal layer this integrates to

\begin{equation*}
p(z) = p_0 \exp\!\left(-\frac{z}{H_s}\right), \qquad H_s = \frac{R_d\, T_v}{g},
\tag{3.4}
\end{equation*}

where the **scale height** $H_s$ is about 8 km. Half of the atmosphere's mass, and nearly all of its water vapor, lies below about 5.5 km.

### Lapse rates and potential temperature

The first law of thermodynamics for an air parcel that rises without exchanging heat (adiabatically) is $c_p\, dT = dp/\rho$. Combined with hydrostatics it gives the dry adiabatic lapse rate $\Gamma_d = g/c_p \approx 9.8$ K km⁻¹. Integrating the adiabatic relation between pressures $p$ and $p_0$ defines the **potential temperature**,

\begin{equation*}
\theta = T\left(\frac{p_0}{p}\right)^{R_d/c_p}, \qquad \frac{R_d}{c_p} \approx 0.286 ,
\tag{3.5}
\end{equation*}

the temperature a parcel would have if brought adiabatically to the reference pressure $p_0$ = 100 kPa. Potential temperature is conserved in dry adiabatic motion, so it labels air parcels the way specific humidity labels their moisture. Near the surface, $\theta \approx T + \Gamma_d z$. Replacing $T$ by $T_v$ gives the **virtual potential temperature** $\theta_v$, which includes the buoyancy of water vapor.

### Static stability

Displace a parcel upward by a small distance. It cools at the adiabatic rate and keeps its $\theta_v$. If the surrounding $\theta_v$ increases with height, the parcel finds itself denser than its new surroundings and sinks back: the layer is **stable**. If $\theta_v$ decreases with height, the parcel is lighter and keeps rising: the layer is **unstable**. If $\theta_v$ is constant, the layer is **neutral**. The criterion is

\begin{equation*}
\frac{\partial \overline{\theta}_v}{\partial z}
\begin{cases} > 0 & \text{stable} \\ = 0 & \text{neutral} \\ < 0 & \text{unstable} \end{cases}
\qquad
N^2 = \frac{g}{\overline{\theta}_v}\frac{\partial \overline{\theta}_v}{\partial z},
\tag{3.6}
\end{equation*}

where $N$, the **Brunt–Väisälä frequency**, is the frequency at which a displaced parcel oscillates in a stable layer. Figure 3.1 shows a typical afternoon sounding over land: an unstable surface layer heated from below, a deep well-mixed layer where $\theta_v$ is nearly constant, and a stable inversion that caps the boundary layer.

```{code-cell} python
:tags: [hide-input]
Rd, cp, g, eps = 287.0, 1005.0, 9.81, 0.622
es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))            # kPa, T in °C

# A synthetic afternoon sounding (heights in m)
z = np.linspace(2, 3000, 600)
zi = 1400.0                                                         # top of the mixed layer
theta = np.where(z < 50, 303.0 - 1.2 * np.log(z / 50), 303.0)       # superadiabatic surface layer
theta = np.where(z > zi, 303.0 + 4.0 * np.tanh((z - zi) / 60) + 0.004 * (z - zi), theta)
q = np.where(z < zi, 0.0120, 0.0120 - 0.0060 * np.tanh((z - zi) / 80) - 1.0e-6 * (z - zi))   # kg/kg
p = 101.0 * np.exp(-z / 8400.0)                                    # kPa
T = theta * (p / 100.0) ** (Rd / cp) - 273.15                       # °C
theta_v = theta * (1 + 0.61 * q)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True)
a1.plot(T, z / 1000, color=ink2, lw=1.6, label="T (°C)")
a1.plot(theta - 273.15, z / 1000, color=c2, lw=2.2, label="θ (°C)")
a1.plot(theta_v - 273.15, z / 1000, color=c1, lw=2.2, ls="--", label="θ$_v$ (°C)")
a1.set_xlabel("Temperature (°C)"); a1.set_ylabel("Height (km)"); a1.legend(loc="upper left", fontsize=9)
a1.axhline(zi / 1000, color=ink2, lw=0.7, ls=":")
a2.plot(q * 1000, z / 1000, color=c3, lw=2.2)
a2.set_xlabel("q (g kg⁻¹)")
a2.text(6.4, 0.6, "mixed\nlayer", fontsize=9.5, color=ink2); a2.text(6.4, 1.8, "free\natmosphere", fontsize=9.5, color=ink2)
a2.text(9.2, zi / 1000 + 0.05, "inversion", fontsize=9, color=ink2)
a1.set_ylim(0, 3)
plt.show()
glue("dthv", float(theta_v[np.searchsorted(z, 900)] - theta[np.searchsorted(z, 900)]), display=False)
```

*Figure 3.1. A synthetic afternoon sounding over land. The temperature falls with height at nearly the dry adiabatic rate in the mixed layer, where the potential temperature θ is constant. Moisture raises the virtual potential temperature θ$_v$ by about {glue:text}`dthv:.1f` K. A strong inversion near 1.4 km caps the boundary layer and traps the moisture evaporated from the surface below it.*

## 3.3 Turbulence and Reynolds averaging

### Why turbulence

The flow of air over the ground is almost always **turbulent**: velocities fluctuate irregularly in time and space, with eddies ranging from the depth of the boundary layer (a kilometer or more) down to millimeters, where viscosity finally dissipates them. Turbulence mixes heat, vapor and momentum far more effectively than molecular diffusion. Molecular diffusion alone would take about a year to carry water vapor through the lowest 10 m of the atmosphere; turbulent eddies do it in seconds.

### Reynolds decomposition

Turbulence cannot be predicted in detail, but its average effect can. Split each variable into a mean and a fluctuation,

\begin{equation*}
w = \overline{w} + w', \qquad q = \overline{q} + q', \qquad \overline{w'} = \overline{q'} = 0 ,
\tag{3.7}
\end{equation*}

where the overbar is an average over a period long enough to include many eddies but short enough that the mean conditions do not change (typically 30 minutes). The average of a product is then

\begin{equation*}
\overline{w\,q} = \overline{w}\;\overline{q} + \overline{w'q'} .
\tag{3.8}
\end{equation*}

The second term, the **covariance** of the fluctuations, survives the averaging even though each fluctuation averages to zero.

### The averaged conservation equation

Apply this to the conservation of water vapor (Equation 1.5) for horizontally uniform conditions. Averaging gives

\begin{equation*}
\frac{\partial \overline{q}}{\partial t} + \overline{w}\,\frac{\partial \overline{q}}{\partial z}
= -\frac{\partial}{\partial z}\left(\overline{w'q'} - D_v\,\frac{\partial \overline{q}}{\partial z}\right) .
\tag{3.9}
\end{equation*}

The **turbulent flux** $\overline{w'q'}$ appears exactly where the molecular flux does, but outside the thin layer touching the surface it is thousands of times larger. Near the surface, where the mean vertical wind is zero and the flux hardly changes with height, the vertical fluxes of vapor, heat and momentum are

\begin{equation*}
E = \overline{\rho}\;\overline{w'q'}, \qquad
H = \overline{\rho}\,c_p\,\overline{w'T'}, \qquad
\tau = -\overline{\rho}\;\overline{u'w'},
\tag{3.10}
\end{equation*}

with $E$ in kg m⁻² s⁻¹ (equal to mm s⁻¹ of water), $H$ in W m⁻² and $\tau$ the momentum flux, or shear stress, in N m⁻². The friction velocity $u_* = (\tau/\rho)^{1/2}$ is the velocity scale of near-surface turbulence (Chapter 4).

### Closing the equations

Equation (3.9) has a new unknown, the turbulent flux. The simplest way to close it is by analogy with molecular diffusion, $\overline{w'q'} = -K_E\,\partial\overline{q}/\partial z$, with an **eddy diffusivity** $K_E$ that depends on the flow rather than the fluid. This **K-theory** underlies the profile and resistance methods of Chapters 4 and 7. Its weakness is that turbulent eddies are not small compared with the gradients they mix, so in convective conditions fluxes can even run against the mean gradient.

### Production of turbulence

Turbulence is maintained by two sources. **Shear production** draws energy from the mean wind, $-\overline{u'w'}\,\partial\overline{u}/\partial z$. **Buoyant production** is $(g/\overline{\theta}_v)\,\overline{w'\theta_v'}$: positive when the surface heats the air (unstable), negative when the surface cools it (stable, as on clear nights). The ratio of the two, the **flux Richardson number**, measures how much buoyancy helps or suppresses turbulence. It leads to the stability parameter of Monin–Obukhov theory in Chapter 4.

## 3.4 Eddy covariance

### The method

Equation (3.10) can be measured directly. A **sonic anemometer** measures the three wind components and the sonic (approximately virtual) temperature, and an infrared **gas analyzer** measures the vapor (and CO₂) concentration, both 10–20 times a second. Over each averaging period the fluctuations are computed and their covariance gives the flux. Since the 1990s, networks of eddy-covariance towers (FLUXNET and its regional networks) have measured evaporation and carbon exchange over every major ecosystem, and they provide the main test of the evaporation theories of Chapters 7 and 8.

### Corrections

The raw covariance must be corrected for several effects.

- **Coordinate rotation.** The anemometer is never perfectly level, and the terrain may slope. A small tilt mixes part of the much larger horizontal wind into the measured vertical wind. The standard remedy rotates the coordinate system so that the mean vertical (and lateral) wind is zero over each period. Example 3.3 shows the size of the error.
- **Averaging period.** Large eddies carry part of the flux. If the averaging period is too short, they are treated as changes in the mean and their contribution is lost. The **ogive**, the cumulative covariance as a function of the time scale of the eddies included, shows whether the flux has converged (Figure 3.3).
- **Density effects.** Heat and vapor fluxes change the density of the air, which creates spurious fluctuations in the measured concentrations. The **Webb–Pearman–Leuning** correction (Webb et al., 1980) removes them; it is small for vapor but large for CO₂.
- **Frequency response.** Sensors that are separated in space, or that respond too slowly, miss the smallest eddies; spectral transfer functions correct for this.

Even after these corrections, eddy-covariance measurements at most sites account for only about 70–90% of the available energy, $R_n - G$. This **energy balance closure** problem is discussed in Chapter 4.

```{code-cell} python
:tags: [hide-input]
# Synthetic 10 Hz turbulence: a shared eddy signal drives both w' and q' (positive vapor flux)
rng = np.random.default_rng(3)
fs, minutes = 10.0, 60
n = int(fs * 60 * minutes)
t = np.arange(n) / fs
def red_noise(n, tau_s, rng):
    a = np.exp(-1 / (fs * tau_s)); x = np.empty(n); x[0] = 0.0
    e = rng.standard_normal(n) * np.sqrt(1 - a ** 2)
    for k in range(1, n): x[k] = a * x[k - 1] + e[k]
    return x
eddy = red_noise(n, 8.0, rng) + 0.6 * red_noise(n, 120.0, rng)    # small and large eddies
w = 0.30 * eddy + 0.20 * rng.standard_normal(n)                   # m/s
w -= w.mean()                                                     # true mean vertical wind is zero
qf = 0.20 * eddy + 0.15 * red_noise(n, 30.0, rng)                 # g/kg
q = 12.0 + qf
u = 3.5 + 0.6 * red_noise(n, 20.0, rng) - 0.5 * eddy               # downdrafts bring faster, drier air

seg = slice(0, int(fs * 120))                                     # first 2 minutes
fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.0), sharex=True)
axes[0].plot(t[seg], w[seg], color=c1, lw=0.7); axes[0].set_ylabel("w (m s⁻¹)")
axes[1].plot(t[seg], q[seg], color=c3, lw=0.7); axes[1].set_ylabel("q (g kg⁻¹)")
prod = (w - w.mean()) * (q - q.mean())
axes[2].plot(t[seg], prod[seg], color=c2, lw=0.7); axes[2].axhline(0, color=ink2, lw=0.8)
axes[2].set_ylabel("w′q′"); axes[2].set_xlabel("Time (s)"); axes[2].set_xlim(0, 120)
for ax, lab in zip(axes, ["(a) Vertical wind", "(b) Specific humidity", "(c) Instantaneous product of fluctuations"]):
    ax.set_title(lab)
plt.show()
cov30 = np.mean([np.cov(w[k:k + 18000], q[k:k + 18000])[0, 1] for k in (0, 18000)])
glue("ec_cov", float(cov30), display=False)
glue("ec_E", float(1.15 * cov30 / 1000 * 3600), display=False)
glue("ec_LE", float(1.15 * cov30 / 1000 * 2.45e6), display=False)
```

*Figure 3.2. Two minutes of synthetic 10 Hz measurements. Updrafts tend to carry moist air and downdrafts drier air, so the product of the fluctuations is mostly positive although it changes sign constantly. Averaged over two 30-minute periods its mean is {glue:text}`ec_cov:.3f` m s⁻¹ g kg⁻¹, a latent heat flux of about {glue:text}`ec_LE:.0f` W m⁻² ({glue:text}`ec_E:.2f` mm h⁻¹) for an air density of 1.15 kg m⁻³.*

```{code-cell} python
:tags: [hide-input]
# Ogive: cumulative cospectrum from high to low frequency, for one 60-minute record
wd, qd = w - w.mean(), q - q.mean()
W, Qs = np.fft.rfft(wd), np.fft.rfft(qd)
freq = np.fft.rfftfreq(n, 1 / fs)
cosp = (W * np.conj(Qs)).real / n ** 2 * 2
cosp[0] = 0.0
og = np.cumsum(cosp[::-1])[::-1]                                  # flux from frequencies ≥ f
period_min = 1 / freq[1:] / 60
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.semilogx(period_min, og[1:] / og[1], color=c1, lw=2.2)
for P, lab in [(5, "5 min"), (30, "30 min")]:
    frac = og[np.searchsorted(freq, 1 / (P * 60))] / og[1]
    ax.axvline(P, color=ink2, lw=0.8, ls=":"); ax.text(P * 1.08, 0.12, lab, color=ink2, fontsize=9.5)
    glue(f"og_{P}", float(frac * 100), display=False)
ax.set_xlabel("Largest eddy time scale included (min)"); ax.set_ylabel("Fraction of total flux")
ax.set_xlim(0.01, 60); ax.set_ylim(0, 1.1)
plt.show()
```

*Figure 3.3. Ogive of the synthetic vapor flux: the fraction of the covariance carried by eddies shorter than a given time scale. Eddies shorter than 5 minutes carry only {glue:text}`og_5:.0f`% of the flux; with a 30-minute window the estimate reaches {glue:text}`og_30:.0f`%. A flat ogive at the chosen window length indicates that the averaging period is long enough.*

## 3.5 Worked examples

**Example 3.1 — Humid air is lighter.** At 30 °C and 100 kPa, compare the density of dry air with that of air at 80% relative humidity, and find the virtual temperature of the humid air.

```{code-cell} python
T, p, RH = 30.0, 100.0, 0.80
e = RH * es(T)
qh = eps * e / (p - (1 - eps) * e)
Tv = (T + 273.15) * (1 + 0.61 * qh)
rho_dry = p * 1e3 / (Rd * (T + 273.15)); rho_moist = p * 1e3 / (Rd * Tv)
print(f"e = {e:.2f} kPa, q = {qh * 1000:.1f} g/kg, Tv = {Tv - 273.15:.2f} °C")
print(f"ρ dry = {rho_dry:.4f}, ρ humid = {rho_moist:.4f} kg/m³ ({100 * (rho_moist / rho_dry - 1):.2f}%)")
```

The vapor makes the air behave as if it were about 4 K warmer.

**Example 3.2 — Pressure and potential temperature aloft.** A radiosonde at 1500 m above a station at sea level (pressure 101.3 kPa) measures 16.0 °C. Using a scale height based on a mean virtual temperature of 293 K, find the pressure there and the potential temperature, and compare with the potential temperature at the surface, where the air is 27.0 °C.

```{code-cell} python
Hs = Rd * 293.0 / g
p15 = 101.3 * np.exp(-1500 / Hs)
th15 = (16.0 + 273.15) * (100 / p15) ** (Rd / cp)
th0 = (27.0 + 273.15) * (100 / 101.3) ** (Rd / cp)
print(f"Hs = {Hs / 1000:.2f} km, p(1500 m) = {p15:.1f} kPa, θ = {th15:.1f} K aloft, {th0:.1f} K at the surface")
print("Stable" if th15 > th0 else "Unstable", f"layer: dθ/dz = {(th15 - th0) / 1.5:.2f} K/km")
```

**Example 3.3 — The cost of a tilted anemometer.** In the synthetic record, the true vertical wind has zero mean. Suppose the anemometer is tilted by 2° in the direction of the wind, so it records $w_m = w\cos 2° + u \sin 2°$. Compute the vapor flux from the raw and the rotated data.

```{code-cell} python
ang = np.radians(2.0)
w_m = w * np.cos(ang) + u * np.sin(ang)
u_m = u * np.cos(ang) - w * np.sin(ang)
k30 = slice(0, 18000)                                              # one 30-minute period
cov = lambda a, b: np.mean((a - a.mean()) * (b - b.mean()))
th = np.arctan2(w_m[k30].mean(), u_m[k30].mean())                 # rotation angle that zeroes mean w
w_r = w_m[k30] * np.cos(th) - u_m[k30] * np.sin(th)
print(f"mean measured w = {w_m[k30].mean():.3f} m/s; rotation angle = {np.degrees(th):.2f}°")
f_true, f_tilt, f_rot = cov(w[k30], q[k30]), cov(w_m[k30], q[k30]), cov(w_r, q[k30])
print(f"flux: true {f_true:.4f}, tilted {f_tilt:.4f} ({100 * (f_tilt / f_true - 1):+.1f}%), rotated {f_rot:.4f} ({100 * (f_rot / f_true - 1):+.1f}%)")
```

Without rotation the mean vertical wind is not zero, and the measured flux includes a fraction $\sin 2°$ of the covariance of the horizontal wind with humidity. Near the ground that covariance is negative, because downdrafts bring faster and drier air from above, so the tilt makes the vapor flux too small. Rotating the coordinates so that the mean vertical wind vanishes removes most of the error; the small remainder arises because the mean vertical wind over a single 30-minute period is not exactly zero even before tilting.

**Example 3.4 — Stability from a sounding.** In Figure 3.1, compute the Brunt–Väisälä frequency in the inversion layer between 1.40 and 1.55 km, and the period of oscillation of a displaced parcel.

```{code-cell} python
i1, i2 = np.searchsorted(z, 1400), np.searchsorted(z, 1550)
dthdz = (theta_v[i2] - theta_v[i1]) / (z[i2] - z[i1])
N = np.sqrt(g / theta_v[i1:i2].mean() * dthdz)
print(f"dθv/dz = {dthdz * 1000:.1f} K/km, N = {N:.4f} 1/s, period = {2 * np.pi / N / 60:.1f} min")
```

## 3.6 Common mistakes

- **Using temperature instead of virtual temperature for buoyancy.** Humid air is lighter; ignoring vapor misjudges stability, especially over wet surfaces.
- **Judging stability from temperature alone.** A layer in which temperature falls with height can still be stable; use $\theta_v$.
- **Averaging over too short a period.** Large eddies then appear as trends in the mean, and part of the flux is lost.
- **Skipping coordinate rotation.** A tilt of one or two degrees produces a spurious mean vertical wind and contaminates every flux.
- **Treating the eddy diffusivity as a property of air.** It depends on height, wind speed and stability.

## 3.7 Problems

```{code-cell} python
:tags: [remove-cell]
e1 = 0.6 * es(25.0)
glue("r1_q", float(1000 * eps * e1 / (101.3 - (1 - eps) * e1)), display=False)
glue("r1_Tv", float((25 + 273.15) * (1 + 0.61 * eps * e1 / (101.3 - (1 - eps) * e1)) - 273.15), display=False)
glue("r2", float(101.3 * np.exp(-5500 / (Rd * 255 / g))), display=False)
glue("r3", float((10 + 273.15) * (100 / 85) ** (Rd / cp) - 273.15), display=False)
glue("r4", 1.2 * 0.15 / 1000 * 2.45e6, display=False)
Nn = np.sqrt(g / 290 * 0.008)
glue("r5", float(2 * np.pi / Nn / 60), display=False)
```

1. Find the specific humidity and virtual temperature of air at 25 °C, 60% relative humidity and 101.3 kPa.
2. With a mean virtual temperature of 255 K, at what pressure is the height of 5.5 km?
3. Air at 85 kPa has a temperature of 10 °C. What is its potential temperature (in °C)?
4. An eddy-covariance system gives $\overline{w'q'}$ = 0.15 m s⁻¹ g kg⁻¹ over a period with air density 1.2 kg m⁻³. Find the latent heat flux.
5. A nocturnal inversion has a potential temperature gradient of 8 K km⁻¹ and a mean potential temperature of 290 K. Find the period of buoyancy oscillations.

```{admonition} Answers
:class: dropdown
1. $q$ = {glue:text}`r1_q:.1f` g kg⁻¹; $T_v$ = {glue:text}`r1_Tv:.1f` °C.
2. About {glue:text}`r2:.0f` kPa (about half of the surface pressure).
3. {glue:text}`r3:.1f` °C.
4. About {glue:text}`r4:.0f` W m⁻².
5. About {glue:text}`r5:.1f` minutes.
```

## 3.8 Open questions

- **Fluxes in complex terrain and at night.** Eddy covariance assumes horizontal homogeneity and steady conditions. On slopes and in stable nights, advection and drainage flows carry fluxes that a single tower misses.
- **Large eddies and the energy gap.** How much of the missing energy in eddy-covariance budgets is carried by slow, organized circulations that a 30-minute average treats as part of the mean?
- **Beyond K-theory.** Convective boundary layers transport heat and vapor in plumes that a local diffusivity cannot describe; nonlocal closures and large-eddy simulation are filling the gap.

## 3.9 Key papers

- Baldocchi, D. D. (2003). Assessing the eddy covariance technique for evaluating carbon dioxide exchange rates of ecosystems: past, present and future. *Global Change Biology*, 9, 479–492.
- Foken, T. (2008). The energy balance closure problem: an overview. *Ecological Applications*, 18, 1351–1367.
- Webb, E. K., Pearman, G. I., and Leuning, R. (1980). Correction of flux measurements for density effects due to heat and water vapour transfer. *Quarterly Journal of the Royal Meteorological Society*, 106, 85–100.
- Wilczak, J. M., Oncley, S. P., and Stage, S. A. (2001). Sonic anemometer tilt correction algorithms. *Boundary-Layer Meteorology*, 99, 127–150.

## 3.10 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The sections on water vapor, stability and turbulent transport.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on principles of surface–atmosphere exchange.
- Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer.
- Garratt, J. R. (1992). *The Atmospheric Boundary Layer*. Cambridge University Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
