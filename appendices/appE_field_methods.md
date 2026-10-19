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

# Appendix E. Field methods

Every number in hydrology starts with a measurement, and every measurement has an uncertainty that propagates into the analyses of the main chapters. This appendix summarizes the principles of the most common field methods, with emphasis on what they measure, their typical errors, and the calculations behind them.

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

## E.1 Stream gauging

### Measuring discharge

Discharge cannot be measured continuously; it is measured occasionally and converted to a continuous record through a relation with water level (stage), which can be logged every few minutes.

- **Velocity–area method.** The cross-section is divided into verticals. At each, the depth is measured and the mean velocity estimated with a current meter, usually at 0.6 of the depth from the surface or as the average at 0.2 and 0.8 of the depth. In the **mid-section** method, each vertical represents a panel extending halfway to its neighbors, and the discharge is the sum of velocity × depth × panel width (Example E.1). Acoustic Doppler current profilers (ADCPs) measure velocity profiles across the whole section from a moving boat and are now the standard in large rivers.
- **Dilution gauging.** A known mass $M$ of a conservative tracer (salt, a fluorescent dye) is injected upstream, and its concentration is measured downstream after complete mixing. Since all the tracer must pass the sampling point, $M = Q\int (c - c_b)\,dt$, so $Q = M/\int(c - c_b)\,dt$, where $c_b$ is the background concentration. It works best in turbulent mountain streams, where current meters struggle.
- **Hydraulic structures.** Weirs and flumes have known stage–discharge relations derived from hydraulics (see the [Hydraulics](https://hydroclimatol.com/hydraulics/) book), and are used in small experimental catchments.
- **Non-contact methods.** Surface velocities from radar or image velocimetry, converted to mean velocity with an assumed profile, are increasingly used during floods, when wading or boat measurements are dangerous.

### Rating curves

Gaugings at different stages define the **rating curve**, usually a power law

\begin{equation*}
Q = a\,(h - h_0)^b ,
\tag{E.1}
\end{equation*}

where $h_0$ is the stage of zero flow and $b$ is typically 1.5–2.5, depending on the shape of the control (Figure E.1). Ratings shift when the channel changes through erosion, deposition or vegetation growth, and they can be **hysteretic** during floods, with higher discharge at the same stage on the rising limb than on the falling limb. The largest uncertainty is usually at high flows, where the curve must be extrapolated beyond the highest gauging. Discharge uncertainties of 5–10% at medium flows and 20–40% for extreme floods are common.

```{code-cell} python
:tags: [hide-input]
from scipy.optimize import curve_fit
rng = np.random.default_rng(5)
a_t, h0_t, b_t = 12.0, 0.25, 1.8
h_g = np.sort(rng.uniform(0.35, 1.8, 18))
Q_g = a_t * (h_g - h0_t) ** b_t * np.exp(0.06 * rng.standard_normal(len(h_g)))
rating = lambda h, a, h0, b: a * np.clip(h - h0, 1e-6, None) ** b
p, cov = curve_fit(rating, h_g, Q_g, p0=(10, 0.2, 1.7))
hh = np.linspace(0.3, 3.0, 200)
draws = rng.multivariate_normal(p, cov, 500)
band = np.percentile([rating(hh, *d) for d in draws], [5, 95], axis=0)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.fill_betweenx(hh, band[0], band[1], color=c1, alpha=0.2, lw=0, label="90% band (parameters)")
ax.plot(rating(hh, *p), hh, color=c1, lw=2.2, label=f"$Q = {p[0]:.1f}\\,(h - {p[1]:.2f})^{{{p[2]:.2f}}}$")
ax.plot(Q_g, h_g, "o", color=ink, ms=5, label="Gaugings (synthetic)")
ax.axhline(h_g.max(), color=c2, lw=1, ls="--"); ax.text(5, h_g.max() + 0.05, "highest gauging", color=c2, fontsize=9.5)
ax.set_xlabel("Discharge (m³ s⁻¹)"); ax.set_ylabel("Stage (m)"); ax.legend(loc="lower right", fontsize=9)
plt.show()
i3 = np.argmin(np.abs(hh - 3.0))
glue("unc_hi", float(100 * (band[1][i3] - band[0][i3]) / 2 / rating(3.0, *p)), display=False)
```

*Figure E.1. A rating curve fitted to 18 synthetic gaugings with 6% scatter. The 90% band reflects parameter uncertainty only. At a stage of 3 m, far above the highest gauging, it is about ±{glue:text}`unc_hi:.0f`%; errors in the form of the curve at high stages, where flow may spill onto the floodplain, add to this.*

## E.2 Precipitation

**Gauges** measure precipitation at a point. Wind deflects drops and especially snowflakes around the gauge orifice, so gauges **undercatch**: by a few percent for rain and by 20–50% or more for snow in windy conditions, unless they are shielded and corrected. Evaporation from the gauge and wetting losses add small negative errors. **Weather radar** estimates precipitation over large areas from the backscatter of drops, through a relation between reflectivity and rain rate that depends on the drop-size distribution; it captures spatial patterns well but needs adjustment to gauges. **Satellite** products combine microwave and infrared observations into global estimates (Appendix F). The areal averaging of point measurements is discussed in Chapter 5.

## E.3 Evaporation and turbulent fluxes

**Eddy covariance** (Chapter 3) measures the turbulent fluxes of heat, water vapor and CO₂ from high-frequency (10–20 Hz) measurements of vertical wind speed with a sonic anemometer and of temperature and humidity with fast sensors. It requires flat, homogeneous terrain within the **footprint**, the upwind area that contributes to the flux, which extends some tens to hundreds of times the measurement height. Its main known error is the **energy balance closure gap**: the sum of measured sensible and latent heat fluxes is typically 10–30% less than the available energy.

Other methods include **weighing lysimeters**, which measure evaporation as the change in mass of an isolated soil block; the **Bowen ratio–energy balance** method, which partitions available energy using temperature and humidity gradients; **scintillometers**, which measure fluxes over paths of hundreds of meters to kilometers; and **sap flow** sensors, which measure transpiration of individual trees. Evaporation pans measure open-water evaporation and are converted to reference evaporation with empirical coefficients.

## E.4 Soil water and groundwater

**Soil water content** is measured gravimetrically (oven-drying samples, the reference method), with dielectric sensors (time-domain reflectometry, capacitance and frequency-domain probes), with neutron probes, and over areas of hectares with **cosmic-ray neutron sensors**, which respond to hydrogen in the top few decimeters of soil. Dielectric sensors need soil-specific calibration for accurate absolute values. **Matric potential** is measured with tensiometers in wet soil and with psychrometers or dielectric potential sensors in dry soil (Chapter 11).

**Groundwater levels** are measured in wells and piezometers with pressure transducers. A **piezometer** is open only over a short screened interval and measures the head there; comparing piezometers at different depths gives vertical gradients. **Hydraulic conductivity** is estimated from slug tests (the recovery of the water level after a sudden change), pumping tests, and permeameter tests on cores, each sampling a different volume of aquifer.

## E.5 Tracers

**Stable isotopes** of water are reported as deviations of the isotope ratio $R$ (²H/¹H or ¹⁸O/¹⁶O) from that of the international standard VSMOW, in parts per thousand:

\begin{equation*}
\delta = \left(\frac{R_{\text{sample}}}{R_{\text{VSMOW}}} - 1\right) \times 1000\ ‰ .
\tag{E.2}
\end{equation*}

Precipitation becomes isotopically lighter (more negative δ) with lower temperature, higher altitude, greater distance from the ocean and larger rain amounts; evaporation enriches the remaining water. These patterns make isotopes useful for hydrograph separation (Chapter 14), for estimating transit times, and for partitioning evaporation and transpiration. **Chemical tracers** such as chloride, silica and electrical conductivity are cheap to measure continuously but are not always conservative. **Temperature** traces groundwater discharge into streams, which is cool in summer and warm in winter. **Artificial tracers** (salt, dyes, deuterium) are used in controlled experiments.

## E.6 Worked examples

**Example E.1 — Mid-section discharge.** A stream gauging gives the following depths and velocities at verticals across the section (distances from the left bank). Compute the discharge.

```{code-cell} python
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])        # m, edges at 0 and 7
d = np.array([0.0, 0.35, 0.60, 0.80, 0.85, 0.70, 0.40, 0.0])   # m
v = np.array([0.0, 0.25, 0.45, 0.60, 0.62, 0.50, 0.30, 0.0])   # m/s, mean in the vertical
w = np.empty_like(x)                                            # panel widths: half-way to neighbors
w[1:-1] = (x[2:] - x[:-2]) / 2; w[0] = (x[1] - x[0]) / 2; w[-1] = (x[-1] - x[-2]) / 2
Q = np.sum(v * d * w)
print(f"Q = {Q:.2f} m3/s; area = {np.sum(d * w):.2f} m2; mean velocity = {Q / np.sum(d * w):.2f} m/s")
```

**Example E.2 — Slug dilution gauging.** 2.0 kg of salt (as NaCl) is injected into a mountain stream. Downstream, the excess concentration over background follows the curve below, sampled every 10 s. Estimate the discharge.

```{code-cell} python
t = np.arange(0, 400, 10.0)                                     # s
c = 55 * (t / 80) ** 2 * np.exp(2 * (1 - t / 80))                # mg/L above background (synthetic curve)
area = np.trapezoid(c, t) * 1e-3                                # (kg/m3) s, since 1 mg/L = 1e-3 kg/m3
print(f"∫c dt = {area:.3f} kg s/m3 -> Q = {2.0 / area:.3f} m3/s")
```

## E.7 Further reading

- Herschy, R. W. (2009). *Streamflow Measurement* (3rd ed.). Taylor & Francis.
- Kendall, C., and McDonnell, J. J. (eds.) (1998). *Isotope Tracers in Catchment Hydrology*. Elsevier.
- Lee, X., Massman, W., and Law, B. (eds.) (2004). *Handbook of Micrometeorology*. Kluwer.
- World Meteorological Organization (2010). *Manual on Stream Gauging*. WMO-No. 1044.

---

*Figures in this appendix are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
