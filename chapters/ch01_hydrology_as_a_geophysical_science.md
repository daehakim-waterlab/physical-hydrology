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

# 1. Hydrology as a geophysical science

```{admonition} Core ideas
:class: tip
1. Hydrology is the science of the water cycle over the continents. It deals with the **transport of water** in its three phases between the atmosphere, the land surface and the subsurface, and with the **global water balance** that links them to the oceans.
2. A store is characterized not only by its mean **residence time** $S/F$ but by its **transit-time distribution**. Stores with the same mean residence time can release a pulse of water, or of a solute, in very different ways.
3. Hydrological variables are measured at one scale and needed at another. Because the governing processes are **nonlinear** and the landscape is **heterogeneous**, a model at a coarse scale cannot simply use averaged parameters. This is the problem of **parameterization**.
4. Every process in this book obeys the same **field equations**: conservation of mass, momentum and energy, written for a fluid moving through space. The chapters that follow simplify them, step by step, for each part of the water cycle.
5. Integrating the field equations over a control volume gives **water balances**. Combining the balance of the land with that of the atmosphere links precipitation minus evaporation to the **convergence of water vapor** transported by the wind.
6. A term computed as the residual of a balance inherits the **absolute** errors of all the other terms. Small relative errors in large fluxes become large relative errors in small ones.
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

## 1.1 What hydrology studies

Water on Earth is constantly on the move. It evaporates from the oceans and the land, is carried by the wind, condenses and falls back, runs over and through the ground, and returns to the sea. **Hydrology** is the science of this circulation over the continents. It studies how water is transported, in liquid, solid and vapor form, between the atmosphere, the land surface, the soil, the aquifers and the channels, and how much of it is stored in each place at each time. It also studies the global water balance, which ties the continental cycle to the oceans and the atmosphere as a whole.

This definition places hydrology among the **geophysical sciences**, alongside meteorology, oceanography and geomorphology, with which it shares both objects and methods. The atmospheric boundary layer is studied by meteorologists and hydrologists alike; groundwater by geologists and hydrologists; channel flow by hydraulic engineers and hydrologists. What distinguishes hydrology is its concern with the **whole path** of the water over the land, and with the fluxes that cross the boundaries between the specialized compartments.

Hydrology is also an engineering discipline, and its methods were long shaped by the need to answer practical questions about floods and water supply before the physics was understood (see the companion book [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/)). This book takes the other route. It starts from the physical laws and asks how far they can be followed before empirical relations become necessary. The answer, as later chapters will show, is often "further than expected": many practical formulas turn out to be approximate solutions of the field equations under well-defined assumptions.

## 1.2 Stores, fluxes and residence times

### The global water cycle

About 97% of the water on Earth is in the oceans, about 2% in ice sheets and glaciers, and less than 1% in groundwater, lakes, soils, rivers, the atmosphere and living things. The fluxes between these stores are large compared with the small stores. Global precipitation is roughly 500,000 km³ per year, about 1 m of water over the whole Earth, while the atmosphere holds only about 25 mm at any time. Over the continents, precipitation exceeds evaporation by roughly 40,000–45,000 km³ per year. The excess returns to the oceans as river and groundwater discharge, and is balanced by an equal net transport of water vapor from the oceans to the land. Estimates of these fluxes from different data sets and methods differ by about 10% (Oki and Kanae, 2006; Rodell et al., 2015); closing the global water budget within its uncertainty is still an active research problem.

### Residence time

For a store with volume $S$ and steady throughflow $F$, the **mean residence time** is

\begin{equation*}
T_r = \frac{S}{F}.
\tag{1.1}
\end{equation*}

It ranges from about nine days for the atmosphere and a few weeks for river channels to thousands of years for the oceans and deep aquifers.

### Transit-time distributions

The mean is not enough. Water that enters a store at a given moment leaves it over a range of times, described by the **transit-time distribution** $p(\tau)$: the probability density of the time $\tau$ that a parcel of water spends in the store. Its mean equals $T_r$ only in steady state. Its shape depends on how the store mixes:

- **perfect mixing** (each parcel equally likely to leave at any moment) gives an exponential distribution, $p(\tau) = T_r^{-1} e^{-\tau/T_r}$;
- **plug flow** (first in, first out) gives every parcel exactly the same transit time, $\tau = T_r$;
- real catchments lie between these extremes, and gamma distributions with a shape parameter below 1 often describe them well, with a heavy tail of very old water.

The distinction matters whenever water carries something with it. A pollutant spilled into a well-mixed lake appears at the outlet at once and decays slowly; spilled into a plug-flow aquifer, it arrives all at once after $T_r$. Figure 1.1 compares the outflow concentration after a pulse input for three stores with the same mean residence time.

```{code-cell} python
:tags: [hide-input]
from scipy.stats import gamma
Tr = 2.0                                   # mean residence time (years)
tau = np.linspace(0, 10, 2001)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for a, col, lab in [(0.5, c2, "heavy-tailed (gamma, shape 0.5)"), (1.0, c1, "well mixed (exponential)"),
                    (6.0, c3, "partly plug flow (gamma, shape 6)")]:
    ax.plot(tau, gamma.pdf(tau, a, scale=Tr / a), color=col, lw=2.2, label=lab)
ax.axvline(Tr, color=ink2, lw=1, ls="--"); ax.text(Tr + 0.1, 1.0, "plug flow:\nall water\nleaves at $T_r$", color=ink2, fontsize=9.5)
ax.set_xlim(0, 10); ax.set_ylim(0, 1.2)
ax.set_xlabel("Transit time τ (years)"); ax.set_ylabel("p(τ) (yr⁻¹)")
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
for a in (0.5, 1.0, 6.0):
    glue(f"young_{str(a).replace('.', '_')}", float(gamma.cdf(0.25, a, scale=Tr / a) * 100), display=False)
```

*Figure 1.1. Transit-time distributions of three stores with the same mean residence time of 2 years. The fraction of water leaving within three months of entering is {glue:text}`young_0_5:.0f`% for the heavy-tailed store, {glue:text}`young_1_0:.0f`% for the well-mixed store, and almost none for the store that behaves partly like plug flow.*

## 1.3 Scales and the problem of parameterization

### A range of scales

Hydrological processes span an enormous range of scales. Water moves through soil pores micrometers wide and through river basins thousands of kilometers across, in times from seconds (raindrop impact, turbulent eddies) to millennia (deep groundwater). No single model can resolve all of them. Every description in this book is therefore written at a chosen scale, and the processes at smaller scales are represented through **effective parameters** or **parameterizations**: functional relations between the variables at the chosen scale, whose form comes from physics and whose coefficients come from data.

Porous media give the classic example. Darcy's law (Chapter 11) does not describe flow in individual pores; it describes the average flow over a volume large enough to contain many pores but small compared with the region of interest, the **representative elementary volume**. Hydraulic conductivity is a property of that volume, not of any pore.

### Why averages are not enough

Moving to a larger scale is not simply a matter of averaging parameters, because the processes are **nonlinear** and the landscape is **heterogeneous**. If a flux $f$ depends nonlinearly on a parameter $k$ that varies in space, then in general

\begin{equation*}
\overline{f(k)} \neq f(\bar{k}),
\tag{1.2}
\end{equation*}

an inequality that, for convex or concave $f$, follows from Jensen's inequality. Figure 1.2 shows a simple case. Rain falls at a uniform rate $i$ on a hillslope whose saturated hydraulic conductivity $K_s$ varies from point to point. Infiltration-excess runoff at a point is $\max(i - K_s, 0)$, a nonlinear function of $K_s$ with a threshold. Using the mean conductivity in that formula can predict no runoff at all, while in reality the patches with low conductivity produce runoff that the patches with high conductivity cannot absorb.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(1)
i_rain, Ks_median = 15.0, 10.0                       # mm/h
sig = np.linspace(0.01, 1.6, 60)                     # standard deviation of ln Ks
z = rng.standard_normal(200_000)
true_mean, from_mean = [], []
for s in sig:
    Ks = Ks_median * np.exp(s * z)
    true_mean.append(np.maximum(i_rain - Ks, 0).mean())
    from_mean.append(max(i_rain - Ks.mean(), 0))
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(sig, true_mean, color=c1, lw=2.4, label="Areal mean of local runoff")
ax.plot(sig, from_mean, color=c2, lw=2.2, ls="--", label="Runoff computed with mean $K_s$")
ax.set_xlabel("Spatial variability of $K_s$ (std. dev. of ln $K_s$)")
ax.set_ylabel("Runoff rate (mm h⁻¹)")
ax.set_xlim(0, 1.6); ax.set_ylim(0, None); ax.legend(loc="lower left", fontsize=9.5)
plt.show()
s_ex = 1.0
Ks = Ks_median * np.exp(s_ex * z)
glue("jen_true", float(np.maximum(i_rain - Ks, 0).mean()), display=False)
glue("jen_meanK", float(Ks.mean()), display=False)
glue("jen_frac", float((Ks < i_rain).mean() * 100), display=False)
```

*Figure 1.2. Infiltration-excess runoff from rain at 15 mm h⁻¹ falling on a hillslope with lognormally distributed saturated conductivity (median 10 mm h⁻¹). With a standard deviation of ln $K_s$ of 1, the mean conductivity is {glue:text}`jen_meanK:.1f` mm h⁻¹, higher than the rainfall rate, so a model using it predicts no runoff; in fact {glue:text}`jen_frac:.0f`% of the hillslope produces runoff and the areal mean is {glue:text}`jen_true:.1f` mm h⁻¹.*

Three responses to this problem recur throughout the book:

1. derive an **effective parameter**, the single value that reproduces the areal mean flux, and accept that it depends on the conditions (here, on the rainfall rate);
2. describe the **distribution** of the parameter and integrate the local process over it, as in the variable-source-area models of Chapter 14;
3. look for **new relations** that hold at the larger scale itself, as in the catchment-scale storage–discharge relations of Chapter 15.

## 1.4 Physical and systems approaches

Two broad strategies have shaped hydrology.

- **The physical approach** starts from the conservation laws, specifies the material properties and the boundary and initial conditions, and solves for the flow. Its strength is that the parameters have physical meaning and can, in principle, be measured. Its weakness is that the required detail is rarely available, and that the equations, valid at small scales, must be applied at the scale of the data.
- **The systems approach** treats the catchment, or one of its parts, as a system that transforms an input (rainfall) into an output (streamflow). It identifies the transformation from observed input–output records with as few assumptions about the inside as possible. The unit hydrograph is the classic example (Chapter 15). Its strength is that it uses only what is measured. Its weakness is that it cannot be applied where no records exist, or under conditions that are absent from the record, such as a changed land use or climate.

Most useful models lie between the two. They keep the conservation of mass exactly, describe the main storages and fluxes with simple physically motivated relations, and estimate a few parameters from data. A good model of this kind is **parsimonious**: it has no more parameters than the data can determine. This book develops the physical description of each process far enough to show which simple relations it justifies, and under which conditions.

## 1.5 The field equations

### The material derivative

Consider a property $C$ of a moving fluid, such as its temperature or the concentration of water vapor, described as a field $C(\mathbf{x}, t)$ with velocity field $\mathbf{u}(\mathbf{x}, t)$. The rate of change of $C$ following a fluid parcel is the **material derivative**

\begin{equation*}
\frac{DC}{Dt} = \frac{\partial C}{\partial t} + \mathbf{u}\cdot\nabla C
= \frac{\partial C}{\partial t} + u\frac{\partial C}{\partial x} + v\frac{\partial C}{\partial y} + w\frac{\partial C}{\partial z}.
\tag{1.3}
\end{equation*}

The first term is the local change at a fixed point; the second, the **advective** change caused by carrying the parcel to a place where $C$ is different.

### Conservation of mass

For a fixed control volume $V$ with surface $A$ and outward unit normal $\mathbf{n}$, the mass inside changes only by the flux of mass through the surface:

\begin{equation*}
\frac{d}{dt}\int_V \rho\, dV = -\oint_A \rho\,\mathbf{u}\cdot\mathbf{n}\, dA .
\end{equation*}

Applying the divergence theorem to the right side and letting the volume shrink to a point gives the **continuity equation**

\begin{equation*}
\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho\,\mathbf{u}) = 0,
\tag{1.4}
\end{equation*}

which for an incompressible fluid reduces to $\nabla\cdot\mathbf{u} = 0$. The same reasoning applied to any substance carried by the fluid, with concentration $c$ (mass per unit volume), a molecular (or turbulent) diffusivity $D$ and a source $s$, gives the **advection–diffusion equation**

\begin{equation*}
\frac{\partial c}{\partial t} + \nabla\cdot(\mathbf{u}\, c) = \nabla\cdot(D\,\nabla c) + s .
\tag{1.5}
\end{equation*}

This single equation, with different meanings for $c$, $\mathbf{u}$ and $D$, describes water vapor in the atmosphere (Chapter 3), solutes in groundwater, and, in a modified form, water itself in unsaturated soil (Chapter 11).

### Conservation of momentum and energy

Newton's second law applied to a fluid parcel gives the momentum equation, which for a Newtonian incompressible fluid is the Navier–Stokes equation,

\begin{equation*}
\rho\,\frac{D\mathbf{u}}{Dt} = -\nabla p + \rho\,\mathbf{g} + \mu\,\nabla^2 \mathbf{u},
\tag{1.6}
\end{equation*}

where $p$ is pressure, $\mathbf{g}$ the acceleration of gravity and $\mu$ the dynamic viscosity (see the [Hydraulics](https://hydroclimatol.com/hydraulics/) book). Conservation of energy, for the purposes of this book, is mostly needed in the form of an advection–diffusion equation for heat, plus the energy budget at the land surface (Chapter 4), where radiation, sensible and latent heat meet.

These equations are rarely solved in full in hydrology. Their value is that every simplification used later can be traced back to them. Averaging over turbulence gives the boundary-layer equations of Chapter 4. Averaging over depth gives the shallow-water equations of Chapter 9. Averaging over pores, and neglecting inertia, gives Darcy's law and the Richards equation of Chapter 11.

### Advection versus diffusion

The relative importance of the two transport terms in Equation (1.5) is measured by the **Péclet number**,

\begin{equation*}
\text{Pe} = \frac{U L}{D},
\tag{1.7}
\end{equation*}

where $U$ is a velocity scale and $L$ a length scale. When Pe is large, a pulse of tracer is carried along almost unchanged; when Pe is small, it spreads out before it travels far. Figure 1.3 shows the one-dimensional solution for an instantaneous release of mass $M$ per unit cross-sectional area,

\begin{equation*}
c(x, t) = \frac{M}{\sqrt{4\pi D t}}\,\exp\!\left[-\frac{(x - U t)^2}{4 D t}\right],
\tag{1.8}
\end{equation*}

for two diffusivities. The same mathematics reappears as the diffusion wave of flood routing in Chapter 10.

```{code-cell} python
:tags: [hide-input]
U, L = 0.5, 100.0                                      # m/s, m (distance to the observation point)
x = np.linspace(-20, 260, 800)
fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True)
for ax, D in zip(axes, (2.0, 50.0)):
    for t, col in [(60, c3), (200, c1), (400, c2)]:
        c = 1.0 / np.sqrt(4 * np.pi * D * t) * np.exp(-(x - U * t) ** 2 / (4 * D * t))
        ax.plot(x, c, color=col, lw=2.2, label=f"t = {t} s")
    ax.set_title(f"D = {D:g} m² s⁻¹, Pe = UL/D = {U * L / D:g}")
    ax.set_ylabel("c (per unit mass)")
    ax.legend(loc="upper right", fontsize=9.5)
axes[1].set_xlabel("Distance x (m)"); axes[1].set_xlim(-20, 260)
plt.show()
```

*Figure 1.3. An instantaneous pulse released at $x$ = 0 in a flow with velocity 0.5 m s⁻¹. With weak diffusion (top, Pe = 25 over 100 m) it travels almost unchanged; with strong diffusion (bottom, Pe = 1) it spreads as fast as it moves.*

## 1.6 Water balances

### The land surface

Integrating the continuity equation for water over a control volume that extends from the top of the vegetation to the base of the active groundwater, and over a catchment area $A$, gives the catchment water balance used throughout hydrology:

\begin{equation*}
\frac{dS}{dt} = P - E - Q - L ,
\tag{1.9}
\end{equation*}

where all terms are averages over the catchment area, expressed as depth rates: $S$ is the water stored per unit area, $P$ precipitation, $E$ evaporation (including transpiration), $Q$ streamflow and $L$ the net groundwater outflow across the catchment boundary. The terms are not equally well known. Streamflow is measured at a single cross-section and is often the most accurate. Precipitation must be interpolated from scattered gauges. Evaporation is usually estimated rather than measured. Storage was long the most poorly known term, until satellite gravimetry (GRACE) began to measure changes in total water storage over large regions in 2002.

### The atmosphere

The same balance can be written for the column of air above the land. Let $W$ be the precipitable water, the mass of water vapor in the column per unit area, and $\mathbf{Q}_v$ the vertically integrated horizontal flux of vapor (kg m⁻¹ s⁻¹). Then

\begin{equation*}
\frac{\partial W}{\partial t} + \nabla\cdot\mathbf{Q}_v = E - P .
\tag{1.10}
\end{equation*}

Averaged over a region and over a long period, the storage term vanishes, and

\begin{equation*}
\overline{P} - \overline{E} = -\overline{\nabla\cdot\mathbf{Q}_v} ,
\tag{1.11}
\end{equation*}

so the excess of precipitation over evaporation equals the **convergence of moisture** carried into the region by the wind. Combining with Equation (1.9), the long-term runoff of a large basin equals the moisture convergence above it. This **atmospheric water balance** gives an estimate of $P - E$ that is independent of rain gauges and of evaporation formulas, using wind and humidity fields from radiosondes or reanalyses. It also shows that land hydrology and atmospheric dynamics are two halves of the same balance.

## 1.7 Measurement and uncertainty

Every term in a water balance carries an error, and terms computed as residuals carry the errors of all the others. If $E$ is estimated as $P - Q - \Delta S$ over a period, and the errors of the three measured terms are independent, the variance of the estimate is

\begin{equation*}
\sigma_E^2 = \sigma_P^2 + \sigma_Q^2 + \sigma_{\Delta S}^2 ,
\tag{1.12}
\end{equation*}

where each $\sigma$ is an **absolute** error (mm). A 5% error in a precipitation of 1300 mm (65 mm) and a 5% error in a streamflow of 800 mm (40 mm) combine into an error of 76 mm in an evaporation of 500 mm, or 15%. The smallest term of a balance is always the least certain when it is computed as a residual. This is why independent estimates, for example of evaporation from eddy covariance (Chapter 3) or of $P - E$ from moisture convergence (Section 1.6), are valuable even when they are individually less precise. Example 1.4 demonstrates the effect by simulation.

## 1.8 Worked examples

**Example 1.1 — Residence and transit times.** A lake holds 0.42 km³ and is fed by a river with a mean inflow of 9.5 m³ s⁻¹. Assume it is well mixed. Find the mean residence time, and the fraction of a conservative tracer added in one day that is still in the lake after one year and after three years.

```{code-cell} python
S, F = 0.42e9, 9.5                                # m³, m³/s
Tr = S / F / (365.25 * 86400)                     # years
for t in (1.0, 3.0):
    print(f"Mean residence time {Tr:.2f} yr; fraction remaining after {t:.0f} yr = {np.exp(-t / Tr):.2f}")
```

About half of the tracer is still in the lake after one year and an eighth after three. In general, a well-mixed store keeps 37% of a pulse after one residence time and 5% after three.

**Example 1.2 — An effective conductivity.** For the hillslope of Figure 1.2 with a standard deviation of ln $K_s$ of 1, find the single "effective" conductivity that reproduces the areal mean runoff for rainfall of 15 mm h⁻¹, and repeat for 30 mm h⁻¹.

```{code-cell} python
from scipy.optimize import brentq
Ks_field = Ks_median * np.exp(1.0 * z)
for i_r in (15.0, 30.0):
    q_mean = np.maximum(i_r - Ks_field, 0).mean()
    K_eff = i_r - q_mean                            # single value giving the same runoff
    print(f"i = {i_r:.0f} mm/h: areal runoff {q_mean:.2f} mm/h -> effective Ks = {K_eff:.2f} mm/h "
          f"(median {Ks_median:.0f}, mean {Ks_field.mean():.1f})")
```

The effective conductivity is not a property of the hillslope alone: it changes with the rainfall rate. A model calibrated with one storm will be biased for another, which is a general feature of effective parameters of nonlinear processes.

**Example 1.3 — Moisture convergence.** Over a square region 500 km on a side, the vertically integrated vapor flux averaged over a month is 150 kg m⁻¹ s⁻¹ eastward across the western boundary and 120 kg m⁻¹ s⁻¹ across the eastern boundary, with a net outflow of 10 kg m⁻¹ s⁻¹ across the northern and southern boundaries together. The precipitable water rose by 2 mm during the month. Estimate $P - E$ for the month.

```{code-cell} python
side = 500e3                                      # m
net_in = (150 - 120 - 10) * side                  # kg/s entering the region
conv = net_in / side ** 2                         # kg m⁻² s⁻¹ = mm/s
days = 30
P_minus_E = conv * 86400 * days - 2.0             # mm per month, minus the storage gain
print(f"Convergence = {conv * 86400:.2f} mm/d; P - E = {P_minus_E:.0f} mm for the month")
```

**Example 1.4 — Error in a residual.** A catchment has a measured annual precipitation of 1300 mm (standard error 5%), streamflow of 800 mm (5%) and storage change of 0 mm (standard error 20 mm). Estimate the evaporation and its uncertainty with Equation (1.12), and check by simulation.

```{code-cell} python
P, Qm, dS = 1300.0, 800.0, 0.0
sP, sQ, sS = 0.05 * P, 0.05 * Qm, 20.0
E = P - Qm - dS
sE = np.sqrt(sP ** 2 + sQ ** 2 + sS ** 2)
rng = np.random.default_rng(7)
E_mc = rng.normal(P, sP, 100_000) - rng.normal(Qm, sQ, 100_000) - rng.normal(dS, sS, 100_000)
print(f"E = {E:.0f} ± {sE:.0f} mm ({100 * sE / E:.0f}%); Monte Carlo: {E_mc.mean():.0f} ± {E_mc.std():.0f} mm")
```

## 1.9 Common mistakes

- **Equating mean residence time with typical transit time.** In a well-mixed store, most water leaves well before $T_r$; in a heavy-tailed store, much earlier still, while some stays far longer.
- **Averaging parameters of nonlinear processes.** The flux computed with an average parameter is not the average flux (Equation 1.2).
- **Treating a calibrated effective parameter as a material property.** It also depends on the forcing and the scale.
- **Forgetting the groundwater term.** In small catchments, or in karst, $L$ in Equation (1.9) can be as large as streamflow.
- **Quoting relative errors of residuals from those of the inputs.** Errors add in absolute terms; a residual's relative error is larger.

## 1.10 Problems

```{code-cell} python
:tags: [remove-cell]
glue("p1_T", 1.2e12 / (4.1e3 * 365.25 * 86400), display=False)
Kf = Ks_median * np.exp(0.5 * z)
glue("p2_q", float(np.maximum(15.0 - Kf, 0).mean()), display=False); glue("p2_m", float(max(15.0 - Kf.mean(), 0)), display=False)
glue("p3_D", 0.5 * 1000 / 10, display=False)
glue("p4_PE", (80 - 65) * 1000e3 / (800e3 * 1000e3) * 86400 * 30, display=False)
sE5 = np.sqrt((0.08 * 1100) ** 2 + (0.03 * 450) ** 2 + 30.0 ** 2)
glue("p5_E", 1100 - 450 - 0.0, display=False); glue("p5_s", float(sE5), display=False)
glue("p5_r", float(sE5 / 650 * 100), display=False)
```

1. A groundwater body holds 1.2 × 10¹² m³ and is recharged at 4.1 × 10³ m³ s⁻¹. What is its mean residence time? Why might the water pumped from a well in it be much older or much younger than this?
2. Repeat the calculation of Figure 1.2 for a standard deviation of ln $K_s$ of 0.5 and rainfall of 15 mm h⁻¹: find the areal mean runoff and the runoff computed with the mean conductivity.
3. For a pulse of tracer in a stream with a velocity of 0.5 m s⁻¹ observed 1 km downstream, what diffusivity gives a Péclet number of 10?
4. A region measures 800 km by 1000 km. Across one of its 1000 km sides water vapor enters at 80 kg m⁻¹ s⁻¹, and across the opposite side it leaves at 65 kg m⁻¹ s⁻¹; there is no net flux across the 800 km sides. Neglecting storage change, estimate $P - E$ in mm over 30 days.
5. A catchment balance has $P$ = 1100 mm ± 8%, $Q$ = 450 mm ± 3%, $\Delta S$ = 0 ± 30 mm. Estimate $E$ and its relative uncertainty. Which measurement would you improve first?

```{admonition} Answers
:class: dropdown
1. About {glue:text}`p1_T:.0f` years. Pumped water is a mixture: the transit-time distribution of an aquifer is broad, and a shallow well draws young water while a deep one draws old water.
2. Areal mean {glue:text}`p2_q:.2f` mm h⁻¹; with the mean $K_s$, {glue:text}`p2_m:.2f` mm h⁻¹.
3. $D$ = {glue:text}`p3_D:.0f` m² s⁻¹.
4. About {glue:text}`p4_PE:.1f` mm.
5. $E$ = {glue:text}`p5_E:.0f` ± {glue:text}`p5_s:.0f` mm ({glue:text}`p5_r:.0f`%). Precipitation, whose absolute error is the largest.
```

## 1.11 Open questions

- **Closing the water budget.** Global and continental budgets from independent data sets still disagree by around 10%. How much of the gap is measurement error, and how much is real storage change?
- **Laws at the catchment scale.** Are there relations that hold at the scale of whole catchments, independent of the details inside them? Chapters 8 and 15 explore two candidates: the Budyko curve and the storage–discharge relation.
- **Ages of water.** Transit-time distributions vary with wetness and season. How should they be represented in models that predict both water quantity and water quality?

## 1.12 Key papers

- Beven, K. (1989). Changing ideas in hydrology: the case of physically-based models. *Journal of Hydrology*, 105, 157–172.
- Blöschl, G., and Sivapalan, M. (1995). Scale issues in hydrological modelling: a review. *Hydrological Processes*, 9, 251–290.
- Dooge, J. C. I. (1986). Looking for hydrologic laws. *Water Resources Research*, 22(9S), 46S–58S.
- Klemeš, V. (1983). Conceptualization and scale in hydrology. *Journal of Hydrology*, 65, 1–23.
- Oki, T., and Kanae, S. (2006). Global hydrological cycles and world water resources. *Science*, 313, 1068–1072.
- Rodell, M., et al. (2015). The observed state of the water cycle in the early twenty-first century. *Journal of Climate*, 28, 8289–8318.

## 1.13 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The introductory chapter.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on basic concepts.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
