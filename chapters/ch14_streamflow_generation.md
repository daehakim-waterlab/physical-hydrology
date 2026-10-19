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

# 14. Streamflow generation

```{admonition} Core ideas
:class: tip
1. Rain becomes storm runoff by several routes: **infiltration excess**, **saturation excess**, and **subsurface stormflow** through soils, macropores and shallow bedrock. Which dominates depends on rain intensity, soil conductivity, storage and topography.
2. In humid, vegetated catchments, storm runoff usually comes from a small, changing fraction of the catchment, the **variable source area**, concentrated near streams and in hollows.
3. Tracers show that most storm runoff in such catchments is **old water**, stored before the event. The stream responds within minutes, but the water it carries is months or years old: pressure waves travel much faster than water particles.
4. Hillslopes often respond with **thresholds**: little runoff until storage is filled, then a lot.
5. **TOPMODEL's** topographic index $\ln(a/\tan\beta)$ maps where saturation is likely, under stated assumptions about the saturated zone.
6. Models of runoff generation should be chosen to match the dominant processes, and simple distribution functions often capture catchment behavior better than detailed point physics.
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

## 14.1 Runoff mechanisms

The question "where does storm runoff come from?" has had different answers over the past century, and each answer still holds somewhere.

**Infiltration-excess overland flow.** Horton's view (Chapter 12): when the rain rate exceeds the soil's infiltrability, the excess runs over the surface. It dominates where infiltrability is low and rain intense: semiarid lands, crusted or compacted soils, urban areas, burned slopes. Because it depends on point-scale intensity, it can occur over much of a hillslope at once, but run-on and patchy conductivity limit how much of it reaches a stream (Chapter 12).

**Saturation-excess overland flow.** In humid, vegetated catchments, measured infiltrabilities usually exceed rain rates, yet streams still respond quickly. Field studies in the 1960s and 1970s (Hewlett and Hibbert, 1967; Dunne and Black, 1970) showed that storm runoff came from areas where the soil was saturated from below: valley bottoms, hollows and concave slopes where subsurface water converges and the water table is shallow. Rain falling on these areas cannot infiltrate, and subsurface water may also **return** to the surface there. These areas expand during storms and wet seasons and shrink in dry periods: the **variable source area** concept.

**Subsurface stormflow.** Water can also move quickly downslope below the surface: through permeable soil above a less permeable layer, along **macropores** and soil pipes, and through fractured bedrock. It reaches the stream directly or emerges as return flow near it.

**Groundwater ridging and displacement.** Near streams, where the water table is close to the surface, a little infiltration can raise it quickly, since the capillary fringe holds the soil nearly saturated. This steepens the gradient toward the stream and pushes stored water out.

Figure 14.1 illustrates the first two mechanisms with simple rules: for a storm of given depth, infiltration excess occurs when the rain rate exceeds what Green–Ampt infiltration (Chapter 12) can absorb, and saturation excess when the infiltrated water exceeds the storage available above an impeding layer. The plot is a cartoon, but it captures a robust pattern: intense rain on slowly permeable soils produces infiltration excess; long, moderate rain on shallow or wet soils produces saturation excess.

```{code-cell} python
:tags: [hide-input]
from scipy.optimize import brentq
Ks, M, P = 10.0, 30.0, 60.0                                       # mm/h, mm (psi_f * dtheta), storm depth (mm)
def ga_I(t):                                                        # Green-Ampt cumulative infiltration under ponding
    return brentq(lambda I: I - M * np.log1p(I / M) - Ks * t, 1e-9, 1e4) if t > 0 else 0.0
def infiltrated(r):
    """Depth infiltrated during a storm of depth P at constant rate r (time-compression approximation)."""
    T = P / r
    if r <= Ks: return P, False
    Ip = M / (r / Ks - 1)
    if Ip >= P: return P, False
    tp = Ip / r; tps = (Ip - M * np.log1p(Ip / M)) / Ks
    return ga_I(T - tp + tps), True
rr = np.logspace(0, 2.3, 160); Sd = np.linspace(5, 120, 150)
cls = np.zeros((len(Sd), len(rr)))
for j, r in enumerate(rr):
    F, ie = infiltrated(r)
    for i, s in enumerate(Sd):
        cls[i, j] = (1 if ie else 0) + (2 if F > s else 0)
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.pcolormesh(rr, Sd, cls, cmap=ListedColormap([bg, "#f7c9b2", "#b9d3f2", "#a6dcc6"]), vmin=-0.5, vmax=3.5, shading="auto")
ax.set_xscale("log"); ax.grid(False)
for txt, x, y in [("No storm runoff", 2, 95), ("Infiltration excess", 60, 95), ("Saturation excess", 2, 20), ("Both", 60, 20)]:
    ax.text(x, y, txt, color=ink, fontsize=10, ha="center")
ax.set_xlabel("Rain rate (mm h⁻¹)"); ax.set_ylabel("Storage deficit (mm)")
plt.show()
```

*Figure 14.1. Runoff mechanism for a 60 mm storm of constant intensity on a soil with $K_s$ = 10 mm h⁻¹ and a Green–Ampt parameter $M = \psi_f\Delta\theta$ = 30 mm, as a function of rain rate and of the storage deficit above an impeding layer. Saturation excess (blue and green) requires the infiltrated water to fill the deficit; infiltration excess (orange and green) requires intense rain. The rules are deliberately simple.*

## 14.2 Riparian zones, headwaters and thresholds

Most of a river network's length is in its headwaters: first- and second-order streams typically make up well over half of it. Storm runoff is therefore generated mainly on hillslopes draining to small channels, through a **riparian zone** that is often wet, organic-rich and close to saturation. The riparian zone buffers hillslope inputs. Small storms produce runoff mainly from the riparian zone and the channel itself. In larger storms the hillslopes become **connected** to the stream by saturated pathways, and their contribution can rise abruptly.

Many hillslope studies have found such **threshold** behavior: subsurface stormflow is negligible until storm rainfall exceeds a threshold, then increases nearly linearly with it. One explanation is **fill and spill** (Tromp-van Meerveld and McDonnell, 2006): water perched on bedrock first fills depressions in the bedrock surface, which must be full before they connect and spill downslope. The threshold depends on antecedent wetness, so the same storm can produce very different runoff in different seasons (Figure 14.2).

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(14)
nev = 160
Pev = rng.gamma(2.0, 25.0, nev)                                     # event rainfall (mm)
wet = rng.uniform(0, 1, nev)                                        # antecedent wetness index
thr = 90 - 50 * wet                                                 # threshold falls as the hillslope wets up
Rev = np.maximum(Pev - thr, 0) * 0.7 + 0.02 * Pev * (1 + 0.3 * rng.standard_normal(nev))
fig, ax = plt.subplots(figsize=(6.4, 3.9))
sc = ax.scatter(Pev, np.maximum(Rev, 0), c=wet, cmap="cividis", s=22, edgecolor="none")
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("Antecedent wetness (0–1)")
ax.set_xlabel("Event rainfall (mm)"); ax.set_ylabel("Subsurface stormflow (mm)")
plt.show()
```

*Figure 14.2. Synthetic illustration of threshold behavior in subsurface stormflow. Below a rainfall threshold that decreases with antecedent wetness, a hillslope yields little flow; above it, flow increases steeply. Real data look similar but noisier.*

## 14.3 Old and new water

### Two-component separation

Chemical and isotopic tracers can separate a storm hydrograph by the **age** of its water rather than by its path. If streamflow during an event is a mixture of **event water** (this storm's rain) and **pre-event water** (stored in the catchment before it), and each component has a distinct, constant tracer concentration, conservation of water and tracer gives

\begin{equation*}
Q = Q_n + Q_o, \qquad Q\,C = Q_n C_n + Q_o C_o
\quad\Longrightarrow\quad
\frac{Q_n}{Q} = \frac{C - C_o}{C_n - C_o} ,
\tag{14.1}
\end{equation*}

where $C$, $C_n$ and $C_o$ are the concentrations in the stream, the event water and the pre-event water. Stable isotopes of water (¹⁸O and ²H, expressed as δ values) are ideal tracers because they are part of the water molecule and rain often differs isotopically from stored water.

The separation is only as good as its assumptions: two components, constant and distinct end-members, and no other sources. The uncertainty can be estimated by propagating errors in the three concentrations through Equation (14.1). It grows sharply when the end-members are close (Genereux, 1998; Figure 14.3).

```{code-cell} python
:tags: [hide-input]
th = np.linspace(0, 48, 289)                                        # hours
Qo = 0.4 + 1.6 * (th / 10) ** 2 * np.exp(-(th - 10) / 6) * (th > 0) / (1 + (th / 10) ** 2) ** 0.8
Qn = 1.0 * np.exp(-0.5 * ((th - 8) / 2.5) ** 2)
Qs = Qo + Qn
Cn, Co = -11.0, -7.5                                                # per mil
Cs = (Qn * Cn + Qo * Co) / Qs + 0.05 * np.random.default_rng(3).standard_normal(len(th))
fn = (Cs - Co) / (Cn - Co)
def f_err(sC, sCo, sCn):                                            # first-order error propagation
    d = Cn - Co
    return np.sqrt((sC / d) ** 2 + (sCo * (Cs - Cn) / d ** 2) ** 2 + (sCn * (Cs - Co) / d ** 2) ** 2)
err = f_err(0.1, 0.2, 0.8)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, height_ratios=[1.4, 1])
a1.plot(th, Qs, color=ink, lw=2.2, label="Streamflow")
a1.fill_between(th, 0, np.clip(fn, 0, 1) * Qs, color=c1, alpha=0.35, lw=0, label="Event water (separated)")
a1.fill_between(th, np.clip(fn - err, 0, 1) * Qs, np.clip(fn + err, 0, 1) * Qs, color=c1, alpha=0.18, lw=0, label="± 1 s.d.")
a1.set_ylabel("Q (mm h⁻¹)"); a1.legend(loc="upper right", fontsize=9)
a2.plot(th, Cs, color=c2, lw=1.8, label="Stream")
a2.axhline(Co, color=ink2, ls="--", lw=1); a2.text(47, Co + 0.15, "pre-event water", ha="right", color=ink2, fontsize=9.5)
a2.axhline(Cn, color=c1, ls="--", lw=1); a2.text(47, Cn + 0.15, "event rain", ha="right", color=c1, fontsize=9.5)
a2.set_ylabel("δ¹⁸O (‰)"); a2.set_xlabel("Time since start of rain (h)"); a2.set_ylim(-11.6, -7)
plt.show()
ipk = np.argmax(Qs)
glue("fn_peak", float(fn[ipk] * 100), display=False); glue("fn_err", float(err[ipk] * 100), display=False)
glue("fn_total", float(100 * np.trapezoid(np.clip(fn, 0, 1) * Qs, th) / np.trapezoid(Qs - 0.4, th)), display=False)
```

*Figure 14.3. Two-component separation of a synthetic storm hydrograph with δ¹⁸O. Event rain is assumed at −11.0 ± 0.8‰ (rainfall varies during storms) and pre-event water at −7.5 ± 0.2‰. At the peak, event water makes up {glue:text}`fn_peak:.0f` ± {glue:text}`fn_err:.0f`% of streamflow; over the event it contributes about {glue:text}`fn_total:.0f`% of the flow above the pre-storm baseflow.*

### The old-water paradox

Isotope studies since the late 1970s (for example Sklash and Farvolden, 1979; Pearce et al., 1986) found that in humid catchments pre-event water typically makes up most of the storm hydrograph, often more than half even at the peak. Streams respond within minutes to rain, yet the water they deliver is old. Kirchner (2003) called this a **double paradox**: catchments store water for long periods but release it promptly, and the chemistry of that water varies with flow in ways that simple mixing cannot explain.

Part of the answer is the difference between the **velocity** of water particles and the **celerity** of the pressure or storage wave that drives the flow (McDonnell and Beven, 2014). In a nearly saturated soil, a small amount of rain raises the water table quickly, because the drainable porosity there is small. The resulting increase in hydraulic gradient pushes stored water out at the stream almost immediately, while the new water itself moves only slowly into the soil. For kinematic subsurface flow on a slope, the wave travels at about $K\sin\vartheta/\phi_e$ and the water at $K\sin\vartheta/n$, where $n$ is the total porosity. Their ratio $n/\phi_e$ can be ten or more near saturation (Example 14.3).

## 14.4 Variable source areas and topographic indices

### TOPMODEL

Where saturation excess dominates, the question is where the soil saturates first. Beven and Kirkby (1979) gave an answer, in a model now called **TOPMODEL**, from three assumptions about the saturated zone:

1. it is in quasi-steady state with a spatially uniform recharge rate $r$, so the downslope flow per unit contour length at a point is $r a$, where $a$ is the upslope area draining through that contour length;
2. the hydraulic gradient equals the local surface slope $\tan\beta$;
3. transmissivity declines exponentially with the local storage deficit $D$, $T = T_0\,e^{-D/m}$.

Equating the flow from (1) and Darcy's law with (2) and (3), $ra = T_0\,e^{-D/m}\tan\beta$, so

\begin{equation*}
D = -m\ln\left(\frac{r\,a}{T_0\tan\beta}\right).
\tag{14.2}
\end{equation*}

Averaging over the catchment eliminates the unknown recharge and gives the local deficit in terms of the mean deficit $\bar D$:

\begin{equation*}
D_i = \bar D + m\left[\lambda - \ln\left(\frac{a}{\tan\beta}\right)_i\right],
\qquad \lambda = \overline{\ln(a/\tan\beta)} .
\tag{14.3}
\end{equation*}

Points with the same **topographic index** $\ln(a/\tan\beta)$ behave alike, and a point is saturated ($D_i \le 0$) when its index exceeds $\lambda + \bar D/m$. As the catchment wets up and $\bar D$ falls, saturation spreads from the cells with the highest index: valley bottoms with large contributing areas and gentle slopes. The model needs only the distribution of the index, computed from a digital elevation model, and a water balance for $\bar D$.

```{code-cell} python
:tags: [hide-input]
import heapq
from scipy.ndimage import gaussian_filter
ny, nx_, dxg = 140, 120, 10.0                                       # grid (m)
yy, xx = np.mgrid[0:ny, 0:nx_] * dxg
gen = np.random.default_rng(41)
xc = nx_ * dxg / 2 + 80 * np.sin(yy / 350)                          # winding valley axis
z = 0.04 * yy + 60 * (1 - np.exp(-((xx - xc) / 250) ** 2)) + 60 * gaussian_filter(gen.standard_normal((ny, nx_)), 6)
nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
filled = np.full(z.shape, np.inf); seen = np.zeros(z.shape, bool); pq = []
for j in range(nx_):                                                 # fill pits, draining to the lower edge
    heapq.heappush(pq, (z[0, j], 0, j)); seen[0, j] = True
while pq:
    e, i, j = heapq.heappop(pq); filled[i, j] = e
    for di, dj in nbr:
        ii, jj = i + di, j + dj
        if 0 <= ii < ny and 0 <= jj < nx_ and not seen[ii, jj]:
            seen[ii, jj] = True; heapq.heappush(pq, (max(z[ii, jj], e + 1e-3), ii, jj))
z = filled
area = np.full(z.shape, dxg * dxg); slope = np.full(z.shape, 1e-3)
for idx in np.argsort(z, axis=None)[::-1]:                           # D8 flow accumulation, highest cell first
    i, j = divmod(idx, nx_); best, bi = 0.0, None
    for di, dj in nbr:
        ii, jj = i + di, j + dj
        if 0 <= ii < ny and 0 <= jj < nx_:
            s = (z[i, j] - z[ii, jj]) / (dxg * np.hypot(di, dj))
            if s > best: best, bi = s, (ii, jj)
    if bi is not None:
        area[bi] += area[i, j]; slope[i, j] = best
TI = np.log(area / dxg / np.maximum(slope, 1e-3))
lam = TI.mean(); m_ = 0.03                                           # m (scale of transmissivity decline)
Dbar = np.linspace(0, 0.15, 60)
fsat = [(TI >= lam + d / m_).mean() for d in Dbar]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 4.4), width_ratios=[1.3, 1])
im = a1.imshow(TI, origin="lower", cmap="Blues", extent=[0, nx_ * dxg / 1000, 0, ny * dxg / 1000])
sat = np.ma.masked_where(TI < lam + 0.05 / m_, np.ones_like(TI))
from matplotlib.colors import ListedColormap
a1.imshow(sat, origin="lower", cmap=ListedColormap([c2]), extent=[0, nx_ * dxg / 1000, 0, ny * dxg / 1000])
a1.set_title("(a) ln(a/tan β)"); a1.set_xlabel("km"); a1.set_ylabel("km"); a1.grid(False)
fig.colorbar(im, ax=a1, shrink=0.7, pad=0.02, location="bottom")
a2.plot(Dbar * 1000, np.array(fsat) * 100, color=c1, lw=2.4)
a2.axvline(50, color=c2, lw=1, ls="--")
a2.set_xlabel("Mean deficit $\\bar D$ (mm)"); a2.set_ylabel("Saturated area (%)"); a2.set_title("(b) Source area")
plt.show()
glue("fsat50", float(100 * (TI >= lam + 0.05 / m_).mean()), display=False)
glue("fsat0", float(100 * (TI >= lam).mean()), display=False)
```

*Figure 14.4. TOPMODEL applied to a synthetic valley, 1.2 km by 1.4 km on a 10 m grid, draining to its lower edge. (a) The topographic index from single-direction (D8) flow accumulation after filling pits; orange cells are predicted to be saturated when the mean deficit is 50 mm with $m$ = 30 mm, {glue:text}`fsat50:.1f`% of the catchment. The straight diagonal lines are artifacts of single-direction routing, one reason multiple-direction algorithms are often preferred. (b) Saturated fraction as a function of the mean deficit; at $\bar D$ = 0 it is {glue:text}`fsat0:.0f`%.*

TOPMODEL's assumptions are strong. The quasi-steady state is reached only in wet periods; surface slope is a poor guide to the hydraulic gradient in flat terrain or where deeper groundwater matters; and the index depends on the flow-routing algorithm and grid size. Yet the idea that topography organizes wetness has been confirmed in many humid catchments, and topographic indices are widely used to map wetlands, soil moisture and source areas.

### Distribution-function models

A related family of models replaces explicit topography with a statistical distribution of storage capacity across the catchment. In the formulation used by the Xinanjiang model (Zhao, 1992) and the Variable Infiltration Capacity land-surface scheme (Wood et al., 1992), the fraction $A_s$ of the area whose storage is full, when the mean storage is $W$ out of a maximum $W_m$, is

\begin{equation*}
A_s = 1 - \left(1 - \frac{W}{W_m}\right)^{\beta_s/(1 + \beta_s)},
\tag{14.4}
\end{equation*}

where $\beta_s$ is a shape parameter. Rain falling on the saturated fraction runs off. Moore (1985) developed the general probability-distributed principle behind such models. They capture the growth of contributing area with wetness with two or three parameters, which is why they are among the most widely used runoff-generation schemes in operational hydrology and climate models.

## 14.5 Choosing a parameterization

No single mechanism or model is right everywhere. A practical approach is to start with a **perceptual model**, a qualitative description of the dominant processes based on field knowledge, and then choose a representation that captures them with as few parameters as the data can support.

- Arid and semiarid catchments, urban areas, burned or crusted soils: infiltration excess, with Green–Ampt or similar losses and kinematic overland flow (Chapters 10 and 12).
- Humid catchments with shallow soils and moderate slopes: saturation excess and subsurface stormflow, with TOPMODEL-type or distribution-function models.
- Steep, forested catchments with permeable soils: subsurface stormflow with thresholds and old-water dominance.
- Large basins: a mixture, for which distribution-function models and calibrated conceptual models are the usual compromise (Chapter 16).

Tracers, soil moisture and water-table observations, and field mapping of saturated areas test whether a model gets the right answer for the right reasons.

## 14.6 Lab (planned)

A Colab notebook will compute flow directions, contributing areas and the topographic index from an openly available digital elevation model, compare single- and multiple-direction flow algorithms, and map saturated areas for a range of mean deficits.

## 14.7 Worked examples

**Example 14.1 — Hydrograph separation.** Before a storm, stream water has δ¹⁸O = −7.8‰. The rain averages −12.4‰, and at the peak the stream reaches −9.6‰ with a discharge of 3.2 m³ s⁻¹. How much of the peak is event water?

```{code-cell} python
Co_, Cn_, Cs_, Qpk = -7.8, -12.4, -9.6, 3.2
f = (Cs_ - Co_) / (Cn_ - Co_)
print(f"event-water fraction = {f:.2f}; event water = {f * Qpk:.2f} m3/s, pre-event = {(1 - f) * Qpk:.2f} m3/s")
```

**Example 14.2 — Saturation by topographic index.** In a catchment with mean index $\lambda$ = 6.2 and $m$ = 25 mm, which cells are saturated when the mean deficit is 40 mm? What mean deficit saturates a cell with index 9.0?

```{code-cell} python
lam_, m2 = 6.2, 25.0
print(f"saturated where ln(a/tanβ) ≥ {lam_ + 40 / m2:.1f}")
print(f"cell with index 9.0 saturates when D̄ ≤ {m2 * (9.0 - lam_):.0f} mm")
```

**Example 14.3 — Celerity and velocity.** A soil on a 20° slope has $K$ = 2 m d⁻¹, porosity 0.45 and drainable porosity 0.04 near saturation. Compare the speed of the subsurface storage wave with the speed of the water.

```{code-cell} python
K3, sl, n3, pe = 2.0, np.radians(20), 0.45, 0.04
c = K3 * np.sin(sl) / pe; v = K3 * np.sin(sl) / n3
print(f"celerity = {c:.1f} m/d, velocity = {v:.2f} m/d, ratio = {c / v:.0f}")
```

A 100 m hillslope responds within a few days, while water entering at its top takes months to reach the stream.

## 14.8 Common mistakes

- **Assuming infiltration excess in humid, vegetated catchments** because rain rates exceed laboratory-measured $K_s$ of the matrix. Check field infiltrability and saturation patterns.
- **Equating fast response with new water.** Tracers usually show the opposite.
- **Ignoring end-member uncertainty** in hydrograph separation; it can exceed the result when end-members are close.
- **Treating the topographic index as model-independent.** It depends on grid size and flow-routing method.
- **Calibrating a model whose mechanism is wrong.** It may fit discharge but fail for water quality, extremes or change.

## 14.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("r1", float((-60 - -48) / (-85 - -48)), display=False)
glue("r2", float(5.8 + 60 / 40), display=False)
glue("r3", float(1 - (1 - 0.6) ** (0.4 / 1.4)), display=False)
glue("r4", float(0.5 / 0.05), display=False)
glue("r5", float(np.sqrt((0.1 / 2.0) ** 2 + (0.5 * 0.5 / 2.0) ** 2 + (0.5 * 0.5 / 2.0) ** 2)), display=False)
```

1. Pre-event water has δ²H = −48‰, rain −85‰, and the stream −60‰. What fraction of streamflow is event water?
2. With $\lambda$ = 5.8 and $m$ = 40 mm, what index threshold marks saturation at a mean deficit of 60 mm?
3. For Equation (14.4) with $\beta_s$ = 0.4, what fraction of the area is saturated when the storage is 60% of its maximum?
4. What is the ratio of celerity to velocity for porosity 0.5 and drainable porosity 0.05?
5. The end-members of a separation differ by 2‰ ($C_n - C_o$ = −2‰) and the stream lies halfway between them. If the stream, pre-event and event concentrations have uncertainties of 0.1, 0.5 and 0.5‰, what is the uncertainty of the event-water fraction?

```{admonition} Answers
:class: dropdown
1. {glue:text}`r1:.2f`.
2. $\ln(a/\tan\beta)$ ≥ {glue:text}`r2:.1f`.
3. {glue:text}`r3:.2f`.
4. {glue:text}`r4:.0f`.
5. About {glue:text}`r5:.2f`, comparable to half of the fraction itself.
```

## 14.10 Open questions

- **Connectivity.** How do hillslopes become connected to streams, and how can connectivity be measured and modeled across scales?
- **Water age.** How do the transit-time distributions of streamflow change with wetness, and what do they imply for the transport of solutes?
- **Transferability.** Can runoff mechanisms be predicted from landscape properties in ungauged catchments, rather than inferred after calibration?

## 14.11 Key papers

- Beven, K. J., and Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24, 43–69.
- Dunne, T., and Black, R. D. (1970). Partial area contributions to storm runoff in a small New England watershed. *Water Resources Research*, 6, 1296–1311.
- Genereux, D. (1998). Quantifying uncertainty in tracer-based hydrograph separations. *Water Resources Research*, 34, 915–919.
- Hewlett, J. D., and Hibbert, A. R. (1967). Factors affecting the response of small watersheds to precipitation in humid areas. In *Forest Hydrology*, Pergamon, 275–290.
- Kirchner, J. W. (2003). A double paradox in catchment hydrology and geochemistry. *Hydrological Processes*, 17, 871–874.
- McDonnell, J. J., and Beven, K. (2014). Debates—The future of hydrological sciences: A (common) path forward? A call to action aimed at understanding velocities, celerities and residence time distributions of the headwater hydrograph. *Water Resources Research*, 50, 5342–5350.
- Moore, R. J. (1985). The probability-distributed principle and runoff production at point and basin scales. *Hydrological Sciences Journal*, 30, 273–297.
- Pearce, A. J., Stewart, M. K., and Sklash, M. G. (1986). Storm runoff generation in humid headwater catchments: 1. Where does the water come from? *Water Resources Research*, 22, 1263–1272.
- Sklash, M. G., and Farvolden, R. N. (1979). The role of groundwater in storm runoff. *Journal of Hydrology*, 43, 45–65.
- Tromp-van Meerveld, H. J., and McDonnell, J. J. (2006). Threshold relations in subsurface stormflow: 2. The fill and spill hypothesis. *Water Resources Research*, 42, W02411.
- Wood, E. F., Lettenmaier, D. P., and Zartarian, V. G. (1992). A land-surface hydrology parameterization with subgrid variability for general circulation models. *Journal of Geophysical Research*, 97(D3), 2717–2728.
- Zhao, R.-J. (1992). The Xinanjiang model applied in China. *Journal of Hydrology*, 135, 371–381.

## 14.12 Further reading

- Beven, K. J. (2012). *Rainfall–Runoff Modelling: The Primer* (2nd ed.). Wiley-Blackwell.
- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The chapter on streamflow generation.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on streamflow generation.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
