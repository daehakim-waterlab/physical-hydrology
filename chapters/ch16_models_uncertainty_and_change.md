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

# 16. Models, uncertainty and change

```{admonition} Core ideas
:class: tip
1. Hydrological models range from data-driven to conceptual to physically based. The right choice depends on the purpose, the data and the scale, not on the amount of physics in the equations.
2. Calibration fits parameters to observations. Many different parameter sets usually fit almost equally well (**equifinality**), and the choice of objective function (NSE, KGE) shapes the result.
3. A model is tested properly only under conditions different from those it was calibrated for. **Differential split-sample tests** often reveal errors in model structure that calibration hides.
4. For climate change, **bottom-up stress tests** map how a system responds over a range of plausible climate changes before asking which changes the climate models project (**scenario-neutral** or decision-scaling assessment).
5. Detecting a change in a hydrological record requires accounting for natural variability and autocorrelation; **attributing** it to climate or land use requires a model of what would have happened otherwise.
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

## 16.1 Purposes and types of hydrological models

A model is a simplified representation of a system, built for a purpose. Hydrological models are used to forecast floods and droughts, to design structures and operate reservoirs, to estimate flows in ungauged catchments, to assess the effects of land-use and climate change, and to test understanding. These purposes demand different things: a flood forecast must be accurate for the next few days at one gauge; a climate-change assessment must remain credible under conditions never observed.

Models are often classified by how they represent processes:

- **Data-driven models** learn the relation between inputs and outputs from records: regressions, transfer functions (Chapter 15) and, increasingly, machine-learning models such as long short-term memory networks, which have proved remarkably accurate for streamflow prediction when trained on many catchments (Kratzert et al., 2019).
- **Conceptual models** represent the catchment as a set of interconnected stores (soil, groundwater, routing), with parameters that are "effective" catchment values, calibrated rather than measured. Examples are the many bucket-type models used operationally around the world.
- **Physically based distributed models** solve the equations of the previous chapters (Richards, Boussinesq, shallow water) on a grid, following the blueprint of Freeze and Harlan (1969). In principle their parameters are measurable; in practice they are measured at scales much smaller than the grid and must also be adjusted (Chapter 11).
- **Land-surface models** in weather and climate models couple the water and energy balances with the atmosphere and carry their own runoff parameterizations (Chapter 14).

These categories overlap, and none is superior in general. Physically based models describe internal states but have many uncertain parameters; conceptual models are parsimonious but their parameters have no direct field meaning; data-driven models can be the most accurate where data are plentiful but may be least reliable outside the conditions they were trained on.

## 16.2 Calibration, equifinality and evaluation

### Objective functions

Calibration adjusts parameters until simulations match observations according to an objective function. The most widely used is the **Nash–Sutcliffe efficiency** (Nash and Sutcliffe, 1970),

\begin{equation*}
\text{NSE} = 1 - \frac{\sum_t (Q_{s,t} - Q_{o,t})^2}{\sum_t (Q_{o,t} - \bar Q_o)^2},
\tag{16.1}
\end{equation*}

which compares the model's squared error with the variance of the observations: NSE = 1 is a perfect fit and NSE = 0 means the model is no better than the observed mean. Gupta et al. (2009) showed that NSE can be decomposed into three terms: the correlation $r$ between simulated and observed flows, the ratio of their standard deviations $\alpha$, and the bias normalized by the observed standard deviation. Maximizing NSE tends to underestimate variability ($\alpha < 1$) and does not penalize bias strongly when flows are variable. The **Kling–Gupta efficiency** treats the three components explicitly,

\begin{equation*}
\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}, \qquad \beta = \bar Q_s / \bar Q_o .
\tag{16.2}
\end{equation*}

Both are dominated by high flows when computed on discharge; computing them on logarithms or on flow-duration curves emphasizes low flows. The objective function must match the purpose.

### Equifinality

Hydrological models usually have more parameters than the data can constrain, and their effects compensate: a larger soil store can be offset by a faster runoff response, for example. As a result, many parameter sets, sometimes widely different, give nearly the same fit. Beven called this **equifinality** and argued that it should be accepted rather than resolved by picking one "optimum" (Beven, 2006). The Generalized Likelihood Uncertainty Estimation (GLUE) method samples parameter sets, keeps the "behavioral" ones that fit acceptably, and uses all of them for prediction (Beven and Binley, 1992).

The experiment below uses a three-parameter conceptual model: a soil store of capacity $S_{\max}$ that releases a fraction $(S/S_{\max})^{\beta}$ of each day's rain as runoff, loses water by evaporation, and passes runoff through a linear reservoir with timescale $k$. It is calibrated against 20 years of synthetic daily streamflow from a "true" catchment with the same structure except in one respect: the truth evaporates at the potential rate until its storage falls below 60% of capacity, while the model's evaporation declines linearly with storage. Every real model is structurally wrong in some way; here we know exactly how. The first ten years of synthetic climate are wet and the last ten are dry, with 27% less precipitation on average.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(16)
ny = 20; nd = 365 * ny; doy = np.arange(nd) % 365; yr = np.arange(nd) // 365
wet = np.zeros(nd, bool)
for t in range(1, nd):                                              # Markov chain of wet and dry days
    wet[t] = rng.random() < (0.65 if wet[t - 1] else 0.25)
P = np.where(wet, rng.exponential(9.0, nd), 0) * np.where(yr < 10, 1.1, 0.8) * (1 + 0.4 * np.cos(2 * np.pi * (doy - 200) / 365))
Ep = 3.0 - 2.2 * np.cos(2 * np.pi * (doy - 15) / 365)             # mm/d, seasonal
def model(P, Ep, Smax, beta, k, lp):
    """Bucket model vectorized over parameter sets (and over forcings if P, Ep are 2-D)."""
    Smax, beta, k, lp = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, float)) for v in (Smax, beta, k, lp)])
    n = Smax.size if np.ndim(P) == 1 else max(Smax.size, P.shape[1])
    S = np.broadcast_to(0.5 * Smax, (n,)).copy(); G = np.zeros(n); Q = np.zeros((len(P), n))
    for t in range(len(P)):
        R = P[t] * np.minimum(S / Smax, 1) ** beta; S = S + P[t] - R
        E = np.minimum(Ep[t] * np.minimum(1, S / (lp * Smax)), S); S = S - E
        over = np.maximum(S - Smax, 0); R = R + over; S = S - over
        G = G + R; q = G / k; G = G - q; Q[t] = q
    return Q
Q_true = model(P, Ep, 150, 2.5, 8, 0.6)[:, 0]
Q_obs = Q_true * np.exp(0.1 * rng.standard_normal(nd))            # 10% observation noise
ns = 4000
Sm, be, kk = rng.uniform(50, 600, ns), rng.uniform(0.5, 6, ns), rng.uniform(1, 30, ns)
Qsim = model(P, Ep, Sm, be, kk, 1.0)
cal, val = (yr >= 1) & (yr < 10), yr >= 10                          # year 0 is warm-up
def nse(o, s): return 1 - ((s - o[:, None]) ** 2).sum(0) / ((o - o.mean()) ** 2).sum()
N_cal, N_val = nse(Q_obs[cal], Qsim[cal]), nse(Q_obs[val], Qsim[val])
beh = np.argsort(N_cal)[::-1][: ns // 20]                           # behavioral: best 5%
fig, axs = plt.subplots(1, 3, figsize=(6.4, 3.2), sharey=True)
for ax, par, lab, tv in zip(axs, [Sm, be, kk], ["$S_{max}$ (mm)", "β", "k (d)"], [150, 2.5, 8]):
    ax.plot(par, N_cal, ".", color="#c9c8c3", ms=2)
    ax.plot(par[beh], N_cal[beh], ".", color=c1, ms=3)
    ax.axvline(tv, color=c2, lw=1.2, ls="--"); ax.set_xlabel(lab)
axs[0].set_ylabel("NSE, calibration"); axs[0].set_ylim(0.3, 1.0)
plt.show()
glue("nse_best", float(N_cal[beh[0]]), display=False); glue("nse_beh", float(N_cal[beh[-1]]), display=False)
glue("sm_lo", float(Sm[beh].min()), display=False); glue("sm_hi", float(Sm[beh].max()), display=False)
```

*Figure 16.1. Dotty plots: calibration NSE of 4000 random parameter sets of the conceptual model, against each parameter. Blue points are the best 5%, with NSE between {glue:text}`nse_beh:.2f` and {glue:text}`nse_best:.2f`. Dashed lines mark the parameters of the true catchment. The behavioral sets span a wide range of soil capacities ({glue:text}`sm_lo:.0f`–{glue:text}`sm_hi:.0f` mm), and the best fits are not near the true values, because the model compensates for its structural error.*

### Testing a model

A model's fit in calibration says little about its predictive ability. Klemeš (1986) proposed a hierarchy of tests. In the **split-sample test**, the record is divided, the model is calibrated on one part and evaluated on the other. In the **differential split-sample test**, the parts are chosen to differ in climate, wet against dry, for example, so the test asks whether the model can be transferred to conditions like those it will face under change. Many conceptual models that pass ordinary split-sample tests fail differential ones, overestimating runoff in dry periods.

```{code-cell} python
:tags: [hide-input]
Qa_obs = Q_obs.reshape(ny, 365).sum(1)
Qa_sim = Qsim[:, beh].reshape(ny, 365, -1).sum(1)
lo, md, hi = np.percentile(Qa_sim, [5, 50, 95], axis=1)
years = np.arange(ny)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True, height_ratios=[1.4, 1])
a1.fill_between(years, lo, hi, color=c1, alpha=0.25, lw=0, label="Behavioral sets, 5–95%")
a1.plot(years, md, color=c1, lw=1.6, label="Behavioral median")
a1.plot(years, Qa_obs, "o", color=ink, ms=4, label="Observed")
a1.axvspan(9.5, 19.5, color="#f1efe9", zorder=0); a1.text(14.5, 950, "evaluation (dry)", ha="center", color=ink2, fontsize=9.5)
a1.text(5, 950, "calibration (wet)", ha="center", color=ink2, fontsize=9.5)
a1.set_ylabel("Annual Q (mm)"); a1.set_ylim(0, 1050); a1.legend(loc="lower left", fontsize=9)
a2.fill_between(years, 100 * (lo / Qa_obs - 1), 100 * (hi / Qa_obs - 1), color=c1, alpha=0.25, lw=0)
a2.plot(years, 100 * (md / Qa_obs - 1), color=c1, lw=1.6)
a2.axhline(0, color=ink2, lw=0.8); a2.axvspan(9.5, 19.5, color="#f1efe9", zorder=0)
a2.set_ylabel("Error (%)"); a2.set_xlabel("Year")
plt.show()
cal_y, val_y = (years >= 1) & (years < 10), years >= 10
b_cal = 100 * (md[cal_y].sum() / Qa_obs[cal_y].sum() - 1); b_val = 100 * (md[val_y].sum() / Qa_obs[val_y].sum() - 1)
w_cal = np.mean((hi - lo)[cal_y] / Qa_obs[cal_y]) * 100; w_val = np.mean((hi - lo)[val_y] / Qa_obs[val_y]) * 100
glue("b_cal", float(b_cal), display=False); glue("b_val", float(b_val), display=False)
glue("w_cal", float(w_cal), display=False); glue("w_val", float(w_val), display=False)
glue("nse_val", float(np.median(N_val[beh])), display=False)
```

*Figure 16.2. Differential split-sample test of the conceptual model. In the wet calibration decade the behavioral median overestimates annual runoff by {glue:text}`b_cal:.0f`%, and the 5–95% band spans {glue:text}`w_cal:.0f`% of the observed value. In the dry evaluation decade the bias is {glue:text}`b_val:.0f`% and the band widens to {glue:text}`w_val:.0f`%. The median daily NSE in evaluation is still {glue:text}`nse_val:.2f`: a good efficiency can hide a large error in the water balance.*

## 16.3 Model structure uncertainty

Figure 16.2 shows the two kinds of uncertainty that matter most. **Parameter uncertainty**, the spread of the behavioral sets, grows when the model is used outside its calibration conditions, because sets that compensate differently in calibration diverge once conditions change. **Structural uncertainty**, the error caused by a wrong representation of a process, appears as a bias that no parameter set removes. Here it is the evaporation function; in real catchments it may be the representation of groundwater, snow, vegetation dynamics or human water use.

Structural uncertainty is the harder of the two, because it cannot be sampled from a known distribution. Practical responses include:

- **multi-model ensembles**, comparing different model structures on the same problem;
- **flexible frameworks** that build models from interchangeable components and test alternative hypotheses about each process;
- **diagnostic evaluation**: testing models against signatures such as recession behavior (Chapter 13), the flow-duration curve, the runoff ratio and its elasticity (Chapter 15), rather than against a single efficiency;
- **multiple data**: evaluating internal states, such as soil moisture, groundwater levels, evaporation or tracer concentrations (Chapter 14), as well as discharge.

Input uncertainty, mainly in areal precipitation (Chapter 5), and uncertainty in the observations used for calibration add to these. A full uncertainty analysis treats all of them, but even a partial one, honestly reported, is better than a single deterministic prediction.

## 16.4 Stress tests and scenario-neutral assessment

The conventional, **top-down** approach to climate-change impact assessment passes the output of global climate models through downscaling and a hydrological model to the impact of interest. Each step adds uncertainty, and the result depends heavily on which climate models and scenarios are chosen, which change as new generations of models appear.

**Bottom-up** approaches reverse the order. They first ask how the system responds to a wide range of plausible changes, then where the climate projections fall within that range. In a **scenario-neutral** assessment (Prudhomme et al., 2010), the observed climate is perturbed systematically, for example by changing precipitation by −30% to +20% and temperature by 0 to +5 °C, and the hydrological model is run for each combination. The result is a **response surface** of the impact over the space of climate changes. Thresholds of concern (a flood level, a supply shortfall, a reliability target) can then be marked on it, and the projections overlaid to judge how likely they are to be crossed. The closely related **decision-scaling** approach starts from the decisions and performance thresholds of stakeholders (Brown et al., 2012). Examples include hourly flood risk (Kim et al., 2018) and the operation of a complex river system (Kim et al., 2019).

The response surface separates the hydrological question (how sensitive is the system?) from the climatological one (how much will the climate change?). It stays valid when new climate projections appear; only the overlay needs updating. Its main limitation is that simple perturbations of the observed climate keep its variability and sequencing unchanged, whereas real changes may alter storm intensity, seasonality and persistence as well as means. Stochastic weather generators can add such changes in a controlled way.

```{code-cell} python
:tags: [hide-input]
p_best = (Sm[beh[0]], be[beh[0]], kk[beh[0]])
dP = np.linspace(-0.30, 0.20, 11); dT = np.linspace(0, 5, 11)
PP, TT = np.meshgrid(dP, dT)
fP, fE = (1 + PP.ravel()), (1 + 0.03 * TT.ravel())                 # Ep assumed to rise 3% per °C
Qgrid = model(P[:, None] * fP, Ep[:, None] * fE, *p_best, 1.0)
Q_base = model(P, Ep, *p_best, 1.0)[365:].mean()
dQ = (Qgrid[365:].mean(0) / Q_base - 1).reshape(TT.shape) * 100
proj_T = rng.normal(2.6, 0.7, 30); proj_P = rng.normal(0.03, 0.07, 30)    # illustrative projections only
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cf = ax.contourf(PP * 100, TT, dQ, levels=np.arange(-80, 51, 10), cmap="RdBu", vmin=-80, vmax=80, extend="both")
cs = ax.contour(PP * 100, TT, dQ, levels=[-20], colors=ink, linewidths=2.0)
ax.clabel(cs, fmt={-20: "−20%"}, fontsize=9.5)
ax.plot(proj_P * 100, proj_T, "o", color=c2, mec=ink, mew=0.6, ms=6, label="Illustrative projections")
cb = fig.colorbar(cf, ax=ax, pad=0.02); cb.set_label("Change in mean annual runoff (%)")
ax.set_xlabel("Change in precipitation (%)"); ax.set_ylabel("Warming (°C)")
ax.legend(loc="lower right", fontsize=9); ax.grid(False)
plt.show()
from scipy.interpolate import RegularGridInterpolator
f_int = RegularGridInterpolator((dT, dP), dQ)
inside = (proj_P > dP[0]) & (proj_P < dP[-1]) & (proj_T > 0) & (proj_T < 5)
frac = float(np.mean(f_int(np.c_[proj_T[inside], proj_P[inside]]) < -20))
glue("frac_thr", 100 * frac, display=False)
```

*Figure 16.3. Scenario-neutral response surface: change in mean annual runoff simulated by the best calibrated model when the observed precipitation is scaled and potential evaporation increased by an assumed 3% per °C of warming. The black line is a threshold of concern, a 20% decrease. The orange points are illustrative projections, not output of real climate models; {glue:text}`frac_thr:.0f`% of them fall beyond the threshold.*

The surface in Figure 16.3 was computed with one parameter set of a model that, as Figure 16.2 showed, overestimates runoff in dry conditions. A credible stress test repeats the analysis with the behavioral ensemble, and ideally with several model structures, and reports how the threshold line moves.

## 16.5 Detecting the effects of land-use and climate change

### Detection

Has the flow regime of a river changed? Answering requires a statistical test against the background of natural variability. The **Mann–Kendall** test (Mann, 1945; Kendall, 1975) checks for a monotonic trend by counting how often later values exceed earlier ones, and the **Theil–Sen** slope (Sen, 1968), the median of the slopes between all pairs of points, estimates its magnitude robustly. Tests for an abrupt change, such as the Pettitt test, locate a single change point. Two cautions apply. Positive autocorrelation, common in annual flows because of storage carryover, inflates the apparent significance of trends and must be accounted for. And testing many stations at once produces some "significant" results by chance, so **field significance** must be assessed.

A simple graphical tool for a change caused by land use is the **double-mass curve**: cumulative runoff plotted against cumulative precipitation. Climate variability moves points along the line; a change in the catchment's response changes its slope.

```{code-cell} python
:tags: [hide-input]
from scipy.stats import kendalltau
rng2 = np.random.default_rng(161)
nyr = 50; years2 = np.arange(1975, 1975 + nyr)
Pann = 1300 * np.exp(0.15 * rng2.standard_normal(nyr))
Epann = 1000.0
def fu(phi, w=2.6): return 1 + phi - (1 + phi ** w) ** (1 / w)
w_series = np.where(years2 < 2005, 2.6, 2.6 + 0.8 * np.minimum((years2 - 2004) / 10, 1))   # afforestation from 2005
Qann = Pann * (1 - fu(Epann / Pann, w_series)) * np.exp(0.04 * rng2.standard_normal(nyr))
tau_, p_mk = kendalltau(years2, Qann)
slopes = [(Qann[j] - Qann[i]) / (j - i) for i in range(nyr) for j in range(i + 1, nyr)]
sen = float(np.median(slopes))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.8))
a1.plot(years2, Qann, "o-", color=c1, ms=3.5, lw=1)
a1.plot(years2, np.median(Qann) + sen * (years2 - years2.mean()), color=c2, lw=1.8, ls="--")
a1.axvline(2005, color=ink2, lw=1, ls=":")
a1.set_xlabel("Year"); a1.set_ylabel("Annual runoff (mm)"); a1.set_title("(a) Annual runoff")
cP, cQ = np.cumsum(Pann) / 1000, np.cumsum(Qann) / 1000
k05 = years2 < 2005
a2.plot(cP, cQ, "o", color=c1, ms=3)
pre = np.polyfit(cP[k05], cQ[k05], 1)
a2.plot(cP, np.polyval(pre, cP), color=ink2, lw=1.2, ls="--", label="Pre-2005 slope")
a2.set_xlabel("Cumulative P (m)"); a2.set_ylabel("Cumulative Q (m)"); a2.set_title("(b) Double-mass curve")
a2.legend(loc="upper left", fontsize=9)
plt.show()
post = np.polyfit(cP[~k05], cQ[~k05], 1)
glue("p_mk", float(p_mk), display=False); glue("sen", sen, display=False)
glue("slope_pre", float(pre[0]), display=False); glue("slope_post", float(post[0]), display=False)
```

*Figure 16.4. Synthetic 50-year record of a catchment afforested gradually from 2005, which shifts its Budyko parameter from 2.6 to 3.4. (a) The Mann–Kendall test gives p = {glue:text}`p_mk:.2f`, and the Theil–Sen slope (dashed) is {glue:text}`sen:.1f` mm per year; with this much interannual variability, the change is hard to establish from runoff alone. (b) The double-mass curve shows it clearly: the runoff ratio falls from {glue:text}`slope_pre:.2f` before 2005 to {glue:text}`slope_post:.2f` after.*

### Attribution

Detecting a change does not explain it. **Attribution** needs an estimate of what the runoff would have been without the suspected cause. Three approaches are common:

- **Paired catchments**: comparing a treated catchment with a nearby control before and after the treatment. Reviews of such experiments found that removing forest increases annual runoff and planting forest decreases it, roughly in proportion to the change in cover (Bosch and Hewlett, 1982).
- **Hydrological modeling**: calibrating a model on the period before the change and running it, with observed climate, through the period after. The difference between observed and simulated runoff is attributed to the change, provided the model passes a differential split-sample test.
- **Elasticity-based decomposition**: the change in mean runoff expected from the change in climate alone is estimated from the elasticities of Chapter 15, $\Delta Q_{\text{clim}} \approx \varepsilon_P\,(\bar Q/\bar P)\,\Delta P + \varepsilon_{E_p}\,(\bar Q/\bar E_p)\,\Delta E_p$, and the remainder of the observed change is attributed to the catchment (Example 16.3).

All three rely on assumptions that should be stated, and their results should be compared when possible.

## 16.6 Worked examples

**Example 16.1 — NSE and KGE.** Observed flows are [2, 5, 9, 4, 3, 2] and simulated flows [2.5, 4, 7, 4.5, 3, 2.5]. Compute NSE and KGE.

```{code-cell} python
o = np.array([2, 5, 9, 4, 3, 2.0]); s = np.array([2.5, 4, 7, 4.5, 3, 2.5])
nse_ = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
r, alpha, beta = np.corrcoef(o, s)[0, 1], s.std() / o.std(), s.mean() / o.mean()
kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
print(f"NSE = {nse_:.2f}; KGE = {kge_:.2f} (r = {r:.2f}, alpha = {alpha:.2f}, beta = {beta:.2f})")
```

**Example 16.2 — Reading a response surface.** In a catchment whose runoff elasticity to precipitation is 2.2 and to potential evaporation −1.2, estimate the change in mean runoff for 5% less precipitation and 2 °C warming, if potential evaporation rises 3% per °C.

```{code-cell} python
epsP, epsE = 2.2, -1.2
dQ_ = epsP * (-0.05) + epsE * 0.03 * 2
print(f"ΔQ/Q ≈ {100 * dQ_:.0f}%")
```

**Example 16.3 — Attribution by elasticities.** Mean annual runoff fell from 520 mm to 430 mm between two 20-year periods, while precipitation fell from 1350 to 1300 mm and potential evaporation rose from 980 to 1010 mm. With $\varepsilon_P$ = 2.0 and $\varepsilon_{E_p}$ = −1.0, how much of the decline is due to climate?

```{code-cell} python
Q1, Q2, P1, P2, E1, E2 = 520.0, 430.0, 1350.0, 1300.0, 980.0, 1010.0
dQ_clim = 2.0 * Q1 / P1 * (P2 - P1) + (-1.0) * Q1 / E1 * (E2 - E1)
print(f"observed ΔQ = {Q2 - Q1:.0f} mm; climate {dQ_clim:.0f} mm; other causes {Q2 - Q1 - dQ_clim:.0f} mm")
```

## 16.7 Common mistakes

- **Equating a high efficiency with a correct model.** Check the water balance, the signatures and the internal states.
- **Reporting one optimal parameter set.** Equifinality makes the optimum fragile; report the behavioral range.
- **Validating only under calibration-like conditions.** Use differential split-sample tests before projecting change.
- **Treating climate-model ensembles as probability distributions.** They are ensembles of opportunity; a response surface keeps sensitivity and projections separate.
- **Claiming a trend from an autocorrelated series** without adjusting the test, or attributing a change without a counterfactual.

## 16.8 Problems

```{code-cell} python
:tags: [remove-cell]
o2 = np.array([10, 20, 30.0]); s2 = np.array([12, 18, 27.0])
glue("t1", float(1 - np.sum((s2 - o2) ** 2) / np.sum((o2 - o2.mean()) ** 2)), display=False)
glue("t2", float(1 - np.sqrt(0.1 ** 2 + 0.2 ** 2 + 0.1 ** 2)), display=False)
glue("t3", float(100 * (2.5 * -0.10 + -1.0 * 0.06)), display=False)
glue("t4", float(-80 - (2.0 * 400 / 1200 * -60)), display=False)
glue("t5", float(1 - 0.95 ** 20), display=False)
```

1. Observed flows [10, 20, 30] are simulated as [12, 18, 27]. Compute NSE.
2. A simulation has $r$ = 0.9, $\alpha$ = 0.8 and $\beta$ = 1.1. Compute KGE.
3. With $\varepsilon_P$ = 2.5 and $\varepsilon_{E_p}$ = −1.0, estimate the runoff change for 10% less precipitation and 6% more potential evaporation.
4. Runoff fell by 80 mm while precipitation fell by 60 mm. With $\bar Q$ = 400 mm, $\bar P$ = 1200 mm, $\varepsilon_P$ = 2.0 and no change in $E_p$, how much of the decline is not explained by precipitation?
5. Twenty independent stations are tested for trends at the 5% level, and none has a real trend. What is the probability that at least one shows a "significant" trend?

```{admonition} Answers
:class: dropdown
1. NSE = {glue:text}`t1:.2f`.
2. KGE = {glue:text}`t2:.2f`.
3. ΔQ/Q ≈ {glue:text}`t3:.0f`%.
4. {glue:text}`t4:.0f` mm.
5. {glue:text}`t5:.2f`.
```

## 16.9 Open questions

- **Physics and learning.** How can data-driven models be combined with physical constraints so that they remain reliable under unprecedented conditions?
- **Models for change.** What evidence would show that a model is fit for projecting the effects of climate and land-use change, rather than only for reproducing the past?
- **Uncertainty that matters.** Which sources of uncertainty dominate decisions, and how should they be communicated to those who make them?

## 16.10 Key papers

- Beven, K. (2006). A manifesto for the equifinality thesis. *Journal of Hydrology*, 320, 18–36.
- Beven, K., and Binley, A. (1992). The future of distributed models: model calibration and uncertainty prediction. *Hydrological Processes*, 6, 279–298.
- Bosch, J. M., and Hewlett, J. D. (1982). A review of catchment experiments to determine the effect of vegetation changes on water yield and evapotranspiration. *Journal of Hydrology*, 55, 3–23.
- Brown, C., Ghile, Y., Laverty, M., and Li, K. (2012). Decision scaling: Linking bottom-up vulnerability analysis with climate projections in the water sector. *Water Resources Research*, 48, W09537.
- Freeze, R. A., and Harlan, R. L. (1969). Blueprint for a physically-based, digitally-simulated hydrologic response model. *Journal of Hydrology*, 9, 237–258.
- Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377, 80–91.
- Kim, D., Chun, J. A., and Aikins, C. M. (2018). An hourly-scale scenario-neutral flood risk assessment in a mesoscale catchment under climate change. *Hydrological Processes*, 32, 3416–3430.
- Kim, D., Chun, J. A., and Choi, S.-J. (2019). Incorporating the logistic regression into a decision-centric assessment of climate change impacts on a complex river system. *Hydrology and Earth System Sciences*, 23, 1145–1162.
- Klemeš, V. (1986). Operational testing of hydrological simulation models. *Hydrological Sciences Journal*, 31, 13–24.
- Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., and Nearing, G. S. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water Resources Research*, 55, 11344–11354.
- Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*, 13, 245–259.
- Nash, J. E., and Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I — A discussion of principles. *Journal of Hydrology*, 10, 282–290.
- Prudhomme, C., Wilby, R. L., Crooks, S., Kay, A. L., and Reynard, N. S. (2010). Scenario-neutral approach to climate change impact studies: Application to flood risk. *Journal of Hydrology*, 390, 198–209.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63, 1379–1389.

## 16.11 Further reading

- Beven, K. J. (2012). *Rainfall–Runoff Modelling: The Primer* (2nd ed.). Wiley-Blackwell.
- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The closing chapters on catchment response and change.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The appendix on hydrologic modeling.
- Kendall, M. G. (1975). *Rank Correlation Methods* (4th ed.). Charles Griffin.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
