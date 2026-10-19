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

# 15. The catchment as a dynamical system

```{admonition} Core ideas
:class: tip
1. A catchment can be studied as a system that transforms an input (precipitation minus evaporation) into an output (streamflow), without describing every process inside it.
2. If the system is **linear and time-invariant**, its behavior is captured by an impulse response, and output is the convolution of input with it. Conceptual models, such as cascades of linear reservoirs, give the response a parametric form.
3. Identifying a response function from noisy data is an **ill-posed** inverse problem; it needs regularization or a parametric form.
4. Real catchments are **nonlinear**: their response depends on how wet they are. If discharge depends on storage alone, a single function $g(Q) = dQ/dS$, estimated from recessions, describes the catchment and can even be run **backward** to infer precipitation from streamflow (Kirchner, 2009).
5. At annual scale, the sensitivity of runoff to precipitation, the **elasticity**, follows from the Budyko framework and is greatest in dry climates.
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

## 15.1 Linear, time-invariant response

### Convolution

The previous chapters described processes: infiltration, subsurface flow, overland flow, channel routing. A complementary view treats the whole catchment as a **system**, characterized by how it transforms its input into its output. This view goes back to the unit hydrograph of Sherman (1932) and was put on a general footing by Dooge (1959) and others (see the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch09_the_catchment_as_a_linear_system.html) book for its design applications).

A system is **linear** if the response to a sum of inputs is the sum of the responses, and **time-invariant** if the response to an input does not depend on when it occurs. For such a system, the response to any input $x(t)$ is the **convolution** of the input with the **impulse response** $u(t)$, the output produced by a unit input applied instantaneously at $t = 0$:

\begin{equation*}
y(t) = \int_0^t u(\tau)\,x(t - \tau)\,d\tau .
\tag{15.1}
\end{equation*}

For rainfall–runoff, $x$ is effective rainfall and $y$ storm runoff, and $u$ is the **instantaneous unit hydrograph**. Conservation of mass requires $\int_0^\infty u\,dt = 1$. Its first moment, the mean delay $\int t\,u\,dt$, is the **lag time** of the catchment, and its higher moments describe the spreading of a pulse.

### Reservoirs and cascades

A **linear reservoir**, with storage proportional to outflow, $S = kQ$, and continuity $dS/dt = x - Q$, has the impulse response $u(t) = e^{-t/k}/k$, with lag $k$. A cascade of $n$ identical linear reservoirs, each feeding the next (Nash, 1957), has the impulse response

\begin{equation*}
u(t) = \frac{1}{k\,\Gamma(n)}\left(\frac{t}{k}\right)^{n-1} e^{-t/k},
\tag{15.2}
\end{equation*}

a gamma distribution with mean $nk$ and variance $nk^2$. Since the moments of a convolution add, the parameters can be estimated from the moments of observed input and output. Linear reservoirs and channels (pure translation) are the building blocks of most conceptual models; their combinations can produce almost any smooth, single-peaked response.

The linear view is useful, but it hides two difficulties. First, the input that matters, effective rainfall, is not measured; it depends on a loss model, which is where most of the nonlinearity lies (Chapters 12 and 14). Second, the response itself changes with wetness (Section 15.4).

## 15.2 Identifying response functions

Given records of input and output, can we find $u$ without assuming its form? In discrete time, with input $x_i$ and output $y_i$ at intervals $\Delta t$ and a response of length $L$, Equation (15.1) becomes a linear system $\mathbf{y} = \mathbf{X}\mathbf{u} + \boldsymbol{\varepsilon}$, where each row of $\mathbf{X}$ contains the $L$ most recent inputs and $\boldsymbol{\varepsilon}$ is noise. The least-squares solution $\hat{\mathbf{u}} = (\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}$ looks straightforward, but the problem is **ill-posed**: neighboring columns of $\mathbf{X}$ are strongly correlated because rainfall is persistent, so small errors in the data produce large, oscillating errors in $\hat{\mathbf{u}}$.

The standard remedy is **regularization**: add a penalty on roughness, minimizing $\lVert\mathbf{y} - \mathbf{X}\mathbf{u}\rVert^2 + \lambda\lVert\mathbf{D}\mathbf{u}\rVert^2$, where $\mathbf{D}$ takes second differences. The solution is $\hat{\mathbf{u}} = (\mathbf{X}^\mathsf{T}\mathbf{X} + \lambda\mathbf{D}^\mathsf{T}\mathbf{D})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}$, and $\lambda$ trades fidelity to the data against smoothness. It can be chosen by cross-validation. Figure 15.1 shows the effect on synthetic data.

```{code-cell} python
:tags: [hide-input]
from scipy.special import gamma as Gfun
rng = np.random.default_rng(15)
N, L = 24 * 40, 48                                                 # 40 days of hourly data, 48 h response
n_true, k_true = 3.0, 4.0
tau = np.arange(L) + 0.5
u_true = (tau / k_true) ** (n_true - 1) * np.exp(-tau / k_true) / (k_true * Gfun(n_true))
u_true /= u_true.sum()
wet = np.zeros(N, bool); i = 0
while i < N:                                                        # storms: alternating wet and dry spells
    i += int(rng.exponential(60)); d = int(rng.exponential(8)) + 1; wet[i:i + d] = True; i += d
x = np.where(wet[:N], rng.exponential(2.0, N), 0.0)                 # effective rain (mm/h)
y_clean = np.convolve(x, u_true)[:N]
y = y_clean + 0.15 * y_clean.std() * rng.standard_normal(N)
X = np.column_stack([np.r_[np.zeros(j), x[:N - j]] for j in range(L)])
u_ols = np.linalg.lstsq(X, y, rcond=None)[0]
Dm = np.diff(np.eye(L), 2, axis=0)
def ridge(lam): return np.linalg.solve(X.T @ X + lam * Dm.T @ Dm, X.T @ y)
half = N // 2                                                       # choose lambda by split-sample validation
Xa, ya, Xb, yb = X[:half], y[:half], X[half:], y[half:]
lams = np.logspace(-1, 4, 30)
err = [np.mean((yb - Xb @ np.linalg.solve(Xa.T @ Xa + l * Dm.T @ Dm, Xa.T @ ya)) ** 2) for l in lams]
lam_best = lams[int(np.argmin(err))]; u_reg = ridge(lam_best)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(tau, u_true, color=ink, lw=2.4, label="True response (gamma, n = 3, k = 4 h)")
ax.plot(tau, u_ols, color=c2, lw=1.2, label="Least squares")
ax.plot(tau, u_reg, color=c1, lw=2.0, ls="--", label=f"Regularized, λ = {lam_best:.0f}")
ax.axhline(0, color=ink2, lw=0.8)
ax.set_xlabel("Lag (h)"); ax.set_ylabel("u (h⁻¹)"); ax.legend(loc="upper right", fontsize=9)
plt.show()
glue("rough_ols", float(np.abs(np.diff(u_ols, 2)).sum() / np.abs(np.diff(u_true, 2)).sum()), display=False)
glue("sum_reg", float(u_reg.sum()), display=False)
```

*Figure 15.1. Identifying an impulse response from 40 days of synthetic hourly effective rainfall and runoff, with noise of 15% of the runoff's standard deviation. Ordinary least squares recovers the general shape but is {glue:text}`rough_ols:.0f` times rougher than the true response; regularization with a second-difference penalty, with λ chosen by split-sample validation, gives a smooth estimate whose ordinates sum to {glue:text}`sum_reg:.2f`.*

## 15.3 Nonlinear storage–discharge systems

### Catchments as simple dynamical systems

Real catchments respond more strongly when wet than when dry. One simple nonlinear description assumes that discharge is a single-valued function of catchment storage, $Q = f(S)$, as for the aquifers of Chapter 13. Continuity, $dS/dt = P - E - Q$, then gives

\begin{equation*}
\frac{dQ}{dt} = \frac{dQ}{dS}\frac{dS}{dt} = g(Q)\,(P - E - Q), \qquad g(Q) = \frac{dQ}{dS} = f'(f^{-1}(Q)) .
\tag{15.3}
\end{equation*}

The **sensitivity function** $g(Q)$ characterizes the catchment completely. Kirchner (2009) pointed out that it can be estimated directly from streamflow during periods when $P$ and $E$ are negligible compared with $Q$, such as rainless nights:

\begin{equation*}
g(Q) \approx \frac{-dQ/dt}{Q}\Big|_{P,\,E\,\ll\,Q} .
\tag{15.4}
\end{equation*}

This is the recession analysis of Chapter 13, with $g(Q) = aQ^{b-1}$ for a power-law recession. In practice $-dQ/dt$ is averaged in bins of $Q$, and $\ln g$ is fitted as a smooth function of $\ln Q$, often a quadratic. Once $g$ is known, Equation (15.3) can be integrated forward to simulate streamflow from $P$ and $E$, with no calibration to the hydrograph itself. It can also be rearranged to infer the catchment-average input from streamflow alone, "doing hydrology backward":

\begin{equation*}
P - E = \frac{dQ/dt}{g(Q)} + Q .
\tag{15.5}
\end{equation*}

Figures 15.2 and 15.3 apply the method to a synthetic catchment.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(9)
nh = 24 * 365
hour = np.arange(nh) % 24
wet = np.zeros(nh, bool); i = 0
while i < nh:
    i += int(rng.exponential(110)); d = int(rng.exponential(7)) + 1; wet[i:i + d] = True; i += d
P = np.where(wet[:nh], rng.exponential(1.6, nh), 0.0)              # mm/h
E_pot = 0.18 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (1.2 - 0.6 * np.cos(2 * np.pi * np.arange(nh) / nh))
a_s, c_s = 1.2e-6, 3.0                                             # Q = a S^c (S in mm, Q in mm/h)
S = 80.0; Q = np.zeros(nh); E = np.zeros(nh)
for t in range(nh):
    for _ in range(10):                                            # 6-minute substeps
        e = E_pot[t] * min(1.0, S / 60.0)
        S += 0.1 * (P[t] - e - a_s * S ** c_s)
    E[t] = e; Q[t] = a_s * S ** c_s
Qobs = Q * np.exp(0.005 * rng.standard_normal(nh))
# Recession selection: rainless hours at night, with no rain in the previous 3 hours
night = (hour < 5) | (hour >= 21)
dry = (P == 0) & (np.r_[0, P[:-1]] == 0) & (np.r_[0, 0, P[:-2]] == 0) & (np.r_[0, 0, 0, P[:-3]] == 0)
sel = night[1:] & dry[1:]
dQdt = np.diff(Qobs)[sel]; Qm = 0.5 * (Qobs[1:] + Qobs[:-1])[sel]
edges = np.logspace(np.log10(Qm.min()), np.log10(Qm.max()), 25)
gb, Qb = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (Qm >= lo) & (Qm < hi)
    if k.sum() >= 8 and dQdt[k].mean() < 0:
        Qb.append(Qm[k].mean()); gb.append(-dQdt[k].mean() / Qm[k].mean())
Qb, gb = np.array(Qb), np.array(gb)
cq = np.polyfit(np.log(Qb), np.log(gb), 2)
g_fit = lambda q: np.exp(np.polyval(cq, np.log(q)))
g_true = lambda q: c_s * a_s ** (1 / c_s) * q ** ((c_s - 1) / c_s)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.loglog(Qm, np.maximum(-dQdt / Qm, 1e-5), ".", color="#c9c8c3", ms=3, label="Individual night-time hours")
ax.loglog(Qb, gb, "o", color=c1, ms=5, label="Binned means")
qq = np.logspace(np.log10(Qb.min()), np.log10(Qb.max()), 50)
ax.loglog(qq, g_fit(qq), color=c2, lw=2.2, label="Quadratic fit in log space")
ax.loglog(qq, g_true(qq), color=ink, lw=1.2, ls="--", label="True g(Q)")
ax.set_ylim(1e-4, 1); ax.set_xlabel("Q (mm h⁻¹)"); ax.set_ylabel("g(Q) = −(dQ/dt)/Q (h⁻¹)")
ax.legend(loc="upper left", fontsize=9)
plt.show()
```

*Figure 15.2. Estimating the sensitivity function $g(Q)$ of a synthetic catchment with $Q \propto S^3$ from one year of hourly streamflow. Only night-time hours at least three hours after rain are used, so that $P$ and $E$ are negligible. Individual hours scatter widely; binned means follow the true function closely.*

```{code-cell} python
:tags: [hide-input]
lnQ = np.log(Qobs[0]); Qsim = np.zeros(nh)
for t in range(nh):                                                 # forward simulation, Eq. (15.3)
    for _ in range(10):
        q = np.exp(lnQ); lnQ += 0.1 * g_fit(q) * (P[t] - E[t] - q) / q
    Qsim[t] = np.exp(lnQ)
nse = 1 - np.sum((Qsim - Qobs) ** 2) / np.sum((Qobs - Qobs.mean()) ** 2)
Qs_ = np.convolve(Qobs, np.ones(3) / 3, mode="same")                # light smoothing before differencing
P_inf = np.r_[0, np.diff(Qs_)] / g_fit(Qs_) + Qs_ + E               # backward inference, Eq. (15.5)
day = np.arange(nh) / 24
w = (day >= 200) & (day < 230)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
a1.plot(day[w], Qobs[w], color=ink, lw=2.0, label="Observed")
a1.plot(day[w], Qsim[w], color=c2, lw=1.4, ls="--", label=f"Simulated from g(Q), NSE = {nse:.2f}")
a1.set_ylabel("Q (mm h⁻¹)"); a1.legend(loc="upper right", fontsize=9)
Pd_true = P[w].reshape(-1, 24).sum(1); Pd_inf = P_inf[w].reshape(-1, 24).sum(1)
dd = day[w].reshape(-1, 24)[:, 0]
a2.bar(dd + 0.3, Pd_true, width=0.4, color=c1, label="True")
a2.bar(dd + 0.7, Pd_inf, width=0.4, color=c3, label="Inferred from Q")
a2.set_ylabel("Daily P (mm)"); a2.set_xlabel("Day of year"); a2.legend(loc="upper right", fontsize=9)
plt.show()
Pd_all = P.reshape(-1, 24).sum(1); Pi_all = P_inf.reshape(-1, 24).sum(1)
glue("nse15", float(nse), display=False)
glue("r_pinf", float(np.corrcoef(Pd_all, Pi_all)[0, 1]), display=False)
glue("ptot_ratio", float(Pi_all.sum() / Pd_all.sum()), display=False)
```

*Figure 15.3. (a) Streamflow simulated forward from precipitation and evaporation with the fitted $g(Q)$ alone, over a 30-day window; the Nash–Sutcliffe efficiency over the whole year is {glue:text}`nse15:.2f`. (b) Daily precipitation inferred backward from streamflow with Equation (15.5). Over the year, inferred and true daily totals have a correlation of {glue:text}`r_pinf:.2f`, and the inferred annual total is {glue:text}`ptot_ratio:.2f` times the true one.*

The synthetic catchment satisfies the method's assumptions exactly, so the results are a best case. In real catchments, discharge is not a single-valued function of storage (hysteresis, spatially variable rain, multiple stores), evaporation is not zero at night, and $g(Q)$ estimates are uncertain at the highest and lowest flows, where data are few. Kirchner's analysis worked well in small, wet catchments with a single dominant store, and has been tested in many others since, with mixed success. Its value lies as much in what it diagnoses, the storage–discharge behavior of the catchment, as in its predictions.

## 15.4 Nonstationary and state-dependent response

Under Equation (15.3), the catchment's response to a given input depends on its state. Linearizing about a flow $Q_0$, a small input decays with a timescale $1/g(Q_0)$: if $g$ increases with $Q$, as for most catchments, the response is faster and higher when the catchment is wet (Figure 15.4). A single unit hydrograph cannot describe this; in practice it is handled by making the loss model, the response, or both, depend on antecedent wetness.

```{code-cell} python
:tags: [hide-input]
def pulse_response(S0, depth=10.0, hours=72):
    S = S0; out = []
    for t in range(hours):
        for _ in range(10):
            S += 0.1 * ((depth if t == 0 else 0) - a_s * S ** c_s)
        out.append(a_s * S ** c_s)
    S = S0; base = []
    for t in range(hours):
        for _ in range(10): S += 0.1 * (-a_s * S ** c_s)
        base.append(a_s * S ** c_s)
    return np.array(out) - np.array(base)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for S0, col in zip([40, 70, 100], [c2, c3, c1]):
    r = pulse_response(S0)
    ax.plot(np.arange(1, 73), r, color=col, lw=2.2, label=f"Initial Q = {a_s * S0 ** c_s:.2f} mm h⁻¹")
ax.set_xlabel("Hours after a 10 mm pulse"); ax.set_ylabel("Added streamflow (mm h⁻¹)")
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
```

*Figure 15.4. Response of the synthetic nonlinear catchment to the same 10 mm input at three initial wetness states. The wetter the catchment, the faster and higher the response: the system is not time-invariant in any useful sense.*

The response can also change over longer periods, through land-use change, urbanization, river regulation, wildfire, or changes in climate that alter vegetation and soil. Detecting such **nonstationarity** in records is the subject of Section 16.5.

A further distinction is between the response of **discharge** and that of **water**. The hydrograph responds through pressure waves, but solutes move with water particles, whose **transit-time distribution** through the catchment is much longer and also varies with wetness (Chapter 14). Describing both within one framework is an active area of research.

## 15.5 Annual runoff from annual precipitation

At annual and longer timescales, storage changes are small and the catchment can be described by its long-term partitioning of precipitation into evaporation and runoff, the Budyko framework of Chapter 8. Writing the evaporative fraction as $E/P = F(\phi)$ with the aridity index $\phi = E_p/P$, the runoff is $Q = P\,[1 - F(\phi)]$. How sensitive is runoff to a change in precipitation? The **precipitation elasticity of runoff** is

\begin{equation*}
\varepsilon_P = \frac{\partial \ln Q}{\partial \ln P} = 1 + \frac{\phi\,F'(\phi)}{1 - F(\phi)},
\tag{15.6}
\end{equation*}

using $\partial\phi/\partial\ln P = -\phi$. Since $F' > 0$, the elasticity always exceeds 1: a 1% change in precipitation changes runoff by more than 1%, because evaporation absorbs part of the change. The amplification is largest in dry climates, where runoff is a small residual (Figure 15.5). With Fu's form of the Budyko curve, $F = 1 + \phi - (1 + \phi^\omega)^{1/\omega}$ (Zhang et al., 2004), the derivative is elementary.

Elasticities are also estimated directly from records. A simple nonparametric estimator (Sankarasubramanian et al., 2001) takes the median over years of $\dfrac{Q_t - \bar Q}{P_t - \bar P}\,\dfrac{\bar P}{\bar Q}$. It is robust but noisy for short records, and it attributes all runoff variation to precipitation, ignoring covarying temperature and storage carryover between years.

```{code-cell} python
:tags: [hide-input]
def fu_F(phi, w): return 1 + phi - (1 + phi ** w) ** (1 / w)
def fu_dF(phi, w): return 1 - phi ** (w - 1) * (1 + phi ** w) ** (1 / w - 1)
phi = np.linspace(0.2, 3.0, 300)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for w, col in zip([1.8, 2.6, 3.5], [c3, c1, c2]):
    eps = 1 + phi * fu_dF(phi, w) / (1 - fu_F(phi, w))
    ax.plot(phi, eps, color=col, lw=2.2, label=f"Fu, ω = {w}")
# synthetic annual record for a catchment with phi = 1.0, omega = 2.6
rng = np.random.default_rng(5)
Pa = 1000 * np.exp(0.15 * rng.standard_normal(40)); Epa = 1000.0
Qa = Pa * (1 - fu_F(Epa / Pa, 2.6)) * np.exp(0.03 * rng.standard_normal(40))
eps_np = np.median((Qa - Qa.mean()) / (Pa - Pa.mean()) * Pa.mean() / Qa.mean())
ax.plot([1.0], [eps_np], "o", color=ink, ms=7, label="Nonparametric estimate, 40 synthetic years")
ax.set_xlabel("Aridity index φ = $E_p/P$"); ax.set_ylabel("Elasticity $\\varepsilon_P$"); ax.set_ylim(1, 4)
ax.legend(loc="upper left", fontsize=9)
plt.show()
glue("eps_np", float(eps_np), display=False)
glue("eps_th", float(1 + 1.0 * fu_dF(1.0, 2.6) / (1 - fu_F(1.0, 2.6))), display=False)
```

*Figure 15.5. Precipitation elasticity of mean annual runoff from Equation (15.6) with Fu's curve. The point is the nonparametric estimate, {glue:text}`eps_np:.2f`, from 40 years of synthetic data for a catchment with φ = 1 and ω = 2.6, whose theoretical elasticity is {glue:text}`eps_th:.2f`.*

## 15.6 Lab (planned)

Two Colab notebooks are planned. The first will apply Kirchner's method to hourly streamflow, precipitation and evaporation records from openly available sources: estimating $g(Q)$ from rainless night-time recessions, simulating streamflow forward and inferring precipitation backward. The second will identify unit hydrographs from storm events by regularized least squares and compare them across events of different antecedent wetness.

## 15.7 Worked examples

**Example 15.1 — Nash cascade from moments.** The impulse response of a catchment has a mean lag of 9 h and a variance of 27 h². Find the Nash cascade parameters.

```{code-cell} python
mean_, var_ = 9.0, 27.0
k_ = var_ / mean_; n_ = mean_ / k_
print(f"k = {k_:.1f} h, n = {n_:.1f}")
```

**Example 15.2 — Recession timescale.** In a catchment, $g(Q) = 0.02\,Q^{0.6}$ h⁻¹ with $Q$ in mm h⁻¹. What is the timescale of response at $Q$ = 0.05 and 1.0 mm h⁻¹?

```{code-cell} python
for q in (0.05, 1.0):
    print(f"Q = {q:4.2f} mm/h: 1/g = {1 / (0.02 * q ** 0.6):6.0f} h")
```

**Example 15.3 — Runoff change from a precipitation change.** A catchment has φ = 1.5 and ω = 2.6. By how much does mean annual runoff change if precipitation falls by 10%?

```{code-cell} python
phi_, w_ = 1.5, 2.6
eps_ = 1 + phi_ * fu_dF(phi_, w_) / (1 - fu_F(phi_, w_))
P0 = 1.0; P1 = 0.9
Q0 = P0 * (1 - fu_F(phi_ / P0, w_)); Q1 = P1 * (1 - fu_F(phi_ / P1, w_))
print(f"elasticity = {eps_:.2f}; linear estimate {-10 * eps_:.0f}%; exact {100 * (Q1 / Q0 - 1):.0f}%")
```

## 15.8 Common mistakes

- **Treating effective rainfall as observed.** The unit hydrograph is only as good as the loss model that defines its input.
- **Unregularized deconvolution.** Least-squares response functions from noisy data oscillate and can be negative.
- **Using daytime or post-storm recessions to estimate $g(Q)$.** Evaporation and quickflow bias the estimate.
- **Assuming the response is fixed.** Wet and dry states respond differently.
- **Reading elasticities across climates as equal.** Runoff in dry climates is far more sensitive to precipitation.

## 15.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("s1", 4 * 2.5, display=False)
glue("s2", float(4 * 2.5 ** 2), display=False)
glue("s3", float(1 / (0.05 * 0.2 ** 0.5)), display=False)
glue("s4", float(0.5 / 0.1 + 0.3), display=False)
glue("s5", float(1 + 2.0 * fu_dF(2.0, 2.6) / (1 - fu_F(2.0, 2.6))), display=False)
```

1. What is the lag time of a Nash cascade with $n$ = 4 and $k$ = 2.5 h?
2. What is the variance of the same cascade?
3. If $g(Q) = 0.05\,Q^{0.5}$ h⁻¹, what is the recession timescale at $Q$ = 0.2 mm h⁻¹?
4. At an instant when $Q$ = 0.3 mm h⁻¹, $dQ/dt$ = 0.5 mm h⁻² and $g(Q)$ = 0.1 h⁻¹, what is $P - E$ by Equation (15.5)?
5. Compute the elasticity for φ = 2 and ω = 2.6.

```{admonition} Answers
:class: dropdown
1. {glue:text}`s1:.0f` h.
2. {glue:text}`s2:.0f` h².
3. {glue:text}`s3:.0f` h.
4. {glue:text}`s4:.1f` mm h⁻¹.
5. ε_P ≈ {glue:text}`s5:.2f`.
```

## 15.10 Open questions

- **Which nonlinearity matters?** How much of a catchment's nonlinearity lies in the generation of effective rainfall, and how much in the routing?
- **Beyond a single store.** How can storage–discharge analysis be extended to catchments with several stores, hysteresis or strong spatial variability?
- **Water age and flow together.** Can one framework predict both hydrographs and transit-time distributions from the same parameters?

## 15.11 Key papers

- Dooge, J. C. I. (1959). A general theory of the unit hydrograph. *Journal of Geophysical Research*, 64, 241–256.
- Kirchner, J. W. (2009). Catchments as simple dynamical systems: Catchment characterization, rainfall-runoff modeling, and doing hydrology backward. *Water Resources Research*, 45, W02429.
- Nash, J. E. (1957). The form of the instantaneous unit hydrograph. *IASH Publication* 45, 114–121.
- Sankarasubramanian, A., Vogel, R. M., and Limbrunner, J. F. (2001). Climate elasticity of streamflow in the United States. *Water Resources Research*, 37, 1771–1781.
- Sherman, L. K. (1932). Streamflow from rainfall by the unit-graph method. *Engineering News-Record*, 108, 501–505.
- Zhang, L., Hickel, K., Dawes, W. R., Chiew, F. H. S., Western, A. W., and Briggs, P. R. (2004). A rational function approach for estimating mean annual evapotranspiration. *Water Resources Research*, 40, W02502.

## 15.12 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The chapter on the catchment response as a dynamical system.
- Dooge, J. C. I. (1973). *Linear Theory of Hydrologic Systems*. Technical Bulletin 1468, US Department of Agriculture.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on streamflow generation and catchment response.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
