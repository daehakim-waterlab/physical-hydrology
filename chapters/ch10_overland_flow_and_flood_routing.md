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

# 10. Overland flow and flood-wave routing

```{admonition} Core ideas
:class: tip
1. On a plane under steady excess rain, the kinematic wave has an exact solution by **characteristics**. The outflow rises as $q = \alpha(i t)^m$ until the **equilibrium time** $t_e$, which is the physically based time of concentration.
2. Integrating the kinematic solution over the plane shows that a hillslope behaves as a **nonlinear reservoir**, $Q \propto S^m$, the catchment-scale relation explored in Chapter 15.
3. Because the kinematic celerity increases with depth, flood fronts **steepen** and can form **kinematic shocks**; the pressure term of the diffusion wave smooths them.
4. The **diffusion wave** adds attenuation through a hydraulic diffusivity $D = Q/(2BS_0)$. The **Muskingum–Cunge** method chooses Muskingum's parameters so that the scheme's numerical diffusion equals this physical diffusion.
5. Routing parameters can be estimated from **channel geometry** (Seddon's law, $c = (1/B)\,dQ/dh$) or from the **moments** of observed inflow and outflow hydrographs. On floodplains the celerity falls, so the largest floods may travel more slowly than bankfull ones.
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
from scipy.optimize import brentq
```

## 10.1 Overland flow on a plane

### The kinematic solution

Consider a plane of length $L$ and slope $S_0$, initially dry, on which rain excess falls at a constant rate $i$ for a duration $D$. With the kinematic approximation of Chapter 9, $q = \alpha h^m$, and Equation (9.6) is

\begin{equation*}
\frac{\partial h}{\partial t} + m\alpha h^{m-1}\frac{\partial h}{\partial x} = i .
\tag{10.1}
\end{equation*}

Along the characteristics $dx/dt = m\alpha h^{m-1}$, the depth obeys $dh/dt = i$ during rain and $dh/dt = 0$ afterward. Two families of characteristics cover the plane: those that start at the top of the plane ($x = 0$) at some time, and those that start at $t = 0$ somewhere on the dry plane. On the latter, the depth simply grows as $h = i t$, independent of position. The outlet therefore sees a rising hydrograph

\begin{equation*}
q(L, t) = \alpha\,(i t)^m \qquad (t \le t_e),
\tag{10.2}
\end{equation*}

until the characteristic that started at the top of the plane at $t = 0$ reaches the outlet. That happens at the **equilibrium time**

\begin{equation*}
t_e = \left(\frac{L}{\alpha\, i^{m-1}}\right)^{1/m} .
\tag{10.3}
\end{equation*}

Afterwards the whole plane is in equilibrium with the rain and $q = iL$. With Manning friction ($m = 5/3$, $\alpha = S_0^{1/2}/n$), $t_e = (nL/(S_0^{1/2} i^{2/3}))^{3/5}$, the physical basis of the kinematic-wave time-of-concentration formulas used in urban drainage (see the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch08_peak_flow_for_small_catchments.html) book). The equilibrium time decreases with rain intensity, unlike the fixed time of concentration of the rational method.

### The recession

When the rain stops at $t = D$, the depth stays constant along each characteristic. A characteristic that leaves the point where the depth was $h$ at time $D$ (in the equilibrium part of the plane, where $h(x) = (ix/\alpha)^{1/m}$) reaches the outlet when

\begin{equation*}
L = \frac{\alpha h^m}{i} + m\alpha h^{m-1}\,(t - D) ,
\tag{10.4}
\end{equation*}

which gives the outlet depth, and hence $q = \alpha h^m$, implicitly as a function of time. If the rain stops before equilibrium ($D < t_e$), the outflow first stays at $\alpha(iD)^m$ for a while, a **partial-equilibrium plateau**, before receding. Figure 10.1 shows both cases.

```{code-cell} python
:tags: [hide-input]
def plane_hydrograph(i, D, L, alpha, m, t):
    """Kinematic outflow (m²/s) from an initially dry plane under rain excess i (m/s) for D seconds."""
    te = (L / (alpha * i ** (m - 1))) ** (1 / m)
    q = np.zeros_like(t)
    for k, tt in enumerate(t):
        if tt <= min(D, te):
            q[k] = alpha * (i * tt) ** m
        elif tt <= D:                                            # equilibrium during rain
            q[k] = i * L
        elif D >= te:                                            # recession from equilibrium
            h = brentq(lambda h: alpha * h ** m / i + m * alpha * h ** (m - 1) * (tt - D) - L, 1e-12, (i * L / alpha) ** (1 / m))
            q[k] = alpha * h ** m
        else:                                                    # rain stopped before equilibrium
            hD = i * D
            t_end_plateau = D + (L - alpha * hD ** m / i) / (m * alpha * hD ** (m - 1))
            if tt <= t_end_plateau:
                q[k] = alpha * hD ** m
            else:
                h = brentq(lambda h: alpha * h ** m / i + m * alpha * h ** (m - 1) * (tt - D) - L, 1e-12, hD)
                q[k] = alpha * h ** m
    return q, te

n_ov, S0, L = 0.15, 0.02, 80.0                                   # grassy plane
alpha, m = S0 ** 0.5 / n_ov, 5 / 3
i = 40 / 1000 / 3600                                             # 40 mm/h
t = np.linspace(0, 3600 * 1.5, 1500)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for D_min, col in [(10, c2), (40, c1)]:
    q, te = plane_hydrograph(i, D_min * 60, L, alpha, m, t)
    ax.plot(t / 60, q / L * 1000 * 3600, color=col, lw=2.2, label=f"rain for {D_min} min")
ax.axhline(40, color=ink2, lw=0.8, ls=":"); ax.text(60, 41.5, "equilibrium: outflow = rain excess", color=ink2, fontsize=9)
ax.axvline(te / 60, color=ink2, lw=0.8, ls="--"); ax.text(te / 60 + 1, 5, f"$t_e$ = {te / 60:.0f} min", color=ink2, fontsize=9.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Outflow per unit area (mm h⁻¹)")
ax.set_xlim(0, 90); ax.set_ylim(0, 48); ax.legend(loc="center right", fontsize=9.5)
plt.show()
glue("te_min", float(te / 60), display=False)
```

*Figure 10.1. Kinematic overland flow from an 80 m grassy plane ($n$ = 0.15, $S_0$ = 0.02) under rain excess of 40 mm h⁻¹. The plane reaches equilibrium after {glue:text}`te_min:.0f` minutes. A 40-minute storm produces the full equilibrium outflow; a 10-minute storm stops short and gives a plateau at a fraction of it before receding.*

## 10.2 From hillslopes to lumped reservoirs

Hydrologic models rarely solve Equation (10.1) on every hillslope. Instead they lump the hillslope into a storage element. The kinematic solution tells what the lumped relation should be. At equilibrium the depth profile is $h(x) = (ix/\alpha)^{1/m}$, so the storage per unit width is

\begin{equation*}
S = \int_0^L h\,dx = \frac{m}{m + 1}\,L\left(\frac{iL}{\alpha}\right)^{1/m},
\qquad Q = iL
\quad\Longrightarrow\quad
Q = \left(\frac{m + 1}{m}\right)^{m}\frac{\alpha}{L^{m}}\,S^m .
\tag{10.5}
\end{equation*}

A hillslope at equilibrium behaves as a **nonlinear reservoir**, $Q = a S^b$ with $b = m$. A lumped model $dS/dt = iL - aS^m$ reproduces the outflow of the distributed kinematic solution reasonably well (Figure 10.2), although it cannot represent the sharp transition at $t_e$, because a lumped store has no travel time. The exponent $b > 1$ means that the response is faster when the hillslope is wetter, the nonlinearity that reappears in the recession analysis of Chapters 13 and 15.

```{code-cell} python
:tags: [hide-input]
a_res = ((m + 1) / m) ** m * alpha / L ** m
D = 40 * 60
dt = 1.0; tt = np.arange(0, 5400, dt)
S, q_res = 0.0, []
for tk in tt:
    rain = i * L if tk < D else 0.0
    q_out = a_res * S ** m
    S += (rain - q_out) * dt
    q_res.append(q_out)
q_kin, _ = plane_hydrograph(i, D, L, alpha, m, tt)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(tt / 60, q_kin / L * 3.6e6, color=c1, lw=2.4, label="Distributed kinematic wave")
ax.plot(tt / 60, np.array(q_res) / L * 3.6e6, color=c2, lw=2.0, ls="--", label="Lumped nonlinear reservoir, $Q = aS^{5/3}$")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Outflow (mm h⁻¹)"); ax.set_xlim(0, 90); ax.legend(loc="center right", fontsize=9.5)
plt.show()
glue("res_err", float(abs(np.trapezoid(q_res, tt) - np.trapezoid(q_kin, tt)) / np.trapezoid(q_kin, tt) * 100), display=False)
```

*Figure 10.2. The plane of Figure 10.1 under the 40-minute storm, computed with the distributed kinematic solution and with a single nonlinear reservoir whose storage–discharge relation is derived from the equilibrium profile (Equation 10.5). The lumped model rises too early and too gradually but conserves volume (difference {glue:text}`res_err:.1f`% within the window) and reproduces the recession closely.*

## 10.3 Flood waves in rivers

### Kinematic steepening

In a river reach without lateral inflow, the kinematic wave equation $\partial Q/\partial t + c(Q)\,\partial Q/\partial x = 0$ says that each discharge travels unchanged at its own celerity. Because $c = mV$ increases with discharge, the high parts of a flood wave travel faster than the low parts. The rising limb steepens and the falling limb flattens. Eventually characteristics cross and a **kinematic shock**, a discontinuity in discharge, forms (Figure 10.3). In real rivers, the pressure-gradient term, neglected by the kinematic wave, prevents the shock and spreads the front.

```{code-cell} python
:tags: [hide-input]
# Characteristics of a flood wave in a wide Manning channel: c(Q) = (5/3) V
n_r, S0r, B = 0.035, 0.0005, 1.0
cel = lambda q: 5 / 3 * q / ((n_r * q / np.sqrt(S0r)) ** 0.6)        # m/s, per unit width
t0 = np.linspace(0, 24, 49) * 3600
q_in = 1.0 + 9.0 * np.exp(-0.5 * ((t0 / 3600 - 8) / 3.0) ** 2)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.linspace(0, 150e3, 2)
for tk, qk in zip(t0, q_in):
    ax.plot(x / 1e3, (tk + x / cel(qk)) / 3600, color=c1 if t0[0] <= tk <= 8 * 3600 else c2, lw=0.8, alpha=0.8)
# first crossing of neighbouring characteristics on the rising limb
xs = []
for k in range(len(t0) - 1):
    c_a, c_b = cel(q_in[k]), cel(q_in[k + 1])
    if c_b > c_a:
        xs.append((t0[k + 1] - t0[k]) / (1 / c_a - 1 / c_b))
x_shock = min(xs)
ax.axvline(x_shock / 1e3, color=ink2, lw=0.9, ls="--"); ax.text(x_shock / 1e3 + 2, 2, f"first crossing\n≈ {x_shock / 1e3:.0f} km", fontsize=9, color=ink2)
ax.set_xlabel("Distance downstream (km)"); ax.set_ylabel("Time (h)"); ax.set_xlim(0, 150); ax.set_ylim(0, 60)
ax.plot([], [], color=c1, label="rising limb"); ax.plot([], [], color=c2, label="falling limb"); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("x_shock", float(x_shock / 1e3), display=False)
```

*Figure 10.3. Kinematic characteristics of a flood wave (peak 10 m² s⁻¹) in a wide channel. Each line carries one discharge at its own celerity. On the rising limb (blue), faster high flows catch up with slower low flows, and the first characteristics cross about {glue:text}`x_shock:.0f` km downstream, where the kinematic wave would form a shock. On the falling limb (orange) the characteristics spread apart.*

### The diffusion wave

Keeping the pressure term but dropping the accelerations, $S_f = S_0 - \partial h/\partial x$. Combined with continuity and linearized about a reference flow $Q_0$ in a channel of width $B$, this gives the advection–diffusion equation

\begin{equation*}
\frac{\partial Q}{\partial t} + c\,\frac{\partial Q}{\partial x} = D_h\,\frac{\partial^2 Q}{\partial x^2},
\qquad c = \frac{1}{B}\frac{dQ}{dh},
\qquad D_h = \frac{Q_0}{2 B S_0} ,
\tag{10.6}
\end{equation*}

the same form as Equation (1.5) of Chapter 1. A pulse of discharge travels at the celerity $c$ and spreads, its variance growing by $2D_h$ per unit time, so that the peak falls. Attenuation is strongest for large discharges on wide, flat rivers, exactly where the kinematic wave is least adequate (Chapter 9).

## 10.4 Muskingum–Cunge from the diffusion wave

The Muskingum method (see the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch10_flood_routing.html) book) routes a flood through a reach of length $\Delta x$ with $O_2 = C_0 I_2 + C_1 I_1 + C_2 O_1$. Cunge (1969) showed that this scheme, viewed as a finite-difference approximation of the kinematic wave equation with $K = \Delta x/c$, is not exact: expanding it in Taylor series shows that it actually solves an advection–diffusion equation with a **numerical diffusivity**

\begin{equation*}
D_{\text{num}} = c\,\Delta x\left(\frac{1}{2} - x\right) .
\tag{10.7}
\end{equation*}

The "error" can be put to use. Choosing $x$ so that the numerical diffusion equals the physical diffusion of Equation (10.6) gives

\begin{equation*}
x = \frac{1}{2}\left(1 - \frac{Q_0}{B S_0\, c\,\Delta x}\right),
\tag{10.8}
\end{equation*}

and the Muskingum scheme then reproduces the diffusion wave, with parameters computed from the channel rather than calibrated. Figure 10.4 checks Equation (10.7) directly: a pulse routed through many Muskingum sub-reaches spreads at exactly the predicted rate.

```{code-cell} python
:tags: [hide-input]
def muskingum(I, K, x, dt):
    den = 2 * K * (1 - x) + dt
    C0, C1, C2 = (dt - 2 * K * x) / den, (dt + 2 * K * x) / den, (2 * K * (1 - x) - dt) / den
    O = np.empty_like(I); O[0] = I[0]
    for k in range(len(I) - 1):
        O[k + 1] = C0 * I[k + 1] + C1 * I[k] + C2 * O[k]
    return O

c, dx, dt = 1.5, 3000.0, 600.0                                   # m/s, m, s
K = dx / c
tt = np.arange(0, 40 * 3600, dt)
pulse = np.exp(-0.5 * ((tt - 4 * 3600) / 1800.0) ** 2)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
out = {}
for xm, col in [(0.0, c2), (0.25, c1), (0.45, c3)]:
    Q = pulse.copy(); var = []
    for r in range(30):
        Q = muskingum(Q, K, xm, dt)
        mu = (tt * Q).sum() / Q.sum(); var.append(((tt - mu) ** 2 * Q).sum() / Q.sum())
    dist = np.arange(1, 31) * dx
    slope_var = np.polyfit(dist, var, 1)[0]                      # s² per m
    D_meas = slope_var * c ** 3 / 2                              # variance in time: 2 D x / c³
    D_theory = c * dx * (0.5 - xm)
    out[xm] = (D_meas, D_theory)
    ax.plot(dist / 1e3, np.sqrt(var) / 3600, color=col, lw=2.2, label=f"x = {xm}: D = {D_meas:,.0f} (theory {D_theory:,.0f}) m² s⁻¹")
ax.set_xlabel("Distance routed (km)"); ax.set_ylabel("Std. dev. of hydrograph (h)")
ax.legend(loc="upper left", fontsize=8.5)
plt.show()
```

*Figure 10.4. A pulse routed through 30 Muskingum sub-reaches of 3 km each ($K$ = 2000 s, $\Delta t$ = 600 s) spreads as it travels. The diffusivity inferred from the growth of the hydrograph's variance matches Equation (10.7) for each value of $x$: $x$ = 0.5 would give pure translation, $x$ = 0 the strongest diffusion.*

## 10.5 Estimating routing parameters

### From channel geometry

The celerity of a flood wave follows from the rating curve. With the kinematic relation $Q(h)$ for the reach, **Seddon's law** gives

\begin{equation*}
c = \frac{1}{B}\frac{dQ}{dh},
\tag{10.9}
\end{equation*}

where $B$ is the top width. In a wide rectangular channel this is $(5/3)V$. When the flood spills onto a floodplain, $B$ jumps while $Q$ grows slowly, so the celerity falls sharply above bankfull. Very large floods can therefore travel **more slowly** than bankfull flows, and floodplain storage attenuates them strongly (Figure 10.5). This is one reason why routing parameters calibrated on moderate floods fail for extreme events.

```{code-cell} python
:tags: [hide-input]
# Compound section: main channel 60 m wide, 4 m deep; floodplains 400 m wide each side
Bm, hb, Bf, n_m, n_f, S0c = 60.0, 4.0, 400.0, 0.03, 0.06, 0.0004
def Q_of_h(h):
    Am = Bm * h; Qm = Am / n_m * (Am / (Bm + 2 * h)) ** (2 / 3) * S0c ** 0.5
    hf = max(h - hb, 0.0)
    Qf = 2 * (Bf * hf) / n_f * hf ** (2 / 3) * S0c ** 0.5 if hf > 0 else 0.0
    return Qm + Qf
def width(h): return Bm if h <= hb else Bm + 2 * Bf
hs = np.linspace(0.3, 7, 400)
Qs = np.array([Q_of_h(h) for h in hs])
cs = np.gradient(Qs, hs) / np.array([width(h) for h in hs])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
a1.plot(Qs, hs, color=c1, lw=2.2); a1.axhline(hb, color=ink2, lw=0.8, ls=":")
a1.set_ylabel("Stage (m)"); a1.set_title("(a) Rating curve"); a1.text(Qs.max() * 0.55, hb + 0.2, "bankfull", fontsize=9, color=ink2)
a2.plot(Qs, cs, color=c2, lw=2.2); a2.axvline(Q_of_h(hb), color=ink2, lw=0.8, ls=":")
a2.set_xlabel("Discharge (m³ s⁻¹)"); a2.set_ylabel("Celerity c (m s⁻¹)"); a2.set_title("(b) Kinematic celerity, Seddon's law")
plt.show()
glue("c_bank", float(cs[np.searchsorted(hs, hb) - 2]), display=False)
glue("c_over", float(cs[np.searchsorted(hs, hb) + 5]), display=False)
glue("Q_bank", float(Q_of_h(hb)), display=False)
```

*Figure 10.5. A compound channel with floodplains. Just below bankfull ({glue:text}`Q_bank:.0f` m³ s⁻¹) the celerity is {glue:text}`c_bank:.2f` m s⁻¹; just above, it drops to {glue:text}`c_over:.2f` m s⁻¹, because the wide floodplain must be filled before the stage rises further.*

### From observed hydrographs

When inflow and outflow have been measured for a reach, the **moments** of the hydrographs estimate the parameters without assuming a scheme. For a linear system the centroid lag and the increase in variance between inflow and outflow are additive, and for the diffusion wave over a reach of length $L$:

\begin{equation*}
\Delta t_c = \frac{L}{c}, \qquad \Delta\sigma_t^2 = \frac{2 D_h L}{c^3} .
\tag{10.10}
\end{equation*}

Equation (10.10) gives $c$ and $D_h$ from the first two moments (Example 10.4), and from them the Muskingum–Cunge parameters.

## 10.6 Worked examples

**Example 10.1 — Equilibrium time of a parking lot.** A paved surface 60 m long ($n$ = 0.013, $S_0$ = 0.01) receives rain excess of 50 mm h⁻¹. Find the equilibrium time and the outflow per unit width at equilibrium. Repeat for 20 mm h⁻¹.

```{code-cell} python
L_p, n_p, S_p = 60.0, 0.013, 0.01
alpha_p = S_p ** 0.5 / n_p
for i_mm in (50, 20):
    i_p = i_mm / 1000 / 3600
    te_p = (L_p / (alpha_p * i_p ** (2 / 3))) ** 0.6
    print(f"i = {i_mm} mm/h: t_e = {te_p / 60:.1f} min, q_eq = {i_p * L_p * 1e4:.1f} cm²/s")
```

**Example 10.2 — The lumped reservoir.** For the grassy plane of Figure 10.1 (L = 80 m, $n$ = 0.15, $S_0$ = 0.02), find the coefficient $a$ of the nonlinear reservoir $Q = aS^{5/3}$ (per unit width) and the storage at equilibrium under 40 mm h⁻¹.

```{code-cell} python
S_eq = m / (m + 1) * L * (i * L / alpha) ** (1 / m)
print(f"a = {a_res:.3e} (SI units), S_eq = {S_eq * 1000 / L:.2f} mm averaged over the plane; check Q = {a_res * S_eq ** m / L * 3.6e6:.1f} mm/h")
```

**Example 10.3 — Muskingum–Cunge parameters.** A river 80 m wide, slope 0.0003, has a reference discharge of 400 m³ s⁻¹ and a celerity of 1.8 m s⁻¹. Compute the hydraulic diffusivity, and $K$ and $x$ for sub-reaches of 10 km and 20 km.

```{code-cell} python
Bw, S0w, Q0, cw = 80.0, 0.0003, 400.0, 1.8
Dh = Q0 / (2 * Bw * S0w)
for dxx in (10e3, 20e3):
    xw = 0.5 * (1 - Q0 / (Bw * S0w * cw * dxx))
    print(f"Δx = {dxx / 1e3:.0f} km: K = {dxx / cw / 3600:.2f} h, x = {xw:.3f}")
print(f"D_h = {Dh:,.0f} m²/s")
```

**Example 10.4 — Celerity and diffusivity from moments.** An inflow hydrograph to a 35 km reach has its centroid at 14.0 h and a standard deviation of 4.2 h; the outflow has its centroid at 19.6 h and a standard deviation of 5.1 h. Estimate $c$ and $D_h$.

```{code-cell} python
L_r = 35e3
lag = (19.6 - 14.0) * 3600; dvar = ((5.1 * 3600) ** 2 - (4.2 * 3600) ** 2)
c_e = L_r / lag
D_e = dvar * c_e ** 3 / (2 * L_r)
print(f"c = {c_e:.2f} m/s, D_h = {D_e:,.0f} m²/s")
```

## 10.7 Common mistakes

- **Using a fixed time of concentration for overland flow.** The kinematic equilibrium time depends on rain intensity: faster in heavy rain.
- **Expecting a lumped reservoir to reproduce travel time.** Lumped stores respond instantly; a delay or cascade is needed for the rising limb.
- **Ignoring kinematic shocks in numerical kinematic models.** Coarse grids smear them, introducing numerical diffusion that is mistaken for physics.
- **Applying bankfull routing parameters to overbank floods.** Celerity and attenuation change abruptly at bankfull.
- **Choosing Muskingum–Cunge sub-reaches so short that $x < 0$.** Then the scheme cannot produce the required diffusion; lengthen the sub-reach or reduce the time step consistently.

## 10.8 Problems

```{code-cell} python
:tags: [remove-cell]
al1 = 0.03 ** 0.5 / 0.2; i1 = 25 / 1000 / 3600
glue("y1", float((50 / (al1 * i1 ** (2 / 3))) ** 0.6 / 60), display=False)
glue("y2", float(0.5 * (1 - 150 / (40 * 0.001 * 2.0 * 5000))), display=False)
glue("y3", float(150 / (2 * 40 * 0.001)), display=False)
glue("y4", float(1.5 * 5000 * (0.5 - 0.2)), display=False)
glue("y5", float(25e3 / (6.5 * 3600)), display=False)
```

1. Find the equilibrium time of a 50 m hillslope with $n$ = 0.2, $S_0$ = 0.03 under rain excess of 25 mm h⁻¹.
2. For a channel 40 m wide with $S_0$ = 0.001, $Q_0$ = 150 m³ s⁻¹ and $c$ = 2.0 m s⁻¹, find the Muskingum–Cunge $x$ for a 5 km sub-reach.
3. Find the hydraulic diffusivity of the channel in Problem 2.
4. What numerical diffusivity does a Muskingum scheme with $c$ = 1.5 m s⁻¹, $\Delta x$ = 5 km and $x$ = 0.2 introduce?
5. The centroid of a flood moves 25 km in 6.5 h. Estimate the celerity.

```{admonition} Answers
:class: dropdown
1. About {glue:text}`y1:.0f` min.
2. $x$ = {glue:text}`y2:.3f`.
3. {glue:text}`y3:,.0f` m² s⁻¹.
4. {glue:text}`y4:,.0f` m² s⁻¹.
5. {glue:text}`y5:.2f` m s⁻¹.
```

## 10.9 Open questions

- **From planes to real hillslopes.** Natural hillslopes converge and diverge, infiltrate unevenly and are partly covered by flow; how much of the kinematic theory survives?
- **Nonlinear routing for extreme floods.** Celerity and diffusivity change with discharge; constant-parameter methods misrepresent the largest floods.
- **Data-driven routing.** Can machine-learning routing models respect conservation of mass and extrapolate beyond observed floods?

## 10.10 Key papers

- Cunge, J. A. (1969). On the subject of a flood propagation computation method (Muskingum method). *Journal of Hydraulic Research*, 7, 205–230.
- Eagleson, P. S. (1970). *Dynamic Hydrology*. McGraw-Hill.
- Lighthill, M. J., and Whitham, G. B. (1955). On kinematic waves, I. Flood movement in long rivers. *Proceedings of the Royal Society of London A*, 229, 281–316.
- Ponce, V. M. (1989). *Engineering Hydrology: Principles and Practices*. Prentice Hall.
- Wooding, R. A. (1965). A hydraulic model for the catchment-stream problem, I. Kinematic-wave theory. *Journal of Hydrology*, 3, 254–267.

## 10.11 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The chapters on overland flow and streamflow routing.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapters on kinematic and dynamic wave routing.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
