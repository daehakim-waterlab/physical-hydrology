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

# 13. Groundwater outflow and baseflow

```{admonition} Core ideas
:class: tip
1. Between storms, rivers are fed by the slow drainage of aquifers next to them. This **baseflow** carries the information about catchment storage contained in streamflow records.
2. In thin, unconfined aquifers flow is nearly horizontal. The **Dupuit–Forchheimer** approximation turns a two-dimensional problem into a one-dimensional one for the water-table height.
3. Conservation of mass then gives the nonlinear **Boussinesq equation**. Its solutions predict how outflow declines: as $t^{-1/2}$ at first, then as $(1 + \alpha t)^{-2}$, or exponentially if the equation is linearized.
4. These solutions have the form $-dQ/dt = aQ^b$ with $b$ = 3, 3/2 or 1. **Recession analysis** (Brutsaert and Nieber, 1977) estimates $a$ and $b$ from streamflow records, and through them aquifer properties.
5. For a linear aquifer, storage is proportional to baseflow, so trends in low flows reveal trends in groundwater storage; these can be compared with gravimetric satellite data.
6. Beyond the riparian zone, groundwater moves in nested **local, intermediate and regional flow systems** set by the shape of the water table.
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

## 13.1 Riparian aquifers and the Dupuit–Forchheimer approximation

A river that keeps flowing through weeks without rain is being fed from storage. Most of this water comes from **unconfined aquifers**, the saturated zone below the water table, that border the channel network. In many landscapes these are thin compared with their length: valley-fill alluvium, glacial deposits, or weathered rock a few meters to tens of meters thick and hundreds of meters wide. The outflow from such aquifers is the **baseflow** of the river, and its decline during dry spells, the **recession**, depends on the aquifer's size and hydraulic properties.

Groundwater flow obeys Darcy's law (Chapter 11), and in a saturated, rigid medium conservation of mass makes the head satisfy Laplace's equation. In a thin aquifer the flow is nearly horizontal, so the head hardly varies with depth and equals the elevation of the water table above the base, $h$. This is the **Dupuit–Forchheimer approximation**, named after Jules Dupuit (1863) and Philipp Forchheimer (1886). With it, the discharge per unit width through a vertical section of the aquifer is

\begin{equation*}
q = -K h\,\frac{\partial h}{\partial x} = -\frac{K}{2}\frac{\partial (h^2)}{\partial x} ,
\tag{13.1}
\end{equation*}

where $K$ is the saturated conductivity and $h$ the saturated thickness above a horizontal impermeable base. The approximation fails close to the stream, where flow converges and has a vertical component (a seepage face forms above the water level), but it describes the aquifer as a whole well when its width is several times its thickness.

As a first application, consider an aquifer between two parallel streams a distance $L$ apart, both at water level $h_0$ above the base, receiving a steady recharge $N$ per unit area. Conservation of mass in steady state gives $dq/dx = N$, and with Equation (13.1),

\begin{equation*}
h^2 = h_0^2 + \frac{N}{K}\,x(L - x), \qquad h_{\max}^2 = h_0^2 + \frac{NL^2}{4K} .
\tag{13.2}
\end{equation*}

The water table forms a mound whose height grows with the recharge and the square of the spacing, and falls with the conductivity. The same result is used to space agricultural drains (Example 13.1).

## 13.2 The Boussinesq equation and its solutions

### The equation

In transient flow, the change of storage in a vertical column of aquifer is $\phi_e\,\partial h/\partial t$, where $\phi_e$ is the **drainable porosity** or specific yield, the volume of water released per unit area per unit drop of the water table. It is smaller than the porosity, since part of the water stays held by capillarity (Chapter 11). Conservation of mass, $\phi_e\,\partial h/\partial t = -\partial q/\partial x + N$, with Equation (13.1) gives the **Boussinesq equation** (Boussinesq, 1904):

\begin{equation*}
\phi_e\,\frac{\partial h}{\partial t} = K\,\frac{\partial}{\partial x}\left(h\,\frac{\partial h}{\partial x}\right) + N .
\tag{13.3}
\end{equation*}

It is a nonlinear diffusion equation: the "diffusivity" $Kh/\phi_e$ is proportional to the saturated thickness. When the thickness varies little, $h$ in the coefficient can be replaced by a constant mean depth $p$, giving the linear equation $\phi_e\,\partial h/\partial t = Kp\,\partial^2 h/\partial x^2 + N$, with diffusivity $Kp/\phi_e$.

### Draining an aquifer into a stream

The basic problem for baseflow is the free drainage of an aquifer of width $B$, initially filled to a uniform height $D$, into a stream at $x = 0$ whose water level is at the base of the aquifer. The far side, $x = B$, is a divide with no flow. The outflow per unit length of stream is $Q = K h\,\partial h/\partial x$ at $x = 0$. There are two stages (Figure 13.1).

**Early times.** Before the drawdown reaches the divide, the aquifer behaves as if it were infinitely wide. As with sorption in Chapter 12, the problem has no length scale of its own, so the solution depends on $x/\sqrt{t}$ alone and the outflow declines as $t^{-1/2}$. The constant was found by Polubarinova-Kochina; it is reproduced numerically below:

\begin{equation*}
Q = 0.332\,(K\phi_e D^3)^{1/2}\,t^{-1/2} .
\tag{13.4}
\end{equation*}

**Late times.** Once the whole aquifer is draining, the solution approaches a shape that keeps its form while its height decreases. Boussinesq found that for Equation (13.3) such a separable solution, $h = h_s(x)\,f(t)$, requires $f \propto (1 + \alpha t)^{-1}$, and therefore

\begin{equation*}
Q = \frac{Q_0}{(1 + \alpha t)^2},
\tag{13.5}
\end{equation*}

with $\alpha$ proportional to $KD/(\phi_e B^2)$. For the linearized equation, the slowest mode decays exponentially, $Q \propto \exp(-\pi^2 Kp\,t/(4\phi_e B^2))$, the behavior of a **linear reservoir**.

```{code-cell} python
:tags: [hide-input]
from scipy.integrate import solve_ivp
from scipy.sparse import diags
Kd, phi_e, B, D = 5.0, 0.10, 200.0, 5.0                        # m/d, -, m, m
nx = 800; dx = B / nx; xc = (np.arange(nx) + 0.5) * dx

def drain(linear=False, p=D / 2, t_end=1e4):
    """Finite-volume solution of the Boussinesq equation; stream at x = 0 (h = 0), divide at x = B."""
    def rhs(t, h):
        u = p * 2 * h if linear else np.maximum(h, 0) ** 2           # K h dh/dx = (K/2) d(u)/dx
        F = np.zeros(nx + 1)                                         # flux toward the stream at each face
        F[0] = Kd * u[0] / dx; F[1:nx] = Kd * (u[1:] - u[:-1]) / (2 * dx)
        return (F[1:] - F[:-1]) / (phi_e * dx)
    tt = np.r_[0, np.logspace(-4, np.log10(t_end), 500)]
    jac = diags([np.ones(nx - 1), np.ones(nx), np.ones(nx - 1)], [-1, 0, 1])
    s = solve_ivp(rhs, (0, t_end), np.full(nx, D), method="BDF", t_eval=tt, jac_sparsity=jac, rtol=1e-7, atol=1e-10)
    h = s.y; u0 = p * 2 * h[0] if linear else h[0] ** 2
    return s.t, h, Kd * u0 / dx                                       # Q in m2/d per m of stream

t, h, Q = drain()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 4.1))
for tv, col in zip([1, 10, 100, 300, 1000], [c2, c1, c3, ink2, ink]):
    k = np.argmin(np.abs(t - tv)); a1.plot(xc, h[:, k], color=col, lw=1.8, label=f"{tv:g} d")
a1.set_xlabel("Distance from stream (m)"); a1.set_ylabel("h (m)"); a1.set_ylim(0, 5.5)
a1.set_title("(a) Water table")
fig.legend(loc="outside lower center", ncol=5, fontsize=9, handlelength=1.5, columnspacing=1.0)
a2.loglog(t[1:], Q[1:], color=c1, lw=2.4, label="Numerical")
te = np.logspace(-2, 2, 50)
a2.loglog(te, 0.332 * np.sqrt(Kd * phi_e * D ** 3 / te), color=c2, lw=1.6, ls="--", label="Eq. (13.4)")
a2.set_xlim(1e-2, 1e4); a2.set_ylim(1e-4, 30)
a2.set_xlabel("Time (d)"); a2.set_ylabel("Q (m² d⁻¹)"); a2.set_title("(b) Outflow")
a2.legend(loc="lower left", fontsize=9)
plt.show()
C_num = float(np.interp(1.0, t, Q * np.sqrt(t)) / np.sqrt(Kd * phi_e * D ** 3))
mb = float(phi_e * D * B - phi_e * h[:, -1].sum() * dx - np.trapezoid(Q, t))
glue("C_num", C_num, display=False); glue("mb13", abs(mb), display=False)
```

*Figure 13.1. Drainage of a horizontal unconfined aquifer (width 200 m, initial thickness 5 m, $K$ = 5 m d⁻¹, $\phi_e$ = 0.1) into a stream, from a numerical solution of the Boussinesq equation. (a) The water table drops first near the stream; after about 100 days the whole aquifer is draining and the profile keeps its shape. (b) Outflow per meter of stream. At early times the numerical solution gives $Q\,t^{1/2}/(K\phi_e D^3)^{1/2}$ = {glue:text}`C_num:.3f`, matching Equation (13.4); later the decline steepens. The water balance closes to {glue:text}`mb13:.3f` m² out of 100 m² drained.*

## 13.3 Recession analysis

### The recession plot

Equations (13.4) and (13.5) can be written as relations between the outflow and its rate of change, eliminating time. Differentiating (13.4) and substituting $t$ gives

\begin{equation*}
-\frac{dQ}{dt} = \frac{Q^3}{2(0.332)^2 K\phi_e D^3} ,
\tag{13.6}
\end{equation*}

and from (13.5), $-dQ/dt = 2\alpha Q_0^{-1/2}\,Q^{3/2}$. The linear reservoir gives $-dQ/dt = Q/k$. All are of the form

\begin{equation*}
-\frac{dQ}{dt} = a\,Q^b ,
\tag{13.7}
\end{equation*}

with $b$ = 3 at early times, 3/2 at late times for the nonlinear equation, and 1 for the linearized equation. Brutsaert and Nieber (1977) proposed plotting $-dQ/dt$ against $Q$ on logarithmic axes from streamflow records. Because time is eliminated, the method needs no knowledge of when each recession began, and recessions of different lengths can be combined. The parameters $a$ and $b$ then relate to the aquifer's $K$, $\phi_e$, $D$ and the total length of stream draining it. Figure 13.2 shows the three regimes for the aquifer of Figure 13.1.

```{code-cell} python
:tags: [hide-input]
tl, hl, Ql = drain(linear=True)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for (tt_, QQ), col, lab in [((t, Q), c1, "Nonlinear Boussinesq"), ((tl, Ql), c3, "Linearized, p = D/2")]:
    dQ = -np.gradient(QQ, tt_); ok = (dQ > 0) & (tt_ > 1e-3) & (QQ > 1e-4)
    ax.loglog(QQ[ok], dQ[ok], color=col, lw=2.4, label=lab)
    if lab.startswith("Nonlinear"): Qn, dQn = QQ[ok], dQ[ok]
order = np.argsort(Qn)
def guide(Q_a, Q_b, b, label, xt):
    y_a = 3 * np.interp(Q_a, Qn[order], dQn[order])                  # anchor 3x above the curve
    Qg = np.logspace(np.log10(Q_a), np.log10(Q_b), 10)
    ax.loglog(Qg, y_a * (Qg / Q_a) ** b, color=ink2, lw=1, ls="--")
    ax.text(Q_a / 1.3, y_a, label, color=ink2, fontsize=10, ha="right", va="center")
guide(2.0, 20.0, 3, "b = 3", 2.0)
guide(3e-4, 1e-2, 1.5, "b = 3/2", 3e-4)
ax.set_xlim(1e-4, 200)
ax.set_xlabel("Q (m² d⁻¹)"); ax.set_ylabel("−dQ/dt (m² d⁻²)")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
```

*Figure 13.2. Recession plot of the outflows in Figure 13.1. The nonlinear solution has slope $b$ = 3 at high flows (early times) and 3/2 at low flows (late times). The linearized solution tends to slope 1, the linear reservoir. Dashed lines are guides with the indicated slopes.*

### Recession analysis in practice

Real hydrographs mix baseflow with quickflow, evaporation from the riparian zone, and measurement noise, so a recession plot of daily data is a cloud of points (Figure 13.3). Common practice is to:

- keep only days of declining flow that are a few days after the last rain, to exclude quickflow;
- compute $-dQ/dt \approx (Q_{i-1} - Q_{i+1})/(2\Delta t)$ or $(Q_{i-1} - Q_i)/\Delta t$ against the corresponding mean flow;
- fit either a regression through the points or a **lower envelope**, on the argument that the slowest recessions are those least affected by evaporation and quickflow.

Results depend on these choices, and published values of $b$ vary widely. Many analyses fix $b$ = 1 at low flows and estimate only the **characteristic drainage timescale** $k = 1/a$, which for many catchments in humid climates is of the order of a month or two.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(13)
ndays = 365 * 15
P = np.where(rng.random(ndays) < 0.3, rng.exponential(10.0, ndays), 0.0)   # synthetic daily rain (mm)
S, Sf, kq = 60.0, 0.0, 0.0005                                             # groundwater store, fast store
Qtot = np.zeros(ndays)
for i in range(ndays):                                                     # nonlinear store Q = kq S^2 -> b = 3/2
    S += 0.35 * P[i]; qb = kq * S ** 2; S -= qb
    Sf += 0.25 * P[i]; qf = 0.6 * Sf; Sf -= qf
    Qtot[i] = (qb + qf) * np.exp(0.03 * rng.standard_normal())
dry = np.r_[np.zeros(3, bool), (P[:-3] == 0) & (P[1:-2] == 0) & (P[2:-1] == 0)] & (P == 0)
dQ = Qtot[:-1] - Qtot[1:]; Qm = 0.5 * (Qtot[:-1] + Qtot[1:])
sel = dry[1:] & (dQ > 0)
bfit, lnafit = np.polyfit(np.log(Qm[sel]), np.log(dQ[sel]), 1)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ok = dQ > 0
ax.loglog(Qm[ok & ~sel], dQ[ok & ~sel], ".", color="#c9c8c3", ms=3, label="All declining days")
ax.loglog(Qm[sel], dQ[sel], ".", color=c1, ms=4, label="Three or more days after rain")
Qg = np.logspace(np.log10(Qm[sel].min()), np.log10(Qm[sel].max()), 20)
ax.loglog(Qg, np.exp(lnafit) * Qg ** bfit, color=c2, lw=2.2, label=f"Regression, b = {bfit:.2f}")
ax.loglog(Qg, 2 * np.sqrt(kq) * Qg ** 1.5, color=ink, lw=1.4, ls="--", label="True baseflow law, b = 1.5")
ax.set_xlabel("Q (mm d⁻¹)"); ax.set_ylabel("−dQ/dt (mm d⁻²)")
ax.legend(loc="upper left", fontsize=9)
plt.show()
glue("bfit", float(bfit), display=False)
```

*Figure 13.3. Recession plot of 15 years of synthetic daily streamflow from a catchment with a nonlinear groundwater store ($b$ = 3/2), a fast quickflow store and 3% multiplicative noise. Days soon after rain (gray) are dominated by quickflow. A regression on the selected days gives $b$ = {glue:text}`bfit:.2f`. The true baseflow law (dashed) runs through the lower part of the cloud: quickflow raises points above it, while noise scatters them both ways.*

### Sloping aquifers

Most hillslope aquifers rest on a sloping base. For a base inclined at angle $\vartheta$, the Dupuit–Forchheimer argument adds a gravity-driven term to Equation (13.3):

\begin{equation*}
\phi_e\,\frac{\partial h}{\partial t} = K\cos\vartheta\,\frac{\partial}{\partial x}\left(h\frac{\partial h}{\partial x}\right) + K\sin\vartheta\,\frac{\partial h}{\partial x} + N ,
\tag{13.8}
\end{equation*}

where $x$ is now measured upslope from the stream along the base. The second term moves water downslope as a **kinematic wave**, the third the familiar diffusion. Scaling $h$ by $D$ and $x$ by $B$, the ratio of the kinematic to the diffusive term is

\begin{equation*}
\frac{K\sin\vartheta\,D/B}{K\cos\vartheta\,D^2/B^2} = \frac{B\tan\vartheta}{D},
\tag{13.9}
\end{equation*}

a **hillslope Péclet number**. For long, steep slopes with thin soils it is large, the diffusive term can be dropped, and the outflow is controlled by the slope rather than the water-table gradient: the **kinematic limit**, analogous to the kinematic wave of Chapter 10. For gentle slopes and thick aquifers it is small and the horizontal solutions apply. Hillslope-storage formulations extend Equation (13.8) to slopes whose width varies along their length, converging or diverging (Troch et al., 2003).

## 13.4 Baseflow decline and groundwater storage change

If an aquifer behaves as a linear reservoir, its drainable storage is proportional to its outflow, $S = kQ$. A trend in the flows of the driest periods, when the river is fed by groundwater alone, then translates into a trend in storage (Brutsaert, 2008):

\begin{equation*}
\frac{dS}{dt} = k\,\frac{dQ_{\text{low}}}{dt} .
\tag{13.10}
\end{equation*}

The method uses only streamflow records, which are often long, and the timescale $k$ from recession analysis. It gives an independent estimate of long-term changes in groundwater storage over areas where wells are sparse. Since 2002, changes in total water storage over large regions have also been measured from space, by the GRACE gravity satellites and their successor GRACE-FO, which sense variations in the Earth's gravity field caused by moving water mass (Tapley et al., 2004). Their resolution is coarse, a few hundred kilometers, and they cannot separate groundwater from soil moisture, snow and surface water without other data, but they have documented major groundwater depletion in irrigated regions.

## 13.5 Regional groundwater in the water cycle

Riparian aquifers are the shallowest part of a deeper system. Tóth (1963) showed how the shape of the water table organizes regional flow. In a basin with a regional slope on which local hills and valleys are superimposed, the head distribution in the aquifer below satisfies Laplace's equation with the water table as its upper boundary. Solving it (here with a cosine series that satisfies no-flow conditions on the sides and the base) shows **nested flow systems** (Figure 13.4): shallow **local** systems from each hill to the adjacent valley, and deeper **intermediate** and **regional** systems that carry water from the main divide to the lowest discharge area. Residence times range from years in local systems to thousands of years in regional ones.

```{code-cell} python
:tags: [hide-input]
Lx, Hd = 6000.0, 1000.0                                            # basin length and depth (m)
xs = np.linspace(0, Lx, 2001)
wt = Hd + 0.02 * xs + 15 * np.sin(2 * np.pi * xs / 1000.0)  # water table: regional slope + local relief
M_ = 200
coef = [np.trapezoid(wt, xs) / Lx] + [2 / Lx * np.trapezoid(wt * np.cos(m * np.pi * xs / Lx), xs) for m in range(1, M_)]
def head(x, z):
    out = coef[0] * np.ones_like(x)
    for m in range(1, M_):
        k = m * np.pi / Lx
        out += coef[m] * np.cos(k * x) * np.exp(k * (z - Hd)) * (1 + np.exp(-2 * k * z)) / (1 + np.exp(-2 * k * Hd))
    return out
X, Z = np.meshgrid(np.linspace(0, Lx, 361), np.linspace(0, Hd, 121))
Hh = head(X, Z)
dHdz, dHdx = np.gradient(Hh, Z[:, 0], X[0])
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.contour(X / 1000, Z, Hh, levels=25, colors="#c9c8c3", linewidths=0.7)
ax.streamplot(X / 1000, Z, -dHdx / 1000, -dHdz, density=(2.2, 1.2), color=c1, linewidth=0.9, arrowsize=0.7)
ax.set_xlim(0, Lx / 1000); ax.set_ylim(0, Hd)
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Height above base (m)")
ax.text(6.0, 1015, "regional divide →", ha="right", va="bottom", color=ink2, fontsize=9.5)
ax.text(0.0, 1015, "← main valley", ha="left", va="bottom", color=ink2, fontsize=9.5)
plt.show()
```

*Figure 13.4. Nested groundwater flow systems in a basin 6 km long and 1 km deep, after Tóth's analysis. The water table (top boundary) falls 120 m toward the main valley at the left and carries local hills 15 m high every 1 km. Gray lines are equipotentials; blue lines show flow paths. The vertical scale is exaggerated about threefold.*

Groundwater is the largest store of liquid fresh water. In humid regions it sustains rivers through dry seasons, and its slow response introduces long memory into catchment behavior (Chapter 15). In dry regions, pumping now exceeds recharge in many major aquifers, lowering water tables and reversing the flow between aquifers and rivers, so that rivers lose water to the ground instead of gaining it.

## 13.6 Lab (planned)

A Colab notebook will extract recession periods from openly available daily streamflow records, build recession plots, estimate $a$, $b$ and the drainage timescale with regression and lower-envelope methods, and test how the results change with the selection rules.

## 13.7 Worked examples

**Example 13.1 — Water-table mound between drains.** Parallel drains 40 m apart keep the water table 0.8 m above an impermeable layer. With $K$ = 1.2 m d⁻¹ and a steady recharge of 5 mm d⁻¹, how high does the water table rise midway between them?

```{code-cell} python
h0, N, K_, L = 0.8, 0.005, 1.2, 40.0
hmax = np.sqrt(h0 ** 2 + N * L ** 2 / (4 * K_))
print(f"h_max = {hmax:.2f} m above the base, {hmax - h0:.2f} m above the drains")
```

**Example 13.2 — Early-time outflow.** An aquifer with $K$ = 8 m d⁻¹, $\phi_e$ = 0.05 and $D$ = 4 m drains into 12 km of stream from both banks. Estimate the baseflow 5 days after a sudden drop of the stream level, while Equation (13.4) applies.

```{code-cell} python
K_, ph, D_, Lst = 8.0, 0.05, 4.0, 12000.0
q1 = 0.332 * np.sqrt(K_ * ph * D_ ** 3 / 5.0)                     # m2/d per m, one bank
print(f"Q = {2 * Lst * q1 / 86400:.3f} m3/s")
```

**Example 13.3 — Storage trend from low flows.** The annual minimum 7-day flow of a river has declined by 0.008 mm d⁻¹ per year over 40 years, and recession analysis gives a drainage timescale of 50 days. Estimate the trend in groundwater storage.

```{code-cell} python
k_, trend = 50.0, -0.008
print(f"dS/dt = {k_ * trend:.2f} mm/yr; over 40 years: {k_ * trend * 40:.0f} mm")
```

## 13.8 Common mistakes

- **Using the porosity instead of the drainable porosity** for the storage change of an unconfined aquifer.
- **Fitting recession parameters to days dominated by quickflow or evaporation.** Select recession days carefully and test the sensitivity.
- **Interpreting $b$ without considering the regime.** The same aquifer can show $b$ = 3 at high flows and 3/2 or 1 at low flows.
- **Applying Dupuit–Forchheimer near the stream or in thick aquifers,** where vertical flow is important.
- **Reading GRACE storage changes as groundwater alone.** They include soil moisture, snow and surface water.

## 13.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("p1", float(np.sqrt(2.0 ** 2 + 0.002 * 100 ** 2 / (4 * 3.0))), display=False)
glue("p2", float(0.332 * np.sqrt(4 * 0.1 * 9 / 2.0)), display=False)
glue("p3", float(1 / (1 / 30)), display=False)
glue("p4", float(40 * 0.12), display=False)
glue("p5", float(1000 * np.tan(np.radians(10)) / 2), display=False)
```

1. Two streams 100 m apart are at 2 m above an aquifer base. With $K$ = 3 m d⁻¹ and recharge 2 mm d⁻¹, find the maximum water-table height.
2. Estimate the early-time outflow per meter of stream (one bank) 2 days after drainage begins, for $K$ = 4 m d⁻¹, $\phi_e$ = 0.1 and $D$ = 3 m.
3. A recession plot at low flows gives $b$ = 1 and $a$ = 1/30 d⁻¹. What is the drainage timescale?
4. With a drainage timescale of 40 days, a decline of 0.12 mm d⁻¹ in low flows corresponds to what decline in storage?
5. A hillslope aquifer is 2 m thick and 1000 m long, on a 10° slope. Compute the hillslope Péclet number (13.9). Which limit applies?

```{admonition} Answers
:class: dropdown
1. $h_{\max}$ = {glue:text}`p1:.2f` m.
2. $Q$ ≈ {glue:text}`p2:.2f` m² d⁻¹.
3. $k$ = {glue:text}`p3:.0f` days.
4. {glue:text}`p4:.1f` mm.
5. About {glue:text}`p5:.0f`; the kinematic limit.
```

## 13.10 Open questions

- **What controls recession behavior?** Catchment recessions often show $b$ values that no single aquifer solution predicts; heterogeneity, the distribution of hillslope lengths and evaporation all play a part.
- **Deep and inter-catchment groundwater.** How much water leaves catchments below the stream gauge, and how much does that bias water balances?
- **Detecting storage change.** How can streamflow-based, well-based and satellite estimates of groundwater storage change be reconciled?

## 13.11 Key papers

- Boussinesq, J. (1904). Recherches théoriques sur l'écoulement des nappes d'eau infiltrées dans le sol et sur le débit des sources. *Journal de Mathématiques Pures et Appliquées*, 5e série, 10, 5–78.
- Brutsaert, W. (2008). Long-term groundwater storage trends estimated from streamflow records: climatic perspective. *Water Resources Research*, 44, W02409.
- Brutsaert, W., and Nieber, J. L. (1977). Regionalized drought flow hydrographs from a mature glaciated plateau. *Water Resources Research*, 13, 637–643.
- Polubarinova-Kochina, P. Ya. (1962). *Theory of Ground Water Movement*. Princeton University Press.
- Tapley, B. D., Bettadpur, S., Ries, J. C., Thompson, P. F., and Watkins, M. M. (2004). GRACE measurements of mass variability in the Earth system. *Science*, 305, 503–505.
- Tóth, J. (1963). A theoretical analysis of groundwater flow in small drainage basins. *Journal of Geophysical Research*, 68, 4795–4812.
- Troch, P. A., Paniconi, C., and van Loon, E. E. (2003). Hillslope-storage Boussinesq model for subsurface flow and variable source areas along complex hillslopes: 1. Formulation and characteristic response. *Water Resources Research*, 39, 1316.

## 13.12 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The chapter on groundwater outflow and base flow.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on groundwater in the hydrologic cycle.
- Freeze, R. A., and Cherry, J. A. (1979). *Groundwater*. Prentice-Hall.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
