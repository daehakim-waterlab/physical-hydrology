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

# 12. Infiltration and exchange at the soil surface

```{admonition} Core ideas
:class: tip
1. Water entering a soil is pulled in by capillarity and pushed down by gravity. Early on capillarity dominates and the infiltration rate is high; later gravity dominates and the rate approaches the saturated conductivity.
2. Capillary uptake alone gives cumulative infiltration proportional to $t^{1/2}$. The constant of proportionality, the **sorptivity**, can be computed from the soil's hydraulic functions.
3. **Philip's** series and the **Green–Ampt** model are approximate solutions of the Richards equation with simple, physically interpretable parameters. Green–Ampt follows from treating the wetting front as a sharp piston.
4. Under rainfall, the surface ponds only when the rain rate exceeds the soil's infiltrability, and only after a **time to ponding** that depends on the rain rate and the initial wetness.
5. At catchment scale, the spatial variability of conductivity produces runoff at rain rates far below the mean conductivity, and point-scale infiltration equations need **effective** parameters.
6. The same physics runs in reverse during **capillary rise** and **soil drying**: a water table can supply evaporation only above a limiting depth, and bare-soil evaporation slows as $t^{-1/2}$ once the surface dries.
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

## 12.1 The infiltration phenomenon

**Infiltration** is the entry of water into the soil through its surface. Its rate partitions rain and snowmelt between the soil, where it may later evaporate, recharge groundwater or feed slow subsurface flow, and the surface, where it may run off quickly as infiltration-excess overland flow (Chapters 10 and 14). Robert Horton made this partition the basis of a theory of runoff in the 1930s (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#horton-and-the-physics-of-runoff)).

When water is ponded on a soil, the infiltration rate is highest at first and decreases with time toward a nearly constant value. The rate the soil can accept under ponding is called its **infiltrability** or infiltration capacity. A soil profile during ponded infiltration has three parts (Figure 12.1): a nearly saturated **transmission zone** near the surface, a **wetting front** where the water content drops steeply to its initial value, and the undisturbed soil below. The front is sharp because the diffusivity of a dry soil is very small (Example 11.3): water cannot move ahead of the front until the soil there has been wetted.

The decline of the rate has a simple explanation. At first the front is close to the surface, so the suction gradient between the wet surface and the dry soil is very steep and capillarity pulls water in rapidly. As the front moves down, the same suction difference acts over a longer distance, its gradient weakens, and eventually only gravity is left. The rate then approaches the saturated conductivity $K_s$, which is reduced in practice by air trapped in the pores.

All examples in this chapter use an illustrative loam described by van Genuchten–Mualem functions (Chapter 11), with $K_s$ = 10 mm h⁻¹ and an initial pressure head of −5 m. The Richards equation is solved with the same implicit, mass-conservative scheme used in Chapter 11.

```{code-cell} python
:tags: [hide-input]
from scipy.linalg import solve_banded
from scipy.integrate import quad
from scipy.optimize import brentq
tr, ts, a_vg, n_vg = 0.078, 0.43, 3.6, 1.56        # illustrative loam
Ks = 10.0 / 1000 / 3600                           # m/s (10 mm/h)
m_vg = 1 - 1 / n_vg
def theta(h): return np.where(h < 0, tr + (ts - tr) / (1 + (a_vg * np.abs(h)) ** n_vg) ** m_vg, ts)
def K(h):
    Se = np.where(h < 0, (1 + (a_vg * np.abs(h)) ** n_vg) ** (-m_vg), 1.0)
    return Ks * Se ** 0.5 * (1 - (1 - Se ** (1 / m_vg)) ** m_vg) ** 2
def C(h, d=1e-6):
    hp = np.minimum(h + d, 0); return np.where(h < 0, (theta(hp) - theta(h - d)) / (hp - h + d), 1e-9)

def picard_step(h, dt, dz, grav, top, r):
    """One implicit step of the mixed-form Richards equation; z positive downward."""
    nz = len(h); th_old = theta(h); hn = h.copy()
    for it in range(30):
        Kn = K(hn); Kh = 0.5 * (Kn[1:] + Kn[:-1]); Cn = C(hn)
        diag = Cn / dt; rhs = -(theta(hn) - th_old) / dt + Cn * hn / dt
        up = np.zeros(nz); lo = np.zeros(nz)
        diag[1:] += Kh / dz ** 2; up[1:] = -Kh / dz ** 2; rhs[1:] += grav * Kh / dz
        diag[:-1] += Kh / dz ** 2; lo[:-1] = -Kh / dz ** 2; rhs[:-1] -= grav * Kh / dz
        rhs[-1] -= grav * Kn[-1] / dz                                  # free drainage (zero flux if grav = 0)
        if top == "head":                                              # ponded surface, h = 0
            diag[0] += Ks / (dz * dz / 2); rhs[0] += grav * Ks / dz        # upstream (saturated) conductivity
        else:                                                          # prescribed rain flux
            rhs[0] += r / dz
        ab = np.zeros((3, nz)); ab[1] = diag; ab[0, 1:] = lo[:-1]; ab[2, :-1] = up[1:]
        h_new = solve_banded((1, 1), ab, rhs)
        err = np.max(np.abs(h_new - hn)); hn = h_new
        if err < 1e-4:
            q0 = Ks * (-hn[0] / (dz / 2) + grav) if top == "head" else r
            return hn, q0, it
    return None, None, it

def infiltrate(T, hi, mode="ponded", r=None, grav=1.0, Lz=1.0, nz=200, save=()):
    """Cumulative infiltration I(t) for ponding or steady rain at rate r (m/s)."""
    dz = Lz / nz; h = np.full(nz, float(hi)); t, dt, I = 0.0, 0.5, 0.0
    tt, II, prof, tp = [0.0], [0.0], {}, None
    save = list(save)
    while t < T - 1e-9:
        dt = min(dt, T - t, (save[0] - t) if save else T)
        hn, q0, it = picard_step(h, dt, dz, grav, "head", r)          # try a ponded surface
        if mode == "rain" and hn is not None:
            if q0 > r: hn, q0, it = picard_step(h, dt, dz, grav, "flux", r)   # soil takes all the rain
            elif tp is None: tp = t                                    # infiltrability has fallen below r
        if hn is None: dt *= 0.5; continue
        h, I, t = hn, I + q0 * dt, t + dt; tt.append(t); II.append(I)
        if save and abs(t - save[0]) < 1e-6: prof[save.pop(0) / 3600] = theta(h).copy()
        dt = min(dt * 1.3, 300.0) if it < 4 else dt * 0.7
    return np.array(tt), np.array(II), (np.arange(nz) + 0.5) * dz, prof, tp

hi = -5.0; th_i = float(theta(np.array(hi)))
t_p, I_p, zc, prof, _ = infiltrate(6 * 3600, hi, save=[900, 3600, 3 * 3600, 6 * 3600])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for (hr, th), col in zip(prof.items(), [c2, c1, c3, ink]):
    ax.plot(th, zc * 100, color=col, lw=2.0, label=f"{hr:g} h")
ax.axvline(th_i, color=ink2, lw=1, ls=":"); ax.text(th_i + 0.005, 95, "initial θ", color=ink2, fontsize=9.5)
ax.set_ylim(100, 0); ax.set_xlabel("θ"); ax.set_ylabel("Depth (cm)")
ax.legend(loc="lower right", fontsize=9.5, title="Time since ponding", title_fontsize=9.5)
plt.show()
fronts = {hr: float(zc[np.argmax(th < th_i + 0.5 * (ts - th_i))] * 100) for hr, th in prof.items()}
glue("front_1", fronts[1.0], display=False); glue("front_6", fronts[6.0], display=False)
```

*Figure 12.1. Water-content profiles during ponded infiltration into the illustrative loam, from a numerical solution of the Richards equation. The wetting front, taken where θ is halfway between its initial and saturated values, is at {glue:text}`front_1:.0f` cm after 1 h and {glue:text}`front_6:.0f` cm after 6 h. The front slows down, but the shape of the profile behind it stays nearly the same.*

## 12.2 Sorption and sorptivity

Imagine water entering a soil column laid horizontally, so that gravity plays no role. This process is called **sorption**. With the surface held at saturation, the water-content form of the Richards equation reduces to nonlinear diffusion,

\begin{equation*}
\frac{\partial\theta}{\partial t} = \frac{\partial}{\partial x}\left[D(\theta)\frac{\partial\theta}{\partial x}\right], \qquad \theta(0, t) = \theta_s, \quad \theta(x, 0) = \theta_i .
\tag{12.1}
\end{equation*}

The problem has no length scale of its own: the only way to combine $x$ and $t$ with $D$ is through $x/\sqrt{t}$. If we look for a solution that depends on the single variable $\lambda = x\,t^{-1/2}$ (the **Boltzmann transformation**), Equation (12.1) becomes an ordinary differential equation in $\lambda$, and the profile at any time is a stretched copy of the profile at any other. The cumulative sorption is then

\begin{equation*}
I = \int_0^\infty (\theta - \theta_i)\,dx = t^{1/2}\int_0^\infty (\theta - \theta_i)\,d\lambda = S\,t^{1/2} ,
\tag{12.2}
\end{equation*}

and the rate $i = \tfrac{1}{2} S\,t^{-1/2}$. The constant $S$ is the **sorptivity**, a property of the soil and of its initial and surface water contents. It is measured directly with short ponded or tension infiltrometer tests, and it can be computed from the diffusivity. A simple, accurate approximation (Parlange, 1975) is

\begin{equation*}
S^2 \approx \int_{\theta_i}^{\theta_s} (\theta_s + \theta - 2\theta_i)\,D(\theta)\,d\theta = \int_{\psi_i}^{0} (\theta_s + \theta - 2\theta_i)\,K(\psi)\,d\psi ,
\tag{12.3}
\end{equation*}

where the second form uses $D\,d\theta = K\,d\psi$ and avoids the diffusivity's steep rise near saturation.

```{code-cell} python
:tags: [hide-input]
S2 = quad(lambda hh: (ts + float(theta(np.array(hh))) - 2 * th_i) * float(K(np.array(hh))), hi, 0, limit=200)[0]
S = np.sqrt(S2)                                                     # m s^-1/2
t_h, I_h, *_ = infiltrate(6 * 3600, hi, grav=0.0, Lz=0.4)           # horizontal column
S_mmh = S * 1000 * 60                                               # mm h^-1/2
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(np.sqrt(t_h / 3600), I_h * 1000, color=c1, lw=2.4, label="Richards equation, horizontal column")
sq = np.linspace(0, np.sqrt(6), 50)
ax.plot(sq, S_mmh * sq, color=c2, lw=1.8, ls="--", label="$S\\,t^{1/2}$ with $S$ from Eq. (12.3)")
ax.set_xlabel("$t^{1/2}$ (h$^{1/2}$)"); ax.set_ylabel("Cumulative sorption (mm)")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
S_num = float(I_h[-1] * 1000 / np.sqrt(6))
glue("S_par", float(S_mmh), display=False); glue("S_num", S_num, display=False)
glue("S_err", float(100 * abs(S_num - S_mmh) / S_num), display=False)
```

*Figure 12.2. Sorption into a horizontal column of the illustrative loam. Plotted against $t^{1/2}$, the numerical solution is a straight line whose slope is the sorptivity, {glue:text}`S_num:.1f` mm h⁻¹ᐟ². Parlange's approximation (12.3) gives {glue:text}`S_par:.1f` mm h⁻¹ᐟ², an error of {glue:text}`S_err:.1f`%.*

The same scaling holds at the start of vertical infiltration, when the front is so close to the surface that gravity is negligible next to the capillary gradient. This is why every physically based infiltration equation behaves as $S\,t^{1/2}$ at short times.

## 12.3 Infiltration with gravity

### Philip's series

Including gravity breaks the similarity. Philip (1957) solved the vertical problem as a power series in $t^{1/2}$, whose first two terms are

\begin{equation*}
I = S\,t^{1/2} + A\,t ,
\tag{12.4}
\end{equation*}

with $A$ a constant that typically lies between about $K_s/3$ and $2K_s/3$, depending on the shape of the soil's diffusivity function. The two-term form is accurate at short and intermediate times. It cannot be right at long times, where the rate should tend to $K_s$, but in practice $A$ is often fitted to data over the time range of interest.

### Green–Ampt

A different route, older and more widely used, replaces the real profile by a **piston**: saturated soil at $\theta_s$ behind a sharp front at depth $Z_f$, initial soil at $\theta_i$ ahead of it (Green and Ampt, 1911). At the front, the soil exerts a fixed suction $\psi_f$, the **wetting-front suction**. With ponding of negligible depth at the surface, Darcy's law across the wet zone gives the infiltration rate

\begin{equation*}
f = K_s\,\frac{Z_f + \psi_f}{Z_f} = K_s\left(1 + \frac{\psi_f\,\Delta\theta}{I}\right), \qquad I = \Delta\theta\,Z_f, \quad \Delta\theta = \theta_s - \theta_i .
\tag{12.5}
\end{equation*}

Writing $M = \psi_f\Delta\theta$ and $f = dI/dt$, Equation (12.5) can be separated as $I\,dI/(I + M) = K_s\,dt$, and integration from $I = 0$ at $t = 0$ gives the implicit solution

\begin{equation*}
I - M\ln\left(1 + \frac{I}{M}\right) = K_s\,t .
\tag{12.6}
\end{equation*}

At short times, expanding the logarithm gives $I \approx \sqrt{2K_sMt} + \tfrac{2}{3}K_s t$: a sorption law with $S^2 = 2K_s\Delta\theta\,\psi_f$, plus a gravity term with $A = 2K_s/3$, the upper end of Philip's range. At long times $f \to K_s$. The model therefore has the right behavior at both ends, and its parameters have a physical meaning. A consistent choice of the front suction is the one that reproduces the sorptivity; to a good approximation this is the integral of the relative conductivity over the suction range, $\psi_f \approx \int_{\psi_i}^0 K_r\,d\psi$, sometimes called the **capillary drive**.

Many other infiltration equations exist. Some are physically based approximations of the Richards equation; others, such as Horton's exponential decay of the rate toward a constant, are empirical curves. The [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch05_soil_water_infiltration_and_groundwater.html) book uses Green–Ampt and Horton for design calculations. Figure 12.3 compares Green–Ampt and Philip's two-term form with the numerical solution.

```{code-cell} python
:tags: [hide-input]
dth = ts - th_i
psi_f = quad(lambda hh: float(K(np.array(hh))) / Ks, hi, 0)[0]       # capillary drive (m)
M = psi_f * dth
def ga_I(t):                                                         # Green-Ampt, Eq. (12.6)
    return np.array([brentq(lambda I: I - M * np.log1p(I / M) - Ks * tt, 1e-12, 10) if tt > 0 else 0.0 for tt in np.atleast_1d(t)])
mask = t_p > 60
A_fit = float(np.sum((I_p[mask] - S * np.sqrt(t_p[mask])) * t_p[mask]) / np.sum(t_p[mask] ** 2))
tg = np.linspace(1, 6 * 3600, 400)
I_ga, I_ph = ga_I(tg), S * np.sqrt(tg) + A_fit * tg
f_num = np.gradient(I_p, t_p)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.8))
a1.plot(t_p / 3600, I_p * 1000, color=c1, lw=2.6, label="Richards")
a1.plot(tg / 3600, I_ga * 1000, color=c2, lw=1.8, ls="--", label="Green–Ampt")
a1.plot(tg / 3600, I_ph * 1000, color=c3, lw=1.8, ls=":", label="Philip, two terms")
a1.set_xlabel("Time (h)"); a1.set_ylabel("I (mm)"); a1.set_title("(a) Cumulative")
a2.loglog(t_p[1:] / 3600, f_num[1:] * 3.6e6, color=c1, lw=2.6)
a2.loglog(tg / 3600, np.gradient(I_ga, tg) * 3.6e6, color=c2, lw=1.8, ls="--")
a2.loglog(tg / 3600, (0.5 * S / np.sqrt(tg) + A_fit) * 3.6e6, color=c3, lw=1.8, ls=":")
a2.axhline(Ks * 3.6e6, color=ink2, lw=1, ls="-."); a2.text(0.012, Ks * 3.6e6 * 1.12, "$K_s$", color=ink2)
a2.set_xlim(0.01, 6); a2.set_ylim(5, 500)
a2.set_xlabel("Time (h)"); a2.set_ylabel("f (mm h⁻¹)"); a2.set_title("(b) Rate")
a1.legend(loc="upper left", fontsize=9)
plt.show()
glue("psi_f_cm", float(psi_f * 100), display=False); glue("dth", float(dth), display=False)
glue("A_ratio", float(A_fit / Ks), display=False)
glue("I6_num", float(I_p[-1] * 1000), display=False); glue("I6_ga", float(ga_I(6 * 3600)[0] * 1000), display=False)
glue("I6_over", float(100 * (ga_I(6 * 3600)[0] / I_p[-1] - 1)), display=False)
```

*Figure 12.3. Ponded infiltration into the illustrative loam: numerical solution, Green–Ampt with $\psi_f$ = {glue:text}`psi_f_cm:.1f` cm (the capillary drive) and $\Delta\theta$ = {glue:text}`dth:.3f`, and Philip's two-term equation with the sorptivity of Figure 12.2 and $A$ fitted by least squares, $A$ = {glue:text}`A_ratio:.2f` $K_s$. After 6 h the numerical solution gives {glue:text}`I6_num:.0f` mm and Green–Ampt {glue:text}`I6_ga:.0f` mm.*

Philip's equation with a fitted $A$ follows the numerical solution closely over the 6 hours. Green–Ampt matches the first minutes, because its front suction was chosen to reproduce the sorptivity, but then overestimates infiltration, by {glue:text}`I6_over:.0f`% after 6 h. The reason is the piston assumption. Green–Ampt treats the soil as if its diffusivity were concentrated at saturation, which gives the largest possible gravity term, $A = 2K_s/3$. In the loam, the profile behind the front is not quite saturated (Figure 12.1) and its rate approaches $K_s$ faster than $K_s(1 + M/I)$ allows. Green–Ampt is closest to the truth for coarse soils with sharp fronts. For other soils its parameters are effective values, and in practice $\psi_f$, $\Delta\theta$ or $K_s$ are often reduced to fit observed infiltration.

## 12.4 Infiltration under rainfall

Under rain of intensity $r$, the soil surface is not saturated at first. If $r < K_s$ it never will be: all rain infiltrates, and the profile approaches a water content at which $K(\theta) = r$. If $r > K_s$, all rain still infiltrates at first, because a dry soil can absorb water faster than it falls. The surface water content rises until it reaches saturation; from that moment, the **time to ponding** $t_p$, the soil's infiltrability is less than the rain rate and the excess starts to pond or run off.

The Green–Ampt model gives a simple estimate (Mein and Larson, 1973). Before ponding, $I = r\,t$. Ponding occurs when the infiltrability (12.5) has fallen to $r$, that is, when

\begin{equation*}
I_p = \frac{M}{r/K_s - 1}, \qquad t_p = \frac{I_p}{r} .
\tag{12.7}
\end{equation*}

After ponding, infiltration proceeds as if the soil had been ponded from the start, but with a time shift: the soil behaves as though it had already been infiltrating for the time $t_p'$ that Equation (12.6) needs to reach $I_p$ under ponding. This **time-compression** idea, $I(t) = I_{\text{ponded}}(t - t_p + t_p')$, is approximate, but it is simple and it is the basis of most infiltration routines in event runoff models.

```{code-cell} python
:tags: [hide-input]
rates = [15, 30, 60]                                                 # mm/h
fig, ax = plt.subplots(figsize=(6.4, 3.9))
tp_rows = []
for rr, col in zip(rates, [c3, c1, c2]):
    r = rr / 1000 / 3600
    t_r, I_r, *_, tp_num = infiltrate(3 * 3600, hi, mode="rain", r=r)
    ax.plot(t_r[1:] / 3600, np.diff(I_r) / np.diff(t_r) * 3.6e6, color=col, lw=2.2, label=f"r = {rr} mm h⁻¹")
    Ip = M / (r / Ks - 1); tp = Ip / r
    tp_star = (Ip - M * np.log1p(Ip / M)) / Ks
    tg = np.linspace(tp, 3 * 3600, 200)
    f_ga = Ks * (1 + M / ga_I(tg - tp + tp_star))
    ax.plot(np.r_[0, tp, tg] / 3600, np.r_[rr, rr, f_ga * 3.6e6], color=col, lw=1.4, ls="--")
    tp_rows.append((rr, tp_num / 60, tp / 60))
ax.set_xlabel("Time since start of rain (h)"); ax.set_ylabel("Infiltration rate (mm h⁻¹)")
ax.set_ylim(0, 65); ax.legend(loc="upper right", fontsize=9.5)
ax.text(1.6, 22, "solid: Richards\ndashed: Green–Ampt, Mein–Larson", color=ink2, fontsize=9.5)
plt.show()
for rr, a_, b_ in tp_rows: print(f"r = {rr:2d} mm/h: time to ponding {a_:5.1f} min (Richards), {b_:5.1f} min (Green–Ampt)")
glue("tp30_num", float(tp_rows[1][1]), display=False); glue("tp30_ga", float(tp_rows[1][2]), display=False)
glue("tp15_ratio", float(tp_rows[0][2] / tp_rows[0][1]), display=False)
```

*Figure 12.4. Infiltration rate under steady rain on the illustrative loam for three rain intensities. Before ponding the rate equals the rain rate; after ponding it follows the soil's declining infiltrability. For 30 mm h⁻¹, ponding occurs after {glue:text}`tp30_num:.0f` min in the numerical solution and {glue:text}`tp30_ga:.0f` min with Green–Ampt. For 15 mm h⁻¹, only slightly above $K_s$, Green–Ampt delays ponding by a factor of about {glue:text}`tp15_ratio:.0f`, because it overestimates the late infiltrability (Figure 12.3).*

## 12.5 Catchment-scale losses and spatial variability

The equations above describe a point. Over a hillslope or catchment, $K_s$ varies by an order of magnitude or more over short distances, with an approximately lognormal distribution (Chapter 11). Two consequences follow.

First, the area-average loss is not given by the point equation with the average parameters. Under steady rain at rate $r$ after ponding, each point infiltrates at about its own $K_s$, if $K_s < r$, or accepts all the rain, if $K_s \ge r$. With a lognormal $K_s$, the areal runoff rate is

\begin{equation*}
\bar R(r) = \int_0^r (r - K_s)\,p(K_s)\,dK_s = r\,\Phi(d) - e^{\mu + \sigma^2/2}\,\Phi(d - \sigma), \qquad d = \frac{\ln r - \mu}{\sigma},
\tag{12.8}
\end{equation*}

where $\mu$ and $\sigma$ are the mean and standard deviation of $\ln K_s$ and $\Phi$ the standard normal distribution function. Figure 12.5 shows that a heterogeneous surface produces runoff at rain rates well below its mean $K_s$, and that the runoff coefficient rises gradually rather than abruptly with rain intensity. A single "effective" $K_s$ calibrated for one storm intensity will therefore be wrong for another.

Second, runoff from a low-conductivity patch can infiltrate further downslope on a high-conductivity patch (**run-on**), so the hillslope loss also depends on the spatial arrangement of the patches and on the flow paths, not only on their distribution. Equation (12.8) ignores run-on and so overestimates runoff.

Event runoff models often bypass these issues with empirical loss functions, such as an initial loss plus a constant rate, or the curve-number method of the [Engineering Hydrology](https://hydroclimatol.com/engineering-hydrology/chapters/ch07_excess_rainfall.html) book. Their parameters are effective catchment values that absorb the effects of heterogeneity, run-on and storm pattern.

```{code-cell} python
:tags: [hide-input]
from scipy.stats import norm
Kmean = 10.0                                                         # mm/h, arithmetic mean
rr = np.linspace(0.1, 60, 400)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(rr, np.maximum(rr - Kmean, 0) / rr, color=ink, lw=1.6, ls="--", label="Uniform, $K_s$ = 10 mm h⁻¹")
for cv, col in zip([0.5, 1.0, 2.0], [c1, c3, c2]):
    sig = np.sqrt(np.log(1 + cv ** 2)); mu = np.log(Kmean) - sig ** 2 / 2
    d = (np.log(rr) - mu) / sig
    R = rr * norm.cdf(d) - np.exp(mu + sig ** 2 / 2) * norm.cdf(d - sig)
    ax.plot(rr, R / rr, color=col, lw=2.2, label=f"Lognormal, CV = {cv:g}")
    if cv == 1.0: glue("rc_at5", float(np.interp(5, rr, R / rr)), display=False)
ax.set_xlabel("Rain rate r (mm h⁻¹)"); ax.set_ylabel("Runoff coefficient R/r")
ax.set_ylim(0, 1); ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 12.5. Steady-state runoff coefficient of a surface whose saturated conductivity is uniform or lognormally distributed with the same mean (10 mm h⁻¹), from Equation (12.8), ignoring run-on. With a coefficient of variation of 1, rain at only half the mean conductivity already produces a runoff coefficient of {glue:text}`rc_at5:.2f`.*

## 12.6 Capillary rise, bare-soil evaporation and soil drying

### Capillary rise from a water table

Water can also move upward, from a shallow water table toward an evaporating surface. In steady state the upward flux $q$ is the same at every height $z$ above the water table, and Darcy's law gives $q = -K(\psi)\,(d\psi/dz + 1)$, so that

\begin{equation*}
dz = -\frac{K(\psi)}{K(\psi) + q}\,d\psi .
\tag{12.9}
\end{equation*}

Integrating from the water table ($\psi = 0$) to a very dry surface ($\psi \to -\infty$) gives the greatest depth from which a flux $q$ can be sustained. Gardner (1958) did this for an exponential conductivity function, $K = K_s e^{\alpha_G\psi}$, for which the integral is elementary: substituting $u = e^{\alpha_G\psi}$,

\begin{equation*}
L = \frac{1}{\alpha_G}\int_0^1 \frac{K_s}{K_s u + q}\,du = \frac{1}{\alpha_G}\ln\left(1 + \frac{K_s}{q}\right)
\quad\Longrightarrow\quad
q_{\max} = \frac{K_s}{e^{\alpha_G L} - 1} .
\tag{12.10}
\end{equation*}

The maximum flux falls exponentially with the water-table depth $L$. However strong the evaporative demand, a soil cannot deliver more than $q_{\max}$ from a water table at depth $L$, and the surface dries out. Coarse soils, with large $\alpha_G$, supply high fluxes from shallow water tables but almost nothing from deep ones; fine soils supply less but from greater depths (Figure 12.6a). Capillary rise sustains evaporation from wetlands, riparian zones and irrigated land with shallow groundwater, and it is responsible for salt accumulation at the surface in dry climates.

### Drying of a bare soil

Without a water table, a wet bare soil dries in two stages. At first, evaporation proceeds at the rate set by the available energy and the air (**stage 1**, Chapters 7 and 8), because the soil can supply water to the surface faster than the atmosphere removes it. As the surface layer dries, its conductivity collapses and the soil can no longer keep up. In **stage 2** the rate is controlled by the soil, by **desorption**, the reverse of sorption: by the argument of Section 12.2, the cumulative evaporation grows as $E = S_d\,t^{1/2}$, where $S_d$ is the **desorptivity**. This $t^{1/2}$ law has been observed in many field and laboratory studies (for example Black et al., 1969).

A simple way to combine the two stages is to note that during desorption the rate depends on how much has already evaporated: from $E = S_d t^{1/2}$, the rate is $e = S_d^2/(2E)$. The actual rate is then the smaller of the potential rate and this soil-limited rate,

\begin{equation*}
e = \min\left(E_p,\ \frac{S_d^2}{2E}\right),
\tag{12.11}
\end{equation*}

so stage 1 ends when the cumulative evaporation reaches $S_d^2/(2E_p)$ (Figure 12.6b). Stage-2 evaporation is slow, which is why a dry surface layer of a few centimeters acts as a mulch protecting the water below, and why the land-surface schemes of Chapter 8 need a soil resistance that grows as the surface dries.

```{code-cell} python
:tags: [hide-input]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.9))
L = np.linspace(0.2, 3.0, 300)
for (lab, Kd, aG), col in zip([("Sand", 1000, 6.0), ("Loam", 250, 2.5), ("Clay", 50, 1.0)], [c2, c1, c3]):
    a1.semilogy(L, Kd / (np.exp(aG * L) - 1), color=col, lw=2.2, label=lab)
a1.axhline(5, color=ink2, lw=1, ls="--"); a1.text(2.0, 6.2, "$E_p$ = 5 mm d⁻¹", color=ink2, fontsize=9.5)
a1.set_ylim(0.01, 1e3); a1.set_xlabel("Water-table depth L (m)"); a1.set_ylabel("$q_{max}$ (mm d⁻¹)")
a1.set_title("(a) Capillary rise"); a1.legend(loc="upper right", fontsize=9)
Ep, days = 5.0, np.linspace(0, 20, 2001); dtd = days[1] - days[0]
for Sd, col in zip([6.0, 3.0], [c1, c2]):
    E = np.zeros_like(days)
    for k in range(1, len(days)):
        rate = Ep if E[k - 1] <= 0 else min(Ep, Sd ** 2 / (2 * E[k - 1]))
        E[k] = E[k - 1] + rate * dtd
    a2.plot(days, E, color=col, lw=2.2, label=f"$S_d$ = {Sd:g} mm d⁻¹ᐟ²")
    if Sd == 3.0: glue("E20_low", float(E[-1]), display=False)
    if Sd == 6.0: glue("E20_high", float(E[-1]), display=False)
a2.plot(days, Ep * days, color=ink2, lw=1, ls="--", label="Potential")
a2.set_ylim(0, 60); a2.set_xlabel("Days of drying"); a2.set_ylabel("Cumulative E (mm)")
a2.set_title("(b) Bare-soil drying"); a2.legend(loc="upper left", fontsize=9)
plt.show()
```

*Figure 12.6. (a) Maximum steady capillary rise from a water table, Equation (12.10), for three illustrative soils described by Gardner's exponential function. (b) Cumulative evaporation from a drying bare soil, Equation (12.11), with $E_p$ = 5 mm d⁻¹. In 20 days, the soil with $S_d$ = 3 mm d⁻¹ᐟ² loses {glue:text}`E20_low:.0f` mm and the one with $S_d$ = 6 mm d⁻¹ᐟ² {glue:text}`E20_high:.0f` mm, against 100 mm of potential evaporation. The parameter values are illustrative.*

## 12.7 Worked examples

**Example 12.1 — Green–Ampt infiltration.** A soil has $K_s$ = 12 mm h⁻¹, a wetting-front suction of 110 mm and $\Delta\theta$ = 0.30. Find the cumulative infiltration and the infiltration rate after 1 h and 3 h of ponding.

```{code-cell} python
Ks_e, psi_e, dth_e = 12.0, 110.0, 0.30
M_e = psi_e * dth_e
for t in (1.0, 3.0):
    I = brentq(lambda I: I - M_e * np.log1p(I / M_e) - Ks_e * t, 1e-9, 1e4)
    print(f"t = {t:.0f} h: I = {I:5.1f} mm, f = {Ks_e * (1 + M_e / I):5.1f} mm/h")
```

**Example 12.2 — Time to ponding.** For the soil of Example 12.1, how long does rain of 40 mm h⁻¹ take to pond the surface, and how much has infiltrated by then? What if the rain is 15 mm h⁻¹?

```{code-cell} python
for r in (40.0, 15.0):
    Ip = M_e / (r / Ks_e - 1)
    print(f"r = {r:.0f} mm/h: I_p = {Ip:5.1f} mm, t_p = {Ip / r * 60:5.1f} min")
```

The weaker rain ponds much later, because it exceeds $K_s$ by only a small margin.

**Example 12.3 — Sorptivity from Green–Ampt.** What sorptivity does the soil of Example 12.1 imply, and how much water does it absorb in the first 5 minutes?

```{code-cell} python
S_e = np.sqrt(2 * Ks_e * M_e)                                         # mm h^-1/2
print(f"S = {S_e:.1f} mm/h^0.5; I(5 min) ≈ {S_e * np.sqrt(5 / 60):.1f} mm")
```

**Example 12.4 — Capillary supply.** A loam with $K_s$ = 250 mm d⁻¹ and $\alpha_G$ = 2.5 m⁻¹ lies above a water table. Below what depth can capillary rise no longer meet an evaporative demand of 4 mm d⁻¹?

```{code-cell} python
Kd, aG, q = 250.0, 2.5, 4.0
print(f"L = {np.log(1 + Kd / q) / aG:.2f} m")
```

## 12.8 Common mistakes

- **Comparing rain intensity with $K_s$ alone.** Early in a storm a dry soil absorbs water far faster than $K_s$; ponding requires both $r > K_s$ and enough cumulative infiltration (Equation 12.7).
- **Using Philip's two-term equation for long times.** It does not tend to $K_s$; fit $A$ over the time range of interest.
- **Choosing Green–Ampt parameters independently.** $\psi_f$, $\Delta\theta$ and $K_s$ together determine the sorptivity; inconsistent values give wrong early-time behavior.
- **Applying point equations with mean parameters to a catchment.** Heterogeneity produces runoff at rates below the mean $K_s$ (Figure 12.5).
- **Assuming a shallow water table always supplies evaporation.** The supply falls exponentially with depth (Equation 12.10).

## 12.9 Problems

```{code-cell} python
:tags: [remove-cell]
glue("q1", float(brentq(lambda I: I - 40 * np.log1p(I / 40) - 8 * 2, 1e-9, 1e4)), display=False)
glue("q2", float(np.sqrt(2 * 5 * 0.25 * 200)), display=False)
glue("q3a", float(80 * 0.35 / (25 / 10 - 1)), display=False); glue("q3b", float(80 * 0.35 / (25 / 10 - 1) / 25 * 60), display=False)
glue("q4", float(500 / (np.exp(4 * 1.5) - 1)), display=False)
glue("q5", float(4 ** 2 / (2 * 6)), display=False)
```

1. A soil has $K_s$ = 8 mm h⁻¹ and $M = \psi_f\Delta\theta$ = 40 mm. Find the Green–Ampt cumulative infiltration after 2 h of ponding.
2. Estimate the sorptivity implied by Green–Ampt for $K_s$ = 5 mm h⁻¹, $\Delta\theta$ = 0.25 and $\psi_f$ = 200 mm.
3. A soil with $K_s$ = 10 mm h⁻¹, $\psi_f$ = 80 mm and $\Delta\theta$ = 0.35 receives rain at 25 mm h⁻¹. Find $I_p$ and $t_p$.
4. With Gardner's function, $K_s$ = 500 mm d⁻¹ and $\alpha_G$ = 4 m⁻¹, what is the maximum capillary rise from a water table at 1.5 m?
5. A bare soil has a desorptivity of 4 mm d⁻¹ᐟ² and a potential evaporation of 6 mm d⁻¹. How much evaporates before stage 2 begins?

```{admonition} Answers
:class: dropdown
1. $I$ ≈ {glue:text}`q1:.1f` mm.
2. $S$ ≈ {glue:text}`q2:.1f` mm h⁻¹ᐟ².
3. $I_p$ = {glue:text}`q3a:.1f` mm, $t_p$ = {glue:text}`q3b:.1f` min.
4. $q_{\max}$ ≈ {glue:text}`q4:.2f` mm d⁻¹.
5. $S_d^2/(2E_p)$ = {glue:text}`q5:.2f` mm.
```

## 12.10 Open questions

- **Infiltration in structured soils.** Macropores can carry water to depth before the matrix wets; how should infiltration models represent this two-domain behavior at field scale?
- **Effective parameters.** How do hillslope-scale infiltration parameters relate to point measurements, and can they be predicted rather than calibrated?
- **Surface sealing and water repellency.** Raindrop impact, fire and drought can change the soil surface during or between storms, making infiltration properties time-dependent.

## 12.11 Key papers

- Black, T. A., Gardner, W. R., and Thurtell, G. W. (1969). The prediction of evaporation, drainage, and soil water storage for a bare soil. *Soil Science Society of America Proceedings*, 33, 655–660.
- Gardner, W. R. (1958). Some steady-state solutions of the unsaturated moisture flow equation with application to evaporation from a water table. *Soil Science*, 85, 228–232.
- Green, W. H., and Ampt, G. A. (1911). Studies on soil physics. 1. The flow of air and water through soils. *Journal of Agricultural Science*, 4, 1–24.
- Mein, R. G., and Larson, C. L. (1973). Modeling infiltration during a steady rain. *Water Resources Research*, 9, 384–394.
- Parlange, J.-Y. (1975). On solving the flow equation in unsaturated soils by optimization: horizontal infiltration. *Soil Science Society of America Proceedings*, 39, 415–418.
- Philip, J. R. (1957). The theory of infiltration: 4. Sorptivity and algebraic infiltration equations. *Soil Science*, 84, 257–264.

## 12.12 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The chapter on infiltration and related unsaturated flows.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on infiltration and water movement in soils.
- Smith, R. E. (2002). *Infiltration Theory for Hydrologic Applications*. American Geophysical Union.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
