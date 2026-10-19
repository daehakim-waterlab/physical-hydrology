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

# 9. Shallow-water flow over land and in channels

```{admonition} Core ideas
:class: tip
1. Water flowing over hillslopes and in channels is **shallow**: its depth is small compared with its horizontal extent. Integrating the equations of motion over the depth gives the **Saint-Venant** (shallow-water) equations for depth and discharge.
2. Friction is represented by an empirical law. **Darcy–Weisbach**, **Chézy** and **Manning** are interchangeable for turbulent flow, and laminar sheet flow under rain follows its own law. Each gives a discharge–depth relation $q = \alpha h^m$, with $m$ = 3 (laminar), 3/2 (Chézy) or 5/3 (Manning).
3. The Saint-Venant equations are **hyperbolic**: disturbances travel along two characteristics at speeds $V \pm \sqrt{gh}$. Flood waves travel instead at the **kinematic celerity** $mV$. When the kinematic celerity exceeds the faster dynamic one, the flow becomes unstable and forms **roll waves**.
4. In most natural floods, gravity and friction dominate the momentum balance. The **kinematic-wave number** and Ponce's criteria tell when the kinematic or diffusion approximations are adequate.
5. Steep channels translate floods almost unchanged; on mild slopes, the pressure-gradient term attenuates them. A numerical solution of the full equations shows both.
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
g = 9.81
```

## 9.1 The depth-averaged equations

### Assumptions

Rain that exceeds the infiltration capacity, and water that emerges from the soil near streams, flows over the surface in thin sheets and small rills before reaching channels (Chapter 14). Over a hillslope tens of meters long, the depth is a few millimeters; in a river kilometers long, a few meters. In both cases the flow is **shallow**, and three assumptions hold well:

- the pressure is **hydrostatic** in the vertical, because vertical accelerations are small;
- the velocity can be represented by its **depth average** $V$, with the non-uniformity absorbed in a friction law;
- the bed slope is small, so that $\sin\theta \approx \tan\theta = S_0$.

### Mass and momentum

Consider a unit width of a wide channel or a plane, with depth $h(x, t)$, discharge per unit width $q = V h$, and a lateral inflow $q_L$ (rain excess, seepage) per unit area. Conservation of mass for a slice of length $dx$ gives

\begin{equation*}
\frac{\partial h}{\partial t} + \frac{\partial q}{\partial x} = q_L .
\tag{9.1}
\end{equation*}

The momentum in the slice changes because of the net momentum flux, the difference in hydrostatic pressure force between its two faces, the weight component along the slope and the bed shear stress $\tau_b = \rho g h S_f$, which defines the **friction slope** $S_f$:

\begin{equation*}
\frac{\partial q}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{q^2}{h} + \frac{g h^2}{2}\right) = g h\,(S_0 - S_f) .
\tag{9.2}
\end{equation*}

The lateral inflow is assumed to enter with no momentum in the flow direction. These are the **Saint-Venant equations** in conservative form, published by Barré de Saint-Venant in 1871 (see [*A Brief History*](https://hydroclimatol.com/water-history/industrial-age-engineering-hydraulics/#pipes-channels-and-groundwater-empirical-hydraulics-matures)). For a channel of arbitrary section, $h$ is replaced by the flow area and $g h^2/2$ by the integral of the hydrostatic pressure over the section. Steady, gradually varied flow, treated in Chapter 6 of the [Hydraulics](https://hydroclimatol.com/hydraulics/chapters/ch06_flow_in_open_channels.html#gradually-varied-flow) book, is the special case without time derivatives.

## 9.2 Friction laws

The friction slope must be related to the flow. Three classical laws are used for turbulent flow, and they are equivalent:

\begin{equation*}
S_f = \frac{f}{8g}\frac{V^2}{h} = \frac{V^2}{C^2 h} = \frac{n^2 V^2}{h^{4/3}},
\qquad C = \sqrt{\frac{8g}{f}} = \frac{h^{1/6}}{n} ,
\tag{9.3}
\end{equation*}

with the Darcy–Weisbach factor $f$, the Chézy coefficient $C$ and Manning's $n$ (for a wide channel, the hydraulic radius equals the depth). Darcy–Weisbach is dimensionally consistent and connects to pipe-flow theory. Manning is the most widely used for channels, with tabulated values of $n$.

### Sheet flow

Thin overland flow on smooth surfaces can be **laminar**, with Reynolds numbers $\text{Re} = 4q/\nu$ of a few hundred. Its friction factor follows $f = K/\text{Re}$, with $K = 24$ for a smooth plane without rain, larger when raindrops disturb the flow, and up to thousands on vegetated surfaces, where most of the resistance comes from stems and litter rather than the bed.

### Discharge–depth relations

In uniform flow, $S_f = S_0$, and each friction law gives a power relation between discharge per unit width and depth:

\begin{equation*}
q = \alpha\,h^m, \qquad
m = 3\ (\text{laminar}), \quad m = \tfrac{3}{2}\ (\text{Chézy}), \quad m = \tfrac{5}{3}\ (\text{Manning}),
\tag{9.4}
\end{equation*}

with $\alpha = 8 g S_0/(K\nu)$, $C S_0^{1/2}$ and $S_0^{1/2}/n$ respectively. The exponent $m$ controls how fast a flood wave travels as the flow grows (Section 9.3) and therefore how nonlinear the catchment response is.

```{code-cell} python
:tags: [hide-input]
S0, nu, K, C_chezy, n_man = 0.02, 1.0e-6, 400.0, 20.0, 0.05
h = np.logspace(-3.5, -1, 200)                                  # m
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for lab, col, alpha, m in [("Laminar (K = 400): m = 3", c3, 8 * g * S0 / (K * nu), 3.0),
                           ("Chézy (C = 20 m½ s⁻¹): m = 3/2", c2, C_chezy * S0 ** 0.5, 1.5),
                           ("Manning (n = 0.05): m = 5/3", c1, S0 ** 0.5 / n_man, 5 / 3)]:
    qq = alpha * h ** m
    ok = (4 * qq / nu < 2000) if m == 3.0 else np.ones_like(h, bool)       # laminar only while Re < 2000
    ax.loglog(h[ok] * 1000, qq[ok] * 1e4, color=col, lw=2.2, label=lab)
ax.set_xlabel("Depth h (mm)"); ax.set_ylabel("q (cm² s⁻¹)"); ax.legend(loc="upper left", fontsize=9.5)
plt.show()
```

*Figure 9.1. Discharge per unit width against depth on a 2% slope for three friction laws. On log–log axes each is a straight line of slope $m$; the laminar law, drawn only up to a Reynolds number of 2000, rises most steeply.*

## 9.3 Characteristics and wave speeds

### Dynamic characteristics

The Saint-Venant equations are a hyperbolic system. Written in terms of $V$ and $h$, they can be combined into two ordinary differential equations along curves in the $(x, t)$ plane, the **characteristics**,

\begin{equation*}
\frac{dx}{dt} = V \pm \sqrt{g h} ,
\tag{9.5}
\end{equation*}

along which the quantities $V \pm 2\sqrt{gh}$ change only because of the source terms (slope, friction, lateral inflow). Small surface disturbances travel at these **dynamic** speeds, upstream and downstream relative to the water. The **Froude number** $F = V/\sqrt{gh}$ tells whether both characteristics point downstream ($F > 1$, supercritical flow) or one points upstream ($F < 1$, subcritical flow, in which downstream conditions such as a dam or a confluence can be felt upstream).

### Kinematic celerity

Flood waves do not travel at either dynamic speed. When friction and gravity dominate, $q \approx \alpha h^m$, and substituting into Equation (9.1) gives the **kinematic-wave** equation

\begin{equation*}
\frac{\partial q}{\partial t} + c_k\,\frac{\partial q}{\partial x} = c_k\,q_L,
\qquad c_k = \frac{dq}{dh} = m\,\alpha h^{m-1} = m\,V ,
\tag{9.6}
\end{equation*}

so the flood wave travels at the **kinematic celerity**, $m$ times the water velocity: 5/3 times faster than the water itself in a wide Manning channel.

### Roll waves

Comparing the two speeds gives a neat stability criterion. A perturbation carried at the kinematic celerity outruns the downstream dynamic characteristic if $mV > V + \sqrt{gh}$, that is if

\begin{equation*}
F > \frac{1}{m - 1} ,
\tag{9.7}
\end{equation*}

or $F > 2$ with Chézy friction and $F > 1.5$ with Manning friction. The dynamic waves can then no longer smooth the flow, and it breaks into a train of **roll waves**, which are seen on steep spillways and paved channels in heavy rain (Figure 9.2).

```{code-cell} python
:tags: [hide-input]
F = np.linspace(0, 3, 300)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(F, F + 1, color=ink, lw=2.2, label="Downstream dynamic: V + √(gh)")
ax.plot(F, F - 1, color=ink2, lw=1.6, ls="--", label="Upstream dynamic: V − √(gh)")
ax.plot(F, 1.5 * F, color=c2, lw=2.2, label="Kinematic, Chézy (m = 3/2)")
ax.plot(F, 5 / 3 * F, color=c1, lw=2.2, label="Kinematic, Manning (m = 5/3)")
ax.axhline(0, color=ink2, lw=0.6)
for Fc, col in [(2.0, c2), (1.5, c1)]:
    ax.axvline(Fc, color=col, lw=0.8, ls=":")
ax.text(2.05, -0.75, "roll waves\n(Chézy)", fontsize=9, color=c2); ax.text(1.55, 3.6, "roll waves\n(Manning)", fontsize=9, color=c1)
ax.set_xlabel("Froude number F"); ax.set_ylabel("Speed / √(gh)")
ax.set_xlim(0, 3); ax.set_ylim(-1.2, 5); ax.legend(loc="upper left", fontsize=9)
plt.show()
```

*Figure 9.2. Wave speeds relative to $\sqrt{gh}$. The kinematic celerity exceeds the downstream dynamic speed beyond F = 2 (Chézy) or F = 1.5 (Manning), where uniform flow becomes unstable to roll waves.*

## 9.4 The hierarchy of approximations

### Scale analysis

Divide Equation (9.2) by $gh$ and compare the terms of the momentum equation:

\begin{equation*}
S_f = S_0 - \underbrace{\frac{\partial h}{\partial x}}_{\text{pressure}} - \underbrace{\frac{V}{g}\frac{\partial V}{\partial x}}_{\text{convective}} - \underbrace{\frac{1}{g}\frac{\partial V}{\partial t}}_{\text{local}} .
\tag{9.8}
\end{equation*}

Keeping all terms gives the **dynamic wave**; dropping the two acceleration terms gives the **diffusion wave**; keeping only $S_f = S_0$ gives the **kinematic wave** (Chapter 10 derives and uses both simplified forms). For a disturbance of length $L$ in flow of normal depth $h_0$ and Froude number $F_0$, the importance of the neglected terms is measured by the **kinematic-wave number** (Woolhiser and Liggett, 1967),

\begin{equation*}
k = \frac{S_0\, L}{h_0\, F_0^2} .
\tag{9.9}
\end{equation*}

For $k \gtrsim 10$–20 the kinematic approximation is accurate. Overland flow on hillslopes almost always satisfies this, because $L$ is many thousands of depths. For flood waves in rivers, Ponce et al. (1978) proposed criteria based on the wave period $T$: the kinematic approximation is adequate when $T S_0 u_0/h_0 \ge 171$, with $u_0$ the mean velocity, and the diffusion approximation when $T S_0 \sqrt{g/h_0} \ge 30$. Steep rivers with long floods satisfy them; mild, large rivers, tidal reaches and rapidly varying flows below dams need the dynamic equations.

### A numerical comparison

Figure 9.3 solves the full Saint-Venant equations with an explicit second-order scheme (MacCormack) for a 12 m² s⁻¹ flood entering a 50 km wide channel. On the steep slope the wave arrives with its peak nearly intact, as the kinematic wave predicts. On the mild slope the pressure gradient across the wave front attenuates the peak substantially and spreads the hydrograph, which the kinematic wave cannot reproduce.

```{code-cell} python
:tags: [hide-input]
def saint_venant(S0, n=0.035, L=50e3, nx=201, Tend=3 * 86400, q0=2.0, qp=12.0, tp=6 * 3600, cfl=0.8):
    """Explicit MacCormack solution of Eqs. (9.1)–(9.2) for a wide channel; returns outlet hydrograph."""
    x = np.linspace(0, L, nx); dx = x[1] - x[0]
    h = np.full(nx, (n * q0 / np.sqrt(S0)) ** 0.6); q = np.full(nx, q0)
    inflow = lambda t: q0 + (qp - q0) * (t / tp) ** 4 * np.exp(4 * (1 - t / tp))
    flux = lambda h, q: (q, q * q / h + 0.5 * g * h * h)
    source = lambda h, q: g * h * (S0 - n * n * q * np.abs(q) / h ** (10 / 3))
    t, T, Qo = 0.0, [], []
    while t < Tend:
        dt = min(cfl * dx / (np.abs(q / h) + np.sqrt(g * h)).max(), Tend - t)
        f1, f2 = flux(h, q)
        hp, qp_ = h.copy(), q.copy()                                         # predictor (forward)
        hp[:-1] = h[:-1] - dt / dx * (f1[1:] - f1[:-1])
        qp_[:-1] = q[:-1] - dt / dx * (f2[1:] - f2[:-1]) + dt * source(h[:-1], q[:-1])
        hp = np.maximum(hp, 1e-3)
        g1, g2 = flux(hp, qp_)
        hn, qn = h.copy(), q.copy()                                          # corrector (backward)
        hn[1:] = 0.5 * (h[1:] + hp[1:] - dt / dx * (g1[1:] - g1[:-1]))
        qn[1:] = 0.5 * (q[1:] + qp_[1:] - dt / dx * (g2[1:] - g2[:-1]) + dt * source(hp[1:], qp_[1:]))
        h, q = np.maximum(hn, 1e-3), qn
        t += dt
        q[0] = inflow(t); h[0] = h[1]                                        # upstream: discharge given
        q[-1], h[-1] = q[-2], h[-2]                                          # downstream: transmissive
        T.append(t); Qo.append(q[-1])
    return np.array(T), np.array(Qo), inflow

fig, ax = plt.subplots(figsize=(6.4, 4.0))
results = {}
tt_in = np.linspace(0, 3 * 86400, 1000)
for S0_c, col in [(0.002, c3), (0.0002, c1)]:
    T, Qo, inflow = saint_venant(S0_c)
    hp_ = (0.035 * 12.0 / np.sqrt(S0_c)) ** 0.6; ck = 5 / 3 * 12.0 / hp_
    ax.plot(T / 3600, Qo, color=col, lw=2.2, label=f"Dynamic, S₀ = {S0_c:g}")
    ax.plot(tt_in / 3600 + 50e3 / ck / 3600, inflow(tt_in), color=col, lw=1.2, ls=":",
            label=f"Kinematic translation, S₀ = {S0_c:g}")
    results[S0_c] = (Qo.max(), T[Qo.argmax()] / 3600, 50e3 / ck / 3600, hp_)
ax.plot(tt_in / 3600, inflow(tt_in), color=ink, lw=1.6, label="Inflow at x = 0")
ax.set_xlabel("Time (h)"); ax.set_ylabel("q (m² s⁻¹)"); ax.set_xlim(0, 40); ax.set_ylim(0, 13.5)
ax.legend(loc="upper right", fontsize=8.5)
plt.show()
for S0_c, key in [(0.002, "steep"), (0.0002, "mild")]:
    glue(f"sv_pk_{key}", float(results[S0_c][0]), display=False)
    glue(f"sv_att_{key}", float(100 * (1 - results[S0_c][0] / 12.0)), display=False)
    hp_ = results[S0_c][3]; Tw = 12 * 3600
    glue(f"ponce_{key}", float(Tw * S0_c * np.sqrt(g / hp_)), display=False)
```

*Figure 9.3. A flood wave entering a 50 km wide channel ($n$ = 0.035), solved with the full Saint-Venant equations (solid) and compared with pure kinematic translation at the celerity of the peak (dotted). On the steep slope the outlet peak is {glue:text}`sv_pk_steep:.1f` m² s⁻¹ ({glue:text}`sv_att_steep:.0f`% attenuation); on the mild slope it is {glue:text}`sv_pk_mild:.1f` m² s⁻¹ ({glue:text}`sv_att_mild:.0f`%). With a wave period of about 12 hours, Ponce's number $T S_0\sqrt{g/h_0}$ is {glue:text}`ponce_steep:.0f` and {glue:text}`ponce_mild:.0f` respectively.*

## 9.5 Worked examples

**Example 9.1 — Converting friction coefficients.** A wide river flows 2.5 m deep with Manning's $n$ = 0.032. Find the equivalent Chézy coefficient and Darcy–Weisbach factor, and the velocity and Froude number on a slope of 0.0004.

```{code-cell} python
hh, n, S0 = 2.5, 0.032, 0.0004
C = hh ** (1 / 6) / n; f = 8 * g / C ** 2
V = C * np.sqrt(hh * S0)
print(f"C = {C:.1f} m^0.5/s, f = {f:.4f}, V = {V:.2f} m/s, F = {V / np.sqrt(g * hh):.2f}, c_k = {5 / 3 * V:.2f} m/s")
```

**Example 9.2 — Laminar sheet flow.** Rain excess of 10 mm h⁻¹ falls on a 40 m long smooth plane with a slope of 0.05 ($K$ = 100, $\nu$ = 1.0 × 10⁻⁶ m² s⁻¹). At equilibrium, find the discharge per unit width, the depth and the Reynolds number at the bottom of the plane.

```{code-cell} python
i_e, Lp, S0p, Kp, nu = 10 / 1000 / 3600, 40.0, 0.05, 100.0, 1.0e-6
q_eq = i_e * Lp
alpha = 8 * g * S0p / (Kp * nu)
h_eq = (q_eq / alpha) ** (1 / 3)
print(f"q = {q_eq * 1e4:.2f} cm²/s, h = {h_eq * 1000:.2f} mm, V = {q_eq / h_eq:.3f} m/s, Re = {4 * q_eq / nu:.0f}")
```

The Reynolds number is below the transition to turbulence (about 500–2000 for sheet flow), so the laminar law applies.

**Example 9.3 — Will roll waves form?** A concrete spillway chute ($n$ = 0.014) on a slope of 0.08 carries 1.5 m² s⁻¹. Check the Froude number in uniform flow against the roll-wave criterion for Manning friction.

```{code-cell} python
q, n, S0 = 1.5, 0.014, 0.08
h_n = (n * q / np.sqrt(S0)) ** 0.6
F_n = q / h_n / np.sqrt(g * h_n)
print(f"h_n = {h_n * 100:.1f} cm, F = {F_n:.2f}; roll waves expected: {F_n > 1.5}")
```

**Example 9.4 — Is the kinematic wave adequate?** Compute the kinematic-wave number for (a) overland flow on a 60 m hillslope with depth 3 mm, slope 0.1 and velocity 0.08 m s⁻¹, and (b) a flood wave 40 km long on a river 4 m deep, slope 0.0003 and velocity 1.5 m s⁻¹.

```{code-cell} python
for name, L, h0, S0, V in [("hillslope", 60.0, 0.003, 0.1, 0.08), ("river", 40e3, 4.0, 0.0003, 1.5)]:
    F0 = V / np.sqrt(g * h0)
    print(f"{name:9s}: F0 = {F0:.2f}, k = {S0 * L / (h0 * F0 ** 2):,.0f}")
```

## 9.6 Common mistakes

- **Confusing the water velocity with the flood-wave celerity.** Flood peaks travel at about $mV$, faster than the water.
- **Using turbulent friction laws for thin sheet flow.** Under light rain on smooth surfaces the flow may be laminar, with a steeper discharge–depth relation.
- **Applying the kinematic wave to mild, large rivers or backwater reaches.** Check the kinematic-wave number or Ponce's criteria first.
- **Treating Manning's $n$ as dimensionless and universal.** It is tied to SI units and to the scale of roughness relative to depth.
- **Explicit numerical schemes with too large a time step.** The time step must satisfy the Courant condition for the fastest characteristic, $V + \sqrt{gh}$.

## 9.7 Problems

```{code-cell} python
:tags: [remove-cell]
glue("x1", float(1.2 ** (1 / 6) / 0.04), display=False)
hn2 = (0.03 * 3.0 / np.sqrt(0.001)) ** 0.6; V2 = 3.0 / hn2
glue("x2_h", float(hn2), display=False); glue("x2_c", float(5 / 3 * V2), display=False)
glue("x2_T", float(20e3 / (5 / 3 * V2) / 3600), display=False)
glue("x3", 1 / (3 - 1), display=False)
glue("x4", float(12 * 3600 * 0.0005 * np.sqrt(g / 3.0)), display=False)
glue("x5", float(4 * 0.5e-4 / 1.0e-6), display=False)
```

1. Find the Chézy coefficient for a wide channel 1.2 m deep with $n$ = 0.04.
2. A wide channel ($n$ = 0.03, $S_0$ = 0.001) carries 3.0 m² s⁻¹. Find the normal depth, the kinematic celerity and the travel time of a flood wave over 20 km.
3. What is the roll-wave criterion for laminar flow?
4. A flood with a period of 12 hours travels down a river with $S_0$ = 0.0005 and depth 3.0 m. Is the kinematic approximation adequate according to Ponce's criterion?
5. Sheet flow carries 0.5 cm² s⁻¹. What is its Reynolds number?

```{admonition} Answers
:class: dropdown
1. $C$ = {glue:text}`x1:.1f` m¹ᐟ² s⁻¹.
2. $h_n$ = {glue:text}`x2_h:.2f` m, $c_k$ = {glue:text}`x2_c:.2f` m s⁻¹, travel time {glue:text}`x2_T:.1f` h.
3. $F$ > {glue:text}`x3:.1f` (with $m$ = 3).
4. $T S_0\sqrt{g/h_0}$ = {glue:text}`x4:.0f` ≥ 30, so the diffusion approximation is adequate. For subcritical flow $u_0 < \sqrt{g h_0}$, so $T S_0 u_0/h_0$ is smaller still and far below 171: the kinematic approximation is not adequate.
5. Re = {glue:text}`x5:.0f`.
```

## 9.8 Open questions

- **Friction on natural hillslopes.** Resistance to overland flow over vegetation and microtopography varies with depth and rain intensity; a general friction law is lacking.
- **Two-dimensional overland flow at catchment scale.** Fine-resolution topography now allows two-dimensional shallow-water simulations of whole catchments; how should infiltration and subgrid microtopography be represented?
- **Numerical diffusion.** Many routing models show attenuation that is partly numerical; separating physical from numerical diffusion matters for model calibration.

## 9.9 Key papers

- Ponce, V. M., Li, R. M., and Simons, D. B. (1978). Applicability of kinematic and diffusion models. *Journal of the Hydraulics Division, ASCE*, 104, 353–360.
- Woolhiser, D. A., and Liggett, J. A. (1967). Unsteady, one-dimensional flow over a plane: the rising hydrograph. *Water Resources Research*, 3, 753–771.
- MacCormack, R. W. (1969). The effect of viscosity in hypervelocity impact cratering. *AIAA Paper* 69-354.

## 9.10 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The chapter on fluid mechanics of free-surface flow.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapters on distributed flow routing.
- Chanson, H. (2004). *The Hydraulics of Open Channel Flow* (2nd ed.). Butterworth-Heinemann.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
