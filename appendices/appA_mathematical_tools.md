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

# Appendix A. Mathematical tools

This appendix collects mathematical results used repeatedly in the book, with short derivations and pointers to the chapters where they appear.

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

## A.1 Differentiation of integrals

Conservation laws for a layer whose boundaries move, such as a water column with a free surface (Chapter 9) or a soil layer with a moving wetting front (Chapter 12), require differentiating an integral whose limits depend on the variable of differentiation. **Leibniz's rule** states that

\begin{equation*}
\frac{\partial}{\partial x}\int_{a(x)}^{b(x)} f(x, z)\,dz = \int_{a}^{b} \frac{\partial f}{\partial x}\,dz + f(x, b)\,\frac{\partial b}{\partial x} - f(x, a)\,\frac{\partial a}{\partial x} .
\tag{A.1}
\end{equation*}

The boundary terms account for the change of the integration interval. The depth-averaged equations of Chapter 9 follow from integrating the three-dimensional equations over the depth of flow and applying (A.1): with the kinematic conditions at the bed and the free surface, the boundary terms of the continuity equation combine into the rate of change of depth.

The three-dimensional analog is the **Reynolds transport theorem**: the rate of change of a quantity in a moving control volume equals its local rate of change plus the flux through the moving boundary. Every budget equation in the book (water, energy, momentum, solutes) is an application of it.

## A.2 Similarity solutions

Several problems in the book have no intrinsic length or time scale: sorption into a semi-infinite soil (Chapter 12), the early stage of aquifer drainage (Chapter 13), diffusion from a sudden change at a boundary. For such problems, a change of variables reduces a partial differential equation in $x$ and $t$ to an ordinary differential equation in a single **similarity variable**.

Consider the diffusion equation $\partial c/\partial t = D\,\partial^2 c/\partial x^2$ with $c(0, t) = c_0$ and $c(x, 0) = 0$. If $x$ is scaled by a factor $s$ and $t$ by $s^2$, the equation and the conditions are unchanged, so the solution can depend on $x$ and $t$ only through $\eta = x/\sqrt{4Dt}$. Substituting $c = c_0 F(\eta)$ gives $F'' + 2\eta F' = 0$, whose solution with $F(0) = 1$ and $F(\infty) = 0$ is

\begin{equation*}
c = c_0\,\text{erfc}\left(\frac{x}{\sqrt{4Dt}}\right) .
\tag{A.2}
\end{equation*}

The same argument applies to nonlinear equations such as the Richards and Boussinesq equations; the resulting ordinary differential equations are then solved numerically or approximately. The key consequence is the scaling: penetration depths grow as $t^{1/2}$, cumulative fluxes as $t^{1/2}$ and rates as $t^{-1/2}$.

## A.3 Linear systems

### Convolution

For a linear, time-invariant system with impulse response $u(t)$, the output $y$ for input $x$ is the convolution (Chapter 15)

\begin{equation*}
y(t) = (u * x)(t) = \int_0^t u(\tau)\,x(t - \tau)\,d\tau .
\tag{A.3}
\end{equation*}

### Laplace and Fourier transforms

The **Laplace transform** $\tilde f(s) = \int_0^\infty f(t)\,e^{-st}\,dt$ turns convolution into multiplication, $\tilde y = \tilde u\,\tilde x$, and differentiation into multiplication by $s$: $\widetilde{df/dt} = s\tilde f - f(0)$. A linear differential equation with constant coefficients therefore becomes an algebraic equation. For a linear reservoir, $k\,dQ/dt + Q = x$ with $Q(0) = 0$ gives $\tilde Q = \tilde x/(1 + ks)$, so the system's **transfer function** is $\tilde u = 1/(1 + ks)$, and a cascade of $n$ reservoirs has $\tilde u = (1 + ks)^{-n}$, whose inverse is Equation (15.2).

The **Fourier transform** plays the same role for periodic and stationary inputs. The response to a sinusoidal input of angular frequency $\omega$ is a sinusoid of the same frequency, multiplied by the **gain** $|\tilde u(i\omega)|$ and shifted by the phase $\arg \tilde u(i\omega)$. For the linear reservoir the gain is $(1 + \omega^2k^2)^{-1/2}$: storage damps fluctuations faster than its timescale $k$ and passes slower ones almost unchanged. This is why groundwater-fed rivers vary less than rainfall, and why soil moisture has longer memory than precipitation.

### Moments

The moments of a convolution are related simply to those of its components. With $M_0 = \int f\,dt$, mean $\mu = \int t f\,dt/M_0$ and variance $\sigma^2$, for $y = u * x$ with $\int u\,dt = 1$,

\begin{equation*}
\mu_y = \mu_u + \mu_x, \qquad \sigma_y^2 = \sigma_u^2 + \sigma_x^2 .
\tag{A.4}
\end{equation*}

Lag times and spreads therefore add along a chain of linear processes, and the moments of the impulse response can be estimated from those of observed input and output.

## A.4 Responses of nonlinear systems

For a nonlinear store $dS/dt = x - f(S)$, the principle of superposition fails, but two tools remain useful. Near a steady state $S_0$ with input $x_0 = f(S_0)$, a small perturbation $s = S - S_0$ obeys the linearized equation $ds/dt = \delta x - f'(S_0)\,s$, so the system behaves locally as a linear reservoir with timescale $1/f'(S_0)$, which depends on the state (Chapter 15). And for the recession, $dS/dt = -f(S)$ can be separated and integrated exactly: $t = \int_{S}^{S_0} dS'/f(S')$. For $f = aS^c$ this gives a power-law decline of storage and outflow, which is the origin of the recession exponents of Chapter 13.

## A.5 Numerical methods for partial differential equations

Most equations in this book are solved numerically. The basic ideas are the same for all of them.

**Discretization.** The domain is divided into cells or nodes, and derivatives are replaced by differences: $\partial c/\partial x \approx (c_{i+1} - c_{i-1})/(2\Delta x)$, $\partial^2 c/\partial x^2 \approx (c_{i+1} - 2c_i + c_{i-1})/\Delta x^2$. In the **finite-volume** form used in Chapters 11–13, the equation is written for each cell as a balance of fluxes through its faces, which guarantees conservation of mass.

**Explicit and implicit time stepping.** An explicit scheme computes the new values from the old ones directly; an implicit scheme solves a system of equations for the new values. For the diffusion equation, the explicit scheme is stable only if $D\,\Delta t/\Delta x^2 \le 1/2$. Halving the grid spacing then requires quartering the time step, which makes explicit schemes expensive for diffusive problems like the Richards equation. Implicit schemes are stable for any time step and are preferred there (Figure A.1). For advection, the explicit stability condition is the **Courant–Friedrichs–Lewy** condition $V\Delta t/\Delta x \le 1$, met in Chapters 9 and 10.

**Nonlinearity.** When coefficients depend on the solution, as $K(\psi)$ does in the Richards equation, an implicit step requires iteration. **Picard** iteration re-evaluates the coefficients with the latest estimate until it converges; **Newton** iteration uses the derivative of the residual and converges faster near the solution. The modified Picard scheme of Celia et al. (1990), used in Chapters 11 and 12, writes the storage term in a way that conserves mass exactly.

**Stiff ordinary differential equations.** The method of lines discretizes space only and passes the resulting system of ordinary differential equations to a solver with adaptive, implicit time stepping (such as the BDF method in `scipy.integrate.solve_ivp`), as done for the Boussinesq equation in Chapter 13.

```{code-cell} python
:tags: [hide-input]
D, L, nx = 1.0, 1.0, 41
dx = L / (nx - 1); x = np.linspace(0, L, nx)
c0 = np.where(np.abs(x - 0.5) < 0.1, 1.0, 0.0)
def run(r, steps, implicit):
    c = c0.copy(); n = nx - 2
    A = (np.diag(np.full(n, 1 + 2 * r)) + np.diag(np.full(n - 1, -r), 1) + np.diag(np.full(n - 1, -r), -1))
    for _ in range(steps):
        if implicit: c[1:-1] = np.linalg.solve(A, c[1:-1])
        else: c[1:-1] = c[1:-1] + r * (c[2:] - 2 * c[1:-1] + c[:-2])
    return c
t_end = 0.004
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)
for ax, r, ttl in zip(axs, [0.4, 0.6], ["(a) r = 0.4", "(b) r = 0.6"]):
    steps = int(round(t_end / (r * dx ** 2)))
    ax.plot(x, c0, color=ink2, lw=1, ls=":", label="Initial")
    ax.plot(x, run(r, steps, False), color=c2, lw=1.8, label="Explicit")
    ax.plot(x, run(r, steps, True), color=c1, lw=1.8, ls="--", label="Implicit")
    ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylim(-0.6, 1.2)
axs[0].set_ylabel("c"); axs[0].legend(loc="upper left", fontsize=9)
plt.show()
```

*Figure A.1. Diffusion of a square pulse computed with explicit and implicit schemes for $D\,\Delta t/\Delta x^2 = r$. (a) With $r$ = 0.4 both are stable and agree. (b) With $r$ = 0.6 the explicit solution develops growing oscillations, while the implicit solution remains stable.*

## A.6 Further reading

- Brutsaert, W. (2023). *Hydrology: An Introduction* (2nd ed.). Cambridge University Press. The appendix on mathematical tools.
- Celia, M. A., Bouloutas, E. T., and Zarba, R. L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. *Water Resources Research*, 26, 1483–1496.
- Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Oxford University Press.
- Dooge, J. C. I. (1973). *Linear Theory of Hydrologic Systems*. Technical Bulletin 1468, US Department of Agriculture.

---

*Figures in this appendix are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
