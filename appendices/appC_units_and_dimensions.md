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

# Appendix C. Units, dimensions and conversion of equations

This book uses SI units throughout. Hydrological literature, data and software, however, use a mixture of units, and many empirical formulas are valid only in the units in which they were derived. This appendix summarizes the units used in the book, the conversions met most often, the principles of dimensional analysis, and a safe procedure for converting empirical equations.

```{code-cell} python
:tags: [remove-cell]
import numpy as np
from IPython.display import HTML, display
def table(header, rows):
    h = "".join(f"<th>{x}</th>" for x in header)
    b = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
    display(HTML(f'<table class="table"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'))
```

## C.1 Units in hydrology

The SI base units used here are the meter, kilogram, second and kelvin; temperatures are usually given in degrees Celsius. Hydrology adds a few conventions of its own:

- **Water depths** express volumes per unit area: 1 mm of water over 1 m² is 1 L, and over 1 km² it is 1000 m³. Precipitation, evaporation, runoff and storage change are all given in millimeters (per hour, day or year), which makes the water balance easy to close.
- **Discharge** is given in m³ s⁻¹ for rivers and in m² s⁻¹ (per unit width) or mm h⁻¹ (per unit area) for hillslopes and aquifers.
- **Energy fluxes** are given in W m⁻² or, for daily totals, MJ m⁻² d⁻¹ (1 W m⁻² sustained for a day is 0.0864 MJ m⁻² d⁻¹).
- **Pressure** is given in kPa for the atmosphere and as **head** (m of water) in soils and aquifers: 1 m of head corresponds to $\rho_w g$ × 1 m ≈ 9.81 kPa.

```{code-cell} python
:tags: [hide-input]
lam, rho = 2.45e6, 1000.0
rows = [
    ("1 mm d⁻¹ of evaporation", f"{lam * rho * 1e-3 / 86400:.1f} W m⁻² (λ = 2.45 MJ kg⁻¹)"),
    ("1 W m⁻² of latent heat", f"{86400 / (lam * rho * 1e-3):.4f} mm d⁻¹"),
    ("1 m³ s⁻¹ from 1 km²", f"{86400 / 1e6 * 1e3:.1f} mm d⁻¹"),
    ("1 mm d⁻¹ over 1 km²", f"{1e-3 * 1e6 / 86400:.4f} m³ s⁻¹"),
    ("1 mm h⁻¹ over 1 km²", f"{1e-3 * 1e6 / 3600:.3f} m³ s⁻¹"),
    ("1 cm d⁻¹ of conductivity", f"{1e-2 / 86400:.3e} m s⁻¹"),
    ("1 m of water head", f"{rho * 9.81 / 1e3:.2f} kPa"),
    ("1 bar (soil physics)", "100 kPa ≈ 10.2 m of head"),
    ("1 inch", "25.4 mm"),
    ("1 foot", "0.3048 m"),
    ("1 cubic foot per second (cfs)", f"{0.3048 ** 3:.5f} m³ s⁻¹"),
    ("1 acre-foot", f"{4046.856 * 0.3048:.1f} m³"),
    ("1 square mile", f"{1.609344 ** 2:.3f} km²"),
    ("1 langley (ly)", "1 cal cm⁻² = 41.84 kJ m⁻²"),
]
table(["Quantity", "Equivalent"], rows)
```

*Table C.1. Conversions used in hydrology.*

## C.2 Dimensions and dimensionless numbers

Every physical equation must be **dimensionally homogeneous**: all its terms must have the same dimensions, expressed in terms of mass [M], length [L], time [T] and temperature [Θ]. Checking dimensions catches many errors, and it underlies the use of dimensionless numbers to characterize processes. The **Buckingham Π theorem** states that a relation among $n$ variables involving $k$ independent dimensions can be rewritten as a relation among $n - k$ dimensionless groups.

As an example, the time $t_d$ for an unconfined aquifer to drain (Chapter 13) may depend on its width $B$, thickness $D$, conductivity $K$ and drainable porosity $\phi_e$. With $n$ = 5 variables and $k$ = 2 dimensions (L and T), there are three groups, for example $t_d K D/(\phi_e B^2)$, $D/B$ and $\phi_e$. If the aspect ratio and porosity enter only through the diffusivity, the drainage time must scale as $t_d \propto \phi_e B^2/(KD)$, the timescale that appears in the late-time solutions of the Boussinesq equation.

```{code-cell} python
:tags: [hide-input]
rows = [
    ("Reynolds, Re", "VL/ν", "inertia / viscosity", "laminar or turbulent flow (Ch. 4, 9, 11)"),
    ("Froude, Fr", "V/√(gh)", "flow speed / wave speed", "sub- or supercritical flow (Ch. 9, 10)"),
    ("Péclet, Pe", "VL/D", "advection / diffusion", "kinematic vs diffusive behavior (Ch. 10, 13)"),
    ("Richardson, Ri", "(g/T)(∂θ/∂z)/(∂u/∂z)²", "buoyancy / shear", "atmospheric stability (Ch. 4)"),
    ("Monin–Obukhov, ζ", "z/L", "height / Obukhov length", "stability in surface layer (Ch. 4)"),
    ("Bowen ratio, β", "H/(λE)", "sensible / latent heat", "surface energy partition (Ch. 2, 7)"),
    ("Aridity index, φ", "Eₚ/P", "energy / water supply", "long-term water balance (Ch. 8, 15)"),
    ("Kinematic-wave number, k", "S₀L/(h₀F₀²)", "gravity–friction / inertia", "validity of kinematic wave (Ch. 9, 10)"),
]
table(["Number", "Definition", "Ratio of", "Used for"], rows)
```

*Table C.2. Dimensionless numbers used in the book.*

## C.3 Converting empirical equations

Many hydrological formulas were fitted to data and are **dimensionally inhomogeneous**: their coefficients carry hidden units. A formula like $V = (1.49/n)\,R^{2/3}S^{1/2}$, Manning's equation in US customary units, cannot be used with SI inputs without changing its coefficient. A safe procedure is:

1. Write the equation with every variable expressed in its original units: $V_{\text{ft/s}} = (1.49/n)\,R_{\text{ft}}^{2/3}S^{1/2}$.
2. Replace each variable by its SI value multiplied by the conversion factor: $V_{\text{ft/s}} = V_{\text{m/s}}/0.3048$ and $R_{\text{ft}} = R_\text{m}/0.3048$.
3. Collect the factors into a new coefficient: $V_{\text{m/s}} = 1.49 \times 0.3048^{1/3}\,(1/n)\,R_\text{m}^{2/3}S^{1/2} = (1.00/n)\,R_\text{m}^{2/3}S^{1/2}$.
4. **Check numerically**: evaluate the original equation with original-unit inputs and the converted equation with SI inputs for a test case, and confirm that the outputs agree after conversion.

The coefficient 1.49 is simply $(1/0.3048)^{1/3}$: Manning's $n$ is treated as having the same value in both systems, with the units absorbed into the coefficient. Step 4 should never be skipped, since sign errors and misplaced exponents are common.

```{code-cell} python
ft = 0.3048
n, R_m, S = 0.035, 1.2, 0.001
V_us = 1.49 / n * (R_m / ft) ** (2 / 3) * S ** 0.5          # ft/s, original formula with inputs in feet
V_si = 1.49 * ft ** (1 / 3) / n * R_m ** (2 / 3) * S ** 0.5  # m/s, converted formula
print(f"converted coefficient = {1.49 * ft ** (1 / 3):.4f}")
print(f"original: {V_us:.4f} ft/s = {V_us * ft:.4f} m/s; converted: {V_si:.4f} m/s")
```

The same procedure applies to any empirical formula, for example time-of-concentration formulas in which length is in feet and time in minutes, or evaporation formulas with wind speed in miles per day. When an equation contains a logarithm or an exponential of a dimensional quantity, the argument must be converted before the function is applied; the coefficient cannot simply be rescaled.

## C.4 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The appendix on units and dimensions.
- Bureau International des Poids et Mesures (2019). *The International System of Units (SI)* (9th ed.). BIPM.

---

*Tables in this appendix are generated by the Python code shown. Text: © Daeha Kim, CC BY-NC 4.0.*
