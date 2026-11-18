# Caputo Fractional Physics Penalties: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See FRACTIONAL_PHYSICS.md](../../math/spectrum/FRACTIONAL_PHYSICS.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Fractional derivative orders extend the `phys()` penalty (integer orders are unchanged):

```python
# Fractional Caputo derivative physics operator
formula = "Y ~ phys(x, basis='fourier', n_coeffs=16, D0_5=1.0)"
```

## Parameter Specification

`phys(x, basis=..., D0_5=1.0, ...)` keeps the parameters of `phys()`; non-integer derivative orders select the Caputo operator.

## Complexity & Performance

- Same cost as the integer-order penalty of `phys()`; the Grünwald–Letnikov weights are built once per penalty matrix.

## State, Complexity and Limits

- **Class and file**: the fractional orders are in `UniversalPhysicsEffect` (`src/tam/model/spectrum/_physics.py`); formula token `phys()` with a non-integer order. Label: (BETA).
- **State**: none. **Integer orders**: unchanged, bit for bit.
- **Spelling in a formula**: `phys(t, basis='fourier', n_coeffs=16, D0_5=1.0)` sets the order 0.5; `D0.5=1.0` is refused by the generic term parser (a dot is not allowed in an argument name). `show_syntax()` gives the example.

## Code

Parsing of the `phys(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_physics>"
:end-before: "#: </parse_physics>"
```
