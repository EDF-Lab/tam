# Exp-Centered Units (ExU): PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See EXU.md](../../math/spectrum/EXU.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Exp-Centered Units are an activation of the neural effect, selected with `activation='exu'`:

```python
# Exp-Centered Units (ExU) inside NeuralEffect for sharp discontinuities
formula = "Y ~ n(x1, others='x2', activation='exu', n_neurons=32)"
```

## Parameter Specification

ExU is selected through `n(..., activation='exu')`; the other parameters are those of the neural effect (see [NEURAL.md](../../math/spectrum/NEURAL.md)).

## Complexity & Performance

- Same cost as the neural effect it extends; the other activations are unchanged.

## State, Complexity and Limits

- **Class and file**: `NeuralEffect` (`src/tam/model/spectrum/_neural.py`) with `activation='exu'`. Label: (BETA).
- **State**: none. **Other activations**: unchanged, bit for bit.

## Code

Parsing of the `n(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_neural>"
:end-before: "#: </parse_neural>"
```
