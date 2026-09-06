# Risk: Static Conformal, Streaming ACI, EVT & Epistemic

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [Risk: conformal, ACI, EVT & epistemic](../../math/statistics/04_risk_conformal_evt.md)

> Implementation only, for the coverage guarantees, the ACI update and the EVT/posterior theory see the [math theory](../../math/statistics/04_risk_conformal_evt.md).

One source of truth for conformal: the static engine `safety.py::SafetyTAM` holds the finite-sample quantile and p-value; the streaming loop and the wrappers build on it. The static (i.i.d.) engine is kept deliberately apart from the streaming, non-stationary adaptation in `statistics/risk/aci.py`.

## The static finite-sample quantile (SafetyTAM)

```{literalinclude} ../../../../src/tam/model/safety.py
:language: python
:pyobject: SafetyTAM.conformal_quantile
```

## Adaptive Conformal Inference (streaming, model-agnostic)

```{literalinclude} ../../../../src/tam/model/statistics/risk/aci.py
:language: python
:pyobject: update_risk_level
```

## Conformalized distributional wrapper (CQR + Mondrian + ACI)

```{literalinclude} ../../../../src/tam/model/statistics/risk/conformal.py
:language: python
:pyobject: ConformalDistributionalTAM
```

## Extreme Value Theory tail scoring

```{literalinclude} ../../../../src/tam/model/statistics/risk/extremes.py
:language: python
:pyobject: GeneralizedParetoTail
```

## Epistemic (parameter) uncertainty

```{literalinclude} ../../../../src/tam/model/statistics/risk/uncertainty.py
:language: python
:pyobject: posterior_prediction
```
