
# DeepGAM Engineering & Sequential Encapsulation [EXP]

> 🧪 **Experimental – Work in Progress**

⚠️ **Status: Incomplete / exploratory code**

- This module is under heavy development or prototyping  
- It may be partially implemented or non-functional  
- Behavior, correctness, and API are not guaranteed  
- Provided for transparency and research exploration only

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/06_deep_gam_backfitting.md)

## Backfitting and the validation guard

`NeuralTAM` first fits its `StaticTAM`, where an `n()` term is a frozen random-feature effect solved in closed form. It then trains one network per group and per `n()` term by backfitting:
the partial residual of a network is the target minus the other effects (the closed-form contribution of that network is added back), the network is trained on it, and the best epoch by validation loss is kept.

* **The closed-form column of a network** is the column of the base decomposition named after the effect: `effect_x` when the network is the only term on `x`, `effect_n_x` when `x` also has a `s()` or `l()` term. `neural_effect_columns(effects)` returns it for every network; the backfitting starts from that contribution.
* **The guard** (`guard=True`, the default): once a network is trained, its validation loss is compared with the loss of the closed-form contribution it would replace, on the same validation rows (the rows of `data_val` when given, otherwise the last `val_split` fraction of the group). The network is kept only if its loss is strictly lower. Otherwise the closed-form column stays in the forecast and the decomposition, `mlps_` has no entry for that group and feature, and `network_used_[group][feature]` is `False`.
* **When no network is kept anywhere**, the model equals its `StaticTAM` and a `UserWarning` says so: give the networks more epochs or neurons, or pass `guard=False` to keep them anyway. A short training (a few epochs, a small network) usually ends there, which is why a `NeuralTAM` is never worse than its `StaticTAM` by more than the noise of the validation rows.
* **`guard=False`** keeps every network, as the model did before the guard existed.
* **Training** of each network: Adam at the fixed learning rate `lr`, the best epoch by validation loss, and early stopping after `patience` epochs without improvement. There is no learning-rate scheduler: the last layer starts at zero, so the validation loss is flat for the first epochs, and a plateau scheduler used to halve the rate to its floor before the network had moved.
