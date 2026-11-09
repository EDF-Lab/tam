# Hierarchical Engineering & Memory Assembly  [BETA]

> 🚧 **Beta – Research Module**

⚠️ **Status: Active development**

- Core functionality is implemented but still evolving  
- Some features may be incomplete or subject to change  
- API is not yet stable  
- Intended for research use only (not production-ready)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/03_hierarchical_joint.md)

## Alignment and groups

`HierarchicalTAM` matches its nodes on `(group, date)`, never on row position (`align='strict'`, the default, raises and lists the first missing keys per node; `align='inner'` keeps the common keys and warns with the rows dropped per node). Each node is then sorted by `(group, date)`, and `additivity_gap_` records, per parent, `max |Y_p - sum Y_c| / mean |Y_p|` (a gap above 0.1 % is warned once at fit). Without a `date_col` the nodes are matched by row position and must have the same number of rows.

With a `group_col` every node keeps its group axis `(G, T, F)`; all nodes must hold the same groups and the same `T` rows per group. The groups do not interact: the global system is `(G, D, D)` with `D` the sum of the node dimensions, and the penalty `P` of shape `(D, D)` is shared. `fit` is `_assemble_system` followed by `_solve`:

```{literalinclude} ../../../../src/tam/model/hierarchical.py
:language: python
:start-after: "#: <fit_hierarchical>"
:end-before: "#: </fit_hierarchical>"
:caption: src/tam/model/hierarchical.py (assembly of the joint system and its three solvers)
```

The memory check counts the `G * D^2 * 8` bytes of the dense covariance against 40 % of the available memory. Direct inversion is one batched solve over the groups; the dense and the matrix-free conjugate-gradient solvers loop over the groups, so one group's system is applied at a time. Both stop at a relative tolerance of `1e-6`, so their coefficients agree with the direct solve to about `1e-3` of the largest coefficient on an ill-conditioned system (predictions to a few `1e-6`).
