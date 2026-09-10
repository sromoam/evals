# Notebooks

Runnable examples. Each notebook is committed **with its outputs**, so the results
are readable without executing anything.

## `StructuredOutput_Evaluator.ipynb`

`StructuredOutput` scored against `Equals` on the same six extraction cases, then
the three surfaces it adds: `per_case()`, `metrics()` and `explain()`.

Offline and deterministic. The extraction is stubbed, so there are no credentials,
no model calls and no cost. Needs the `stickler` extra:

```bash
pip install -e ".[stickler]"
```

Run it with:

```bash
jupyter lab examples/notebooks/StructuredOutput_Evaluator.ipynb
```

To re-execute in place after a change:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  examples/notebooks/StructuredOutput_Evaluator.ipynb
```
