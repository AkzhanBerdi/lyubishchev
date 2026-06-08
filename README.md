# lyubishchev

**Quantitative taxonomy methods of Alexander Alexandrovich Lyubishchev (1890–1972)**

```bash
pip install lyubishchev
```

[![PyPI](https://img.shields.io/pypi/v/lyubishchev)](https://pypi.org/project/lyubishchev/)
[![Python](https://img.shields.io/pypi/pyversions/lyubishchev)](https://pypi.org/project/lyubishchev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Open a Python terminal and type:

```python
from scipy.spatial.distance import sokalsneath, sokalmichener
```

Two distance functions named after Sokal, memorialized in the standard scientific computing library. They compute dissimilarity between organisms described as binary character vectors — presence or absence of a trait.

Now type:

```python
from lyubishchev import divergence_coefficient, scatter_ellipse, classify
```

Lyubishchev was doing the same thing in 1943. With continuous measurements. With full covariance structure. Twenty years before Sokal & Sneath published *Principles of Numerical Taxonomy* (1963).

He deserves a citation. This library is it.

---

## What this is

A regularized-by-default Gaussian classifier and multivariate taxonomy toolkit, implementing Lyubishchev's methods from his 1943 manuscript:

> Lyubishchev, A.A. (1943). *Programma obshchey sistematiki* [Program of General Systematics].
> Manuscript, 22 November 1943. Digitized by ZIN RAS Coleoptera Laboratory.
> http://www.zin.ru/animalia/coleoptera/rus/lyubis05.htm

His Western publication, one year before the canonical Sokal & Sneath text:

> Lubischew, A.A. (1962). On the use of discriminant functions in taxonomy.
> *Biometrics*, 18(4), 455–477.

**Benchmark results and the full story:** [baddogdata.com/lyubishchev-pypi-benchmarks](https://baddogdata.com/lyubishchev-pypi-benchmarks)

---

## When to use this library

**Use it when:**
- Your features are **continuous and approximately Gaussian** — measurements, scores, financial metrics, biological data
- Your data is **low-to-moderate dimensional** (up to a few hundred features)
- You want a classifier that **never crashes** — `reg_covar` is structural, not a patch
- sklearn's `QuadraticDiscriminantAnalysis` is failing on your data — this is mathematically equivalent but robust across all sklearn versions
- You need **well-calibrated posterior probabilities** on clean, well-separated classes
- You want to **validate that segments actually exist** in your data before fitting a model (`divergence_coefficient`)

**Do not use it when:**
- Your features are **categorical or mixed** (one-hot encoded, binary flags) — tree-based methods are better there
- You have **interaction-driven signals** (XOR-like feature combinations) — gradient boosting will win
- You need **maximum accuracy on general tasks** — SVM or gradient boosting typically leads
- Your data is **very noisy** — full covariance estimation degrades faster than LDA under measurement noise

---

## The QDA version problem this solves

sklearn's `QuadraticDiscriminantAnalysis` behavior on singular covariance matrices is version-dependent:

```python
# This may run, warn, or crash depending on your sklearn version
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis(reg_param=0.0)
qda.fit(X, y)  # Silent on 1.6, crashes on 1.9+
```

`LyubishchevClassifier` applies `reg_covar` explicitly and structurally — it behaves the same on sklearn 1.2 through 1.9 and beyond. No silent version-dependent behavior.

---

## `LyubishchevClassifier` — scikit-learn estimator

A drop-in sklearn classifier. Plugs into `Pipeline`, `GridSearchCV`, `cross_val_score`.

```python
from lyubishchev import LyubishchevClassifier

clf = LyubishchevClassifier(standardize=True)
clf.fit(X_train, y_train)

labels = clf.predict(X_test)
proba  = clf.predict_proba(X_test)
score  = clf.score(X_test, y_test)
```

Works in a Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('clf', LyubishchevClassifier())])

grid = GridSearchCV(pipe, {
    'clf__reg_covar': [1e-6, 1e-4, 1e-2, 1e-1],
    'clf__standardize': [True, False],
}, cv=5)
grid.fit(X_train, y_train)
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `standardize` | `False` | Z-score features using training mean/std before fitting. Use when measurements are on different scales. |
| `reg_covar` | `1e-6` | Ridge added to each class covariance diagonal. Prevents `LinAlgError` on collinear or high-dimensional data. Tune via `GridSearchCV` on high-dim problems. |
| `priors` | `'uniform'` | `'uniform'` (equal priors), `'empirical'` (observed class frequencies), or an explicit array. |

Passes sklearn's `check_estimator` conformance suite. Input validation via `check_X_y` / `check_array`.

---

## Lower-level functions

### `divergence_coefficient(a, b)`

Lyubishchev's D — numerical measure of separation between two groups on continuous features:

```
D = (M₁ - M₂)² / (σ₁² + σ₂²)
```

When D > 1.0, groups are cleanly separated. When D < 0.5, your segments do not exist in the data — no algorithm will find them reliably.

```python
from lyubishchev import divergence_coefficient
import numpy as np

rng = np.random.default_rng(42)
group_a = rng.multivariate_normal([3.2, 1.5], [[0.04, 0.01], [0.01, 0.04]], 20)
group_b = rng.multivariate_normal([3.8, 1.9], [[0.04, 0.01], [0.01, 0.04]], 20)

D = divergence_coefficient(group_a, group_b)
print(f"D = {D:.3f}")  # D >> 1.0 → clean separation
```

**Practical use:** run this before building a segmentation model. If D < 0.5 on your key features, your segments are not in the data.

---

### `scatter_ellipse(X, y)`

Fit covariance ellipses per class — the geometric object Lyubishchev drew by hand in Fig. 1 of his 1943 manuscript.

```python
from lyubishchev import scatter_ellipse

X = np.vstack([group_a, group_b])
y = ['a'] * 20 + ['b'] * 20

ellipses = scatter_ellipse(X, y)
print(ellipses['a']['mean'])   # centroid
print(ellipses['a']['cov'])    # covariance matrix
```

---

### `transgression(ellipses, class_a, class_b)`

Check whether two scatter ellipses overlap. Lyubishchev called overlap "transgression" — the failure of measurement space to separate two taxa.

```python
from lyubishchev import transgression

result = transgression(ellipses, 'a', 'b')
print(result['transgression'])      # False → clean separation
print(result['separation_ratio'])   # > 1.0 → well separated
```

---

### `classify(specimen, ellipses)`

Bayesian classification of a new specimen — posterior probability of belonging to each class.

```python
from lyubishchev import classify

specimen = np.array([3.75, 1.85])
result = classify(specimen, ellipses)

best = max(result, key=lambda k: result[k]['posterior'])
print(best)                              # 'b'
print(result['b']['posterior'])          # e.g. 0.923
print(result['b']['mahalanobis_distance'])
```

---

## Plotting

Requires `pip install lyubishchev[plot]`.

```python
from lyubishchev.plot import plot_ellipses, plot_classification
from lyubishchev import scatter_ellipse, LyubishchevClassifier

# Reproduce Lyubishchev's 1943 Fig. 1 digitally
ellipses = scatter_ellipse(X, y)
plot_ellipses(ellipses, X, y)

# Decision regions for a fitted classifier (2-D)
clf = LyubishchevClassifier().fit(X, y)
plot_classification(clf, X, y)
```

---

## Comparison with SciPy

| | `scipy.sokalsneath` / `sokalmichener` | `lyubishchev` |
|---|---|---|
| Input | Binary presence/absence vectors | Continuous measurements |
| Covariance | Not modeled (independence assumed) | Full covariance matrix |
| Distance metric | Simple matching / weighted mismatch | Mahalanobis distance |
| Classification | No | Yes — posterior probability per class |
| sklearn compatible | No | Yes — Pipeline, GridSearchCV, cross_val_score |
| Primary source | Sokal & Sneath (1963) | Lyubishchev (1943, 1962) |

---

## Installation

```bash
pip install lyubishchev           # core
pip install lyubishchev[plot]     # with plotting
pip install lyubishchev[dev]      # with test dependencies
```

**Requirements:** Python ≥ 3.9 · numpy ≥ 1.24 · scipy ≥ 1.10 · scikit-learn ≥ 1.2

---

## License

MIT

---

## Citation

If you use this library, you are citing Lyubishchev. That is the point.

```bibtex
@misc{lyubishchev1943,
  author = {Lyubishchev, Alexander Alexandrovich},
  title  = {Programma obshchey sistematiki [Program of General Systematics]},
  year   = {1943},
  note   = {Manuscript, 22 November 1943. Digitized by ZIN RAS Coleoptera Laboratory.
            Available at: http://www.zin.ru/animalia/coleoptera/rus/lyubis05.htm}
}

@article{lubischew1962,
  author  = {Lubischew, A.A.},
  title   = {On the use of discriminant functions in taxonomy},
  journal = {Biometrics},
  year    = {1962},
  volume  = {18},
  number  = {4},
  pages   = {455--477},
}

@software{lyubishchev_python,
  author = {Berdeyev, Akzhan},
  title  = {lyubishchev: Quantitative taxonomy methods of A.A. Lyubishchev},
  year   = {2026},
  url    = {https://github.com/AkzhanBerdi/lyubishchev},
}
```

---

*Bad Dog Data — [baddogdata.com](https://baddogdata.com)*
