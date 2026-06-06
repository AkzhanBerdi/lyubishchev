# lyubishchev

**Quantitative taxonomy methods of Alexander Alexandrovich Lyubishchev (1890–1972)**

```
pip install lyubishchev
```

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

Implementations of Lyubishchev's multivariate classification methods for biological taxonomy, as described in his 1943 manuscript:

> Lyubishchev, A.A. (1943). *Programma obshchey sistematiki* [Program of General Systematics].  
> Manuscript, 22 November 1943. Digitized by ZIN RAS Coleoptera Laboratory.  
> http://www.zin.ru/animalia/coleoptera/rus/lyubis05.htm

His Western publication, one year before the canonical Sokal & Sneath text:

> Lubischew, A.A. (1962). On the use of discriminant functions in taxonomy.  
> *Biometrics*, 18(4), 455–477.

---

## Functions

### `divergence_coefficient(a, b)`

Lyubishchev's D from the 1943 manuscript:

```
D = (M₁ - M₂)² / (σ₁² + σ₂²)
```

Squared distance between two group means, normalized by pooled variance. Works on continuous measurements. When D > 1.0, groups are cleanly separated. When D < 0.5, you have transgression — the boundary between taxa breaks down.

```python
import numpy as np
from lyubishchev import divergence_coefficient

rng = np.random.default_rng(42)
oleracea  = rng.multivariate_normal([3.2, 1.5], [[0.04, 0.01], [0.01, 0.04]], 20)
carduorum = rng.multivariate_normal([3.8, 1.9], [[0.04, 0.01], [0.01, 0.04]], 20)

D = divergence_coefficient(oleracea, carduorum)
print(f"D = {D:.3f}")  # D >> 1.0 → clean separation
```

---

### `scatter_ellipse(X, y)`

Fit covariance ellipses per class — the geometric object Lyubishchev drew by hand in Fig. 1 of his 1943 manuscript. Each class is represented by its centroid and covariance matrix. Overlap between ellipses is transgression.

```python
from lyubishchev import scatter_ellipse

X = np.vstack([oleracea, carduorum])
y = ['oleracea'] * 20 + ['carduorum'] * 20

ellipses = scatter_ellipse(X, y)
print(ellipses['oleracea']['mean'])   # centroid
print(ellipses['oleracea']['cov'])    # covariance matrix
```

---

### `transgression(ellipses, class_a, class_b)`

Compute whether two scatter ellipses overlap. Lyubishchev called this transgression — the failure of measurement space to cleanly separate two taxa.

```python
from lyubishchev import transgression

result = transgression(ellipses, 'oleracea', 'carduorum')
print(result['transgression'])        # False → clean separation
print(result['separation_ratio'])     # > 1.0 → well separated
```

---

### `classify(specimen, ellipses)`

The Edgeworth-Pearson multivariate probability function — the mathematical core of Lyubishchev's paper nomograms (Fig. 3, 1943 manuscript). Given a specimen's measurements and a set of reference groups, returns posterior probability of belonging to each group.

This is what his field nomograms computed by hand. This is a deployed model on paper.

```python
from lyubishchev import classify

specimen = np.array([3.75, 1.85])
result = classify(specimen, ellipses)

best = max(result, key=lambda k: result[k]['posterior'])
print(best)                                          # 'carduorum'
print(result['carduorum']['posterior'])              # e.g. 0.923
print(result['carduorum']['mahalanobis_distance'])   # distance to centroid
```

---

## `LyubishchevClassifier` — scikit-learn estimator

New in 0.2.0. The 1943 method wrapped as a drop-in scikit-learn classifier, so it plugs straight into `Pipeline`, `GridSearchCV`, `cross_val_score`, and the rest of the ecosystem. Each class is modeled by its centroid and full (regularized) covariance; a specimen is assigned to the class under which its measurements are most probable — the multivariate posterior Lyubishchev computed with paper nomograms.

```python
from lyubishchev import LyubishchevClassifier

clf = LyubishchevClassifier(standardize=True)   # z-score features first
clf.fit(X_train, y_train)

labels = clf.predict(X_test)
proba  = clf.predict_proba(X_test)              # vectorized over all rows
score  = clf.score(X_test, y_test)
```

Parameters:

- `standardize` (default `False`) — z-score each feature using the training mean/std before fitting. Use it when measurements live on different scales (mm vs kg) so the covariance structure isn't dominated by one feature.
- `reg_covar` (default `1e-6`) — ridge added to each class covariance diagonal, preventing `LinAlgError` when features are collinear or a class has fewer samples than features (the same trick `GaussianNB` and `GaussianMixture` use).
- `priors` (default `'uniform'`) — `'uniform'` (Lyubishchev's equal-prior assumption), `'empirical'` (observed class frequencies), or an explicit array.

Input is validated through `check_X_y` / `check_array`: NaN, infinite values, single-class targets, and feature-count mismatches at predict time all raise. The estimator passes scikit-learn's `check_estimator` conformance suite.

```python
from sklearn.model_selection import cross_val_score
cross_val_score(LyubishchevClassifier(standardize=True), X, y, cv=5)
```

---

## Plotting

New in 0.2.0. A digital reproduction of Lyubishchev's Fig. 1 (Рис. 1) — covariance ellipses per taxon — plus decision-region plots for a fitted classifier. Requires the `[plot]` extra.

```python
from lyubishchev import scatter_ellipse, LyubishchevClassifier
from lyubishchev.plot import plot_ellipses, plot_classification

ellipses = scatter_ellipse(X, y)
plot_ellipses(ellipses, X, y)                   # the 1943 figure, digitally

clf = LyubishchevClassifier().fit(X, y)
plot_classification(clf, X, y)                  # decision regions (2-D)
```

---

## Comparison with SciPy

| | `scipy.sokalsneath` / `sokalmichener` | `lyubishchev` |
|---|---|---|
| Input | Binary presence/absence vectors | Continuous measurements |
| Covariance | Not modeled (independence assumed) | Full covariance matrix |
| Distance metric | Simple matching / weighted mismatch | Mahalanobis distance |
| Classification | No | Yes — posterior probability per class |
| Primary source | Sokal & Sneath (1963) | Lyubishchev (1943, 1962) |

Lyubishchev explicitly criticized the independence assumption in his 1943 manuscript. The covariance structure between measurements is not noise — it is information.

---

## Installation

```bash
pip install lyubishchev
```

With plotting support:

```bash
pip install lyubishchev[plot]
```

---

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- scipy ≥ 1.10
- scikit-learn ≥ 1.2
- matplotlib ≥ 3.6 (optional, for the `[plot]` extra)

---

## License

MIT

---

## Citation

If you use this library, you are citing Lyubishchev. That is the point.

```bibtex
@misc{lyubishchev1943,
  author    = {Lyubishchev, Alexander Alexandrovich},
  title     = {Programma obshchey sistematiki
               [Program of General Systematics]},
  year      = {1943},
  note      = {Manuscript, 22 November 1943.
               Digitized by ZIN RAS Coleoptera Laboratory.
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
  author  = {Berdeyev, Akzhan},
  title   = {lyubishchev: Quantitative taxonomy methods of A.A. Lyubishchev},
  year    = {2026},
  url     = {https://github.com/akzhanberdi/lyubishchev},
}
```

---

*Bad Dog Data — baddogdata.com*
