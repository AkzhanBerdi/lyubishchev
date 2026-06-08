# Changelog

All notable changes to this project will be documented here.

---

## [0.2.2] — 2026-06-07

### Fixed
- **Predict-time performance** — replaced `np.einsum` Mahalanobis path with Cholesky whitening. Predict is now at parity with sklearn's QDA (ratio ≤ 2× at scale, was 75–82× in 0.2.1).

---

## [0.2.1] — 2026-06-05

### Fixed
- Critical performance fix: Cholesky whitening in covariance computation (initial implementation).

---

## [0.2.0] — 2026-06-04

### Added
- `LyubishchevClassifier` — scikit-learn compatible estimator (`BaseEstimator`, `ClassifierMixin`).
  - `fit`, `predict`, `predict_proba`, `score`
  - `standardize`, `reg_covar`, `priors` parameters
  - Input validation via `check_X_y` / `check_array`
  - Passes `check_estimator` conformance suite
  - Works in `Pipeline`, `GridSearchCV`, `cross_val_score`
- `lyubishchev.plot` module — `plot_ellipses`, `plot_classification`
- `[plot]` and `[dev]` optional extras in `pyproject.toml`
- Full test suite for estimator (62 tests total)

---

## [0.1.0] — 2026-06-01

### Added
- `divergence_coefficient(a, b)` — Lyubishchev's D formula from the 1943 manuscript
- `scatter_ellipse(X, y)` — covariance ellipses per class
- `transgression(ellipses, class_a, class_b)` — overlap detection between ellipses
- `classify(specimen, ellipses)` — Edgeworth-Pearson multivariate posterior classification
- Initial PyPI release
