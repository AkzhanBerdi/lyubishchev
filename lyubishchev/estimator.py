"""
lyubishchev.estimator
~~~~~~~~~~~~~~~~~~~~~~

A scikit-learn compatible classifier built on Lyubishchev's 1943
multivariate method: model each taxon by its centroid and full
covariance matrix, then assign a specimen to the class under which
its measurements are most probable (Edgeworth-Pearson posterior with
equal priors).

Mathematically this is Quadratic Discriminant Analysis — but it is
the formulation Lyubishchev set down in *Programma obshchey
sistematiki* (22 November 1943), two decades before the methods that
gave the field the name "numerical taxonomy".
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data

    def _check_X(estimator, X):
        return _validate_data(estimator, X, reset=False)
except ImportError:  # older scikit-learn
    def _check_X(estimator, X):
        return estimator._validate_data(X, reset=False)


class LyubishchevClassifier(ClassifierMixin, BaseEstimator):
    """
    Multivariate Gaussian classifier after Lyubishchev (1943).

    Each class is summarized by a mean vector and a (regularized)
    covariance matrix. Prediction assigns a specimen to the class
    with the highest posterior probability, assuming equal priors and
    a multivariate normal density per class. This reproduces the
    classification implied by Lyubishchev's scatter ellipses and
    paper nomograms.

    Parameters
    ----------
    standardize : bool, default=False
        If True, z-score each feature using the training mean and
        standard deviation before fitting and prediction. Use this
        when measurements are on different scales (e.g. mm vs kg) so
        that the covariance structure is not dominated by the
        largest-variance feature.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of each
        class covariance matrix. Prevents ``LinAlgError`` from
        singular covariance when features are collinear or a class
        has fewer samples than features (cf. ``reg_covar`` in
        ``sklearn.mixture.GaussianMixture`` and ``var_smoothing`` in
        ``GaussianNB``).
    priors : {'uniform', 'empirical'} or array-like of shape \
(n_classes,), default='uniform'
        Class priors. ``'uniform'`` gives every class equal prior
        (Lyubishchev's original assumption). ``'empirical'`` uses the
        observed class frequencies. An array is normalized to sum to
        one, ordered by ``classes_``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels seen during :meth:`fit`.
    means_ : dict
        Per-class mean vector (in the fitted, possibly standardized,
        space).
    covariances_ : dict
        Per-class regularized covariance matrix.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    scale_mean_, scale_std_ : ndarray or None
        Standardization parameters when ``standardize=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from lyubishchev import LyubishchevClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([
    ...     rng.multivariate_normal([0, 0], np.eye(2), 30),
    ...     rng.multivariate_normal([4, 4], np.eye(2), 30),
    ... ])
    >>> y = np.array([0] * 30 + [1] * 30)
    >>> clf = LyubishchevClassifier(standardize=True).fit(X, y)
    >>> clf.predict([[4, 4]])
    array([1])
    """

    def __init__(self, standardize=False, reg_covar=1e-6, priors="uniform"):
        self.standardize = standardize
        self.reg_covar = reg_covar
        self.priors = priors

    # -- sklearn tags (graceful across versions) --------------------
    def _more_tags(self):
        return {"requires_y": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    # -- fitting ----------------------------------------------------
    def fit(self, X, y):
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training measurements.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
        """
        if self.reg_covar < 0:
            raise ValueError(
                f"reg_covar must be non-negative, got {self.reg_covar}."
            )

        X, y = check_X_y(X, y, ensure_min_samples=1, dtype=np.float64)
        check_classification_targets(y)

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            raise ValueError(
                "Classifier can't train when only one class is present: "
                f"the data contains only one class ({self.classes_[0]!r})."
            )

        self.n_features_in_ = X.shape[1]

        # Standardization parameters from the full training set.
        if self.standardize:
            self.scale_mean_ = X.mean(axis=0)
            std = X.std(axis=0, ddof=0)
            std[std == 0] = 1.0  # leave constant features unscaled
            self.scale_std_ = std
            Xs = (X - self.scale_mean_) / self.scale_std_
        else:
            self.scale_mean_ = None
            self.scale_std_ = None
            Xs = X

        self.priors_ = self._resolve_priors(y_idx, n_classes)

        self.means_ = {}
        self.covariances_ = {}
        self._cov_inv = {}
        self._log_det = {}
        eye = np.eye(self.n_features_in_)

        for k, cls in enumerate(self.classes_):
            Xc = Xs[y_idx == k]
            mean = Xc.mean(axis=0)
            if Xc.shape[0] > 1:
                cov = np.cov(Xc, rowvar=False)
                cov = np.atleast_2d(cov)
            else:
                # Single sample: no covariance information.
                cov = np.zeros((self.n_features_in_, self.n_features_in_))
            cov = cov + self.reg_covar * eye

            self.means_[cls] = mean
            self.covariances_[cls] = cov
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                # Fall back to a stronger ridge if still degenerate.
                cov = cov + max(self.reg_covar, 1e-6) * eye
                self.covariances_[cls] = cov
                sign, log_det = np.linalg.slogdet(cov)
            self._cov_inv[cls] = np.linalg.pinv(cov)
            self._log_det[cls] = float(log_det)

        return self

    def _resolve_priors(self, y_idx, n_classes):
        priors = self.priors
        if isinstance(priors, str):
            if priors == "uniform":
                return np.full(n_classes, 1.0 / n_classes)
            if priors == "empirical":
                counts = np.bincount(y_idx, minlength=n_classes)
                return counts / counts.sum()
            raise ValueError(
                "priors must be 'uniform', 'empirical', or an array; "
                f"got {priors!r}."
            )
        priors = np.asarray(priors, dtype=np.float64)
        if priors.shape != (n_classes,):
            raise ValueError(
                f"priors has shape {priors.shape}, expected ({n_classes},)."
            )
        if np.any(priors < 0):
            raise ValueError("priors must be non-negative.")
        total = priors.sum()
        if total <= 0:
            raise ValueError("priors must sum to a positive value.")
        return priors / total

    # -- prediction -------------------------------------------------
    def _log_likelihood(self, X):
        """Per-class log-likelihood, vectorized over samples."""
        n_samples = X.shape[0]
        k = self.n_features_in_
        const = -0.5 * k * np.log(2.0 * np.pi)
        log_lik = np.empty((n_samples, self.classes_.shape[0]))

        for j, cls in enumerate(self.classes_):
            diff = X - self.means_[cls]                      # (n, k)
            cov_inv = self._cov_inv[cls]
            # Mahalanobis^2 for every row without a Python loop.
            maha_sq = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            log_lik[:, j] = const - 0.5 * self._log_det[cls] - 0.5 * maha_sq

        return log_lik

    def _joint_log_posterior(self, X):
        log_lik = self._log_likelihood(X)
        return log_lik + np.log(self.priors_)

    def _prepare(self, X):
        check_is_fitted(self)
        X = _check_X(self, X)
        X = check_array(X, dtype=np.float64)
        if self.standardize:
            X = (X - self.scale_mean_) / self.scale_std_
        return X

    def predict(self, X):
        """
        Predict the most probable class for each specimen.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels from ``classes_``.
        """
        X = self._prepare(X)
        scores = self._joint_log_posterior(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """
        Posterior class probabilities for each specimen.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Each row sums to 1, columns ordered as ``classes_``.
        """
        X = self._prepare(X)
        scores = self._joint_log_posterior(X)
        # Stable softmax across classes.
        scores -= scores.max(axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=1, keepdims=True)
        return scores

    def predict_log_proba(self, X):
        """
        Natural log of :meth:`predict_proba`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
        """
        return np.log(self.predict_proba(X))
