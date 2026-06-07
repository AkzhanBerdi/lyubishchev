"""
Tests for LyubishchevClassifier and the plotting module.

Where possible these use the Haltica oleracea / Haltica carduorum
setup from Lyubishchev's 1943 manuscript: two species inseparable on
single characters but cleanly separated in 2-D measurement space.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from lyubishchev import LyubishchevClassifier

RNG = np.random.default_rng(42)

OLERACEA = RNG.multivariate_normal([3.2, 1.5], [[0.04, 0.01], [0.01, 0.04]], 30)
CARDUORUM = RNG.multivariate_normal([3.8, 1.9], [[0.04, 0.01], [0.01, 0.04]], 30)
X = np.vstack([OLERACEA, CARDUORUM])
Y = np.array(["oleracea"] * 30 + ["carduorum"] * 30)


class TestFitPredict:
    def test_fit_returns_self(self):
        clf = LyubishchevClassifier()
        assert clf.fit(X, Y) is clf

    def test_predict_shape(self):
        clf = LyubishchevClassifier().fit(X, Y)
        pred = clf.predict(X)
        assert pred.shape == (X.shape[0],)

    def test_classifies_centroids(self):
        clf = LyubishchevClassifier().fit(X, Y)
        assert clf.predict([[3.2, 1.5]])[0] == "oleracea"
        assert clf.predict([[3.8, 1.9]])[0] == "carduorum"

    def test_high_train_accuracy(self):
        clf = LyubishchevClassifier().fit(X, Y)
        assert clf.score(X, Y) > 0.9

    def test_classes_attribute(self):
        clf = LyubishchevClassifier().fit(X, Y)
        np.testing.assert_array_equal(clf.classes_, ["carduorum", "oleracea"])

    def test_n_features_in(self):
        clf = LyubishchevClassifier().fit(X, Y)
        assert clf.n_features_in_ == 2

    def test_integer_labels(self):
        yi = np.array([0] * 30 + [1] * 30)
        clf = LyubishchevClassifier().fit(X, yi)
        assert set(clf.predict(X)).issubset({0, 1})


class TestPredictProba:
    def setup_method(self):
        self.clf = LyubishchevClassifier().fit(X, Y)

    def test_proba_shape(self):
        proba = self.clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_proba_rows_sum_to_one(self):
        proba = self.clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_proba_nonnegative(self):
        proba = self.clf.predict_proba(X)
        assert np.all(proba >= 0)

    def test_predict_matches_argmax_proba(self):
        proba = self.clf.predict_proba(X)
        argmax = self.clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(argmax, self.clf.predict(X))

    def test_log_proba_matches(self):
        lp = self.clf.predict_log_proba(X)
        np.testing.assert_allclose(np.exp(lp), self.clf.predict_proba(X), atol=1e-10)

    def test_vectorized_matches_singleton(self):
        # Batch prediction must equal row-by-row prediction.
        batch = self.clf.predict_proba(X)
        single = np.vstack([self.clf.predict_proba(x.reshape(1, -1)) for x in X])
        np.testing.assert_allclose(batch, single, atol=1e-12)

    def test_mahalanobis_matches_explicit_quadratic_form(self):
        # The Cholesky-whitening predict path must produce exactly the
        # same log-likelihoods as the textbook diff @ cov^-1 @ diff form.
        # This guards against a future refactor silently changing the math.
        ll = self.clf._log_likelihood(X)
        k = self.clf.n_features_in_
        const = -0.5 * k * np.log(2.0 * np.pi)
        expected = np.empty_like(ll)
        for j, cls in enumerate(self.clf.classes_):
            cov = self.clf.covariances_[cls]
            cov_inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            diff = X - self.clf.means_[cls]
            maha = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            expected[:, j] = const - 0.5 * logdet - 0.5 * maha
        np.testing.assert_allclose(ll, expected, rtol=1e-9, atol=1e-9)


class TestStandardize:
    def test_standardize_scale_invariance(self):
        # Blow up the second feature by 1000x; standardize should
        # recover essentially the same predictions.
        Xs = X.copy()
        Xs[:, 1] *= 1000.0
        clf = LyubishchevClassifier(standardize=True).fit(Xs, Y)
        assert clf.score(Xs, Y) > 0.9

    def test_scale_params_set(self):
        clf = LyubishchevClassifier(standardize=True).fit(X, Y)
        assert clf.scale_mean_ is not None
        assert clf.scale_std_ is not None

    def test_scale_params_none_when_off(self):
        clf = LyubishchevClassifier(standardize=False).fit(X, Y)
        assert clf.scale_mean_ is None

    def test_constant_feature_no_nan(self):
        Xc = np.hstack([X, np.ones((X.shape[0], 1))])  # constant column
        clf = LyubishchevClassifier(standardize=True).fit(Xc, Y)
        assert not np.any(np.isnan(clf.predict_proba(Xc)))


class TestRegularization:
    def test_collinear_features_no_error(self):
        # Perfectly correlated features -> singular covariance.
        Xcol = np.hstack([X[:, :1], X[:, :1]])
        clf = LyubishchevClassifier(reg_covar=1e-6).fit(Xcol, Y)
        proba = clf.predict_proba(Xcol)
        assert not np.any(np.isnan(proba))

    def test_n_samples_lt_n_features(self):
        Xw = RNG.normal(size=(6, 10))
        yw = np.array([0, 0, 0, 1, 1, 1])
        clf = LyubishchevClassifier(reg_covar=1e-3).fit(Xw, yw)
        assert clf.predict(Xw).shape == (6,)

    def test_negative_reg_covar_raises(self):
        with pytest.raises(ValueError):
            LyubishchevClassifier(reg_covar=-1.0).fit(X, Y)


class TestPriors:
    def test_uniform_default(self):
        clf = LyubishchevClassifier().fit(X, Y)
        np.testing.assert_allclose(clf.priors_, [0.5, 0.5])

    def test_empirical(self):
        yimb = np.array(["a"] * 40 + ["b"] * 20)
        clf = LyubishchevClassifier(priors="empirical").fit(X, yimb)
        np.testing.assert_allclose(sorted(clf.priors_), [1 / 3, 2 / 3])

    def test_array_priors_normalized(self):
        clf = LyubishchevClassifier(priors=[3, 1]).fit(X, Y)
        np.testing.assert_allclose(clf.priors_.sum(), 1.0)

    def test_bad_prior_string_raises(self):
        with pytest.raises(ValueError):
            LyubishchevClassifier(priors="nonsense").fit(X, Y)

    def test_wrong_prior_shape_raises(self):
        with pytest.raises(ValueError):
            LyubishchevClassifier(priors=[1, 1, 1]).fit(X, Y)


class TestInputValidation:
    def test_nan_raises(self):
        Xn = X.copy()
        Xn[0, 0] = np.nan
        with pytest.raises(ValueError):
            LyubishchevClassifier().fit(Xn, Y)

    def test_inf_raises(self):
        Xi = X.copy()
        Xi[0, 0] = np.inf
        with pytest.raises(ValueError):
            LyubishchevClassifier().fit(Xi, Y)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            LyubishchevClassifier().fit(X, np.zeros(X.shape[0]))

    def test_predict_before_fit_raises(self):
        from sklearn.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            LyubishchevClassifier().predict(X)

    def test_wrong_n_features_raises(self):
        clf = LyubishchevClassifier().fit(X, Y)
        with pytest.raises(ValueError):
            clf.predict(np.zeros((3, 5)))

    def test_predict_nan_raises(self):
        clf = LyubishchevClassifier().fit(X, Y)
        with pytest.raises(ValueError):
            clf.predict([[np.nan, 1.0]])


class TestMulticlass:
    def test_three_classes(self):
        X3 = np.vstack([
            RNG.multivariate_normal([0, 0], np.eye(2) * 0.5, 25),
            RNG.multivariate_normal([5, 0], np.eye(2) * 0.5, 25),
            RNG.multivariate_normal([0, 5], np.eye(2) * 0.5, 25),
        ])
        y3 = np.array(["a"] * 25 + ["b"] * 25 + ["c"] * 25)
        clf = LyubishchevClassifier().fit(X3, y3)
        assert clf.predict([[4.9, 0.1]])[0] == "b"
        assert clf.predict_proba(X3).shape == (75, 3)


class TestSklearnCompat:
    def test_clone(self):
        clf = LyubishchevClassifier(standardize=True, reg_covar=1e-4)
        cloned = clone(clf)
        assert cloned.get_params() == clf.get_params()

    def test_get_set_params(self):
        clf = LyubishchevClassifier()
        clf.set_params(standardize=True)
        assert clf.get_params()["standardize"] is True

    def test_pipeline(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LyubishchevClassifier()),
        ])
        pipe.fit(X, Y)
        assert pipe.score(X, Y) > 0.9

    def test_cross_val_score(self):
        scores = cross_val_score(LyubishchevClassifier(), X, Y, cv=3)
        assert scores.mean() > 0.8

    def test_grid_search(self):
        grid = GridSearchCV(
            LyubishchevClassifier(),
            {"standardize": [True, False], "reg_covar": [1e-6, 1e-3]},
            cv=3,
        )
        grid.fit(X, Y)
        assert hasattr(grid, "best_params_")

    def test_check_estimator(self):
        # Full scikit-learn estimator API conformance.
        check_estimator(LyubishchevClassifier())


class TestPlot:
    def test_plot_ellipses_runs(self):
        mpl = pytest.importorskip("matplotlib")
        mpl.use("Agg")
        from lyubishchev import scatter_ellipse
        from lyubishchev.plot import plot_ellipses
        ax = plot_ellipses(scatter_ellipse(X, Y), X, Y)
        assert ax is not None

    def test_plot_classification_runs(self):
        mpl = pytest.importorskip("matplotlib")
        mpl.use("Agg")
        from lyubishchev.plot import plot_classification
        clf = LyubishchevClassifier().fit(X, Y)
        ax = plot_classification(clf, X, Y)
        assert ax is not None

    def test_plot_classification_rejects_3d(self):
        pytest.importorskip("matplotlib")
        from lyubishchev.plot import plot_classification
        X3 = RNG.normal(size=(20, 3))
        y3 = np.array([0] * 10 + [1] * 10)
        clf = LyubishchevClassifier().fit(X3, y3)
        with pytest.raises(ValueError):
            plot_classification(clf, X3, y3)
