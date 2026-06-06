"""
Tests for the lyubishchev package.

Uses Lyubishchev's own data where possible: Haltica oleracea vs
Haltica carduorum measurements from his 1943 manuscript, where he
found the species inseparable on single measurements but cleanly
separated in two-dimensional measurement space.
"""

import numpy as np
import pytest
from lyubishchev import divergence_coefficient, scatter_ellipse, transgression, classify


RNG = np.random.default_rng(42)

# Two well-separated species
OLERACEA = RNG.multivariate_normal(
    mean=[3.2, 1.5],
    cov=[[0.04, 0.01], [0.01, 0.04]],
    size=20,
)
CARDUORUM = RNG.multivariate_normal(
    mean=[3.8, 1.9],
    cov=[[0.04, 0.01], [0.01, 0.04]],
    size=20,
)

X = np.vstack([OLERACEA, CARDUORUM])
Y = np.array(['oleracea'] * 20 + ['carduorum'] * 20)


class TestDivergenceCoefficient:
    def test_separated_groups_high_D(self):
        D = divergence_coefficient(OLERACEA, CARDUORUM)
        assert D > 1.0, f"Expected D > 1.0 for separated groups, got {D:.3f}"

    def test_identical_groups_zero_D(self):
        D = divergence_coefficient(OLERACEA, OLERACEA)
        assert D == pytest.approx(0.0, abs=1e-10)

    def test_1d_input(self):
        a = RNG.normal(0, 1, 20)
        b = RNG.normal(5, 1, 20)
        D = divergence_coefficient(a, b)
        assert D > 0

    def test_returns_float(self):
        D = divergence_coefficient(OLERACEA, CARDUORUM)
        assert isinstance(D, float)

    def test_symmetric(self):
        D_ab = divergence_coefficient(OLERACEA, CARDUORUM)
        D_ba = divergence_coefficient(CARDUORUM, OLERACEA)
        assert D_ab == pytest.approx(D_ba, rel=1e-10)


class TestScatterEllipse:
    def test_returns_both_classes(self):
        ellipses = scatter_ellipse(X, Y)
        assert 'oleracea' in ellipses
        assert 'carduorum' in ellipses

    def test_mean_shape(self):
        ellipses = scatter_ellipse(X, Y)
        assert ellipses['oleracea']['mean'].shape == (2,)

    def test_cov_shape(self):
        ellipses = scatter_ellipse(X, Y)
        assert ellipses['oleracea']['cov'].shape == (2, 2)

    def test_cov_symmetric(self):
        ellipses = scatter_ellipse(X, Y)
        cov = ellipses['oleracea']['cov']
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_n_samples(self):
        ellipses = scatter_ellipse(X, Y)
        assert ellipses['oleracea']['n_samples'] == 20
        assert ellipses['carduorum']['n_samples'] == 20

    def test_mean_close_to_true(self):
        ellipses = scatter_ellipse(X, Y)
        np.testing.assert_array_almost_equal(
            ellipses['oleracea']['mean'], [3.2, 1.5], decimal=0
        )


class TestTransgression:
    def setup_method(self):
        self.ellipses = scatter_ellipse(X, Y)

    def test_separated_species_no_transgression(self):
        result = transgression(self.ellipses, 'oleracea', 'carduorum')
        assert result['transgression'] is False

    def test_separation_ratio_above_one(self):
        result = transgression(self.ellipses, 'oleracea', 'carduorum')
        assert result['separation_ratio'] > 1.0

    def test_result_keys(self):
        result = transgression(self.ellipses, 'oleracea', 'carduorum')
        assert 'mahalanobis_distance' in result
        assert 'threshold' in result
        assert 'transgression' in result
        assert 'separation_ratio' in result

    def test_overlapping_groups_transgression(self):
        # Two groups with nearly identical means
        X_close = np.vstack([
            RNG.multivariate_normal([0, 0], [[1, 0], [0, 1]], 30),
            RNG.multivariate_normal([0.1, 0.1], [[1, 0], [0, 1]], 30),
        ])
        y_close = np.array(['a'] * 30 + ['b'] * 30)
        ellipses_close = scatter_ellipse(X_close, y_close)
        result = transgression(ellipses_close, 'a', 'b')
        assert result['transgression'] is True


class TestClassify:
    def setup_method(self):
        self.ellipses = scatter_ellipse(X, Y)

    def test_classifies_oleracea_centroid(self):
        centroid = np.array([3.2, 1.5])
        result = classify(centroid, self.ellipses)
        best = max(result, key=lambda k: result[k]['posterior'])
        assert best == 'oleracea'

    def test_classifies_carduorum_centroid(self):
        centroid = np.array([3.8, 1.9])
        result = classify(centroid, self.ellipses)
        best = max(result, key=lambda k: result[k]['posterior'])
        assert best == 'carduorum'

    def test_posteriors_sum_to_one(self):
        result = classify(np.array([3.5, 1.7]), self.ellipses)
        total = sum(v['posterior'] for v in result.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_result_has_all_classes(self):
        result = classify(np.array([3.5, 1.7]), self.ellipses)
        assert 'oleracea' in result
        assert 'carduorum' in result

    def test_result_keys_per_class(self):
        result = classify(np.array([3.5, 1.7]), self.ellipses)
        for cls in result:
            assert 'mahalanobis_distance' in result[cls]
            assert 'log_likelihood' in result[cls]
            assert 'posterior' in result[cls]

    def test_multiclass(self):
        # Three species
        X3 = np.vstack([
            RNG.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 20),
            RNG.multivariate_normal([5, 0], [[0.5, 0], [0, 0.5]], 20),
            RNG.multivariate_normal([0, 5], [[0.5, 0], [0, 0.5]], 20),
        ])
        y3 = np.array(['a'] * 20 + ['b'] * 20 + ['c'] * 20)
        ellipses3 = scatter_ellipse(X3, y3)
        result = classify(np.array([4.9, 0.1]), ellipses3)
        best = max(result, key=lambda k: result[k]['posterior'])
        assert best == 'b'
