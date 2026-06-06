"""
lyubishchev.core
~~~~~~~~~~~~~~~~

Implementations of Alexander Alexandrovich Lyubishchev's (1890–1972)
quantitative taxonomy methods, as described in his 1943 manuscript:

    Lyubishchev, A.A. (1943). Programma obshchey sistematiki
    [Program of General Systematics]. Manuscript, 22 November 1943.
    Digitized by ZIN RAS Coleoptera Laboratory.
    http://www.zin.ru/animalia/coleoptera/rus/lyubis05.htm

These methods predate and are mathematically more general than the
binary-character similarity coefficients of Sokal & Sneath (1963),
which are memorialized in scipy.spatial.distance as sokalsneath and
sokalmichener. Lyubishchev worked with continuous measurements and
full covariance structure — the formulation now standard in
multivariate statistics and machine learning.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


def divergence_coefficient(a, b):
    """
    Compute Lyubishchev's divergence coefficient D between two groups.

    Defined in his 1943 manuscript as:

        D = (M₁ - M₂)² / (σ₁² + σ₂²)

    where M₁, M₂ are group means and σ₁², σ₂² are group variances.
    For multivariate data, D is computed per dimension and summed.

    When D is large, the groups are cleanly separated in measurement
    space. When D is small, you have transgression — the classical
    boundary between taxa breaks down.

    Parameters
    ----------
    a : array-like, shape (n_samples,) or (n_samples, n_features)
        Measurements for group A (e.g. one species).
    b : array-like, shape (n_samples,) or (n_samples, n_features)
        Measurements for group B (e.g. another species).

    Returns
    -------
    D : float
        Divergence coefficient. Values above 1.0 indicate clean
        separation. Values below 0.5 indicate strong transgression.

    Examples
    --------
    >>> import numpy as np
    >>> from lyubishchev import divergence_coefficient
    >>> rng = np.random.default_rng(42)
    >>> haltica_oleracea = rng.normal(loc=[3.2, 1.5], scale=0.2, size=(20, 2))
    >>> haltica_carduorum = rng.normal(loc=[3.8, 1.9], scale=0.2, size=(20, 2))
    >>> divergence_coefficient(haltica_oleracea, haltica_carduorum)
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))

    if a.shape[0] == 1:
        a = a.T
    if b.shape[0] == 1:
        b = b.T

    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    var_a = np.var(a, axis=0, ddof=1)
    var_b = np.var(b, axis=0, ddof=1)

    pooled_var = var_a + var_b
    # Avoid division by zero for constant features
    mask = pooled_var > 0
    if not np.any(mask):
        return 0.0

    D = np.sum((mean_a[mask] - mean_b[mask]) ** 2 / pooled_var[mask])
    return float(D)


def scatter_ellipse(X, y):
    """
    Fit covariance ellipses per class, as Lyubishchev did graphically
    in his 1943 manuscript (Fig. 1 — Рис. 1).

    Each class is represented by its centroid and covariance matrix,
    defining an ellipse of equal probability density in measurement
    space. Overlap between ellipses is the multivariate equivalent of
    Lyubishchev's "transgression" — the failure of a single character
    to separate two taxa.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Measurement matrix. Each row is a specimen, each column a
        morphological measurement.
    y : array-like, shape (n_samples,)
        Class labels (taxon names or integer codes).

    Returns
    -------
    ellipses : dict
        Keys are unique class labels. Values are dicts with:
            'mean'       : ndarray, shape (n_features,)
            'cov'        : ndarray, shape (n_features, n_features)
            'n_samples'  : int

    Examples
    --------
    >>> import numpy as np
    >>> from lyubishchev import scatter_ellipse
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([
    ...     rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 30),
    ...     rng.multivariate_normal([3, 3], [[1, 0.5], [0.5, 1]], 30),
    ... ])
    >>> y = ['Haltica oleracea'] * 30 + ['Haltica carduorum'] * 30
    >>> ellipses = scatter_ellipse(X, y)
    >>> ellipses['Haltica oleracea']['mean']
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)

    ellipses = {}
    for cls in classes:
        mask = y == cls
        X_cls = X[mask]
        ellipses[cls] = {
            'mean': np.mean(X_cls, axis=0),
            'cov': np.cov(X_cls, rowvar=False) if X_cls.shape[0] > 1 else np.eye(X.shape[1]),
            'n_samples': int(np.sum(mask)),
        }
    return ellipses


def transgression(ellipses, class_a, class_b, confidence=0.95):
    """
    Compute the transgression (overlap) between two scatter ellipses.

    Lyubishchev defined transgression as the proportion of specimens
    that fall within the boundary region of both groups. This function
    estimates it by computing the Mahalanobis distance between the two
    group centroids and comparing it to the chi-squared threshold for
    the given confidence level.

    Parameters
    ----------
    ellipses : dict
        Output of scatter_ellipse().
    class_a : label
        First class label.
    class_b : label
        Second class label.
    confidence : float, default 0.95
        Confidence level for the ellipse boundary.

    Returns
    -------
    result : dict
        'mahalanobis_distance' : float
            Distance between centroids in Mahalanobis units.
        'threshold' : float
            Chi-squared threshold at the given confidence level.
        'transgression' : bool
            True if the ellipses overlap (distance < threshold).
        'separation_ratio' : float
            mahalanobis_distance / threshold. Values > 1.0 mean
            clean separation. Values < 1.0 mean transgression.

    Examples
    --------
    >>> result = transgression(ellipses, 'Haltica oleracea', 'Haltica carduorum')
    >>> result['transgression']
    False
    """
    ea = ellipses[class_a]
    eb = ellipses[class_b]

    n_features = len(ea['mean'])
    threshold = np.sqrt(chi2.ppf(confidence, df=n_features))

    # Pooled covariance
    na, nb = ea['n_samples'], eb['n_samples']
    pooled_cov = (na * ea['cov'] + nb * eb['cov']) / (na + nb)

    try:
        dist = mahalanobis(ea['mean'], eb['mean'], np.linalg.inv(pooled_cov))
    except np.linalg.LinAlgError:
        dist = np.linalg.norm(ea['mean'] - eb['mean'])

    return {
        'mahalanobis_distance': float(dist),
        'threshold': float(threshold),
        'transgression': bool(dist < threshold),
        'separation_ratio': float(dist / threshold) if threshold > 0 else 0.0,
    }


def classify(specimen, ellipses):
    """
    Classify a specimen using the Edgeworth-Pearson multivariate
    probability function, as described by Lyubishchev in his 1943
    manuscript.

    This is the mathematical core of his paper nomograms (Fig. 3 —
    Рис. 3): given a specimen's measurements and a set of reference
    groups with known means and covariance matrices, return the
    posterior probability of belonging to each group (assuming equal
    priors).

    Parameters
    ----------
    specimen : array-like, shape (n_features,)
        Measurements of the specimen to classify.
    ellipses : dict
        Output of scatter_ellipse(). Each entry must have 'mean',
        'cov', and 'n_samples'.

    Returns
    -------
    result : dict
        Keys are class labels. Values are dicts with:
            'mahalanobis_distance' : float
                Distance from specimen to class centroid.
            'log_likelihood'       : float
            'posterior'            : float
                Posterior probability (sums to 1.0 across classes).

    Examples
    --------
    >>> from lyubishchev import scatter_ellipse, classify
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> X = np.vstack([
    ...     rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 20),
    ...     rng.multivariate_normal([4, 4], [[1, 0.3], [0.3, 1]], 20),
    ... ])
    >>> y = ['oleracea'] * 20 + ['carduorum'] * 20
    >>> ellipses = scatter_ellipse(X, y)
    >>> specimen = np.array([3.8, 3.9])
    >>> result = classify(specimen, ellipses)
    >>> max(result, key=lambda k: result[k]['posterior'])
    'carduorum'
    """
    specimen = np.asarray(specimen, dtype=float)
    log_likelihoods = {}

    for cls, params in ellipses.items():
        mean = params['mean']
        cov = params['cov']
        n = params['n_samples']
        k = len(mean)

        try:
            cov_inv = np.linalg.inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                raise np.linalg.LinAlgError("Non-positive definite covariance")
        except np.linalg.LinAlgError:
            cov_inv = np.eye(k)
            log_det = 0.0

        diff = specimen - mean
        maha = float(np.sqrt(diff @ cov_inv @ diff))
        log_ll = -0.5 * (k * np.log(2 * np.pi) + log_det + diff @ cov_inv @ diff)

        log_likelihoods[cls] = {
            'mahalanobis_distance': maha,
            'log_likelihood': float(log_ll),
        }

    # Softmax over log-likelihoods for numerical stability
    max_ll = max(v['log_likelihood'] for v in log_likelihoods.values())
    exp_lls = {cls: np.exp(v['log_likelihood'] - max_ll)
               for cls, v in log_likelihoods.items()}
    total = sum(exp_lls.values())

    result = {}
    for cls in log_likelihoods:
        result[cls] = {
            **log_likelihoods[cls],
            'posterior': float(exp_lls[cls] / total),
        }

    return result
