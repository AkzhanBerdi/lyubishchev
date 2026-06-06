"""
lyubishchev.plot
~~~~~~~~~~~~~~~~~

Visualization of Lyubishchev's scatter ellipses — a digital
reproduction of Fig. 1 (Рис. 1) from his 1943 manuscript, where two
beetle species inseparable on any single character separate cleanly
as overlapping (or disjoint) covariance ellipses in two dimensions.

Requires matplotlib (install with ``pip install lyubishchev[plot]``).
"""

import numpy as np

from lyubishchev.core import scatter_ellipse


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.patches import Ellipse  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Plotting requires matplotlib. Install it with "
            "`pip install lyubishchev[plot]` or `pip install matplotlib`."
        ) from exc


def _ellipse_patch(mean, cov, confidence, **kwargs):
    """Build a matplotlib Ellipse for a confidence contour."""
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2

    # Eigen-decomposition gives axis directions and lengths.
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0, None)

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(chi2.ppf(confidence, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def plot_ellipses(
    ellipses,
    X=None,
    y=None,
    confidence=0.95,
    ax=None,
    feature_names=None,
    title="Lyubishchev scatter ellipses (after Рис. 1, 1943)",
):
    """
    Draw per-class covariance ellipses, optionally over the raw points.

    Parameters
    ----------
    ellipses : dict
        Output of :func:`lyubishchev.scatter_ellipse`. Only the first
        two feature dimensions are drawn.
    X : array-like of shape (n_samples, n_features), optional
        Raw measurements to scatter beneath the ellipses.
    y : array-like of shape (n_samples,), optional
        Class labels matching ``X``.
    confidence : float, default=0.95
        Confidence level of the drawn contour.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    feature_names : sequence of str, optional
        Axis labels for the first two features.
    title : str
        Plot title.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    labels = list(ellipses.keys())
    cmap = plt.get_cmap("tab10")
    colors = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    if X is not None and y is not None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        for lab in labels:
            pts = X[y == lab]
            if pts.size:
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=24, alpha=0.55, color=colors[lab],
                    edgecolors="none",
                )

    for lab in labels:
        mean = np.asarray(ellipses[lab]["mean"], dtype=float)[:2]
        cov = np.asarray(ellipses[lab]["cov"], dtype=float)[:2, :2]
        patch = _ellipse_patch(
            mean, cov, confidence,
            facecolor="none", edgecolor=colors[lab], lw=2,
        )
        ax.add_patch(patch)
        ax.scatter(*mean, marker="x", s=70, color=colors[lab], label=str(lab))

    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_classification(
    clf,
    X,
    y=None,
    resolution=300,
    ax=None,
    feature_names=None,
    title="Lyubishchev classification regions",
):
    """
    Plot decision regions of a fitted :class:`LyubishchevClassifier`.

    Only the first two features are used; the classifier must have
    been fitted on two-dimensional data for the regions to be exact.

    Parameters
    ----------
    clf : LyubishchevClassifier
        A fitted classifier.
    X : array-like of shape (n_samples, 2)
        Points to scatter and to bound the region grid.
    y : array-like of shape (n_samples,), optional
        Labels for coloring the scattered points.
    resolution : int, default=300
        Grid resolution per axis for the region mesh.
    ax : matplotlib.axes.Axes, optional
    feature_names : sequence of str, optional
    title : str

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise ValueError(
            "plot_classification needs 2-D data; got "
            f"{X.shape[1]} features."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    pad_x = 0.5 * (X[:, 0].max() - X[:, 0].min() + 1e-9)
    pad_y = 0.5 * (X[:, 1].max() - X[:, 1].min() + 1e-9)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, resolution),
        np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    classes = list(clf.classes_)
    pred = clf.predict(grid)
    idx = np.searchsorted(clf.classes_, pred).reshape(xx.shape)

    n = len(classes)
    base = plt.get_cmap("tab10")
    region_cmap = ListedColormap([base(i % 10) for i in range(n)])
    ax.contourf(
        xx, yy, idx, levels=np.arange(-0.5, n, 1),
        cmap=region_cmap, alpha=0.25,
    )

    if y is not None:
        y = np.asarray(y)
        for i, cls in enumerate(classes):
            pts = X[y == cls]
            if pts.size:
                ax.scatter(
                    pts[:, 0], pts[:, 1], s=28,
                    color=base(i % 10), edgecolors="k", linewidths=0.4,
                    label=str(cls),
                )
        ax.legend(loc="best", frameon=False)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=20, color="k", alpha=0.6)

    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    ax.set_title(title)
    return ax


__all__ = ["plot_ellipses", "plot_classification"]
