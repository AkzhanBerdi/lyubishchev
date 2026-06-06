"""
lyubishchev
~~~~~~~~~~~

Quantitative taxonomy methods of Alexander Alexandrovich Lyubishchev
(1890–1972), implemented for the modern Python scientific stack.

Lyubishchev described multivariate classification by covariance
structure in his 1943 manuscript *Programma obshchey sistematiki*
(Program of General Systematics) — twenty years before Sokal &
Sneath's *Principles of Numerical Taxonomy* (1963), whose binary
similarity coefficients are memorialized in scipy.spatial.distance
as ``sokalsneath`` and ``sokalmichener``. This package puts
Lyubishchev's name into the same ecosystem.

Primary source:
    Lyubishchev, A.A. (1943). Programma obshchey sistematiki.
    Manuscript, 22 November 1943. Digitized by ZIN RAS Coleoptera
    Laboratory. http://www.zin.ru/animalia/coleoptera/rus/lyubis05.htm

Western publication:
    Lubischew, A.A. (1962). On the use of discriminant functions in
    taxonomy. Biometrics, 18(4), 455–477.
"""

from lyubishchev.core import (
    classify,
    divergence_coefficient,
    scatter_ellipse,
    transgression,
)
from lyubishchev.estimator import LyubishchevClassifier

__version__ = "0.2.0"

__all__ = [
    "divergence_coefficient",
    "scatter_ellipse",
    "transgression",
    "classify",
    "LyubishchevClassifier",
    "__version__",
]
