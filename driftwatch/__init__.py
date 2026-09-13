"""driftwatch - drift monitoring for models already in production.

No scipy, no sklearn: the statistics used here (Kolmogorov, chi-square, incomplete gamma,
Benjamini-Hochberg, AUC, Page-Hinkley) are implemented in :mod:`driftwatch.stats` and
:mod:`driftwatch.metrics` and checked against published critical values in the test suite.
That keeps the service image small and makes every number auditable.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
