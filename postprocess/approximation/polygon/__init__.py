"""Per-frame polygon approximation algorithms."""

from .rdp import OpenCvRdpApproximator, PolygonApproximator, approximate_sqlite

__all__ = ["OpenCvRdpApproximator", "PolygonApproximator", "approximate_sqlite"]
