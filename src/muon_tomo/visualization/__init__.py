"""Presentation-only views; prediction arrays are never modified."""

from .sections import plot_comparison_slab
from .viewer import plot_comparison_3d

__all__ = ["plot_comparison_3d", "plot_comparison_slab"]
