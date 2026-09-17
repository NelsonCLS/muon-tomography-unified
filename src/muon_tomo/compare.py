"""Backward-compatible evaluation entry point."""

from .evaluation.metrics import compare_to_truth

__all__ = ["compare_to_truth"]
