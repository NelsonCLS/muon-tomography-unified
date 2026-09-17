"""Evaluation-only functions; never imported by reconstruction algorithms."""

from .metrics import compare_to_truth

__all__ = ["compare_to_truth"]
