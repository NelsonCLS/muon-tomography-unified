"""Custom material grids and an optional Geant4-to-TFRecord workflow.

Importing this package does not import TensorFlow or require Geant4.
"""

from .objects import create_voxels, load_voxels, save_voxels

__all__ = ["create_voxels", "load_voxels", "save_voxels"]
