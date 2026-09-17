"""Read-only legacy and detector-only adapters for supported formats."""

from .detector import load_detector_npz, save_detector_npz
from .root import load_root_detector_tracks, load_root_tracks
from .tfrecord import (
    inspect_tfrecord,
    load_tfrecord_detector_tracks,
    load_tfrecord_record,
    load_tfrecord_truth,
)

__all__ = [
    "inspect_tfrecord",
    "load_detector_npz",
    "load_root_detector_tracks",
    "load_root_tracks",
    "load_tfrecord_detector_tracks",
    "load_tfrecord_record",
    "load_tfrecord_truth",
    "save_detector_npz",
]
