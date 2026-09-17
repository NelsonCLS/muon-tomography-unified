"""Read-only adapters for supported simulation formats."""

from .root import load_root_tracks
from .tfrecord import inspect_tfrecord, load_tfrecord_record

__all__ = ["inspect_tfrecord", "load_root_tracks", "load_tfrecord_record"]
