"""Opt-in, simulation-explicit momentum-policy experiments; no production changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from muon_tomo.constants import MATERIAL_NAMES, labels_to_inverse_radiation_length
from muon_tomo.geometry import GridGeometry
from muon_tomo.reconstruction.munet import (
    MUNET_TRAINING_MOMENTUM_MEAN_MEV,
    MUNET_TRAINING_MOMENTUM_STD_MEV,
    MuNetReconstructor,
)
from muon_tomo.reconstruction.paths import (
    closest_approach,
    physical_grid_intersections,
    physical_track_order,
    prepare_poca_paths,
)

HISTORICAL_MEV = 1743.4601815051958
ZENITH_EDGES = np.asarray([*range(0, 50, 5), 90.0])
POLICIES = {
    "historical": ("Historical 1.743 GeV", "historical simulation prior"),
    "ref_arithmetic": ("Reference arithmetic", "independent simulation prior"),
    "ref_geometric": ("Reference geometric", "independent simulation prior"),
    "ref_percentile10": ("Reference 10th percentile", "independent simulation prior"),
    "ref_median": ("Reference median", "independent simulation prior"),
    "ref_inverse_square": ("Reference inverse-square", "independent simulation prior"),
    "ref_zenith_arithmetic": ("Reference zenith arithmetic", "independent simulation prior"),
    "ref_zenith_inverse_square": (
        "Reference zenith inverse-square",
        "independent simulation prior",
    ),
    "truth_outgoing": ("Truth outgoing / track", "event oracle"),
    "truth_incoming": ("Truth initial / track", "event oracle"),
    "truth_endpoint_geometric": ("Truth initial/out geometric", "event oracle"),
    "object_arithmetic": ("Object arithmetic", "object oracle"),
    "object_geometric": ("Object geometric", "object oracle"),
    "object_inverse_square": ("Object inverse-square", "object oracle"),
    "object_zenith_arithmetic": ("Object zenith arithmetic", "object oracle"),
    "object_zenith_inverse_square": ("Object zenith inverse-square", "object oracle"),
}


def positive_momenta(p, n=None):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not p.size or (n is not None and len(p) != n):
        raise ValueError("momenta must be a nonempty vector with one entry per track")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("momenta must be finite and positive")
    return p


def momentum_statistics(p):
    p = positive_momenta(p)
    return {
        "arithmetic": float(p.mean()),
        "geometric": float(np.exp(np.log(p).mean())),
        "percentile10": float(np.percentile(p, 10, method="linear")),
        "median": float(np.median(p)),
        "inverse_square": float(1.0 / np.sqrt(np.mean(p**-2))),
    }


def initial_momentum_from_record(x):
    """Legacy x[:,9:12] is a unit direction; x[:,12] is vertex kinetic energy.

    Return initial/source momentum, not a momentum measured at the upper plane.
    Muon rest mass is 105.6583755 MeV/c^2. The conversion is p=sqrt(T*(T+2m)).
    """
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] < 13:
        raise ValueError("initial-momentum oracle requires kinetic-energy column 12")
    kinetic = positive_momenta(x[:, 12])
    return np.sqrt(kinetic * (kinetic + 2 * 105.6583755))


def incoming_zenith(tracks):
    _, incoming, _, _ = physical_track_order(tracks)
    return np.degrees(np.arccos(np.clip(np.abs(incoming[:, 2]), 0, 1)))


def zenith_bins(angles):
    angles = np.asarray(angles, dtype=float)
    if not np.all(np.isfinite(angles)) or np.any((angles < 0) | (angles > 90)):
        raise ValueError("zenith angles must lie in [0,90] degrees")
    return np.minimum(
        np.searchsorted(ZENITH_EDGES[1:], angles, side="right"), len(ZENITH_EDGES) - 2
    )


@dataclass
class MomentumPrior:
    global_stats: dict
    bin_stats: dict
    counts: np.ndarray
    minimum_count: int

    @classmethod
    def fit(cls, p, angles, minimum_count=100):
        p = positive_momenta(p)
        bins = zenith_bins(angles)
        if bins.shape != p.shape:
            raise ValueError("angle and momentum arrays must match")
        stats = momentum_statistics(p)
        counts = np.bincount(bins, minlength=len(ZENITH_EDGES) - 1)
        per_bin = {key: np.full(len(counts), value) for key, value in stats.items()}
        for j, count in enumerate(counts):
            if count >= minimum_count:
                for key, value in momentum_statistics(p[bins == j]).items():
                    per_bin[key][j] = value
        return cls(stats, per_bin, counts, minimum_count)

    def assign(self, kind, angles, conditional=False):
        bins = zenith_bins(angles)
        return (
            self.bin_stats[kind][bins].copy()
            if conditional
            else np.full(bins.shape, self.global_stats[kind])
        )

    def metadata(self):
        return {
            "global_mev": self.global_stats,
            "bin_mev": {k: v.tolist() for k, v in self.bin_stats.items()},
            "edges_degrees": ZENITH_EDGES.tolist(),
            "counts": self.counts.tolist(),
            "minimum_count": self.minimum_count,
            "global_fallback_bins": np.flatnonzero(self.counts < self.minimum_count).tolist(),
        }


def assigned_policies(reference, angles, p_out, p_in):
    """Explicit oracle inputs are never used to fit the independent reference."""
    p_out = positive_momenta(p_out)
    p_in = positive_momenta(p_in, len(p_out)) if p_in is not None else None
    local = MomentumPrior.fit(p_out, angles, minimum_count=20)
    assigned = {"historical": np.full(len(p_out), HISTORICAL_MEV)}
    for prefix, prior in (("ref", reference), ("object", local)):
        for kind in ("arithmetic", "geometric", "inverse_square"):
            assigned[f"{prefix}_{kind}"] = prior.assign(kind, angles)
        for kind in ("arithmetic", "inverse_square"):
            assigned[f"{prefix}_zenith_{kind}"] = prior.assign(kind, angles, True)
    assigned["ref_percentile10"] = reference.assign("percentile10", angles)
    assigned["ref_median"] = reference.assign("median", angles)
    assigned["truth_outgoing"] = p_out.copy()
    if p_in is not None:
        assigned.update(truth_incoming=p_in.copy(), truth_endpoint_geometric=np.sqrt(p_in * p_out))
    return {k: assigned[k] for k in POLICIES if k in assigned}, local


def momentum_features(tracks, p):
    """Preserve all 13 other production features bit-for-bit."""
    p = positive_momenta(p, len(tracks))
    features = MuNetReconstructor()._features(tracks)
    features[:, 12] = (
        (p - MUNET_TRAINING_MOMENTUM_MEAN_MEV) / MUNET_TRAINING_MOMENTUM_STD_MEV
    ).astype(np.float32)
    return features


class CachedPoCA:
    """Production 32³/sigma=6.25 cm calculation with geometry reused."""

    def __init__(self, tracks):
        geometry = GridGeometry()
        paths, lengths, _, _ = prepare_poca_paths(tracks, geometry)
        starts, incoming, ends, outgoing = physical_grid_intersections(tracks, geometry)
        dtx, dty = tracks.projected_scattering_angles()
        self.q_per_p2 = 0.5 * (dtx**2 + dty**2) / 15.0**2
        self.n = len(tracks)
        self.event_cells = np.full(self.n, -1, dtype=int)
        exposure = np.zeros(32**3, dtype=float)
        counts = np.zeros(32**3, dtype=np.int64)
        for i, (ids, lens) in enumerate(zip(paths, lengths, strict=True)):
            if not len(ids):
                continue
            coarse = np.asarray(np.unravel_index(ids, geometry.shape)).T // 2
            np.add.at(exposure, np.ravel_multi_index(coarse.T, (32,) * 3), lens)
            point = closest_approach(starts[i], incoming[i], ends[i], outgoing[i])
            if point is None or not bool(geometry.contains(point)):
                continue
            flat = int(np.ravel_multi_index(geometry.point_to_index(point) // 2, (32,) * 3))
            self.event_cells[i] = flat
            counts[flat] += 1
        self.denominator = gaussian_filter(exposure.reshape((32,) * 3), 2.0, mode="constant")
        self.counts = self.expand(counts.reshape((32,) * 3))
        self.valid_mask = self.expand(self.denominator > 0)

    @staticmethod
    def expand(x):
        return x.repeat(2, 0).repeat(2, 1).repeat(2, 2)

    def predict(self, p):
        p = positive_momenta(p, self.n)
        keep = self.event_cells >= 0
        signal = np.zeros(32**3, dtype=float)
        np.add.at(signal, self.event_cells[keep], self.q_per_p2[keep] * p[keep] ** 2)
        numerator = gaussian_filter(signal.reshape((32,) * 3), 2.0, mode="constant")
        raw = np.divide(
            numerator, self.denominator, out=np.zeros_like(numerator), where=self.denominator > 0
        )
        return self.expand(raw).astype(np.float32)


class CachedMuNet:
    """Frozen production architecture; seed is reset after loading, per policy."""

    def __init__(self, weights_path=None):
        import tensorflow as tf

        from muon_tomo._vendor.munet.models.agg_3d import Agg3D

        self.weights_path = MuNetReconstructor(weights_path=weights_path).weights_path
        self.model = Agg3D(
            point_size=1,
            downward_convs=[1, 2, 3, 4, 5],
            downward_filters=[8, 16, 32, 64, 128],
            upward_convs=[4, 3, 2, 1],
            upward_filters=[64, 32, 16, 8],
            resolution=64,
            threshold=1e-8,
        )
        self.model(tf.zeros((1, 1024, 14), dtype=tf.float32), training=False)
        self.model.load_weights(str(self.weights_path))

    def predict(self, tracks, p, seed=0):
        import tensorflow as tf

        features = momentum_features(tracks, p)
        if len(features) % 1024:
            raise ValueError("study requires complete 1024-track chunks")
        tf.random.set_seed(seed)
        total = np.zeros((64, 64, 64), dtype=np.float64)
        for chunk in features.reshape(-1, 1024, 14):
            total += self.model(tf.convert_to_tensor(chunk[None]), training=False).numpy()[
                0, ..., 0
            ]
        raw = (total / (len(features) // 1024)).astype(np.float32)
        return np.clip(raw, 0, None), raw


def evaluate(prediction, labels):
    truth = labels_to_inverse_radiation_length(labels).astype(float)
    pred = prediction.astype(float)
    error = pred - truth
    obj, bg = labels > 0, labels == 0
    metrics = {
        "mse_full": float(np.mean(error**2)),
        "mse_object": float(np.mean(error[obj] ** 2)) if obj.any() else None,
        "mse_background": float(np.mean(error[bg] ** 2)) if bg.any() else None,
        "object_bias": float(np.mean(error[obj])) if obj.any() else None,
        "object_mean": float(np.mean(pred[obj])) if obj.any() else None,
        "object_truth_mean": float(np.mean(truth[obj])) if obj.any() else None,
        "background_fp_fraction_gt_0p8": float(np.mean(pred[bg] > 0.8)) if bg.any() else None,
        "object_fraction": float(obj.mean()),
        "zero_mse_full": float(np.mean(truth**2)),
        "zero_mse_object": float(np.mean(truth[obj] ** 2)) if obj.any() else None,
        "max_prediction": float(pred.max()),
    }
    material_metrics = []
    for material in np.unique(labels):
        mask = labels == material
        material_metrics.append(
            {
                "label": int(material),
                "material": MATERIAL_NAMES[material],
                "voxels": int(mask.sum()),
                "truth": float(truth[mask][0]),
                "mean": float(pred[mask].mean()),
                "bias": float(error[mask].mean()),
                "mse": float(np.mean(error[mask] ** 2)),
            }
        )
    return metrics, material_metrics
