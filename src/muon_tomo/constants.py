"""Physical constants and the Geant4 material-label contract."""

from __future__ import annotations

import numpy as np

GRID_SHAPE = (64, 64, 64)
AXIS_ORDER = "xyz"
GRID_LOWER_CM = (-50.0, -50.0, -50.0)
GRID_UPPER_CM = (50.0, 50.0, 50.0)
MUNET_POSITION_SCALE_CM = 100.0
MUNET_POSITION_OFFSET = 0.5
TRACK_PLANE_LOWER_CM = -100.0
TRACK_PLANE_UPPER_CM = 100.0
HIGHLAND_CONSTANT_MEV = 15.0
REFERENCE_MOMENTUM_MEV = 3000.0

MATERIAL_NAMES = (
    "air", "benzene", "methanol", "water", "magnesium", "concrete",
    "gypsum", "calcium", "sulfur", "silicon", "aluminium", "caesium",
    "manganese", "iron", "iodine", "nickel", "molybdenum", "silver",
    "polonium", "lead", "gold", "uranium",
)

# Radiation lengths in centimetres, copied from the Mu-Net truth definition.
RADIATION_LENGTH_CM = np.asarray(
    [
        np.inf,
        49.834983498349835,
        49.82309830679809,
        36.08,
        14.385057471264368,
        11.552173913043479,
        10.607758620689655,
        10.412903225806451,
        9.75,
        9.368827823100043,
        8.895887365690998,
        4.436732514682328,
        1.967741935483871,
        1.7576835153670307,
        1.7200811359026373,
        1.4243990114580993,
        0.9589041095890413,
        0.8542857142857143,
        0.6609442060085837,
        0.5612334801762114,
        0.33436853002070394,
        0.316622691292876,
    ],
    dtype=np.float64,
)
INVERSE_RADIATION_LENGTH_CM_INV = np.divide(
    1.0,
    RADIATION_LENGTH_CM,
    out=np.zeros_like(RADIATION_LENGTH_CM),
    where=np.isfinite(RADIATION_LENGTH_CM),
)


def labels_to_inverse_radiation_length(labels: np.ndarray) -> np.ndarray:
    """Map integer material labels to canonical 1/X0 in cm^-1."""
    labels = np.asarray(labels)
    if labels.shape != GRID_SHAPE:
        raise ValueError(f"truth labels must have shape {GRID_SHAPE}, got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("truth labels must be integers")
    if labels.size and (labels.min() < 0 or labels.max() >= len(RADIATION_LENGTH_CM)):
        raise ValueError("truth contains a material label outside the material table")
    return INVERSE_RADIATION_LENGTH_CM_INV[labels].astype(np.float32)


def inverse_to_reference_scattering_density(
    inverse_x0_cm_inv: np.ndarray,
    *,
    reference_momentum_mev: float = REFERENCE_MOMENTUM_MEV,
    highland_constant_mev: float = HIGHLAND_CONSTANT_MEV,
) -> np.ndarray:
    """Convert 1/X0 to reference scattering density in mrad^2/cm."""
    factor = 1.0e6 * (highland_constant_mev / reference_momentum_mev) ** 2
    return np.asarray(inverse_x0_cm_inv) * factor


def reference_scattering_density_to_inverse(
    scattering_mrad2_per_cm: np.ndarray,
    *,
    reference_momentum_mev: float = REFERENCE_MOMENTUM_MEV,
    highland_constant_mev: float = HIGHLAND_CONSTANT_MEV,
) -> np.ndarray:
    """Convert reference scattering density in mrad^2/cm to 1/X0."""
    factor = 1.0e6 * (highland_constant_mev / reference_momentum_mev) ** 2
    return np.asarray(scattering_mrad2_per_cm) / factor
