"""Render Funny 0 PoCA with 32^3 path-exposure normalization and no pooling."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

with np.load(HERE / "record_0_sigma_0.0.npz", allow_pickle=False) as data:
    prediction = data["prediction"].astype(np.float64)

with np.load(ROOT / "outputs/funny-current-best/record_0_truth.npz", allow_pickle=False) as data:
    truth = data["inverse_radiation_length_cm_inv"].astype(np.float64)

assert prediction.shape == truth.shape == (64, 64, 64)
mse = float(np.mean((prediction - truth) ** 2))
assert np.isclose(mse, 35.2951, rtol=0, atol=5e-5), mse

slab = prediction[:, :, 29:36].mean(axis=2).T

fig, ax = plt.subplots(figsize=(7.5, 6.6))
image = ax.imshow(
    slab,
    origin="lower",
    extent=(-50, 50, -50, 50),
    cmap="inferno",
    vmin=0,
    vmax=3.2,
    interpolation="nearest",
)
ax.set_title(
    "Funny 0 — PoCA exposure normalization only\n"
    "32³ estimate, no Gaussian pooling\n"
    f"Full 64³ raw MSE = {mse:.4f} cm⁻²; mean z layers 29–35",
    fontsize=15,
)
ax.set_xlabel("x (cm)", fontsize=13)
ax.set_ylabel("y (cm)", fontsize=13)
ax.set_xticks((-50, -25, 0, 25, 50))
ax.set_yticks((-50, -25, 0, 25, 50))
colorbar = fig.colorbar(image, ax=ax, shrink=0.84, extend="max")
colorbar.set_label("1/X₀ (cm⁻¹)", fontsize=13)
fig.tight_layout()
output = HERE / "record_0_exposure_only_cross_section.png"
fig.savefig(output, dpi=200)
plt.close(fig)

print(f"output={output}")
print(f"mse={mse:.12f}")

