"""Matched Funny 0 production PoCA grid/sigma sweep; no source changes."""
from pathlib import Path
import csv
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from muon_tomo.plotting import COMPARISON_VMAX_CM_INV
import numpy as np

from muon_tomo.cli import _track_fingerprint, _save_truth
from muon_tomo.constants import MATERIAL_NAMES
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.reconstruction.poca import PoCAReconstructor
from muon_tomo.types import ReconstructionResult

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'dataset/voxels_prediction_funny.tfrecord'
MOMENTUM = 1743.4601815051958
SIGMAS = (6.25, 3.125, 1.5625)
tracks, truth = load_tfrecord_record(SOURCE, 0, limit=8192, seed=0)
# Match the historical runner's final TrackTable construction exactly.
tracks = tracks.take_first(8192)
fingerprint = _track_fingerprint(tracks)
np.save(OUT / 'selected_event_ids.npy', tracks.event_id)
_save_truth(truth, OUT / 'truth.npz')
protocol = dict(dataset=str(SOURCE), dataset_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    record_index=0, tracks=len(tracks), selection_seed=0, track_fingerprint_sha256=fingerprint,
    grids=[32,64], sigma_cm=list(SIGMAS), momentum_mev=MOMENTUM,
    momentum_provenance='Historical fixed inverse-square effective momentum derived from Funny 0 simulation; retained to isolate grid and sigma.',
    per_track_momentum_used=False, calibration_applied=False, truth_used_for_evaluation_only=True,
    prediction_unit='cm^-1', mse_unit='cm^-2', all_scoring_on_full_64_cubed_before_display_cuts=True,
    source_sha256=hashlib.sha256((ROOT/'src/muon_tomo/reconstruction/poca.py').read_bytes()).hexdigest())
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
results = {}
rows = []
for grid in (32,64):
    for sigma in SIGMAS:
        start=time.perf_counter()
        recon=PoCAReconstructor(coarse_grid_size=grid,pooling_sigma_cm=sigma,
            momentum_policy='mean',average_momentum_mev=MOMENTUM).reconstruct(tracks)
        recon.metadata.update(protocol, tested_grid_size=grid, tested_sigma_cm=sigma)
        name=f'poca_grid_{grid}_sigma_{sigma:g}'
        recon.save(OUT/f'{name}.npz')
        volume=recon.inverse_radiation_length_cm_inv.astype(float)
        assert volume.shape==(64,64,64) and np.isfinite(volume).all() and (volume>=0).all()
        if grid==32:
            for axis in range(3):
                np.testing.assert_array_equal(np.take(volume,range(0,64,2),axis=axis),np.take(volume,range(1,64,2),axis=axis))
        y=truth.inverse_radiation_length_cm_inv.astype(float)
        err=volume-y
        row=dict(grid=grid,sigma_cm=sigma,sigma_grid_voxels=sigma/(100/grid),
            mse=float(np.mean(err**2)),object_mse=float(np.mean(err[y>0]**2)),
            background_mse=float(np.mean(err[y==0]**2)),maximum_cm_inv=float(volume.max()),
            voxels_above_08=int(np.sum(volume>0.8)),elapsed_seconds=time.perf_counter()-start,
            material_statistics={})
        for label in np.unique(truth.material_labels):
            mask=truth.material_labels==label
            row['material_statistics'][MATERIAL_NAMES[label]]=dict(voxels=int(mask.sum()),
                truth_mean_cm_inv=float(y[mask].mean()),mean_cm_inv=float(volume[mask].mean()),
                spatial_std_cm_inv=float(volume[mask].std()),bias_cm_inv=float(err[mask].mean()))
        if grid==32 and sigma==6.25:
            reference=ReconstructionResult.load(ROOT/'outputs/poca-exposure-study/record_0_poca.npz')
            assert reference.metadata['track_fingerprint_sha256']==fingerprint
            np.testing.assert_allclose(volume,reference.inverse_radiation_length_cm_inv,rtol=1e-6,atol=1e-7)
            row['historical_baseline_max_abs_difference']=float(np.max(np.abs(volume-reference.inverse_radiation_length_cm_inv)))
        results[grid,sigma]=volume
        rows.append(row)
        (OUT/'metrics.json').write_text(json.dumps(dict(protocol=protocol,zero_mse=float(np.mean(y**2)),results=rows),indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='material_statistics'}),flush=True)

with (OUT/'metrics.csv').open('w') as f:
    fields=['grid','sigma_cm','sigma_grid_voxels','mse','object_mse','background_mse','maximum_cm_inv','voxels_above_08']
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

vmax=COMPARISON_VMAX_CM_INV
for axis in (2,1,0):
    axes_names={2:('x','y','z'),1:('x','z','y'),0:('y','z','x')}
    a,b,c=axes_names[axis]
    def slab(v):
        return np.take(v,range(29,36),axis=axis).mean(axis=axis).T
    fig,axs=plt.subplots(2,4,figsize=(16,8.4),layout='constrained')
    for r,grid in enumerate((32,64)):
        for col in range(4):
            ax=axs[r,col]
            if col==0:
                data=y;title=f'Truth — reference for {grid}³ row'
            else:
                sigma=SIGMAS[col-1];data=results[grid,sigma]
                row=next(x for x in rows if x['grid']==grid and x['sigma_cm']==sigma)
                title=f'PoCA {grid}³ · σ = {sigma:g} cm\nFull 64³ MSE = {row["mse"]:.6f} cm⁻²'
            im=ax.imshow(slab(data),origin='lower',extent=(-50,50,-50,50),cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
            ax.set_title(title,fontsize=10);ax.set_xlabel(f'{a} (cm)');ax.set_ylabel(f'{b} (cm)')
    fig.colorbar(im,ax=axs,label='Mean 1/X₀ (cm⁻¹); values above scale saturated',shrink=.8,extend='max')
    fig.suptitle(f'Funny 0 — {c} layers 29–35 averaged · same 8,192 tracks, seed 0\nFixed p = 1743.46 MeV/c (historical simulation-derived prior) · exposure normalization + Gaussian pooling',fontsize=13)
    fig.savefig(OUT/f'comparison_{c}_slab.png',dpi=170)
    if axis==2:fig.savefig(OUT/'comparison_z_slab.pdf')
    plt.close(fig)
print('COMPLETE',flush=True)
