"""Fixed-input PoCA comparison: only grid size and Gaussian sigma vary."""
from pathlib import Path
import csv
import json
import shutil
import time
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from muon_tomo.cli import _track_fingerprint, _save_truth
from muon_tomo.io.tfrecord import load_tfrecord_record
from muon_tomo.plotting import COMPARISON_VMAX_CM_INV
from muon_tomo.reconstruction.poca import PoCAReconstructor
from muon_tomo.simulation.pipeline import sha256
from muon_tomo.types import ReconstructionResult

ROOT=Path.cwd()
OUT=ROOT/'outputs/lead-block-25cm-poca-grid-sigma'
OUT.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,OUT/'run.py')
BASE=ROOT/'outputs/lead-block-25cm-8192'
SOURCE=ROOT/'dataset/generated_custom/lead_block_25cm_filtered_8192.tfrecord'
config=yaml.safe_load((BASE/'config.yaml').read_text())
shutil.copy2(BASE/'config.yaml',OUT/'fixed_config.yaml')
tracks,truth=load_tfrecord_record(SOURCE,0,limit=8192,seed=0)
tracks=tracks.take_first(8192)
assert len(tracks)==8192
fingerprint=_track_fingerprint(tracks)
reference=ReconstructionResult.load(BASE/'record_0_poca.npz')
assert fingerprint==reference.metadata['track_fingerprint_sha256']
np.save(OUT/'selected_event_ids.npy',tracks.event_id)
_save_truth(truth,OUT/'truth.npz')
fixed=dict(highland_constant_mev=config['physics']['highland_constant_mev'],minimum_coverage=config['poca']['minimum_coverage'],momentum_policy=config['physics']['classical_momentum_policy'],average_momentum_mev=config['physics']['average_momentum_mev'])
protocol=dict(dataset=str(SOURCE),dataset_sha256=sha256(SOURCE),track_count=len(tracks),track_fingerprint_sha256=fingerprint,record_index=0,selection_seed=0,grids=[32,64],sigma_cm=[6.25,3.125,1.5625],fixed_reconstructor_arguments=fixed,only_varied=['coarse_grid_size','pooling_sigma_cm'],calibration='none',per_track_momentum_used=False,truth_used_for_evaluation_only=True,production_poca_sha256=sha256(ROOT/'src/muon_tomo/reconstruction/poca.py'),display_color_scale_cm_inv=[0,COMPARISON_VMAX_CM_INV],mse_scope='All 64^3 voxels before display clipping; 32^3 outputs expanded by repeating cells',simulation_note='Same custom simulator dataset as previous comparison; training-distribution equivalence unverified.')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
y=truth.inverse_radiation_length_cm_inv.astype(float)
lead=truth.material_labels==19
air=truth.material_labels==0
volumes={};rows=[]
for grid in (32,64):
 for sigma in (6.25,3.125,1.5625):
  start=time.perf_counter()
  result=PoCAReconstructor(coarse_grid_size=grid,pooling_sigma_cm=sigma,**fixed).reconstruct(tracks)
  result.metadata.update(track_fingerprint_sha256=fingerprint,track_count=len(tracks),input_source=str(SOURCE),calibration='none')
  volume=result.inverse_radiation_length_cm_inv.astype(float)
  assert volume.shape==(64,64,64) and np.isfinite(volume).all() and (volume>=0).all()
  if grid==32:
   for axis in range(3):
    np.testing.assert_array_equal(np.take(volume,range(0,64,2),axis=axis),np.take(volume,range(1,64,2),axis=axis))
  error=volume-y
  row=dict(grid=grid,sigma_cm=sigma,sigma_grid_voxels=sigma/(100/grid),full_volume_mse=float(np.mean(error**2)),lead_mse=float(np.mean(error[lead]**2)),air_mse=float(np.mean(error[air]**2)),lead_mean_cm_inv=float(volume[lead].mean()),lead_std_cm_inv=float(volume[lead].std()),air_mean_cm_inv=float(volume[air].mean()),maximum_cm_inv=float(volume.max()),voxels_above_3_2=int((volume>3.2).sum()),seconds=time.perf_counter()-start)
  if grid==32 and sigma==6.25:
   np.testing.assert_array_equal(result.inverse_radiation_length_cm_inv,reference.inverse_radiation_length_cm_inv)
   protocol['baseline_matches_previous_poca_exactly']=True
  result.save(OUT/f'poca_grid_{grid}_sigma_{sigma:g}.npz')
  volumes[grid,sigma]=volume;rows.append(row)
  print(json.dumps(row),flush=True)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
with (OUT/'metrics.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=dict(protocol=protocol,lead_truth_cm_inv=float(y[lead][0]),lead_voxels=int(lead.sum()),results=rows)
report['best_full_volume_mse']=min(rows,key=lambda row:row['full_volume_mse'])
(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
for axis,horizontal,vertical,normal in ((2,'x','y','z'),(1,'x','z','y'),(0,'y','z','x')):
 fig,axs=plt.subplots(2,4,figsize=(16,8.5),layout='constrained')
 for row_idx,grid in enumerate((32,64)):
  for col in range(4):
   ax=axs[row_idx,col]
   if col==0:
    volume=y; title=f'Ground truth · {grid}³ comparison\nLead: 1/X₀ = {float(y[lead][0]):.3f} cm⁻¹'
   else:
    sigma=(6.25,3.125,1.5625)[col-1];volume=volumes[grid,sigma]
    stats=next(r for r in rows if r['grid']==grid and r['sigma_cm']==sigma)
    title=f'PoCA {grid}³ · σ = {sigma:g} cm\nFull-volume MSE = {stats["full_volume_mse"]:.5f} cm⁻²'
   slab=np.take(volume,range(29,36),axis=axis).mean(axis=axis).T
   im=ax.imshow(slab,origin='lower',extent=(-50,50,-50,50),cmap='magma',vmin=0,vmax=COMPARISON_VMAX_CM_INV,interpolation='nearest')
   ax.set_title(title,fontsize=11);ax.set_xlabel(f'{horizontal} [cm]');ax.set_ylabel(f'{vertical} [cm]')
   ax.set_xticks([-40,-20,0,20,40]);ax.set_yticks([-40,-20,0,20,40])
 fig.colorbar(im,ax=axs.ravel().tolist(),label='1/X₀ [cm⁻¹]',ticks=np.linspace(0,3.2,5),extend='max',shrink=.88)
 fig.suptitle(f'Centred 25 cm lead cube · same 8,192 filtered tracks\n{horizontal.upper()}{vertical.upper()} cross-section · mean {normal} layers 29–35 · only grid size and pooling σ vary',fontsize=15)
 fig.supxlabel('Fixed p = 1.743 GeV/c · pooled scattering / pooled path exposure · no calibration\nAll MSE values use the full 64³ volume; display colors saturate at 3.2 cm⁻¹.',fontsize=11)
 fig.savefig(OUT/f'comparison_{normal}_slab.png',dpi=180)
 fig.savefig(OUT/f'comparison_{normal}_slab.pdf')
 plt.close(fig)
print('COMPLETE '+str(OUT),flush=True)
