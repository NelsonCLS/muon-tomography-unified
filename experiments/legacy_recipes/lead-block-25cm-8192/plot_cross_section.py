from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from muon_tomo.plotting import COMPARISON_VMAX_CM_INV
from muon_tomo.cli import _load_truth
from muon_tomo.types import ReconstructionResult

out=Path.cwd()/'outputs/lead-block-25cm-8192'
truth=_load_truth(out/'record_0_truth.npz')
report=json.loads((out/'metrics.json').read_text())
volumes=[truth.inverse_radiation_length_cm_inv]
for m in ('munet','poca','likelihood'):
 volumes.append(ReconstructionResult.load(out/f'record_0_{m}.npz').inverse_radiation_length_cm_inv)
vmax=COMPARISON_VMAX_CM_INV
fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
for ax,volume,name,method in zip(axes.flat,volumes,('Ground truth','MuNet','PoCA','Likelihood'),(None,'munet','poca','likelihood'),strict=True):
 slab=volume[:,:,29:36].mean(axis=2)
 im=ax.imshow(slab.T,origin='lower',extent=(-50,50,-50,50),cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
 ax.set_xlabel('x [cm]');ax.set_ylabel('y [cm]')
 subtitle='Lead: 1/X₀ = 1.782 cm⁻¹' if method is None else f"Full-volume MSE = {report['methods'][method]['mse_full_grid_cm_inv_squared']:.5f} (cm⁻¹)²"
 ax.set_title(name+'\n'+subtitle,fontsize=12)
 ax.set_xticks([-40,-20,0,20,40]);ax.set_yticks([-40,-20,0,20,40])
 if method:
  report['methods'][method]['displayed_slab_voxels_above_color_scale']=int((slab>vmax).sum())
fig.colorbar(im,ax=axes.ravel().tolist(),label='Inverse radiation length 1/X₀ [cm⁻¹]',extend='max',ticks=np.linspace(0, vmax, 5),shrink=.92)
fig.suptitle('Centred 25 cm solid lead cube · 8,192 filtered tracks\nXY cross-section: mean of z layers 29–35 · common color scale',fontsize=16)
fig.supxlabel('Uncalibrated · fixed p = 1.743 GeV/c · PoCA: 32³, σ = 6.25 cm · likelihood: 10 median updates\nColors saturate above 3.2 cm⁻¹; MSE uses all 64³ voxels without display clipping.',fontsize=10)
fig.savefig(out/'cross_section.png',dpi=180)
fig.savefig(out/'cross_section.pdf')
plt.close(fig)
report['display_color_scale_cm_inv']=[0,vmax]
report['display_values_above_scale']='saturated; upper colorbar extension marks clipping; MSE unchanged'
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
if Path(__file__).resolve() != (out/'plot_cross_section.py').resolve():
 shutil.copy2(__file__,out/'plot_cross_section.py')
print(out/'cross_section.png')
