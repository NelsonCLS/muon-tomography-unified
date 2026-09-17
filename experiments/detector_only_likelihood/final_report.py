"""Render the complete fixed-parameter comparison; no reconstruction tuning."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/muon-mpl')
from pathlib import Path
import json,hashlib,html
import numpy as np
import matplotlib.pyplot as plt
from muon_tomo.plotting import COMPARISON_VMAX_CM_INV
import plotly.graph_objects as go
from muon_tomo.plotting3d import _voxel_cube_mesh
from muon_tomo.geometry import GridGeometry
P=Path(__file__).resolve().parent;R=P/'results'
base=json.loads((R/'inverse_scores.json').read_text());pool=json.loads((R/'pooled_control_scores.json').read_text())
rows=[]
for b,s in zip(base,pool):
 assert b['case']==s['case']
 name=b['case'];d=R/name
 t=np.load(d/'evaluation_truth.npz')['inverse_radiation_length_cm_inv']
 raw=np.load(d/'inverse_radiation_length.npz')['baseline_cm_inv']
 pred=np.load(d/'pooled_likelihood.npz')['inverse_radiation_length_cm_inv']
 fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
 for ax,v,title in zip(axes,[t,raw,pred],['Truth','Existing median + external prior','Exposure-pooled likelihood']):
  mse=float(np.mean((v-t)**2));im=ax.imshow(v[:,:,29:36].mean(axis=2).T,origin='lower',extent=(-50,50,-50,50),cmap='magma',vmin=0,vmax=COMPARISON_VMAX_CM_INV)
  ax.set_title(f'{title}\nFull 64³ MSE = {mse:.4g} (cm⁻¹)²');ax.set_xlabel('x [cm]');ax.set_ylabel('y [cm]')
 fig.colorbar(im,ax=axes,label='1/X0 [cm⁻¹]; colors saturate at truth maximum',shrink=.7)
 fig.suptitle(f'{name} · 8,192 identical tracks · z slab 29–35\nFixed external spectrum; fixed 3.125 cm pooling; no truth-based calibration')
 fig.savefig(d/'pooled_cross_section.png',dpi=160);plt.close(fig)
 row=dict(case=name,baseline_mse=b['baseline_mse'],pooled_mse=s['mse'],marginalized_mse=b['candidate_mse'],zero_mse=b['zero_mse'],baseline_auc=b['baseline_auc'],pooled_auc=s['auc'],baseline_reserved_nll=b['baseline_mean_test_nll'],pooled_reserved_nll=s['pooled_mean_test_nll'])
 rows.append(row)
 if name=='funny_0':
  fig=go.Figure();g=GridGeometry();counts={}
  titles=['Truth','Existing likelihood + external prior','Exposure-pooled likelihood']
  for title,v in zip(titles,[t,raw,pred]):
   idx=np.argwhere(v>.8);vals=v[tuple(idx.T)];counts[title]=len(idx)
   mesh=_voxel_cube_mesh(idx,vals,g.lower_cm,tuple(g.voxel_size_cm))
   fig.add_trace(go.Mesh3d(**mesh,name=title,visible=title==titles[2],opacity=.55,colorscale='Magma',cmin=0,cmax=float(max(t.max(),raw.max(),pred.max())),colorbar=dict(title='1/X0 [cm⁻¹]')))
  buttons=[dict(label=title,method='update',args=[dict(visible=[i==j for j in range(3)]),dict(title=f'{title} · funny 0 · all {counts[title]:,} voxels >0.8 cm⁻¹')]) for i,title in enumerate(titles)]
  fig.update_layout(title=f'Exposure-pooled likelihood · funny 0 · all {counts[titles[2]]:,} voxels >0.8 cm⁻¹',updatemenus=[dict(buttons=buttons,active=2)],scene=dict(aspectmode='cube',xaxis=dict(title='x [cm]',range=[-50,50]),yaxis=dict(title='y [cm]',range=[-50,50]),zaxis=dict(title='z [cm]',range=[-50,50])),margin=dict(l=0,r=0,b=0,t=95))
  fig.write_html(d/'pooled_viewer.html',include_plotlyjs=True)
  (d/'pooled_viewer_metadata.json').write_text(json.dumps(dict(threshold_strictly_greater_than_cm_inv=.8,counts=counts,surface_only=False,no_voxel_count_cap=True),indent=2))
(R/'final_comparison.json').write_text(json.dumps(rows,indent=2))
summary=dict(cases=9,baseline_mean_mse=float(np.mean([r['baseline_mse'] for r in rows])),pooled_mean_mse=float(np.mean([r['pooled_mse'] for r in rows])),pooled_mse_wins=sum(r['pooled_mse']<r['baseline_mse'] for r in rows),reserved_nll_wins=sum(r['pooled_reserved_nll']<r['baseline_reserved_nll'] for r in rows),production_unchanged=True)
(R/'final_summary.json').write_text(json.dumps(summary,indent=2))
body='''<!doctype html><meta charset="utf-8"><title>Detector-only likelihood experiment</title><style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#202020;background:#fafafa}h1{font-size:28px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%}a{color:#17518b}p{line-height:1.55}.note{background:#eee;padding:15px}details{margin:20px 0}</style><h1>Likelihood reconstruction: fixed external spectrum</h1>
<p><strong>Production likelihood code remains unchanged.</strong> The useful experimental variant is exposure-weighted spatial pooling of the existing full median likelihood. All predictions remain raw 1/X0 [cm⁻¹].</p>
<p>Same external Reyna spectrum, same 3.125 cm Gaussian width, same 32³ pooling grid, same 10 likelihood iterations, and 8,192 tracks per object. Every coarse prediction is repeated into identical 2³ blocks on 64³. No input momentum magnitudes, truth masks, per-object tuning, fitted slopes or intercepts.</p>
<p class="note">Absolute density assumes the published sea-level spectrum applies to the detected muons. The model excludes momenta below 1 GeV/c and does not account for object-dependent attenuation, slowing, or detector acceptance. This is an experimental approximation, not general calibration-free material identification.</p>
<h2>Raw full-volume MSE [(cm⁻¹)²]</h2><p>Both baseline and pooled variant use the same external spectrum. These baseline values therefore differ from earlier runs using the simulation-derived 1,743.46 MeV/c setting. MSE includes all 262,144 voxels before display thresholds.</p><table><thead><tr><th>Case</th><th>Existing + external prior</th><th>Exposure pooled</th><th>Empty volume</th><th>Marginalized candidate (failed)</th></tr></thead><tbody>'''
for r in rows:body+=f"<tr><td>{r['case']}</td><td>{r['baseline_mse']:.5f}</td><td><b>{r['pooled_mse']:.5f}</b></td><td>{r['zero_mse']:.5f}</td><td>{r['marginalized_mse']:.5f}</td></tr>"
body+='''</tbody></table><p>The pooled result improves MSE and reserved-track conditional likelihood in all nine cases. Record 3 barely improves on an empty volume. This panel evaluates a fixed method; it is not evidence that all unseen objects or real detectors will perform equally well.</p>
<h2>What changed</h2><p>First reconstruct with the existing full covariance median update, replacing hidden momentum with the expected inverse-square momentum coefficient from the fixed external spectrum conditioned on measured incoming zenith. Then accumulate total path exposure E and E times the reconstructed density in 32³ cells. Apply the same Gaussian to each sum and divide. This is spatial regularization of an existing likelihood estimate, not an exact new maximum-likelihood update. The existing 20 cm⁻¹ cap is retained before pooling; no extra clipping or calibration is applied afterward.</p>
<h2>Validation and limitations</h2><p>Reserved-track scores use 6,144 training tracks and 2,048 reserved tracks; only rays crossing the volume have modeled covariance. Final plotted maps use all 8,192 tracks. Paths use measured PoCA, so these are conditional plug-in likelihood scores, not unconditional detector likelihoods. Truth labels are opened separately for evaluation after fitting. Direction vectors encoded with momentum scale are normalized; their magnitudes and all other simulation columns are discarded.</p>
<p>The separate pooled Student-t and full external-spectrum marginalization approach improved shape localization but worsened absolute-density MSE. It is retained as a failed experiment. Pooling width, prior coefficients, integration limits, and iteration counts were not swept against truth. The simpler control was evaluated after the earlier candidate failed; therefore approach selection used this panel, and a fully independent validation set remains necessary.</p>
<p><a href="README.md">Method and reproduction notes</a> · <a href="exposure_pooled_likelihood.py">Useful experimental method</a> · <a href="results/funny_0/pooled_viewer.html">Funny 0 3D viewer</a> · <a href="https://arxiv.org/pdf/hep-ph/0604145">External spectrum: Reyna (2006)</a></p>'''
for r in rows:body+=f"<details {'open' if r['case']=='funny_0' else ''}><summary>{r['case']} cross-section</summary><img src='results/{r['case']}/pooled_cross_section.png' alt='{r['case']} truth and likelihood comparison'><p>Shared display scale saturates at the truth maximum; saved predictions and MSE are not clipped by the color scale.</p></details>"
(P/'report.html').write_text(body)
old=json.loads((R/'frozen_protocol.json').read_text())['production_sha256'];actual=hashlib.sha256(Path('src/muon_tomo/reconstruction/likelihood.py').read_bytes()).hexdigest();assert old==actual
integrity=dict(production_likelihood_sha256=actual,production_matches_before_experiment=True,experimental_sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('*.py')})
(R/'integrity.json').write_text(json.dumps(integrity,indent=2))
print(json.dumps(summary))
