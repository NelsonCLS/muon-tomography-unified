from pathlib import Path
import json
import shutil
import time

import numpy as np
import tensorflow as tf
from muon_tomo.cli import main, _load_truth
from muon_tomo.plotting import plot_comparison_slab
from muon_tomo.simulation.pipeline import sha256
from muon_tomo.types import ReconstructionResult

root = Path.cwd()
out = root/'outputs/lead-block-25cm-8192'
out.mkdir(parents=True, exist_ok=False)
data = root/'dataset/generated_custom/lead_block_25cm_filtered_8192.tfrecord'
config = out/'config.yaml'
shutil.copy2(root/'configs/default.yaml', config)
shutil.copy2('/private/tmp/reconstruct_lead_block.py', out/'run_reconstruction.py')
seed=20260912
tf.random.set_seed(seed)
start=time.monotonic()
code=main(['reconstruct', str(data), '--config', str(config), '--record-index', '0', '--method', 'all', '--limit', '8192', '--seed', '0', '--output-dir', str(out)])
if code: raise SystemExit(code)
results=[ReconstructionResult.load(out/f'record_0_{m}.npz') for m in ('munet','poca','likelihood')]
assert all(r.metadata['track_count']==8192 for r in results)
assert len({r.metadata['track_fingerprint_sha256'] for r in results})==1
assert all(np.isfinite(r.inverse_radiation_length_cm_inv).all() for r in results)
truth=_load_truth(out/'record_0_truth.npz')
report=plot_comparison_slab(truth,results,output=out/'cross_section.png',center_index=32,slab_layers=7)
plot_comparison_slab(truth,results,output=out/'cross_section.pdf',center_index=32,slab_layers=7)
report.update({'input_sha256':sha256(data),'config_sha256':sha256(config),'tensorflow_seed':seed,'track_count':8192,'same_tracks_all_methods':True,'calibration':'none','elapsed_seconds':time.monotonic()-start,'simulation_note':'Current custom simulator settings; training-distribution equivalence is unverified.'})
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
