"""Build the physics report, exact score tables and standalone scientific plots."""
import base64
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/muon-mpl')
import matplotlib.pyplot as plt
import numpy as np
from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
NAMES = dict(raw10='Raw median · 10', pool10='Exposure pool · 10', pool40='Exposure pool · 40',
    local_pool10='Transverse pool · 10', local_pool40='Transverse pool · 40',
    mean_local_pool40='Transverse mean · 40', gaussian_ml='Gaussian ML',
    gaussian_quadratic='Gaussian + quadratic', gaussian_huber='Gaussian + Huber',
    student_huber='Student + Huber', angular_huber='Angular + Huber')


def table(headers, rows):
    head = ''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)
    body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def inline_image(path, alt):
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}">'


def main():
    rows = json.loads((OUT/'metrics.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    val = summary['validation']['methods']
    methods = list(NAMES)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), layout='constrained')
    for ax, metric, title in zip(axes, ['mse', 'material_mse', 'background_mse'],
                                ['Full-volume MSE', 'Material-only MSE', 'Air-only MSE']):
        colors = ['#167d8d' if m == 'local_pool40' else '#718096' for m in methods]
        ax.barh(np.arange(len(methods)), [val[m][metric] for m in methods], color=colors)
        ax.set_yticks(np.arange(len(methods)), [NAMES[m] for m in methods] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Mean squared error [cm⁻²] · lower is better')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_ylabel('Reconstruction method')
    fig.suptitle('Six additional validation objects · equal weight per object\n8,192 selected tracks per object · original unfiltered data · 15 September 2026', fontsize=13)
    fig.savefig(OUT/'validation_comparison.png', dpi=160)
    fig.savefig(OUT/'validation_comparison.pdf')
    plt.close(fig)

    shown = ['raw10', 'pool10', 'local_pool40', 'gaussian_huber', 'student_huber']
    images = {}
    for row in rows:
        p = OUT/row['case']
        truth = np.load(p/'truth_evaluation_only.npz')['rho']
        maps = np.load(p/'full_maps.npz')
        arrays = [truth]+[maps[m] for m in shown]
        labels = ['Truth (evaluation only)']+[NAMES[m] for m in shown]
        fig, axes = plt.subplots(2, 6, figsize=(18, 6.7), layout='constrained')
        for j, (v, label) in enumerate(zip(arrays, labels)):
            projections = [v.mean(axis=2).T, v.mean(axis=1).T]
            for i, a in enumerate(projections):
                ax = axes[i, j]
                ref = truth.mean(axis=2 if i == 0 else 1)
                upper = max(float(ref.max()), .05)
                im = ax.imshow(a, origin='lower', extent=(-50, 50, -50, 50),
                    cmap='magma', vmin=0, vmax=upper, interpolation='nearest')
                ax.set_xlabel('x [cm]')
                ax.set_ylabel('y [cm]' if i == 0 else 'z [cm]')
                ax.set_title(label if i == 0 else f'MSE {row["models"][shown[j-1]]["mse"]:.4f}' if j else 'Same truth used for all scores', fontsize=10)
                if j == 5:
                    fig.colorbar(im, ax=axes[i, :], shrink=.7, label='Mean 1/X₀ along omitted axis [cm⁻¹]', extend='max')
        fig.suptitle(f'{row["case"]} · mean projections through the complete 100 cm depth\nSame color scale for all methods within each row; brighter extremes saturate only the display.', fontsize=13)
        fig.savefig(p/'projections.png', dpi=140)
        if row['case'] in ('funny_0', 'validation_11'):
            fig.savefig(p/'projections.pdf')
        plt.close(fig)
        images[row['case']] = p/'projections.png'

    raw, pooled, best = val['raw10']['mse'], val['pool10']['mse'], val['local_pool40']['mse']
    result_text = f'''# Findings and practical recommendation

The largest improvement comes from matching spatial resolution to the available information. On six additional validation objects, mean full-volume MSE changes from **{raw:.6f} to {pooled:.6f} cm⁻²** with exposure pooling, a **{100*(1-pooled/raw):.2f}% reduction**. This reproduces the earlier pooling finding on additional records.

The incoming-transverse, 40-iteration median-plus-pooling variant gives the lowest mean MSE in the frozen panel: **{best:.6f} cm⁻²**, **{100*(1-best/pooled):.2f}% below Pool10** and **{100*(1-best/raw):.2f}% below Raw10**. It beats Pool10 on **3/6** additional objects, rather than uniformly. Across all nine objects its MSE is {summary['all']['methods']['local_pool40']['mse']:.6f}, versus {summary['all']['methods']['pool10']['mse']:.6f} for Pool10.

Compared with Pool10, the transverse 40-iteration variant reduces mean material-only MSE from **{val['pool10']['material_mse']:.6f} to {val['local_pool40']['material_mse']:.6f}** and raises localization AUC from **{val['pool10']['auc']:.4f} to {val['local_pool40']['auc']:.4f}**. Its air-only MSE rises from **{val['pool10']['background_mse']:.6f} to {val['local_pool40']['background_mse']:.6f}**. This is a material-recovery versus false-positive tradeoff.

The 10-iteration transverse variant has the lowest air-only MSE among these candidates, **{val['local_pool10']['background_mse']:.6f}**, but leaves more material underestimated. For balanced density error, the 40-iteration version is the provisional candidate. For suppressing apparent material in empty space, the 10-iteration version merits comparison. Neither has established reliable small-feature resolution.

The all-zero-volume control has mean MSE **{summary['validation']['zero_mse']:.6f}**. The provisional candidate beats it on all six additional objects. The paired mean MSE difference between transverse Pool40 and Pool10 is {best-pooled:.6f} cm⁻²; the exploratory object-bootstrap 95% interval is [{val['local_pool40']['paired_mse_delta_vs_pool10_bootstrap95'][0]:.6f}, {val['local_pool40']['paired_mse_delta_vs_pool10_bootstrap95'][1]:.6f}]. This is not a simultaneous confidence statement over 11 inspected candidates.

## Which ideas failed in this test

- **More iterations alone:** Pool40 reaches {val['pool40']['mse']:.6f}, slightly worse full-volume error than Pool10. The coordinate change and the iteration count must be judged together.
- **Mean update:** transverse mean-plus-pooling reaches {val['mean_local_pool40']['mse']:.6f}; median remains slightly better on average.
- **Direct Gaussian ML:** {val['gaussian_ml']['mse']:.6f}. It assigns excessive density to poorly constrained coarse voxels.
- **Quadratic and Huber penalties:** {val['gaussian_quadratic']['mse']:.6f} and {val['gaussian_huber']['mse']:.6f}. These fixed settings do not control the failure sufficiently. This does not rule out other regularization strengths or better forward models.
- **Student plus Huber:** {val['student_huber']['mse']:.6f}. Robust tails help relative to direct Gaussian fitting but fail to beat simple Pool10 on average.
- **Angular only:** {val['angular_huber']['mse']:.6f}. Removing displacement is not the answer under these settings.

## Why likelihood score and image accuracy disagree

The common reserved-track Gaussian score is best for Pool40, whereas the common Student score is best for transverse Pool10. The lowest mean image error is achieved by transverse Pool40. The score measures consistency under a conditional approximate forward model; the image metric measures physical density relative to simulation truth. These are different targets, especially with unmeasured momenta and observation-derived paths.

## Implementation status

All work is isolated under `experiments/likelihood_physics_study`. Production reconstruction and configuration remain unchanged. An opt-in detector-only function, `reconstruct_local_pooled`, provides the provisional method, the physical output units and its provenance. It is not fitted to material labels. No new affine calibration, threshold, material assignment or per-object parameter is applied.

The complete evidence is in `results/metrics.json`, `results/summary.json`, the saved full/training maps and reserved-track scores. `results/protocol.json` records the initial experimental settings. `results/convergence_protocol.json` records the added optimizer-budget diagnostic before new validation evaluation.
'''
    # Check statements against every object rather than infer from mean metrics.
    assert all(r['models']['local_pool40']['mse'] < r['zero_mse'] for r in rows if r['group'] == 'validation')
    convergence_rows, convergence_data = [], []
    for case in ('funny_0', 'validation_11'):
        p = OUT/case
        old, new = np.load(p/'full_maps.npz'), np.load(p/'convergence_maps.npz')
        truth = np.load(p/'truth_evaluation_only.npz')['rho']
        meta = json.loads((p/'convergence.json').read_text())
        old_scores = np.load(p/'reserved_scores.npz')
        new_scores = np.load(p/'convergence_reserved_scores.npz')
        for m in new.files:
            a, b = float(np.mean((old[m]-truth)**2)), float(np.mean((new[m]-truth)**2))
            entry = dict(case=case, method=m, mse160=a, mse_extended=b,
                full_optimizer_success=meta[f'full__{m}']['success'],
                training_optimizer_success=meta[f'training__{m}']['success'],
                additional_full_iterations=meta[f'full__{m}']['iterations'],
                test_student_nll160=float(old_scores[m+'__student'].mean()),
                test_student_nll_extended=float(new_scores[m+'__student'].mean()))
            convergence_data.append(entry)
            convergence_rows.append([case, NAMES[m], f'{a:.6f}', f'{b:.6f}',
                str(entry['additional_full_iterations']), 'yes' if entry['full_optimizer_success'] else 'no'])
    (OUT/'convergence_evaluation.json').write_text(json.dumps(convergence_data, indent=2))
    result_text += '''
## Optimization and verification

All 160-step direct fits reached their iteration budget. Continuing all five methods on Funny 0 and validation 11 for up to 1,000 additional steps changes the full-volume MSE by less than 0.0001 in nine of ten comparisons. The exception is Student–Huber on validation 11, which worsens by about 0.0041. Some optimizers report convergence and others still hit the limit. These checks support the observed image-error conclusion on those two objects; they do not establish global likelihood optima.

The ten focused numerical tests pass: four likelihood-gradient cases, both penalty gradients, observation rotation invariance, synthetic density recovery and density–momentum degeneracy, coarse covariance integral preservation, and parity with the installed median solver. The existing suite passes 44 tests; three optional data integration tests are skipped in that run. A separate run enabling the real Funny TFRecord integration test passes all 11 tests (10 focused plus 1 integration). The experimental study itself performs real TFRecord extraction and reconstruction on six additional records.

**Recommendation:** retain exposure pooling as the main improvement, provide transverse-coordinate Pool40 as an experimental option, and preserve Pool10/Local Pool10 as explicit controls. Before promoting a new default, validate the accepted momentum spectrum and detector covariance on independent data and quantify the feature-size versus false-positive tradeoff. The present results support a useful regularized reconstruction, not uniquely calibrated material identification.
'''
    physics = (HERE/'PHYSICS.md').read_text()
    (HERE/'REPORT.md').write_text(physics+'\n\n---\n\n'+result_text)
    renderer = MarkdownIt('commonmark', {'html': True})
    scoreboard = table(['Method', 'Full MSE ↓', 'Material MSE ↓', 'Air MSE ↓', 'AUC ↑', 'Wins vs Pool10'],
        [[NAMES[m], f'{val[m]["mse"]:.6f}', f'{val[m]["material_mse"]:.6f}', f'{val[m]["background_mse"]:.6f}',
          f'{val[m]["auc"]:.4f}', f'{val[m]["mse_wins_over_pool10"]}/6'] for m in methods])
    allcases = table(['Object', 'Zero volume', 'Raw10', 'Pool10', 'Transverse Pool40', 'Change vs Pool10'],
        [[r['case'], f'{r["zero_mse"]:.6f}', *[f'{r["models"][m]["mse"]:.6f}' for m in ('raw10','pool10','local_pool40')],
          f'{100*(r["models"]["local_pool40"]["mse"]/r["models"]["pool10"]["mse"]-1):+.1f}%'] for r in rows])
    track_scores = table(['Method', 'Gaussian NLL ↓', 'Student NLL ↓'],
        [[NAMES[m], f'{val[m]["test_gaussian_nll"]:.5f}', f'{val[m]["test_student_nll"]:.5f}'] for m in methods])
    gallery = ''.join(f'<section class="case" id="{r["case"]}" {"" if i == 0 else "hidden"}>'
        + inline_image(images[r['case']], f'{r["case"]}: truth and five reconstruction methods; full-depth xy and xz mean projections')+'</section>' for i, r in enumerate(rows))
    select = '<select id="case-selector" aria-label="Select reconstructed object">'+''.join(f'<option>{r["case"]}</option>' for r in rows)+'</select>'
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Likelihood reconstruction — physics and validation</title>
<style>body{margin:0;background:#f6f5f1;color:#202d38;font:16px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:36px 28px 80px}header{border-bottom:2px solid #167d8d;padding-bottom:24px}h1{font-size:30px;line-height:1.25;margin:18px 0}h2{font-size:23px;margin-top:40px}h3{font-size:18px}p,li{max-width:960px}a{color:#07697a}nav{display:flex;flex-wrap:wrap;gap:20px;margin-top:22px}.eyebrow{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#526273}pre{overflow:auto;padding:20px;background:#e9edf0;line-height:1.55}code{font-size:14px}img{display:block;width:100%;height:auto;margin:18px 0}.table-wrap{overflow-x:auto;margin:24px 0}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #d7dcdf;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e9edf0}caption,.caption{font-size:13px;color:#566675}.lead{font-size:19px;max-width:950px}.facts{display:flex;gap:35px;flex-wrap:wrap;margin:24px 0}.fact strong{display:block;font-size:28px;color:#0b7180}.fact span{font-size:14px}select{font:inherit;padding:8px 18px;border:1px solid #a6b3bc;background:white}details{margin:18px 0}summary{cursor:pointer;font-weight:600}section{scroll-margin-top:20px}[hidden]{display:none!important}</style></head><body><main>
<header><div class="eyebrow">Unified muon tomography · 15 September 2026</div><h1>Likelihood reconstruction, starting from the physics</h1><p class="lead">Spatial support is the dominant practical limit. Exposure pooling produces the largest gain; transverse coordinates and more iterations add a smaller, nonuniform improvement.</p>
'''
    document += f'<div class="facts"><div class="fact"><strong>11</strong><span>fixed reconstruction variants</span></div><div class="fact"><strong>6 + 3</strong><span>additional validation + development objects</span></div><div class="fact"><strong>{100*(1-best/raw):.1f}%</strong><span>validation MSE reduction vs unpooled control</span></div><div class="fact"><strong>{100*(1-best/pooled):.1f}%</strong><span>increment beyond existing pooling</span></div></div>'
    document += '<nav><a href="#physics">Physical motivation</a><a href="#findings">Findings</a><a href="#scores">All methods</a><a href="#images">Reconstruction images</a><a href="#optimization">Convergence</a></nav></header>'
    document += '<section id="physics">'+renderer.render(physics)+'</section>'
    document += '<section id="findings">'+renderer.render(result_text)+'</section>'
    document += '<section id="scores"><h2>Validation: complete comparison</h2><p>Six objects, equally weighted. MSE units are cm⁻²; no truth smoothing or fitted density scaling.</p>'+scoreboard+inline_image(OUT/'validation_comparison.png','Validation comparison of full-volume, material-only and air-only MSE for all eleven methods')
    document += '<h2>Every object, including failures to improve</h2><p>Positive change means worse MSE. The first three rows are development objects; all others are additional validation records.</p>'+allcases
    document += '<h2>Reserved-track consistency</h2><p>Mean per-object conditional NLL, in natural-log units per usable reserved track. Compare within a column only. The scoring frame, noise and events are identical across methods.</p>'+track_scores+'</section>'
    document += '<section id="images"><h2>Inspect the actual reconstructions</h2><p>Full-depth mean projections avoid selecting only a favorable slice. Each row shares its truth-derived display range; saturated bright values remain unchanged in the saved volumes.</p>'+select+gallery+'</section>'
    document += '<section id="optimization"><h2>Longer-budget check</h2><p>The optimizer was allowed up to 1,000 additional steps from each 160-step solution. “Converged” is the optimizer flag, not proof of a global optimum.</p>'+table(['Object','Method','MSE at 160','Extended MSE','Extra steps','Converged'],convergence_rows)+'</section>'
    document += '<footer><h2>Reproducibility</h2><p>Experimental sources, detector-only entry point and commands: <a href="README.md">README</a>. Full text: <a href="REPORT.md">REPORT.md</a>. Exact data: <a href="results/metrics.json">metrics.json</a> and <a href="results/summary.json">summary.json</a>. Scientific figure: <a href="results/validation_comparison.pdf">PDF</a>. Production source hashes were verified unchanged.</p></footer></main><script>document.getElementById("case-selector").addEventListener("change",e=>document.querySelectorAll(".case").forEach(c=>c.hidden=c.id!==e.target.value));</script></body></html>'
    (HERE/'report.html').write_text(document)
    current = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')}
    (OUT/'final_source_sha256.json').write_text(json.dumps(current, indent=2))
    print('Wrote', HERE/'report.html', flush=True)


if __name__ == '__main__':
    main()
