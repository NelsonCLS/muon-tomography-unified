"""Summarize saved momentum runs; never rerun inference or rescale predictions."""
# ruff: noqa: RUF001 -- Intentional scientific symbols and publication typography.

from __future__ import annotations

import argparse
import csv
import json
import re
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muon_tomo.constants import MATERIAL_NAMES, labels_to_inverse_radiation_length
from muon_tomo.geometry import GridGeometry
from muon_tomo.research.momentum_study import POLICIES
from muon_tomo.types import ReconstructionResult, TruthVolume
from muon_tomo.visualization.viewer import plot_comparison_3d


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    groups = {
        "legacy": sorted({r["case"] for r in rows if r["group"] == "legacy"}),
        "new_source": ["lead25"],
        "validation_test": sorted(
            {r["case"] for r in rows if r["case"].startswith(("val", "test"))}
        ),
    }
    for group, cases in groups.items():
        for method in ("munet", "poca"):
            lookup = {(r["case"], r["policy"]): r for r in rows if r["method"] == method}
            for policy in POLICIES:
                available = [c for c in cases if (c, policy) in lookup]
                if not available:
                    continue
                ids = np.random.default_rng(20261020).integers(
                    0, len(available), size=(10000, len(available))
                )
                selected = [lookup[c, policy] for c in available]
                baseline = np.array([lookup[c, "historical"]["mse_full"] for c in available])
                values = np.array([r["mse_full"] for r in selected])
                differences = values - baseline
                boot = differences[ids].mean(axis=1)
                summary.append(
                    {
                        "group": group,
                        "objects": len(available),
                        "method": method,
                        "policy": policy,
                        "information": POLICIES[policy][1],
                        "macro_mse_full": float(values.mean()),
                        "macro_mse_object": float(np.mean([r["mse_object"] for r in selected])),
                        "macro_mse_background": float(
                            np.mean([r["mse_background"] for r in selected])
                        ),
                        "macro_object_bias": float(np.mean([r["object_bias"] for r in selected])),
                        "macro_background_fp_fraction_gt_0p8": float(
                            np.mean([r["background_fp_fraction_gt_0p8"] for r in selected])
                        ),
                        "median_mse_ratio_to_historical": float(np.median(values / baseline)),
                        "improved_objects": int(np.sum(differences < 0)),
                        "macro_delta_mse_to_historical": float(differences.mean()),
                        "delta_mse_ci_low": float(np.percentile(boot, 2.5))
                        if len(available) > 1
                        else None,
                        "delta_mse_ci_high": float(np.percentile(boot, 97.5))
                        if len(available) > 1
                        else None,
                        "macro_zero_mse_full": float(
                            np.mean([r["zero_mse_full"] for r in selected])
                        ),
                    }
                )
    return summary


def heatmap(run, rows, cases):
    fig, axes = plt.subplots(1, 2, figsize=(17, 10), constrained_layout=True)
    for ax, method in zip(axes, ("munet", "poca"), strict=True):
        lut = {(r["case"], r["policy"]): r["mse_full"] for r in rows if r["method"] == method}
        ratios = np.array(
            [[lut.get((c, p), np.nan) / lut[c, "historical"] for c in cases] for p in POLICIES]
        )
        img = ax.imshow(np.log10(ratios), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(POLICIES)):
            for j in range(len(cases)):
                r = ratios[i, j]
                text = "N/A" if not np.isfinite(r) else f"{r:.2f}" if r < 10 else f"{r:.0f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(np.log10(r)) > 0.6 else "black",
                )
        ax.set_xticks(range(len(cases)), cases, rotation=50, ha="right")
        ax.set_yticks(range(len(POLICIES)), [v[0] for v in POLICIES.values()], fontsize=9)
        ax.set_title("MuNet" if method == "munet" else "PoCA")
        ax.set_xlabel("Object (lead25 uses a different simulation source)")
        ax.set_ylabel("Momentum policy")
        for boundary in [0.5, 6.5, 9.5]:
            ax.axhline(boundary, color="gray", linewidth=0.7)
    cbar = fig.colorbar(img, ax=axes, shrink=0.65, ticks=[-1, 0, 1], extend="both")
    cbar.ax.set_yticklabels(["0.1×", "1×", "10×"])
    cbar.set_label("Full-grid MSE / historical baseline MSE (log colour scale)")
    fig.suptitle(
        "Momentum sensitivity: values below 1 improve MSE\n12 fixed objects · 8,192 filtered tracks each · frozen settings",
        fontsize=15,
    )
    fig.savefig(run / "mse_ratios.png", dpi=170)
    fig.savefig(run / "mse_ratios.pdf")
    plt.close(fig)


def cross_sections(run, rows, cases):
    index = {(r["case"], r["method"], r["policy"]): r for r in rows}
    sets = {
        "population": [
            "historical",
            "ref_arithmetic",
            "ref_geometric",
            "ref_median",
            "ref_percentile10",
            "ref_inverse_square",
            "ref_zenith_arithmetic",
            "ref_zenith_inverse_square",
        ],
        "oracle": [
            "historical",
            "truth_outgoing",
            "truth_incoming",
            "truth_endpoint_geometric",
            "object_arithmetic",
            "object_geometric",
            "object_inverse_square",
        ],
    }
    case_info = []
    for case in cases:
        labels = np.load(run / case / "input.npz")["labels"]
        truth = labels_to_inverse_radiation_length(labels)
        # Select a representative slice only for display, after prediction.
        z = int(np.clip(np.rint(np.argwhere(labels > 0)[:, 2].mean()), 0, 63))
        slice_cm = -50 + (z + 0.5) * 100 / 64
        material_labels = np.unique(labels[labels > 0])
        case_info.append(
            {
                "case": case,
                "slice_z_index": z,
                "slice_z_cm": slice_cm,
                "materials": [MATERIAL_NAMES[k] for k in material_labels],
                "object_fraction": float(np.mean(labels > 0)),
                "truth_max": float(truth.max()),
            }
        )
        for group, base_policies in sets.items():
            policies = [p for p in base_policies if (case, "poca", p) in index]
            if group == "oracle" and len(policies) < 7:
                policies += ["object_zenith_arithmetic", "object_zenith_inverse_square"]
            fig, axes = plt.subplots(
                2,
                len(policies) + 1,
                figsize=(2.5 * (len(policies) + 1), 6),
                constrained_layout=True,
            )
            for row, method in enumerate(("munet", "poca")):
                for col, policy in enumerate([None, *policies]):
                    ax = axes[row, col]
                    if policy is None:
                        volume = truth
                        title = "Truth"
                    else:
                        volume = ReconstructionResult.load(
                            run / case / f"{method}__{policy}.npz"
                        ).inverse_radiation_length_cm_inv
                        title = (
                            POLICIES[policy][0]
                            .replace("Reference ", "Ref. ")
                            .replace("Object ", "Obj. ")
                            .replace("inverse-square", "inv. square")
                        )
                        title += f"\nMSE={index[case, method, policy]['mse_full']:.4g}"
                    img = ax.imshow(
                        volume[:, :, z].T,
                        origin="lower",
                        extent=[-50, 50, -50, 50],
                        cmap="inferno",
                        vmin=0,
                        vmax=3.2,
                        interpolation="nearest",
                    )
                    ax.set_title(title, fontsize=9)
                    ax.set_xticks([-50, 0, 50])
                    ax.set_yticks([-50, 0, 50])
                    ax.tick_params(labelsize=8)
                    ax.set_xlabel("x (cm)", fontsize=9)
                    ax.set_ylabel(
                        ("MuNet" if row == 0 else "PoCA") + "\ny (cm)" if col == 0 else "y (cm)",
                        fontsize=9,
                    )
            cb = fig.colorbar(img, ax=axes, shrink=0.7, extend="max", pad=0.01)
            cb.set_label("1/X₀ (cm⁻¹); saturates at 3.2")
            fig.suptitle(
                f"{case}: {'population priors' if group == 'population' else 'simulation oracle diagnostics'} | z={slice_cm:.2f} cm\n8,192 identical tracks · full-grid MSE in cm⁻² · all panels share 0–3.2 cm⁻¹",
                fontsize=14,
            )
            fig.savefig(run / case / f"{group}_sections.png", dpi=150)
            fig.savefig(run / case / f"{group}_sections.pdf")
            plt.close(fig)
        # Four-panel deliverables for the three primary population choices, per method.
        for method in ("munet", "poca"):
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)
            for ax, policy in zip(
                axes, [None, "ref_arithmetic", "ref_geometric", "ref_inverse_square"], strict=True
            ):
                volume = (
                    truth
                    if policy is None
                    else ReconstructionResult.load(
                        run / case / f"{method}__{policy}.npz"
                    ).inverse_radiation_length_cm_inv
                )
                img = ax.imshow(
                    volume[:, :, z].T,
                    origin="lower",
                    extent=[-50, 50, -50, 50],
                    cmap="inferno",
                    vmin=0,
                    vmax=3.2,
                )
                ax.set_title(
                    "Truth"
                    if policy is None
                    else POLICIES[policy][0]
                    + f"\nMSE {index[case, method, policy]['mse_full']:.4g}",
                    fontsize=10,
                )
                ax.set_xlabel("x (cm)")
                ax.set_ylabel("y (cm)")
            fig.colorbar(img, ax=axes, extend="max", shrink=0.75).set_label("1/X₀ (cm⁻¹)")
            fig.suptitle(f"{case} · {method} · z={slice_cm:.2f} cm · MSE in cm⁻²", fontsize=12)
            fig.savefig(run / case / f"{method}_four_panel.png", dpi=160)
            plt.close(fig)
    dump(run / "objects.json", case_info)
    return case_info


def momentum_plot(run, reference):
    prior = reference
    centers = (np.array(prior["edges_degrees"][:-1]) + np.array(prior["edges_degrees"][1:])) / 2
    counts = np.array(prior["counts"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, label in [
        ("arithmetic", "Arithmetic"),
        ("geometric", "Geometric"),
        ("inverse_square", "Inverse-square effective"),
        ("percentile10", "10th percentile"),
    ]:
        vals = np.array(prior["bin_mev"][key]) / 1000
        valid = counts >= prior["minimum_count"]
        axes[0].plot(centers[valid], vals[valid], marker="o", label=label)
    axes[0].set(
        xlabel="Incoming zenith bin centre (degrees)",
        ylabel="Assigned momentum (GeV/c)",
        title="Reference conditional momenta",
    )
    axes[0].legend()
    axes[1].bar(centers, counts, width=np.diff(prior["edges_degrees"]) * 0.9, color="gray")
    axes[1].set(
        xlabel="Incoming zenith bin centre (degrees)",
        ylabel="Accepted reference tracks per bin",
        title=f"Independent training records 0–31: {sum(counts):,} tracks",
    )
    fig.savefig(run / "reference_momenta.png", dpi=170)
    plt.close(fig)


def html_report(run, rows, summary, objects, reference):
    payload = json.dumps(
        {
            "rows": rows,
            "summary": summary,
            "objects": objects,
            "policies": POLICIES,
            "reference": reference,
        },
        allow_nan=False,
    ).replace("</", "<\\/")
    template = """<!doctype html><html><meta charset="utf-8"><title>MuNet and PoCA momentum study</title>
<style>body{max-width:1240px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#252b33;background:#fafafa}h1{font-size:30px}h2{margin-top:32px}select{font:inherit;padding:7px;margin:8px 20px 12px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:right;padding:8px;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}thead{background:#eee}img{width:100%;height:auto}small{color:#58606a}.notice{border-left:3px solid #405b78;padding:8px 16px}a{color:#215c99}.links a{margin-right:20px}</style>
<h1>How momentum assumptions change MuNet and PoCA</h1>
<p>12 fixed objects, {len(POLICIES)} momentum policies and 2 methods: <b>{len(rows)} reconstructions</b>. Two optional source-energy diagnostics are unavailable for Custom 0 because its legacy record lacks initial energy. Each uses the same 8,192 filtered tracks per object. Frozen MuNet weights; PoCA 32³, Gaussian σ=6.25 cm.</p>
<p class="notice">Independent reference priors come from simulation, not an external momentum measurement. Event and object oracle policies use test-track truth. These are separate information settings. No truth-based calibration or retraining.</p>
<p class="links"><a href="metrics.csv">All metrics (CSV)</a><a href="summary.csv">Group summary and paired intervals</a><a href="materials.csv">Per-material results</a><a href="protocol.md">Protocol</a><a href="manifest.json">Run provenance</a><a href="parity.json">Parity checks</a></p>
<h2>Compare groups and uncertainty</h2><label>Group <select id="group"><option value="legacy">11 legacy-source objects</option><option value="validation_test">8 validation/test objects</option><option value="new_source">Lead cube: separate source</option></select></label><label>Method <select id="method"><option value="munet">MuNet</option><option value="poca">PoCA</option></select></label>
<div style="overflow:auto"><table id="aggregate"></table></div>
<small>Full and object MSE in cm⁻²; bias in cm⁻¹. Macro means weight each object equally. Δ is policy minus historical MSE; negative improves. Intervals: paired object bootstrap, 10,000 draws; exploratory, conditional on these selected objects. No independent track-subset repeats.</small>
<p id="zero"></p><h2>Every object and momentum policy</h2><img src="mse_ratios.png" alt="MSE ratios for both methods across all objects"><p>Numbers below 1 improve on the historical baseline. Colour saturates beyond 0.1× and 10×; printed numbers are untruncated. Only cross-section density colours saturate at 3.2 cm⁻¹.</p>
<h2>Inspect an object</h2><label>Object <select id="object"></select></label><p id="description"></p><div style="overflow:auto"><table id="detail"></table></div>
<p id="viewer"></p><h3>Independent reference priors</h3><img id="population" alt="Population-prior cross-section comparisons"><h3>Truth-based diagnostics</h3><img id="oracle" alt="Oracle cross-section comparisons">
<small>For display only, slice z is chosen at the mean occupied truth-voxel z index after predictions. All policies use the same slice and fixed 0–3.2 cm⁻¹ colours. Scores use the entire 64³ volume, never the slice or display threshold.</small>
<h2>What the averages mean</h2><p>The choice of inverse-square averaging follows the leading momentum dependence of multiple-scattering variance; see <a href="https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf">PDG, Passage of Particles Through Matter, section 34.3</a>. It is an approximation: energy loss, path exposure, acceptance and PoCA localization can still produce bias.</p><p>Arithmetic: E[p]. Geometric: exp(E[log p]). Median: 50th percentile. The 10th-percentile policy assigns the global reference outgoing-momentum quantile (linear interpolation) to every track. Inverse-square effective: E[p⁻²]⁻¹ᐟ². Zenith policies replace each track's momentum with a fixed-bin conditional average using its incoming direction. Bins have 5° width to 45°, then 45–90°. Sparse bins use the global statistic.</p><img src="reference_momenta.png" alt="Reference momenta by zenith angle and sample counts">
<p>True outgoing momentum matches the legacy model feature. True initial/source and √(p<sub>initial</sub>p<sub>out</sub>) test energy-loss sensitivity; neither gives the exact scattering-path momentum integral. Per-track truth is not guaranteed to minimize MSE for a biased approximate estimator or a frozen model.</p>
<p>The historical 1.743 GeV/c prior came from Funny 0, so that case is not independent of that baseline. The 25 cm lead cube uses a different source spectrum and simulator version from the legacy data. All inputs are ideal simulated tracks: these results do not include RPC spatial or timing errors. The random seed is reset after model loading for each momentum policy.</p>
<script>const D=__DATA__;
const fmt=x=>x===null?'—':Number(x).toPrecision(4);
function table(id,heads,rows){document.getElementById(id).innerHTML='<thead><tr>'+heads.map(s=>'<th>'+s+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(s=>'<td>'+s+'</td>').join('')+'</tr>').join('')+'</tbody>';}
const group=document.getElementById('group'),method=document.getElementById('method'),obj=document.getElementById('object');
obj.innerHTML=D.objects.map(o=>'<option>'+o.case+'</option>').join('');
function render(){const s=D.summary.filter(r=>r.group===group.value&&r.method===method.value);table('aggregate',['Policy / information','Full MSE','Object MSE','Object bias','Air FP >0.8','Δ full MSE [95% CI]','Improved objects'],s.map(r=>[D.policies[r.policy][0]+'<br><small>'+r.information+'</small>',fmt(r.macro_mse_full),fmt(r.macro_mse_object),fmt(r.macro_object_bias),(100*r.macro_background_fp_fraction_gt_0p8).toFixed(2)+'%',fmt(r.macro_delta_mse_to_historical)+' ['+fmt(r.delta_mse_ci_low)+', '+fmt(r.delta_mse_ci_high)+']',r.improved_objects+'/'+r.objects]));
document.getElementById('zero').textContent='All-zero prediction macro full-grid MSE: '+fmt(s.find(r=>r.policy==='historical').macro_zero_mse_full)+' cm⁻². Policies requiring initial energy use 10 legacy objects; other policies use 11.';
const o=D.objects.find(o=>o.case===obj.value);document.getElementById('description').textContent=o.materials.join(', ')+'; object fills '+(100*o.object_fraction).toFixed(2)+'% of volume; displayed z='+o.slice_z_cm.toFixed(2)+' cm.';
const rs=D.rows.filter(r=>r.case===obj.value&&r.method===method.value);document.getElementById('description').textContent+=' All-zero MSE: full '+fmt(rs[0].zero_mse_full)+', object '+fmt(rs[0].zero_mse_object)+' cm⁻².';table('detail',['Policy','Full MSE','Object MSE','Object mean','Object truth mean','Air MSE','Air FP >0.8'],rs.map(r=>[D.policies[r.policy][0],fmt(r.mse_full),fmt(r.mse_object),fmt(r.object_mean),fmt(r.object_truth_mean),fmt(r.mse_background),(100*r.background_fp_fraction_gt_0p8).toFixed(2)+'%']));
document.getElementById('population').src=obj.value+'/population_sections.png';document.getElementById('oracle').src=obj.value+'/oracle_sections.png';document.getElementById('viewer').innerHTML=['lead25','funny0'].includes(obj.value)?'<a href="'+obj.value+'/viewer_3d.html">Open 3D comparison: all voxels >0.8 cm⁻¹, interiors included</a>':'';}
for(const el of [group,method,obj])el.onchange=render;render();</script></html>"""
    narrative = run / "findings.json"
    if narrative.is_file():
        paragraphs = json.loads(narrative.read_text())
        findings = "<h2>Findings and practical limits</h2>" + "".join(
            "<p>" + escape(p) + "</p>" for p in paragraphs
        )
        template = template.replace(
            "<h2>Compare groups and uncertainty</h2>",
            findings + "<h2>Compare groups and uncertainty</h2>",
        )
    template = template.replace(
        '<a href="parity.json">Parity checks</a>',
        '<a href="parity.json">Parity checks</a><a href="validation.json">Complete validation</a>',
    )
    (run / "report.html").write_text(
        template.replace("__DATA__", payload)
        .replace("{len(POLICIES)}", str(len(POLICIES)))
        .replace("{len(rows)}", str(len(rows)))
    )


def set_viewer_color_limit(path, metadata):
    """Apply the study's 3.2 display scale without touching scientific arrays."""
    old_max = metadata["color_scale_cm_inv"][1]
    html, replacements = re.subn(
        r'"cmax"\s*:\s*' + re.escape(str(old_max)), '"cmax":3.2', path.read_text()
    )
    if replacements == 0:
        raise ValueError("No viewer trace colour limits found")
    path.write_text(html)
    metadata["color_scale_cm_inv"] = [0.0, 3.2]
    metadata["opacity_normalization_cm_inv"] = old_max
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--skip-viewers", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if not manifest.get("completed"):
        raise ValueError("Cannot publish an incomplete study")
    rows = json.loads((run / "metrics.json").read_text())
    materials = json.loads((run / "materials.json").read_text())
    reference = json.loads((run / "reference.json").read_text())
    cases = [s[0] for s in manifest["cases"]]
    expected = sum(
        2
        * (
            len(POLICIES)
            - len(json.loads((run / c / "input.json").read_text())["unavailable_policies"])
        )
        for c in cases
    )
    assert len(rows) == expected
    assert len({(r["case"], r["method"], r["policy"]) for r in rows}) == expected
    summary = summarize(rows)
    dump(run / "summary.json", summary)
    for name, data in [("metrics", rows), ("materials", materials), ("summary", summary)]:
        csv_write(run / f"{name}.csv", data)
    heatmap(run, rows, cases)
    objects = cross_sections(run, rows, cases)
    momentum_plot(run, reference)
    if not args.skip_viewers:
        for case in ("lead25", "funny0"):
            labels = np.load(run / case / "input.npz")["labels"]
            truth = TruthVolume(labels, labels_to_inverse_radiation_length(labels), GridGeometry())
            volumes = []
            for policy in ("historical", "ref_inverse_square", "truth_outgoing"):
                for method in ("munet", "poca"):
                    result = ReconstructionResult.load(run / case / f"{method}__{policy}.npz")
                    result.metadata["display_name"] = f"{method}: {POLICIES[policy][0]}"
                    result.metadata["method_id"] = f"{method}_{policy}"
                    volumes.append(result)
            metadata = plot_comparison_3d(
                truth,
                volumes,
                output=run / case / "viewer_3d.html",
                threshold_cm_inv=0.8,
                max_cubes=None,
                surface_only=False,
            )
            metadata = set_viewer_color_limit(run / case / "viewer_3d.html", metadata)
            dump(run / case / "viewer_3d.json", metadata)
    html_report(run, rows, summary, objects, reference)
    print("Report saved", run / "report.html")
    for method in ("munet", "poca"):
        print(
            method,
            "legacy macro MSE",
            [
                (r["policy"], round(r["macro_mse_full"], 6))
                for r in summary
                if r["group"] == "legacy" and r["method"] == method
            ],
        )


if __name__ == "__main__":
    main()
