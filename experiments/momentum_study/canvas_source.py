"""Prepare a self-contained canvas source from completed scientific results."""
# ruff: noqa: RUF001 -- Intentional mathematical notation in user-facing text.

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Staging text file for the managed canvas writer"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    if not json.loads((run / "manifest.json").read_text()).get("completed"):
        raise ValueError("Refuse incomplete scientific results")
    from muon_tomo.research.momentum_study import POLICIES

    summary = json.loads((run / "summary.json").read_text())
    rows = json.loads((run / "metrics.json").read_text())
    objects = json.loads((run / "objects.json").read_text())
    payload = {
        "policies": {key: list(value) for key, value in POLICIES.items()},
        "objects": objects,
        "summary": [
            {
                k: r[k]
                for k in [
                    "group",
                    "objects",
                    "method",
                    "policy",
                    "macro_mse_full",
                    "macro_mse_object",
                    "macro_object_bias",
                    "macro_background_fp_fraction_gt_0p8",
                    "macro_delta_mse_to_historical",
                    "delta_mse_ci_low",
                    "delta_mse_ci_high",
                    "improved_objects",
                    "macro_zero_mse_full",
                ]
            }
            for r in summary
        ],
        "rows": [
            {
                k: r[k]
                for k in [
                    "case",
                    "method",
                    "policy",
                    "mse_full",
                    "mse_object",
                    "mse_background",
                    "object_mean",
                    "object_truth_mean",
                    "zero_mse_full",
                    "zero_mse_object",
                    "background_fp_fraction_gt_0p8",
                ]
            }
            for r in rows
        ],
    }
    template = """import { H1, H2, Text, Row, Stack, Select, Table, Button, useState, useHostTheme, useCanvasAction } from "cursor/canvas";
const D = __DATA__;
const ROOT = __ROOT__;
const fmt = (n: number | null) => n === null ? "—" : n.toPrecision(4);
const policies: Record<string, string[]> = D.policies;
export default function MomentumStudy() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();
  const [method,setMethod] = useState("munet");
  const [group,setGroup] = useState("legacy");
  const [caseName,setCase] = useState("lead25");
  const aggregates = D.summary.filter(r=>r.method===method && r.group===group);
  const selected = D.rows.filter(r=>r.method===method && r.case===caseName);
  const object = D.objects.find(o=>o.case===caseName)!;
  const baseline = aggregates.find(r=>r.policy==="historical")!;
  const open = (file: string) => dispatch({type:"openFile", path:ROOT+"/"+file});
  return <Stack gap={20} style={{padding:28,maxWidth:1400,margin:"auto",color:theme.text.primary}}>
    <H1>Momentum sensitivity: MuNet and PoCA</H1>
    <Text>356 reconstructions · 12 objects · 8,192 filtered tracks per object · 15 momentum choices. Two optional initial-energy diagnostics are unavailable for Custom 0. Fixed PoCA 32³ / σ=6.25 cm; frozen MuNet, eight 1024-track chunks averaged.</Text>
    <Text tone="secondary">Reference priors use 474,603 accepted tracks from independent training records 0–31. Event and object oracle policies use test-track truth. No retraining or truth-based rescaling. Lower full-grid MSE can coexist with stronger material underestimation.</Text>
    <Row gap={12}><Button onClick={()=>open("report.html")}>Open full report and plots</Button><Button onClick={()=>open("metrics.csv")}>All results as CSV</Button><Button onClick={()=>open("mse_ratios.png")}>Compare all objects</Button></Row>
    <H2>Performance across objects</H2>
    <Row gap={12}><Text>Method</Text><Select value={method} onChange={setMethod} options={[{value:"munet",label:"MuNet"},{value:"poca",label:"PoCA"}]}/><Text>Object group</Text><Select value={group} onChange={setGroup} options={[{value:"legacy",label:"11 legacy objects"},{value:"validation_test",label:"8 validation/test objects"},{value:"new_source",label:"Lead cube: different source"}]}/></Row>
    <Text>All-zero prediction: mean full-grid MSE {fmt(baseline.macro_zero_mse_full)} cm⁻². Each object has equal weight. Optional initial-energy policies have 10 available legacy objects; other policies have 11.</Text>
    <Table headers={["Policy", "Information", "Objects", "Full MSE (cm⁻²)", "Object MSE (cm⁻²)", "Object bias (cm⁻¹)", "Δ full MSE [95% interval]", "Improved"]}
      rows={aggregates.map(r=>[policies[r.policy][0],policies[r.policy][1],r.objects,fmt(r.macro_mse_full),fmt(r.macro_mse_object),fmt(r.macro_object_bias),`${fmt(r.macro_delta_mse_to_historical)} [${fmt(r.delta_mse_ci_low)}, ${fmt(r.delta_mse_ci_high)}]`,`${r.improved_objects}/${r.objects}`])} striped columnAlign={["left","left","right","right","right","right","right","right"]}/>
    <Text tone="secondary" size="small">Δ compares each policy with the historical 1.743 GeV/c prior on the same available objects; negative improves. Intervals use 10,000 paired object-bootstrap draws. Exploratory intervals apply to this selected small object set, not future detector data. No independent track-subset repeats.</Text>
    <H2>Inspect material recovery</H2>
    <Row gap={12}><Select value={caseName} onChange={setCase} options={D.objects.map(o=>({value:o.case,label:o.case}))}/><Button onClick={()=>open(caseName+"/population_sections.png")}>Population cross sections</Button><Button onClick={()=>open(caseName+"/oracle_sections.png")}>Oracle cross sections</Button></Row>
    <Text>{object.materials.join(", ")} · object occupies {(object.object_fraction*100).toFixed(2)}% of the volume. Displayed z={object.slice_z_cm.toFixed(2)} cm. All cross-section colours saturate at 3.2 cm⁻¹. MSE uses the full unmasked 64³ array.</Text>
    <Table headers={["Policy", "Full MSE (cm⁻²)", "Object MSE (cm⁻²)", "Object mean (cm⁻¹)", "Truth mean (cm⁻¹)", "Air MSE (cm⁻²)", "Air >0.8"]}
      rows={selected.map(r=>[policies[r.policy][0],fmt(r.mse_full),fmt(r.mse_object),fmt(r.object_mean),fmt(r.object_truth_mean),fmt(r.mse_background),(100*r.background_fp_fraction_gt_0p8).toFixed(2)+"%"])} striped columnAlign={["left","right","right","right","right","right","right"]}/>
    <Text tone="secondary" size="small">Arithmetic = E[p]; geometric = exp(E[log p]); inverse-square effective = E[p⁻²]⁻¹ᐟ². Zenith means use physical incoming direction and fixed bins. Truth outgoing is the legacy feature's momentum; initial/source momentum is derived from kinetic energy. Initial/outgoing geometric momentum is only an energy-loss proxy.</Text>
    <Text tone="secondary" size="small">Source: local 2026-10-20 study, corrected run v3. Historical 1.743 GeV/c was derived from Funny 0. Lead25 uses a different simulator/source spectrum. These ideal simulated tracks do not model RPC resolution, timing or noise.</Text>
  </Stack>;
}
"""
    source = template.replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    ).replace("__ROOT__", json.dumps(str(run)))
    args.output.write_text(source)
    print("Prepared canvas source from", len(rows), "completed reconstructions")


if __name__ == "__main__":
    main()
