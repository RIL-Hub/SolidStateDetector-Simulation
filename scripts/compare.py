#!/usr/bin/env python3
"""
Compare waveforms from SolidStateDetectors.jl and CTSI simulators.
Generates an interactive HTML report with overlaid Plotly traces.

Usage:
    python3 scripts/compare.py \
        --ssd output/ssd_waveforms.json \
        --ctsi ctsi/output/interactiveOut.txt \
        --ctsi-root ctsi \
        --output output/comparison.html
"""

import argparse
import json
import sys
import os

def parse_ssd(path):
    """Load SSD waveform JSON export."""
    with open(path) as f:
        return json.load(f)

def parse_ctsi(output_path, ctsi_root):
    """Parse CTSI interactiveOut.txt using its Python toolkit."""
    sys.path.insert(0, os.path.abspath(ctsi_root))
    from tools.ctsi_toolkit.parser import parse_interactive_output
    result = parse_interactive_output(output_path, mode="full")

    # Load detector spec for metadata
    spec = {}
    spec_path = os.path.join(ctsi_root, "config", "detectorSpec.txt")
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("//")[0].strip().split()
                if len(parts) >= 2:
                    spec[parts[0]] = parts[1]

    return result, spec

def build_html(ssd_data, ctsi_result, ctsi_spec, output_path):
    """Generate comparison HTML with Plotly overlays."""

    # -- Extract CTSI waveforms --
    ctsi_time = ctsi_result.time_vec * 1e9  # s → ns
    ctsi_time_preamp = ctsi_result.time_vec_preamp * 1e9

    # Find the collecting anode (center) and neighbors
    n_anodes = int(ctsi_spec.get("NUM_ANODES", 39))
    center_anode_ctsi = n_anodes // 2  # 0-indexed: anode 19 for 39 anodes

    # Build CTSI traces
    ctsi_traces = {}

    # Anode waveforms (raw current)
    for i, collector_id in enumerate(ctsi_result.an_collector_id):
        offset = collector_id - center_anode_ctsi
        if abs(offset) <= 2:
            label = f"ctsi_anode_{offset:+d}" if offset != 0 else "ctsi_anode_center"
            t = ctsi_time[:len(ctsi_result.an_vs_time[i])]
            ctsi_traces[label] = {
                "raw_time_ns": t.tolist(),
                "raw_current": ctsi_result.an_vs_time[i].tolist(),
                "preamp_time_ns": ctsi_time_preamp[:len(ctsi_result.an_vs_time_preamp[i])].tolist(),
                "preamp_signal": ctsi_result.an_vs_time_preamp[i].tolist(),
                "type": "anode",
                "offset": offset,
            }

    # Cathode waveforms
    for i, collector_id in enumerate(ctsi_result.ca_collector_id):
        label = f"ctsi_cathode_{collector_id}"
        t = ctsi_time[:len(ctsi_result.ca_vs_time[i])]
        ctsi_traces[label] = {
            "raw_time_ns": t.tolist(),
            "raw_current": ctsi_result.ca_vs_time[i].tolist(),
            "preamp_time_ns": ctsi_time_preamp[:len(ctsi_result.ca_vs_time_preamp[i])].tolist(),
            "preamp_signal": ctsi_result.ca_vs_time_preamp[i].tolist(),
            "type": "cathode",
            "offset": collector_id,
        }

    # -- Extract SSD waveforms --
    ssd_wf = ssd_data.get("waveforms", {})

    # Map SSD contacts to comparison roles
    ssd_mapping = {
        "anode_3": ("ssd_anode_center", 0),
        "anode_2": ("ssd_anode_-1", -1),
        "anode_4": ("ssd_anode_+1", +1),
        "anode_1": ("ssd_anode_-2", -2),
        "anode_5": ("ssd_anode_+2", +2),
        "cathode_2": ("ssd_cathode", None),
        "cathode_1": ("ssd_cathode_1", None),
        "steering": ("ssd_steering", None),
    }

    ssd_traces = {}
    for contact_name, (label, offset) in ssd_mapping.items():
        if contact_name in ssd_wf:
            wf = ssd_wf[contact_name]
            ssd_traces[label] = {
                "raw_time_ns": wf["raw_time_ns"],
                "raw_current": wf["raw_current"],
                "preamp_time_ns": wf["preamp_time_ns"],
                "preamp_signal": wf["preamp_signal"],
                "type": wf["contact_type"],
                "offset": offset,
            }

    # -- Build comparison panels --
    # Panel 1: Collecting anode (raw current)
    # Panel 2: Collecting anode (preamp)
    # Panel 3: Neighbor anodes (raw current)
    # Panel 4: Cathode (raw current)
    # Panel 5: Cathode (preamp)

    def js_arr(arr):
        return json.dumps(arr)

    def make_trace(name, t, y, color, dash="solid", width=2):
        return (
            f"{{x:{js_arr(t)},y:{js_arr(y)},"
            f"type:'scatter',mode:'lines',name:'{name}',"
            f"line:{{color:'{color}',width:{width},dash:'{dash}'}}}}"
        )

    panels = []

    # Panel 1: Collecting anode - raw current
    traces_1 = []
    if "ssd_anode_center" in ssd_traces:
        d = ssd_traces["ssd_anode_center"]
        traces_1.append(make_trace("SSD Anode 3", d["raw_time_ns"], d["raw_current"], "#e74c3c"))
    if "ctsi_anode_center" in ctsi_traces:
        d = ctsi_traces["ctsi_anode_center"]
        traces_1.append(make_trace("CTSI Anode (center)", d["raw_time_ns"], d["raw_current"], "#3498db"))
    panels.append(("collect_raw", "Collecting Anode — Raw Current", "Time (ns)", "Induced current", traces_1))

    # Panel 2: Collecting anode - preamp
    traces_2 = []
    if "ssd_anode_center" in ssd_traces:
        d = ssd_traces["ssd_anode_center"]
        traces_2.append(make_trace("SSD Anode 3", d["preamp_time_ns"], d["preamp_signal"], "#e74c3c"))
    if "ctsi_anode_center" in ctsi_traces:
        d = ctsi_traces["ctsi_anode_center"]
        traces_2.append(make_trace("CTSI Anode (center)", d["preamp_time_ns"], d["preamp_signal"], "#3498db"))
    panels.append(("collect_preamp", "Collecting Anode — Preamp Output", "Time (ns)", "Preamp output", traces_2))

    # Panel 3: Neighbor anodes - raw current
    traces_3 = []
    ssd_colors = {"ssd_anode_-1": "#27ae60", "ssd_anode_+1": "#2ecc71",
                  "ssd_anode_-2": "#16a085", "ssd_anode_+2": "#1abc9c"}
    ctsi_colors = {"ctsi_anode_-1": "#8e44ad", "ctsi_anode_+1": "#9b59b6",
                   "ctsi_anode_-2": "#6c3483", "ctsi_anode_+2": "#a569bd"}
    for label, color in ssd_colors.items():
        if label in ssd_traces:
            d = ssd_traces[label]
            offset = label.split("_")[-1]
            traces_3.append(make_trace(f"SSD ({offset})", d["raw_time_ns"], d["raw_current"], color, width=1.5))
    for label, color in ctsi_colors.items():
        if label in ctsi_traces:
            d = ctsi_traces[label]
            offset = label.split("_")[-1]
            traces_3.append(make_trace(f"CTSI ({offset})", d["raw_time_ns"], d["raw_current"], color, "dash", 1.5))
    panels.append(("neighbor_raw", "Neighbor Anodes — Raw Current", "Time (ns)", "Induced current", traces_3))

    # Panel 4: Cathode - raw current
    traces_4 = []
    if "ssd_cathode" in ssd_traces:
        d = ssd_traces["ssd_cathode"]
        traces_4.append(make_trace("SSD Cathode 2", d["raw_time_ns"], d["raw_current"], "#e74c3c"))
    if "ssd_cathode_1" in ssd_traces:
        d = ssd_traces["ssd_cathode_1"]
        traces_4.append(make_trace("SSD Cathode 1", d["raw_time_ns"], d["raw_current"], "#c0392b", "dot"))
    for label, d in ctsi_traces.items():
        if d["type"] == "cathode":
            traces_4.append(make_trace(f"CTSI {label.replace('ctsi_','')}", d["raw_time_ns"], d["raw_current"], "#3498db"))
    panels.append(("cathode_raw", "Cathode — Raw Current", "Time (ns)", "Induced current", traces_4))

    # Panel 5: Cathode - preamp
    traces_5 = []
    if "ssd_cathode" in ssd_traces:
        d = ssd_traces["ssd_cathode"]
        traces_5.append(make_trace("SSD Cathode 2", d["preamp_time_ns"], d["preamp_signal"], "#e74c3c"))
    for label, d in ctsi_traces.items():
        if d["type"] == "cathode":
            traces_5.append(make_trace(f"CTSI {label.replace('ctsi_','')}", d["preamp_time_ns"], d["preamp_signal"], "#3498db"))
    panels.append(("cathode_preamp", "Cathode — Preamp Output", "Time (ns)", "Preamp output", traces_5))

    # -- Parameters table --
    ssd_pos = ssd_data.get("position_mm", {})
    ctsi_mu_e = ctsi_spec.get("MU_E", "?")
    ctsi_mu_h = ctsi_spec.get("MU_H", "?")
    # Format lifetimes from seconds to readable units
    try:
        ctsi_tau_e_s = float(ctsi_spec.get("TAU_E", "0"))
        ctsi_tau_e = f"{ctsi_tau_e_s*1e6:.0f} &mu;s"
    except ValueError:
        ctsi_tau_e = ctsi_spec.get("TAU_E", "?")
    try:
        ctsi_tau_h_s = float(ctsi_spec.get("TAU_H", "0"))
        ctsi_tau_h = f"{ctsi_tau_h_s*1e6:.1f} &mu;s"
    except ValueError:
        ctsi_tau_h = ctsi_spec.get("TAU_H", "?")

    # -- Generate HTML --
    panel_divs = []
    panel_js = []
    for div_id, title, xlabel, ylabel, traces in panels:
        panel_divs.append(f'<div class="pb"><div id="{div_id}" style="height:400px"></div></div>')
        trace_str = ",\n".join(traces)
        panel_js.append(f"""Plotly.newPlot('{div_id}',[{trace_str}],{{
  title:'{title}',
  xaxis:{{title:'{xlabel}'}},yaxis:{{title:'{ylabel}'}},
  margin:{{t:40,b:50,l:70,r:20}},hovermode:'x unified'
}},C);""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulator Comparison: SSD vs CTSI</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,"Segoe UI",Roboto,sans-serif;max-width:1200px;
       margin:0 auto;padding:20px;color:#1a1a1a;background:#f8f9fa}}
  h1{{font-size:1.5em;border-bottom:3px solid #2c3e50;padding-bottom:8px}}
  h2{{font-size:1.15em;margin-top:32px;color:#2c3e50}}
  .sub{{color:#666;margin-bottom:24px}}
  table{{border-collapse:collapse;width:100%;margin:8px 0 16px}}
  th,td{{text-align:left;padding:7px 12px;border-bottom:1px solid #dee2e6}}
  th{{background:#e9ecef;font-weight:600}}
  .val{{font-family:"SF Mono",Menlo,monospace;font-size:0.95em}}
  .g2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .pb{{background:#fff;border:1px solid #dee2e6;border-radius:6px;padding:8px;margin-bottom:16px}}
  .ssd{{color:#e74c3c;font-weight:600}}
  .ctsi{{color:#3498db;font-weight:600}}
  footer{{margin-top:40px;padding-top:12px;border-top:1px solid #dee2e6;font-size:0.85em;color:#888}}
  @media(max-width:800px){{.g2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<h1>Simulator Comparison: SSD vs CTSI</h1>
<p class="sub">{ssd_data.get('energy_keV', 662)} keV photoelectric interaction &bull;
<span class="ssd">SSD</span> = SolidStateDetectors.jl &bull;
<span class="ctsi">CTSI</span> = C++ Transport Simulator</p>

<h2>Simulation Parameters</h2>
<div class="g2">
<table>
<tr><th></th><th class="ssd">SSD</th><th class="ctsi">CTSI</th></tr>
<tr><td>Crystal</td><td class="val">40x40x5 mm CZT</td><td class="val">40x40x5 mm CZT</td></tr>
<tr><td>&mu;<sub>e</sub></td><td class="val">1000 cm&sup2;/Vs</td><td class="val">{ctsi_mu_e} cm&sup2;/Vs</td></tr>
<tr><td>&mu;<sub>h</sub></td><td class="val">50 cm&sup2;/Vs</td><td class="val">{ctsi_mu_h} cm&sup2;/Vs</td></tr>
<tr><td>&tau;<sub>e</sub></td><td class="val">10 &mu;s</td><td class="val">{ctsi_tau_e}</td></tr>
<tr><td>&tau;<sub>h</sub></td><td class="val">1 &mu;s</td><td class="val">{ctsi_tau_h}</td></tr>
<tr><td>Carriers</td><td class="val">{ssd_data.get('n_carriers', 50)}</td><td class="val">50</td></tr>
<tr><td>Time step</td><td class="val">{ssd_data.get('dt_ns', '?')} ns</td><td class="val">adaptive</td></tr>
<tr><td>Preamp &tau;</td><td class="val">140 &mu;s</td><td class="val">140 &mu;s</td></tr>
</table>
<table>
<tr><th></th><th class="ssd">SSD</th><th class="ctsi">CTSI</th></tr>
<tr><td>Anodes</td><td class="val">5 (100 &mu;m, 1mm pitch)</td><td class="val">{ctsi_spec.get('NUM_ANODES','?')} (100 &mu;m, 1mm pitch)</td></tr>
<tr><td>Cathodes</td><td class="val">2</td><td class="val">{ctsi_spec.get('NUM_CATHODES','?')}</td></tr>
<tr><td>Bias</td><td class="val">-600 V cathode</td><td class="val">{ctsi_spec.get('BIAS','?')} V/cm (600 V)</td></tr>
<tr><td>Steering</td><td class="val">-80 V</td><td class="val">N/A</td></tr>
<tr><td>Position</td>
    <td class="val">({ssd_pos.get('x','?')}, {ssd_pos.get('y','?')}, {ssd_pos.get('z','?')}) mm</td>
    <td class="val">center, mid-depth</td></tr>
<tr><td>Grid</td><td class="val">Adaptive (0.05V refine)</td><td class="val">0.01 cm uniform</td></tr>
</table>
</div>

<h2>Waveform Comparison</h2>
<div class="g2">
{chr(10).join(panel_divs[:2])}
</div>
{chr(10).join(panel_divs[2:3])}
<div class="g2">
{chr(10).join(panel_divs[3:])}
</div>

<footer>SolidStateDetectors.jl vs CTSI comparison</footer>

<script>
var C={{responsive:true,displaylogo:false}};
{chr(10).join(panel_js)}
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Wrote {output_path} ({len(html)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Compare SSD and CTSI waveforms")
    parser.add_argument("--ssd", required=True, help="Path to SSD waveform JSON")
    parser.add_argument("--ctsi", required=True, help="Path to CTSI interactiveOut.txt")
    parser.add_argument("--ctsi-root", required=True, help="Path to CTSI repo root")
    parser.add_argument("--output", default="output/comparison.html", help="Output HTML path")
    args = parser.parse_args()

    print("Loading SSD data …")
    ssd_data = parse_ssd(args.ssd)

    print("Loading CTSI data …")
    ctsi_result, ctsi_spec = parse_ctsi(args.ctsi, args.ctsi_root)

    print("Building comparison report …")
    build_html(ssd_data, ctsi_result, ctsi_spec, args.output)

if __name__ == "__main__":
    main()
