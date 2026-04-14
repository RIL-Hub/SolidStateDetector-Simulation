#!/usr/bin/env python3
"""
Compare waveforms from SolidStateDetectors.jl and CTSI simulators.
Generates an interactive HTML report with overlaid Plotly traces.

Usage:
    # Single-event comparison (default):
    python3 scripts/compare.py \
        --ssd output/ssd_waveforms.json \
        --ctsi ctsi/output/interactiveOut.txt \
        --ctsi-root ctsi \
        --output output/comparison.html

    # Z-scan comparison:
    python3 scripts/compare.py --mode zscan \
        --ssd output/ssd_zscan.json \
        --ctsi-dir ctsi/output/zscan \
        --ctsi-root ctsi \
        --output output/zscan_comparison.html
"""

import argparse
import json
import sys
import os

import numpy as np


# ── Z-scan coordinate mapping ──
# CTSI config has Event_z_pos_scale_factor=-1 and Event_z_pos_offset=0.5,
# so z_internal = -z_input + 0.5.  Near-cathode in CTSI is z_input=0.45
# (→ z_internal=0.05), not z_input=0.05.
Z_FROM_CATHODE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
CTSI_Z_CM = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

# Contacts to compare in z-scan: (SSD contact name, display label)
ZSCAN_ELECTRODES = [
    ("anode_3", "Anode (collecting)"),
    ("cathode_2", "Cathode"),
]

# Minimum signal amplitude to include in Pearson calculation.
# CTSI outputs in Coulombs (~1e-14), SSD in arbitrary units (~1e9).
# Use a very small threshold to avoid skipping valid CTSI signals.
SIGNAL_THRESHOLD = 1e-20


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


def extract_ctsi_raw(ctsi_result, ctsi_spec, contact_name):
    """Extract raw induced current for a specific contact from CTSI result.

    Returns (time_ns, signal) or (None, None) if not found.
    """
    n_anodes = int(ctsi_spec.get("NUM_ANODES", 39))
    center_anode = n_anodes // 2

    t_ns = ctsi_result.time_vec * 1e9

    if contact_name == "anode_3":
        for i, cid in enumerate(ctsi_result.an_collector_id):
            if cid == center_anode:
                sig = ctsi_result.an_vs_time[i]
                return t_ns[:len(sig)].copy(), sig.copy()
    elif contact_name == "cathode_2":
        # Find the collecting cathode closest to center
        # CTSI cathode IDs are 0-indexed; for 8 cathodes, center ≈ 3 or 4
        n_cathodes = int(ctsi_spec.get("NUM_CATHODES", 8))
        center_cathode = n_cathodes // 2
        for i, cid in enumerate(ctsi_result.ca_collector_id):
            if cid == center_cathode:
                sig = ctsi_result.ca_vs_time[i]
                return t_ns[:len(sig)].copy(), sig.copy()
        # Fallback: use first cathode
        if len(ctsi_result.ca_collector_id) > 0:
            sig = ctsi_result.ca_vs_time[0]
            return t_ns[:len(sig)].copy(), sig.copy()

    return None, None


def interpolate_to_common_grid(t1, s1, t2, s2, dt_ns=1.0):
    """Resample two waveforms onto a common time grid.

    Returns (t_common, s1_resampled, s2_resampled).
    """
    t_min = max(t1[0], t2[0])
    t_max = min(t1[-1], t2[-1])
    if t_max <= t_min:
        return None, None, None

    t_common = np.arange(t_min, t_max, dt_ns)
    s1_interp = np.interp(t_common, t1, s1)
    s2_interp = np.interp(t_common, t2, s2)
    return t_common, s1_interp, s2_interp


def pearson_r(a, b):
    """Pearson correlation coefficient between two arrays."""
    if len(a) < 2:
        return float("nan")
    r = np.corrcoef(a, b)[0, 1]
    return float(r)


def amplitude_ratio(a, b):
    """Ratio of max absolute amplitudes: max(|a|) / max(|b|)."""
    max_a = np.max(np.abs(a))
    max_b = np.max(np.abs(b))
    if max_b < SIGNAL_THRESHOLD:
        return float("nan")
    return float(max_a / max_b)


# ═══════════════════════════════════════════════════════════════════════
# Z-SCAN MODE
# ═══════════════════════════════════════════════════════════════════════

def run_zscan(args):
    """Run z-scan comparison: 9 depths × 2 electrodes."""
    print("Loading SSD z-scan data …")
    ssd_data = parse_ssd(args.ssd)
    ssd_zscan = ssd_data.get("zscan", {})

    # Load CTSI toolkit
    ctsi_root = args.ctsi_root
    sys.path.insert(0, os.path.abspath(ctsi_root))
    from tools.ctsi_toolkit.parser import parse_interactive_output

    # Load detector spec
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

    # Parse CTSI z-scan outputs (one interactiveOut per z position)
    ctsi_dir = args.ctsi_dir
    ctsi_results = {}
    for i, z_cm in enumerate(CTSI_Z_CM):
        z_key = f"{Z_FROM_CATHODE[i]}"
        # Try naming conventions: z_0.05.txt, interactiveOut_z0.05.txt, etc.
        candidates = [
            os.path.join(ctsi_dir, f"z_{z_cm:.2f}", "interactiveOut.txt"),
            os.path.join(ctsi_dir, f"interactiveOut_z{z_cm:.2f}.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"  Loading CTSI z={z_cm:.2f} cm from {path}")
                ctsi_results[z_key] = parse_interactive_output(path, mode="full")
                break
        else:
            print(f"  WARNING: No CTSI output for z={z_cm:.2f} cm")

    # ── Compute metrics and collect traces ──
    # metrics[z_key][electrode_label] = {pearson, amp_ratio}
    metrics = {}
    panels = []  # (div_id, title, ssd_t, ssd_s, ctsi_t, ctsi_s)

    for i, z_depth in enumerate(Z_FROM_CATHODE):
        z_key = str(z_depth)
        metrics[z_key] = {}
        ssd_wf = ssd_zscan.get(z_key, {})
        ctsi_res = ctsi_results.get(z_key)

        for contact_name, electrode_label in ZSCAN_ELECTRODES:
            # SSD raw induced current
            ssd_contact = ssd_wf.get(contact_name, {})
            ssd_t = ssd_contact.get("time_ns")
            ssd_s = ssd_contact.get("current")

            # CTSI raw induced current
            ctsi_t, ctsi_s = None, None
            if ctsi_res is not None:
                ctsi_t, ctsi_s = extract_ctsi_raw(ctsi_res, spec, contact_name)

            # Store panel data
            panel_id = f"z{z_key.replace('.', '_')}_{contact_name}"
            title = f"z={z_depth} mm — {electrode_label}"
            panels.append((panel_id, title, ssd_t, ssd_s, ctsi_t, ctsi_s))

            # Compute Pearson if both signals exist
            r_val = float("nan")
            amp_r = float("nan")
            if ssd_t is not None and ctsi_t is not None:
                ssd_arr = np.array(ssd_t), np.array(ssd_s)
                ctsi_arr = np.array(ctsi_t), np.array(ctsi_s)

                # Skip negligible signals
                if (np.max(np.abs(ssd_arr[1])) > SIGNAL_THRESHOLD and
                        np.max(np.abs(ctsi_arr[1])) > SIGNAL_THRESHOLD):
                    t_com, s1, s2 = interpolate_to_common_grid(
                        ssd_arr[0], ssd_arr[1], ctsi_arr[0], ctsi_arr[1])
                    if t_com is not None:
                        r_val = pearson_r(s1, s2)
                        amp_r = amplitude_ratio(s1, s2)

            metrics[z_key][electrode_label] = {
                "pearson": r_val,
                "amp_ratio": amp_r,
            }

    # ── Build HTML ──
    build_zscan_html(ssd_data, spec, metrics, panels, args.output)


def build_zscan_html(ssd_data, ctsi_spec, metrics, panels, output_path):
    """Generate z-scan comparison HTML."""

    def js_arr(arr):
        if arr is None:
            return "[]"
        return json.dumps(list(arr) if not isinstance(arr, list) else arr)

    def make_trace(name, t, y, color, dash="solid", width=2):
        return (
            f"{{x:{js_arr(t)},y:{js_arr(y)},"
            f"type:'scatter',mode:'lines',name:'{name}',"
            f"line:{{color:'{color}',width:{width},dash:'{dash}'}}}}"
        )

    # ── Summary table ──
    electrode_labels = [label for _, label in ZSCAN_ELECTRODES]
    all_pearsons = []

    table_rows = []
    for z_depth in Z_FROM_CATHODE:
        z_key = str(z_depth)
        cells = [f"<td class=\"val\">{z_depth}</td>"]
        for label in electrode_labels:
            m = metrics.get(z_key, {}).get(label, {})
            r = m.get("pearson", float("nan"))
            amp = m.get("amp_ratio", float("nan"))
            if np.isnan(r):
                cells.append('<td class="val">—</td>')
            else:
                pct = r * 100
                all_pearsons.append(r)
                if pct >= 90:
                    cls = "good"
                elif pct >= 70:
                    cls = "ok"
                else:
                    cls = "bad"
                amp_str = f" ({amp:.2f}x)" if not np.isnan(amp) else ""
                cells.append(f'<td class="val {cls}">{pct:.1f}%{amp_str}</td>')
        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    overall = np.mean(all_pearsons) * 100 if all_pearsons else float("nan")
    if np.isnan(overall):
        overall_str = "N/A"
        overall_cls = ""
    elif overall >= 90:
        overall_str = f"{overall:.1f}%"
        overall_cls = "good"
    elif overall >= 70:
        overall_str = f"{overall:.1f}%"
        overall_cls = "ok"
    else:
        overall_str = f"{overall:.1f}%"
        overall_cls = "bad"

    header_cells = "".join(f"<th>{l}</th>" for l in electrode_labels)
    summary_table = f"""
<table>
<tr><th>Depth (mm)</th>{header_cells}</tr>
{chr(10).join(table_rows)}
<tr style="border-top:2px solid #2c3e50;font-weight:600">
  <td>Overall mean</td>
  <td class="val {overall_cls}" colspan="{len(electrode_labels)}">{overall_str}</td>
</tr>
</table>"""

    # ── Plot panels ──
    # Normalize both signals to peak amplitude so they're visually comparable.
    # SSD outputs ~1e9 (arb. units), CTSI outputs ~1e-14 (Coulombs).
    def normalize(signal, flip=False):
        """Normalize signal to [-1, 1] by dividing by peak absolute value."""
        if signal is None:
            return None
        arr = np.array(signal) if not isinstance(signal, np.ndarray) else signal
        peak = np.max(np.abs(arr))
        if peak < SIGNAL_THRESHOLD:
            return arr
        normed = arr / peak
        if flip:
            normed = -normed
        return normed.tolist()

    panel_divs = []
    panel_js = []
    for panel_id, title, ssd_t, ssd_s, ctsi_t, ctsi_s in panels:
        traces = []
        if ssd_t is not None:
            # Flip SSD polarity so signals align with CTSI convention
            traces.append(make_trace("SSD", ssd_t, normalize(ssd_s, flip=True), "#e74c3c"))
        if ctsi_t is not None:
            traces.append(make_trace("CTSI", ctsi_t, normalize(ctsi_s), "#3498db"))

        panel_divs.append(
            f'<div class="pb"><div id="{panel_id}" style="height:350px"></div></div>')
        trace_str = ",\n".join(traces)
        panel_js.append(f"""Plotly.newPlot('{panel_id}',[{trace_str}],{{
  title:'{title}',
  xaxis:{{title:'Time (ns)'}},yaxis:{{title:'Normalized induced current'}},
  margin:{{t:40,b:50,l:70,r:20}},hovermode:'x unified'
}},C);""")

    # Group panels into 2-column grid (anode + cathode per row)
    panel_grid = []
    for i in range(0, len(panel_divs), 2):
        pair = panel_divs[i:i + 2]
        panel_grid.append(f'<div class="g2">{chr(10).join(pair)}</div>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Z-Scan Comparison: SSD vs CTSI</title>
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
  .good{{background:#d4edda;color:#155724}}
  .ok{{background:#fff3cd;color:#856404}}
  .bad{{background:#f8d7da;color:#721c24}}
  footer{{margin-top:40px;padding-top:12px;border-top:1px solid #dee2e6;font-size:0.85em;color:#888}}
  @media(max-width:800px){{.g2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<h1>Z-Scan Comparison: SSD vs CTSI</h1>
<p class="sub">662 keV photoelectric &bull; 9 depths (0.5&ndash;4.5 mm from cathode) &bull;
<span class="ssd">SSD</span> = SolidStateDetectors.jl &bull;
<span class="ctsi">CTSI</span> = C++ Transport Simulator</p>

<h2>Pearson Correlation Summary</h2>
<p>Shape similarity of raw induced current waveforms. Amplitude ratio (SSD/CTSI) shown in parentheses.</p>
{summary_table}

<h2>Waveform Overlays</h2>
{chr(10).join(panel_grid)}

<footer>SolidStateDetectors.jl vs CTSI z-scan comparison</footer>

<script>
var C={{responsive:true,displaylogo:false}};
{chr(10).join(panel_js)}
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Wrote {output_path} ({len(html)} bytes)")


# ═══════════════════════════════════════════════════════════════════════
# SINGLE-EVENT MODE (original)
# ═══════════════════════════════════════════════════════════════════════

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
    parser.add_argument("--mode", default="single", choices=["single", "zscan"],
                        help="Comparison mode: single event or z-scan")
    parser.add_argument("--ssd", required=True, help="Path to SSD waveform JSON")
    parser.add_argument("--ctsi", help="Path to CTSI interactiveOut.txt (single mode)")
    parser.add_argument("--ctsi-dir", help="Path to CTSI z-scan output directory (zscan mode)")
    parser.add_argument("--ctsi-root", required=True, help="Path to CTSI repo root")
    parser.add_argument("--output", default="output/comparison.html", help="Output HTML path")
    args = parser.parse_args()

    if args.mode == "zscan":
        if not args.ctsi_dir:
            parser.error("--ctsi-dir is required for zscan mode")
        run_zscan(args)
    else:
        if not args.ctsi:
            parser.error("--ctsi is required for single mode")
        print("Loading SSD data …")
        ssd_data = parse_ssd(args.ssd)

        print("Loading CTSI data …")
        ctsi_result, ctsi_spec = parse_ctsi(args.ctsi, args.ctsi_root)

        print("Building comparison report …")
        build_html(ssd_data, ctsi_result, ctsi_spec, args.output)


if __name__ == "__main__":
    main()
