#!/usr/bin/env python3
"""
Build sanitized datasets and paper-matched TikZ/Linux-Libertine SVG figures
for Not-Just-Speedup.

Follows the figure rules:
- One committed palette module (scripts/palette.py) matching the paper's mint-green theme.
- Color follows the entity across every figure.
- Data ink only: axes, series, ticks, and legend -- no titles, subtitles, or notes painted
  inside the image.
- Numbers read strictly from committed datasets; rendered with pdflatex + Libertine + TikZ
  and converted to SVG via pdftocairo, matching the paper's own figures.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as PAL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
ARTIFACT = os.path.join(WS, "Agent_Bench_Artifact")
PAPER = os.path.join(WS, "icpe27")

DATA_DIR = os.path.join(ROOT, "data")
ASSET_DIR = os.path.join(ROOT, "docs", "assets")

MODEL_DISPLAY = {
    "fable-5-2-claude-code": ("Fable 5.2", "Claude Code", "closed"),
    "fable-claude-code": ("Fable 5.1", "Claude Code", "closed"),
    "opus-5-5-claude-code": ("Opus 5.5", "Claude Code", "closed"),
    "sonnet-5-5-claude-code": ("Sonnet 5.5", "Claude Code", "closed"),
    "haiku-5-5-claude-code": ("Haiku 5.5", "Claude Code", "closed"),
    "haiku-4-5-claude-code": ("Haiku 4.5", "Claude Code", "closed"),
    "gemini-4-neon-xhigh-antigravity": ("Gemini 4 Neon", "Antigravity", "closed"),
    "gemini-4-carbon-high-antigravity": ("Gemini 4 Carbon", "Antigravity", "closed"),
    "gemini-4-argon-high-antigravity": ("Gemini 4 Argon", "Antigravity", "closed"),
    "astra-6-2-codex-cli": ("GPT-6.2 Astra", "Codex", "closed"),
    "astra-codex-cli": ("GPT-6 Astra", "Codex", "closed"),
    "sol-6-2-codex-cli": ("GPT-6.2 Sol", "Codex", "closed"),
    "sol-6-1-codex-cli": ("GPT-6.1 Sol", "Codex", "closed"),
    "sol-codex-cli": ("GPT-6 Sol", "Codex", "closed"),
    "terra-codex-cli": ("GPT-5.6 Terra", "Codex", "closed"),
    "luna-codex-cli": ("GPT-6 Luna", "Codex", "closed"),
    "kimi-opencode": ("Kimi K3", "OpenCode", "open"),
    "kimi-k3-1-pro-opencode": ("Kimi K3.1 Pro", "OpenCode", "open"),
    "glm-5-2-opencode": ("GLM-5.2", "OpenCode", "open"),
    "glm-5-3-flash-opencode": ("GLM-5.3-Flash", "OpenCode", "open"),
    "deepseek-flash-opencode": ("DeepSeek-V4-Flash", "OpenCode", "open"),
    "deepseek-flash-reasoning-opencode": ("DeepSeek-V4-Flash-R", "OpenCode", "open"),
    "qwen3-8-27b-opencode": ("Qwen3.8-27B", "OpenCode", "open"),
    "qwen3-6-35b-a3b-opencode": ("Qwen3.6-35B-A3B", "OpenCode", "open"),
    "mimo-v2-6-pro-opencode": ("MiMo-V2.6-Pro", "OpenCode", "open"),
    "seed-oss-36b-opencode": ("Seed-OSS-36B", "OpenCode", "open"),
    "nemotron-3-ultra-opencode": ("Nemotron 3 Ultra", "OpenCode", "open"),
    "nemotron-3-5-super-opencode": ("Nemotron 3.5 Super", "OpenCode", "open"),
}


def sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r";?\s*(user|owner)\s+2026-\d{2}-\d{2}:.*", "", s, flags=re.I)
    s = re.sub(r"\((user|owner)\s+2026-\d{2}-\d{2}:[^)]*\)", "", s, flags=re.I)
    s = " ".join(s.split())
    return s.strip(" ;")


def classify_root_cause(what: str, defect: str, test_note: str) -> str:
    blob = f"{what} {defect} {test_note}".lower()
    if "launch grid over the limit" in blob or "very large dimension" in blob or "1.4 billion rows" in blob:
        return "Extreme-shape grid or allocation limit"
    if "zero launch grid" in blob:
        return "Zero launch grid on small sizes"
    if "second pb_compute" in blob or "repeated pb_compute" in blob or "wrong step count" in blob:
        return "Repeated-call state corruption"
    if "race" in blob or "cp.async" in blob or "shfl_" in blob or "warp_max" in blob or "co-resident" in blob or "sharing the gpu" in blob or "another kernel holding an sm" in blob:
        return "Sync, warp/PTX, or concurrency race"
    if "out-of-bounds" in blob or "uninitialised" in blob or "unwritten buffer" in blob or "beta = 0" in blob:
        return "Uninitialized memory or OOB read"
    if "hang" in blob or "deadlock" in blob or "call counter" in blob:
        return "Deadlock or call-counter hang"
    if "illegal address" in blob or "allocation failure" in blob or "failed optional allocation" in blob or "refuses an allowed size" in blob:
        return "Crash, illegal address, or OOM"
    if "padded" in blob or "aliases" in blob or "grid-stride" in blob or "wave_kernel" in blob or "fewer than 32" in blob or "fixed scale" in blob or "slower" in blob or "714" in blob:
        return "Tiling, padding, BLAS alias, or fallback"
    return "Other defect"


def render_tikz_svg(body_lines: list[str], out_svg: str) -> None:
    tmp = tempfile.mkdtemp(prefix="njs-tikz-")
    try:
        tex_path = os.path.join(tmp, "fig.tex")
        pdf_path = os.path.join(tmp, "fig.pdf")
        with open(tex_path, "w") as f:
            f.write(PAL.TIKZ_DOC.replace("BODY", "\n".join(body_lines)))
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"pdflatex failed for {out_svg}:\n{res.stdout[-2000:]}")
        res2 = subprocess.run(
            ["pdftocairo", "-svg", pdf_path, out_svg],
            capture_output=True,
            text=True,
        )
        if res2.returncode != 0:
            raise RuntimeError(f"pdftocairo failed for {out_svg}:\n{res2.stderr}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_attainment() -> tuple[list[dict], dict[str, tuple[str, str, str]]]:
    src = os.path.join(PAPER, "figures", "attainment", "attainment.csv")
    rows = list(csv.DictReader(open(src)))
    case_map = {}
    for r in rows:  # failed rows carry no binding; keep the first row that has one
        if r["binding"] or r["case"] not in case_map:
            if not (r["case"] in case_map and case_map[r["case"]][2]):
                case_map[r["case"]] = (r["kernel"], r["category"], r["binding"])

    keep_cols = [
        "case", "kernel", "category", "agent", "model", "framework", "weight_class",
        "state", "time_ms", "compute_share", "hbm_share",
        "attainment", "binding", "compute_peak", "libraries", "approach",
        "algorithm_changed", "tensor_cores", "dpx", "emulation",
        "precision_kinds", "temporal_blocking", "bug_marked", "bug_verification",
    ]
    out_path = os.path.join(DATA_DIR, "attainment.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep_cols)
        w.writeheader()
        for r in rows:
            disp, fw, wc = MODEL_DISPLAY[r["agent"]]
            rec = {k: r.get(k, "") for k in keep_cols}
            rec["model"] = disp
            rec["framework"] = fw
            rec["weight_class"] = wc
            rec["bug_verification"] = sanitize_text(rec.get("bug_verification", ""))
            w.writerow(rec)
    return rows, case_map


def copy_clean_csvs() -> None:
    for rel, dst_name in [
        ("figures/behavior/behavior.csv", "behavior.csv"),
        ("figures/ablation/profiler-pairs.csv", "profiler-pairs.csv"),
        ("figures/ablation/no-gpu.csv", "no-gpu.csv"),
    ]:
        src = os.path.join(PAPER, rel)
        dst = os.path.join(DATA_DIR, dst_name)
        if dst_name != "behavior.csv":
            shutil.copyfile(src, dst)
            continue
        # The analyzer column holds internal report paths that name the host site; drop it.
        rows = list(csv.DictReader(open(src)))
        cols = [c for c in rows[0].keys() if c != "analyzer"]
        with open(dst, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def build_bugs(att_rows: list[dict], case_map: dict[str, tuple[str, str, str]]) -> tuple[list[dict], dict]:
    cl_map = json.load(open(os.path.join(ARTIFACT, "notes", "2026-11-19-round-020", "bug-classes.json")))["claims"]
    # Kimi-K3.1-Pro (round 022): the class is recorded in notes/2026-11-23-round-022/README.md
    # ("Class: a computation error"); the owner counted it as a failure on 2026-11-24.
    cl_map.setdefault("kimi-k3-1-pro-opencode.case-29 c1", {
        "class": "computation", "test": "fails",
        "what": "wrong values at the left tile edge of a later wave: blocks read halos that earlier blocks already overwrote (race between thread blocks)",
        "test_note": "",
    })
    claims_meta = {}
    for lst in ("bugv-list-20261120-all-66.json", "bugv-list-20261123-round022.json"):
        for entry in json.load(open(os.path.join(ARTIFACT, "notes", "2026-10-02-round-011b-second-operator", lst))):
            for c in entry["claims"]:
                claims_meta[(entry["name"], c["id"])] = c

    l1_pass = collections.Counter()
    final_pass = collections.Counter()
    for r in att_rows:
        ag = r["agent"]
        if r["state"] == "pass":
            final_pass[ag] += 1
            l1_pass[ag] += 1
        elif (r.get("reason") or "").lower().startswith("bug verification:"):
            l1_pass[ag] += 1

    records = []
    fpaths = []
    for d in ("2026-11-20-bugv-v1-claude-default", "2026-11-23-bugv-v1-round022-claude-default"):
        fpaths += sorted(glob.glob(os.path.join(ARTIFACT, "analysis", d, "*", "*", "analysis", "out", "report.json")))
    for fpath in fpaths:
        trial_name = fpath.split("/analysis/", 1)[1].split("/")[1]
        agent_raw, case_raw = trial_name.split(".case-")
        case_id = "case-" + case_raw.split("@")[0]
        kernel, category, binding = case_map[case_id]
        disp, fw, wc = MODEL_DISPLAY[agent_raw]
        rep = json.load(open(fpath))
        for c in rep["goals"]["B1"]["claims"]:
            cid = c["id"]
            key = f"{trial_name} {cid}"
            cl = cl_map.get(key, {})
            cm = claims_meta.get((trial_name, cid), {})
            verdict = c["verdict"]
            bclass = cl.get("class", "")
            btest = cl.get("test", "")
            what = sanitize_text(cl.get("what", ""))
            test_note = sanitize_text(cl.get("test_note", ""))
            defect = sanitize_text(cm.get("defect", ""))

            if verdict != "reproduced":
                outcome = verdict
                root_cause = ""
            elif bclass == "waived":
                outcome = "waived"
                root_cause = "Waived (sign of zero in unwritten border)"
            elif bclass == "computation" and btest != "passes":
                outcome = "fail_computation"
                root_cause = classify_root_cause(what, defect, test_note)
            elif bclass == "engineering" and btest == "fails":
                outcome = "fail_engineering"
                root_cause = classify_root_cause(what, defect, test_note)
            elif bclass == "computation" and btest == "passes":
                outcome = "kept_computation_borderline"
                root_cause = classify_root_cause(what, defect, test_note)
            elif bclass == "engineering" and btest == "passes":
                outcome = "kept_engineering_borderline"
                root_cause = classify_root_cause(what, defect, test_note)
            else:
                raise ValueError(f"Unexpected bug state: {key} {verdict} {bclass} {btest}")

            records.append({
                "agent": agent_raw,
                "model": disp,
                "framework": fw,
                "weight_class": wc,
                "case": case_id,
                "kernel": kernel,
                "kernel_category": category,
                "binding": binding,
                "claim_id": cid,
                "verdict": verdict,
                "bug_class": bclass,
                "scoring_outcome": outcome,
                "root_cause": root_cause,
                "summary": what if what else defect[:220],
                "borderline_note": test_note,
            })

    out_csv = os.path.join(DATA_DIR, "bugs.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    outcomes = collections.Counter(r["scoring_outcome"] for r in records)
    counted = [r for r in records if r["verdict"] == "reproduced" and r["scoring_outcome"] != "waived"]
    by_root = collections.defaultdict(lambda: {"fail": 0, "kept": 0})
    for r in counted:
        if r["scoring_outcome"].startswith("fail_"):
            by_root[r["root_cause"]]["fail"] += 1
        else:
            by_root[r["root_cause"]]["kept"] += 1

    by_model = {}
    for ag, (disp, fw, wc) in MODEL_DISPLAY.items():
        sub = [r for r in records if r["agent"] == ag]
        fc = sum(1 for r in sub if r["scoring_outcome"] == "fail_computation")
        fe = sum(1 for r in sub if r["scoring_outcome"] == "fail_engineering")
        kc = sum(1 for r in sub if r["scoring_outcome"] == "kept_computation_borderline")
        ke = sum(1 for r in sub if r["scoring_outcome"] == "kept_engineering_borderline")
        nr = sum(1 for r in sub if r["scoring_outcome"] in ("not-reproduced", "not-a-bug", "cannot-test"))
        failed_solutions = len({r["case"] for r in sub if r["scoring_outcome"].startswith("fail_")})
        kept_solutions = len({r["case"] for r in sub if r["scoring_outcome"].startswith("kept_")})
        by_model[ag] = {
            "model": disp,
            "framework": fw,
            "weight_class": wc,
            "l1_pass": l1_pass[ag],
            "final_pass": final_pass[ag],
            "claims_total": len(sub),
            "fail_computation": fc,
            "fail_engineering": fe,
            "kept_computation": kc,
            "kept_engineering": ke,
            "unconfirmed_or_rejected": nr,
            "failed_solutions": failed_solutions,
            "kept_solutions": kept_solutions,
            "fail_rate_pct": round(100.0 * failed_solutions / l1_pass[ag], 1) if l1_pass[ag] else 0.0,
        }

    closed_l1 = sum(v["l1_pass"] for v in by_model.values() if v["weight_class"] == "closed")
    open_l1 = sum(v["l1_pass"] for v in by_model.values() if v["weight_class"] == "open")
    closed_fail = sum(v["failed_solutions"] for v in by_model.values() if v["weight_class"] == "closed")
    open_fail = sum(v["failed_solutions"] for v in by_model.values() if v["weight_class"] == "open")

    summary = {
        "total_trials_examined": len({(r["agent"], r["case"]) for r in records}),
        "total_claims": len(records),
        "reproduced_total": sum(1 for r in records if r["verdict"] == "reproduced"),
        "reproduced_counted": len(counted),
        "outcomes": dict(outcomes),
        "closed_vs_open": {
            "closed_l1_passes": closed_l1,
            "closed_failed_solutions": closed_fail,
            "closed_fail_rate_pct": round(100.0 * closed_fail / closed_l1, 2),
            "open_l1_passes": open_l1,
            "open_failed_solutions": open_fail,
            "open_fail_rate_pct": round(100.0 * open_fail / open_l1, 2),
        },
        "root_causes": dict(sorted(by_root.items(), key=lambda kv: -(kv[1]["fail"] + kv[1]["kept"]))),
        "by_kernel": dict(collections.Counter(r["kernel"] for r in counted).most_common()),
        "by_model": by_model,
    }
    with open(os.path.join(DATA_DIR, "bugs_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return records, summary


def draw_bugs_funnel(summary: dict) -> None:
    """Data-ink-only proportional stacked bars in TikZ + Linux Libertine."""
    oc = summary["outcomes"]
    n_tot = summary["total_claims"]
    n_rep = summary["reproduced_total"]
    X0, W, BH = 78.0, 310.0, 14.0
    P: list[str] = []

    # Row 1 (top): Bug Verifier execution outcome (n = 76)
    y1 = 44.0
    P.append(rf"\node[anchor=east, inner sep=0pt] at ({X0 - 6:.1f},{y1 + BH/2:.1f}) {{Bug Verifier ($n={n_tot}$)}};")
    s1 = [
        ("reproduced", n_rep, PAL.BUG_OUTCOME_COLORS["reproduced"], True),
        ("not-reproduced", oc["not-reproduced"], PAL.BUG_OUTCOME_COLORS["not-reproduced"], False),
        ("not-a-bug", oc["not-a-bug"], PAL.BUG_OUTCOME_COLORS["not-a-bug"], False),
        ("cannot-test", oc["cannot-test"], PAL.BUG_OUTCOME_COLORS["cannot-test"], False),
    ]
    cx = X0
    for key, val, col, light_txt in s1:
        sw = W * val / n_tot
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.3pt] ({cx:.2f},{y1:.2f}) rectangle ({cx + sw:.2f},{y1 + BH:.2f});")
        if sw >= 22:
            tc = "white" if light_txt else "black"
            P.append(rf"\node[inner sep=0pt, text={tc}] at ({cx + sw/2:.2f},{y1 + BH/2:.2f}) {{{val}}};")
        cx += sw

    # Legend 1 directly above bar 1
    leg1 = [
        (PAL.BUG_OUTCOME_COLORS["reproduced"], f"Reproduced ({n_rep})"),
        (PAL.BUG_OUTCOME_COLORS["not-reproduced"], f"Not reproduced ({oc['not-reproduced']})"),
        (PAL.BUG_OUTCOME_COLORS["not-a-bug"], f"Not a bug ({oc['not-a-bug']})"),
        (PAL.BUG_OUTCOME_COLORS["cannot-test"], f"Cannot test ({oc['cannot-test']})"),
    ]
    lx, ly = X0, y1 + BH + 10.0
    for col, lbl in leg1:
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.3pt] ({lx:.1f},{ly - 2.5:.1f}) rectangle ({lx + 5:.1f},{ly + 2.5:.1f});")
        P.append(rf"\node[anchor=west, inner sep=0pt] at ({lx + 7.5:.1f},{ly:.1f}) {{{PAL.tex_escape(lbl)}}};")
        lx += 78.0 if "Not reproduced" in lbl else 73.0

    # Row 2 (bottom): Scoring outcome of reproduced claims (n = 55)
    y2 = 10.0
    P.append(rf"\node[anchor=east, inner sep=0pt] at ({X0 - 6:.1f},{y2 + BH/2:.1f}) {{Reproduced ($n={n_rep}$)}};")
    s2 = [
        ("fail_computation", oc["fail_computation"], PAL.BUG_OUTCOME_COLORS["fail_computation"], True),
        ("fail_engineering", oc["fail_engineering"], PAL.BUG_OUTCOME_COLORS["fail_engineering"], False),
        ("kept_computation_borderline", oc["kept_computation_borderline"], PAL.BUG_OUTCOME_COLORS["kept_computation_borderline"], False),
        ("kept_engineering_borderline", oc["kept_engineering_borderline"], PAL.BUG_OUTCOME_COLORS["kept_engineering_borderline"], True),
        ("waived", oc["waived"], PAL.BUG_OUTCOME_COLORS["waived"], False),
    ]
    cx = X0
    for key, val, col, light_txt in s2:
        sw = W * val / n_rep
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.3pt] ({cx:.2f},{y2:.2f}) rectangle ({cx + sw:.2f},{y2 + BH:.2f});")
        if sw >= 16:
            tc = "white" if light_txt else "black"
            P.append(rf"\node[inner sep=0pt, text={tc}] at ({cx + sw/2:.2f},{y2 + BH/2:.2f}) {{{val}}};")
        cx += sw

    # Legend 2 directly below bar 2 (two tidy rows)
    leg2 = [
        (PAL.BUG_OUTCOME_COLORS["fail_computation"], f"Fail: computation ({oc['fail_computation']})"),
        (PAL.BUG_OUTCOME_COLORS["fail_engineering"], f"Fail: engineering ({oc['fail_engineering']})"),
        (PAL.BUG_OUTCOME_COLORS["kept_computation_borderline"], f"Kept: borderline comp. ({oc['kept_computation_borderline']})"),
        (PAL.BUG_OUTCOME_COLORS["kept_engineering_borderline"], f"Kept: borderline eng. ({oc['kept_engineering_borderline']})"),
        (PAL.BUG_OUTCOME_COLORS["waived"], f"Waived ({oc['waived']})"),
    ]
    for i, (col, lbl) in enumerate(leg2):
        row = i // 3
        col_i = i % 3
        lx = X0 + col_i * 105.0
        ly = y2 - 11.0 - row * 10.0
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.3pt] ({lx:.1f},{ly - 2.5:.1f}) rectangle ({lx + 5:.1f},{ly + 2.5:.1f});")
        P.append(rf"\node[anchor=west, inner sep=0pt] at ({lx + 7.5:.1f},{ly:.1f}) {{{PAL.tex_escape(lbl)}}};")

    render_tikz_svg(P, os.path.join(ASSET_DIR, "bugs_funnel.svg"))


def draw_bugs_by_model(summary: dict) -> None:
    """Horizontal stacked bars per model in TikZ + Linux Libertine, data ink only."""
    models = [v for v in summary["by_model"].values()
              if (v["fail_computation"] + v["fail_engineering"] + v["kept_computation"] + v["kept_engineering"]) > 0]
    closed = sorted([m for m in models if m["weight_class"] == "closed"],
                    key=lambda m: (-(m["fail_computation"] + m["fail_engineering"]), -(m["kept_computation"] + m["kept_engineering"]), m["model"]))
    opened = sorted([m for m in models if m["weight_class"] == "open"],
                    key=lambda m: (-(m["fail_computation"] + m["fail_engineering"]), -(m["kept_computation"] + m["kept_engineering"]), m["model"]))

    cvo = summary["closed_vs_open"]
    items = [("SEP", f"Closed models ({cvo['closed_failed_solutions']}/{cvo['closed_l1_passes']} = {cvo['closed_fail_rate_pct']:.1f}%)")] + \
            [("M", m) for m in closed] + \
            [("SEP", f"Open-weight models ({cvo['open_failed_solutions']}/{cvo['open_l1_passes']} = {cvo['open_fail_rate_pct']:.1f}%)")] + \
            [("M", m) for m in opened]

    X0, W, MAXV = 96.0, 210.0, 7
    RH = 11.5
    N = len(items)
    TOP = N * RH
    P: list[str] = []

    # Legend at top
    leg = [
        (PAL.BUG_OUTCOME_COLORS["fail_computation"], "Fail: computation"),
        (PAL.BUG_OUTCOME_COLORS["fail_engineering"], "Fail: engineering"),
        (PAL.BUG_OUTCOME_COLORS["kept_computation_borderline"], "Kept: borderline comp."),
        (PAL.BUG_OUTCOME_COLORS["kept_engineering_borderline"], "Kept: borderline eng."),
    ]
    lx = 6.0
    ly = TOP + 16.0
    for col, lbl in leg:
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.3pt] ({lx:.1f},{ly - 2.5:.1f}) rectangle ({lx + 5:.1f},{ly + 2.5:.1f});")
        P.append(rf"\node[anchor=west, inner sep=0pt] at ({lx + 7.5:.1f},{ly:.1f}) {{{PAL.tex_escape(lbl)}}};")
        lx += 94.0

    # Column header for right rate column
    RX = X0 + W + 14.0
    P.append(rf"\node[anchor=west, inner sep=0pt] at ({RX:.1f},{TOP + 4.5:.1f}) {{\textbf{{Fail / Verifier pass}}}};" )

    # Vertical grid lines
    for t in range(0, MAXV + 1):
        gx = X0 + W * t / MAXV
        P.append(rf"\draw[line width=0.2pt, dash pattern=on 1pt off 1.5pt, color=black!30] ({gx:.2f},0) -- ({gx:.2f},{TOP:.2f});")
        P.append(rf"\node[anchor=north, inner sep=0pt] at ({gx:.2f},-3) {{{t}}};")
    P.append(rf"\draw[line width=0.3pt] ({X0:.2f},0) -- ({X0 + W:.2f},0);")
    P.append(rf"\node[anchor=north, inner sep=0pt] at ({X0 + W/2:.2f},-13) {{Reproduced bug claims}};")

    # Rows top-down
    for idx, (kind, payload) in enumerate(items):
        ym = TOP - (idx + 0.5) * RH
        if kind == "SEP":
            if idx > 0:
                P.append(rf"\draw[line width=0.35pt] (6,{ym + RH/2 - 0.5:.2f}) -- ({RX + 68:.2f},{ym + RH/2 - 0.5:.2f});")
            P.append(rf"\node[anchor=west, inner sep=1.2pt, fill=white] at (6,{ym:.2f}) {{\textit{{{PAL.tex_escape(payload)}}}}};")
        else:
            m = payload
            P.append(rf"\node[anchor=east, inner sep=0pt] at ({X0 - 5:.2f},{ym:.2f}) {{{PAL.tex_escape(m['model'])}}};")
            cx = X0
            segs = [
                (m["fail_computation"], PAL.BUG_OUTCOME_COLORS["fail_computation"], True),
                (m["fail_engineering"], PAL.BUG_OUTCOME_COLORS["fail_engineering"], False),
                (m["kept_computation"], PAL.BUG_OUTCOME_COLORS["kept_computation_borderline"], False),
                (m["kept_engineering"], PAL.BUG_OUTCOME_COLORS["kept_engineering_borderline"], True),
            ]
            bh = 7.4
            for val, col, light_txt in segs:
                if val <= 0:
                    continue
                sw = W * val / MAXV
                P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.25pt] ({cx:.2f},{ym - bh/2:.2f}) rectangle ({cx + sw:.2f},{ym + bh/2:.2f});")
                tc = "white" if light_txt else "black"
                P.append(rf"\node[inner sep=0pt, text={tc}, font=\fontsize{{6}}{{7}}\selectfont] at ({cx + sw/2:.2f},{ym:.2f}) {{{val}}};")
                cx += sw
            rate_str = f"{m['failed_solutions']}/{m['l1_pass']} ({m['fail_rate_pct']:.1f}%)"
            P.append(rf"\node[anchor=west, inner sep=0pt] at ({RX:.2f},{ym:.2f}) {{{PAL.tex_escape(rate_str)}}};")

    render_tikz_svg(P, os.path.join(ASSET_DIR, "bugs_by_model.svg"))


def draw_bugs_taxonomy(summary: dict) -> None:
    """Two-panel TikZ + Linux Libertine chart: (a) root causes, (b) top 12 kernels."""
    rc_items = list(summary["root_causes"].items())
    n_fail = sum(v["fail"] for _, v in rc_items)
    n_kept = sum(v["kept"] for _, v in rc_items)
    k_items = list(summary["by_kernel"].items())[:12]

    P: list[str] = []
    H = 148.0

    # Left panel: Root causes
    LX0, LW = 132.0, 96.0
    max_rc = max(v["fail"] + v["kept"] for _, v in rc_items)
    rh_l = H / len(rc_items)
    P.append(rf"\node[anchor=west, inner sep=0pt] at (0,{H + 10:.1f}) {{\textbf{{(a) Root cause}}}};" )
    for t in [0, 4, 8, 12, 16]:
        if t > max_rc:
            continue
        gx = LX0 + LW * t / max_rc
        P.append(rf"\draw[line width=0.2pt, dash pattern=on 1pt off 1.5pt, color=black!30] ({gx:.2f},0) -- ({gx:.2f},{H:.2f});")
        P.append(rf"\node[anchor=north, inner sep=0pt] at ({gx:.2f},-3) {{{t}}};")
    P.append(rf"\draw[line width=0.3pt] ({LX0:.2f},0) -- ({LX0 + LW:.2f},0);")
    P.append(rf"\node[anchor=north, inner sep=0pt] at ({LX0 + LW/2:.2f},-13) {{Count}};")

    for i, (name, cnts) in enumerate(rc_items):
        ym = H - (i + 0.5) * rh_l
        P.append(rf"\node[anchor=east, inner sep=0pt] at ({LX0 - 4:.2f},{ym:.2f}) {{{PAL.tex_escape(name)}}};")
        cx = LX0
        bh = 8.4
        for key, col, light_txt in [("fail", PAL.AMBER, True), ("kept", PAL.THEME_MID, False)]:
            v = cnts[key]
            if v <= 0:
                continue
            sw = LW * v / max_rc
            P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.25pt] ({cx:.2f},{ym - bh/2:.2f}) rectangle ({cx + sw:.2f},{ym + bh/2:.2f});")
            if sw >= 8:
                tc = "white" if light_txt else "black"
                P.append(rf"\node[inner sep=0pt, text={tc}, font=\fontsize{{6}}{{7}}\selectfont] at ({cx + sw/2:.2f},{ym:.2f}) {{{v}}};")
            cx += sw
        tot = cnts["fail"] + cnts["kept"]
        P.append(rf"\node[anchor=west, inner sep=0pt] at ({cx + 3:.2f},{ym:.2f}) {{{tot}}};")

    # Compact legend under panel (a)
    for j, (col, lbl) in enumerate([(PAL.AMBER, f"Fails scoring ({n_fail})"), (PAL.THEME_MID, f"Kept borderline ({n_kept})")]):
        lx = 6.0 + j * 95.0
        ly = -26.0
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.3pt] ({lx:.1f},{ly - 2.5:.1f}) rectangle ({lx + 5:.1f},{ly + 2.5:.1f});")
        P.append(rf"\node[anchor=west, inner sep=0pt] at ({lx + 7.5:.1f},{ly:.1f}) {{{PAL.tex_escape(lbl)}}};")

    # Divider
    P.append(rf"\draw[line width=0.3pt, color=black!35] (248,-18) -- (248,{H + 14:.1f});")

    # Right panel: Top 12 kernels
    RX0, RW = 310.0, 78.0
    max_k = max(v for _, v in k_items)
    rh_r = H / len(k_items)
    P.append(rf"\node[anchor=west, inner sep=0pt] at (258,{H + 10:.1f}) {{\textbf{{(b) Top 12 kernels}}}};" )
    for t in [0, 2, 4, 6]:
        if t > max_k:
            continue
        gx = RX0 + RW * t / max_k
        P.append(rf"\draw[line width=0.2pt, dash pattern=on 1pt off 1.5pt, color=black!30] ({gx:.2f},0) -- ({gx:.2f},{H:.2f});")
        P.append(rf"\node[anchor=north, inner sep=0pt] at ({gx:.2f},-3) {{{t}}};")
    P.append(rf"\draw[line width=0.3pt] ({RX0:.2f},0) -- ({RX0 + RW:.2f},0);")
    P.append(rf"\node[anchor=north, inner sep=0pt] at ({RX0 + RW/2:.2f},-13) {{Reproduced bugs}};")

    for i, (kname, cnt) in enumerate(k_items):
        ym = H - (i + 0.5) * rh_r
        P.append(rf"\node[anchor=east, inner sep=0pt] at ({RX0 - 4:.2f},{ym:.2f}) {{\texttt{{{PAL.tex_escape(kname)}}}}};")
        sw = RW * cnt / max_k
        bh = 6.8
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(PAL.THEME_DARK)}, draw=black, line width=0.25pt] ({RX0:.2f},{ym - bh/2:.2f}) rectangle ({RX0 + sw:.2f},{ym + bh/2:.2f});")
        P.append(rf"\node[anchor=west, inner sep=0pt] at ({RX0 + sw + 3:.2f},{ym:.2f}) {{{cnt}}};")

    render_tikz_svg(P, os.path.join(ASSET_DIR, "bugs_taxonomy.svg"))


def draw_analyzer_agreement() -> None:
    """Claude vs. Codex 180-trial inter-rater agreement in TikZ + Linux Libertine."""
    readme = open(os.path.join(ARTIFACT, "notes", "2026-11-19-l2-v10-codex-vs-claude", "README.md")).read()
    parsed = []
    for line in readme.splitlines():
        m = re.match(r"^\|\s*(G\d+\s+[^|]+?)\s*\|\s*(\d+)\s*\((\d+)\s*%\)\s*\|\s*([^|]+?)\s*\|$", line)
        if m:
            parsed.append((m.group(1).strip(), int(m.group(2)), int(m.group(3))))

    X0, W = 166.0, 175.0
    RH = 11.2
    H = len(parsed) * RH
    P: list[str] = []

    # Top legend
    leg = [
        (PAL.AGREEMENT_BAND_COLORS["high"], r"$\ge 85\%$"),
        (PAL.AGREEMENT_BAND_COLORS["mid"], r"70--84\%"),
        (PAL.AGREEMENT_BAND_COLORS["low"], r"$< 70\%$ (mostly ``unknown'')"),
    ]
    lx = X0
    ly = H + 11.0
    for idx, (col, lbl) in enumerate(leg):
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.3pt] ({lx:.1f},{ly - 2.5:.1f}) rectangle ({lx + 5:.1f},{ly + 2.5:.1f});")
        P.append(rf"\node[anchor=west, inner sep=0pt] at ({lx + 7.5:.1f},{ly:.1f}) {{{lbl}}};")
        lx += [44.0, 52.0, 95.0][idx]

    # Grid & axis
    for pct in [0, 25, 50, 75, 100]:
        gx = X0 + W * pct / 100.0
        P.append(rf"\draw[line width=0.2pt, dash pattern=on 1pt off 1.5pt, color=black!30] ({gx:.2f},0) -- ({gx:.2f},{H:.2f});")
        P.append(rf"\node[anchor=north, inner sep=0pt] at ({gx:.2f},-3) {{{pct}\%}};")
    g80 = X0 + W * 0.80
    P.append(rf"\draw[line width=0.45pt, densely dashed, color=black] ({g80:.2f},0) -- ({g80:.2f},{H:.2f});")
    P.append(rf"\draw[line width=0.3pt] ({X0:.2f},0) -- ({X0 + W:.2f},0);")
    P.append(rf"\node[anchor=north, inner sep=0pt] at ({X0 + W/2:.2f},-13) {{Exact agreement across 180 trials}};")

    for i, (field, same, pct) in enumerate(parsed):
        field = field.replace("judgement", "judgment")
        ym = H - (i + 0.5) * RH
        col = PAL.AGREEMENT_BAND_COLORS["high"] if pct >= 85 else (PAL.AGREEMENT_BAND_COLORS["mid"] if pct >= 70 else PAL.AGREEMENT_BAND_COLORS["low"])
        P.append(rf"\node[anchor=east, inner sep=0pt] at ({X0 - 5:.2f},{ym:.2f}) {{{PAL.tex_escape(field)}}};")
        sw = W * (same / 180.0)
        bh = 6.8
        P.append(rf"\filldraw[fill={PAL.tikz_rgb(col)}, draw=black, line width=0.25pt] ({X0:.2f},{ym - bh/2:.2f}) rectangle ({X0 + sw:.2f},{ym + bh/2:.2f});")
        P.append(rf"\node[anchor=west, inner sep=1.2pt, fill=white] at ({X0 + sw + 2.5:.2f},{ym:.2f}) {{{pct}\% ({same}/180)}};")

    render_tikz_svg(P, os.path.join(ASSET_DIR, "analyzer_agreement.svg"))


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ASSET_DIR, exist_ok=True)
    att_rows, case_map = load_attainment()
    copy_clean_csvs()
    _, summary = build_bugs(att_rows, case_map)
    draw_bugs_funnel(summary)
    draw_bugs_by_model(summary)
    draw_bugs_taxonomy(summary)
    draw_analyzer_agreement()
    print("Built sanitized data and 4 paper-matched TikZ/Libertine SVG figures.")
    print("Closed vs Open:", summary["closed_vs_open"])


if __name__ == "__main__":
    main()
