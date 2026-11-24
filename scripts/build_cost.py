#!/usr/bin/env python3
"""
Per-session cost, tokens and time for the Cost page (data/cost.csv and three figures).

Sources, the same as the paper's Table 1 (tables/make_tables.py, models_table):
- tokens and USD: the trial records' unified cost files (trial-costs-*.json, the latest file with
  records wins), list prices as of November 2026;
- Kimi K3.1 Pro has no cost record: tokens from its preprocessed trace, priced at 2 / 0.2 / 2 / 10
  USD per 1M (input / cache read / cache write / output incl. reasoning);
- Qwen3.8-27B cache reads repriced at the input price (Alibaba lists no cache price);
- time: active time T_e2e, LLM time and tool time from the paper's behavior.csv.

Figures are TikZ in Linux Libertine, rendered to SVG with pdftocairo, in a red ramp reserved for expense.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as PAL
from build_bug_analysis import MODEL_DISPLAY, render_tikz_svg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
ARTIFACT = os.path.join(WS, "Agent_Bench_Artifact")
PAPER = os.path.join(WS, "icpe27")
DATA_DIR = os.path.join(ROOT, "data")
ASSET_DIR = os.path.join(ROOT, "docs", "assets")

# column order of the paper's attainment figure: (framework, [(agent, short name)])
FRAMEWORKS = [
    ("Claude Code", [("fable-5-2-claude-code", "Fable 5.2"), ("fable-claude-code", "Fable 5.1"),
                     ("opus-5-5-claude-code", "Opus 5.5"), ("sonnet-5-5-claude-code", "Sonnet 5.5"),
                     ("haiku-5-5-claude-code", "Haiku 5.5"), ("haiku-4-5-claude-code", "Haiku 4.5")]),
    ("Antigravity", [("gemini-4-neon-xhigh-antigravity", "Gemini 4N"), ("gemini-4-carbon-high-antigravity", "Gemini 4C"),
                     ("gemini-4-argon-high-antigravity", "Gemini 4A")]),
    ("Codex", [("astra-6-2-codex-cli", "GPT-6.2A"), ("astra-codex-cli", "GPT-6A"), ("sol-6-2-codex-cli", "GPT-6.2S"),
               ("sol-6-1-codex-cli", "GPT-6.1S"), ("sol-codex-cli", "GPT-6S"), ("terra-codex-cli", "GPT-5.6T"),
               ("luna-codex-cli", "GPT-6L")]),
    ("OpenCode", [("kimi-k3-1-pro-opencode", "Kimi K3.1P"), ("kimi-opencode", "Kimi K3"), ("glm-5-2-opencode", "GLM-5.2"),
                  ("glm-5-3-flash-opencode", "GLM-5.3F"), ("deepseek-flash-opencode", "DS-V4F"),
                  ("deepseek-flash-reasoning-opencode", "DS-V4F-R"), ("qwen3-8-27b-opencode", "Qwen3.8"),
                  ("qwen3-6-35b-a3b-opencode", "Qwen3.6"), ("mimo-v2-6-pro-opencode", "MiMo-2.6P"),
                  ("seed-oss-36b-opencode", "Seed-36B"), ("nemotron-3-ultra-opencode", "Nemo 3U"),
                  ("nemotron-3-5-super-opencode", "Nemo 3.5S")]),
]
CATEGORIES = [("linear-algebra/blas", "BLAS"), ("linear-algebra/kernels", "Kernels"),
              ("linear-algebra/solvers", "Solvers"), ("datamining", "Data"),
              ("medley", "Medley"), ("stencils", "Stencils")]
REPRICE_CACHE_READ = {"qwen3-8-27b-opencode": 0.50 - 0.10}
K31_PRICE = (2.0, 0.2, 2.0, 10.0)  # input, cache read, cache write, output (USD per 1M)


def build_rows() -> list[dict]:
    norm = lambda q: q.split("rounds/", 1)[1] if "rounds/" in q else q
    cost = {}
    for f in sorted(glob.glob(os.path.join(ARTIFACT, "notes/2026-11-13-unified-cost/trial-costs-*.json"))):
        for t in json.load(open(f)):
            if t.get("records"):
                cost[norm(t["trial"])] = t
    beh = {(b["agent"], b["kernel"]): b for b in csv.DictReader(open(os.path.join(PAPER, "figures/behavior/behavior.csv")))}
    rows = []
    for r in csv.DictReader(open(os.path.join(PAPER, "figures/attainment/attainment.csv"))):
        ag = r["agent"]
        t = cost.get(norm(r["trial"]))
        if t is None and ag == "kimi-k3-1-pro-opencode":
            k = json.load(open(glob.glob(os.path.join(
                ARTIFACT, "preprocessing/2026-11-23-round-022-kimi-k31-pro", f"{ag}.{r['case']}*", "SESSION-STATISTICS.json"))[0]))["tokens"]
            o = (k.get("output") or 0) + (k.get("reasoning") or 0)
            inp, cr, cw = k.get("input") or 0, k.get("cache_read") or 0, k.get("cache_write") or 0
            t = dict(input_tokens=inp, cache_read_tokens=cr, cache_write_5m_tokens=cw, cache_write_1h_tokens=0,
                     output_billed=o, reasoning_tokens=k.get("reasoning") or 0,
                     usd=(inp * K31_PRICE[0] + cr * K31_PRICE[1] + cw * K31_PRICE[2] + o * K31_PRICE[3]) / 1e6)
        if t is None:
            sys.exit(f"no cost record for {r['trial']}")
        inp = int(t.get("input_tokens") or 0)
        cr = int(t.get("cache_read_tokens") or 0)
        cw = int(t.get("cache_write_5m_tokens") or 0) + int(t.get("cache_write_1h_tokens") or 0)
        out = int(t.get("output_billed") or 0)
        usd = None if t.get("usd") in ("", None) else float(t["usd"]) + REPRICE_CACHE_READ.get(ag, 0.0) * cr / 1e6
        b = beh[(ag, r["kernel"])]
        disp, fw, wc = MODEL_DISPLAY[ag]
        f4 = lambda v: "" if v is None else f"{v:.4f}"
        rows.append({
            "agent": ag, "model": disp, "framework": fw, "weight_class": wc,
            "case": r["case"], "kernel": r["kernel"], "category": r["category"], "state": r["state"],
            "attainment": r["attainment"], "bug_marked": r.get("bug_marked", ""),
            "input_tokens": inp, "cache_read_tokens": cr, "cache_write_tokens": cw,
            "output_tokens": out, "reasoning_tokens": int(t.get("reasoning_tokens") or 0),
            "input_tokens_total": inp + cr + cw, "usd": f4(usd),
            "active_min": f"{float(b['active_s']) / 60:.2f}", "llm_min": f"{float(b['llm_s']) / 60:.2f}",
            "tool_min": f"{float(b['tool_s']) / 60:.2f}",
        })
    with open(os.path.join(DATA_DIR, "cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------- figures
def _ramp(v: float, lo: float, hi: float, R: list[str]) -> tuple[str, bool]:
    """Log-scale colour from a light-to-dark ramp; returns (hex, dark) where dark asks for white text."""
    x = (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    x = min(max(x, 0.0), 1.0) * (len(R) - 1)
    i = min(int(x), len(R) - 2)
    f = x - i
    a, b = R[i], R[i + 1]
    mix = "".join(f"{round(int(a[k:k + 2], 16) * (1 - f) + int(b[k:k + 2], 16) * f):02X}" for k in (0, 2, 4))
    return mix, x > 2.9


def _fmt(v: float) -> str:
    if v >= 9.95:
        return f"{v:.0f}"
    if v >= 0.995:
        return f"{v:.1f}"
    if v >= 0.0095:
        return f"{v:.2f}".lstrip("0")
    return "<.01"


GEOM = dict(CAT_W=13.0, NAME_W=56.0, COL_W=14.6, GAP=4.0, ROW_H=9.2, HEAD_H=44.0)
GREY_NA = "EEEEEE"


def heatmap(rows: list[dict], key: str, lo: float, hi: float, unit: str, out: str, ticks: list[float],
            ramp: list[str], scale: float = 1.0, unpriced_legend: bool = False) -> None:
    """Kernels x models; cell value = row[key] * scale. A black corner marks a failed or incorrect session,
    a dashed frame a session with a bug marked (reproduced outside the task's setting, kept and scored), as in the
    paper's attainment figure. Footer rows: geometric and arithmetic mean over the model's 30 sessions."""
    G = GEOM
    cells = {(r["agent"], r["kernel"]): r for r in rows}
    val = lambda r: None if r[key] in ("", None) else float(r[key]) * scale
    kernels = []
    for cat, lab in CATEGORIES:
        ks = []
        for r in rows:
            if r["category"] == cat and r["kernel"] not in ks:
                ks.append(r["kernel"])
        kernels.append((cat, lab, sorted(ks, key=lambda k: min(x["case"] for x in rows if x["kernel"] == k))))
    xs, x = [], G["CAT_W"] + G["NAME_W"]
    fw_spans = []
    for fw, models in FRAMEWORKS:
        x0 = x
        for ag, short in models:
            xs.append((ag, short, x))
            x += G["COL_W"]
        fw_spans.append((fw, x0, x))
        x += G["GAP"]
    width = x - G["GAP"]
    P = []
    y_top = 0.0
    W, H = G["COL_W"], G["ROW_H"]
    small = r"\fontsize{5.6}{6}\selectfont"

    def cell(cx, y, v, bold=False):
        if v is None:
            P.append(rf"\fill[fill={PAL.tikz_rgb(GREY_NA)}] ({cx:.2f},{y:.2f}) rectangle ({cx + W:.2f},{y + H:.2f});")
            P.append(rf"\node[text={PAL.tikz_rgb('8A8A8A')}, inner sep=0pt, font={small}] at ({cx + W / 2:.2f},{y + H / 2:.2f}) {{--}};")
            return
        col, dark = _ramp(v, lo, hi, ramp)
        P.append(rf"\fill[fill={PAL.tikz_rgb(col)}] ({cx:.2f},{y:.2f}) rectangle ({cx + W:.2f},{y + H:.2f});")
        f = (r"\bfseries" if bold else "") + small
        P.append(rf"\node[text={'white' if dark else 'black'}, inner sep=0pt, font={f}] at ({cx + W / 2:.2f},{y + H / 2:.2f}) {{{_fmt(v)}}};")

    for fw, a, b in fw_spans:
        P.append(rf"\node[anchor=south, inner sep=1pt] at ({(a + b) / 2:.2f},{y_top + G['HEAD_H'] - 6:.2f}) {{{fw}}};")
        P.append(rf"\draw[line width=0.4pt] ({a:.2f},{y_top + G['HEAD_H'] - 6:.2f}) -- ({b:.2f},{y_top + G['HEAD_H'] - 6:.2f});")
    for ag, short, cx in xs:
        P.append(rf"\node[anchor=west, rotate=90, inner sep=0.5pt, font=\fontsize{{6.5}}{{7}}\selectfont] at ({cx + W / 2:.2f},{y_top + 1:.2f}) {{{PAL.tex_escape(short)}}};")
    y = y_top
    marks = []
    for cat, lab, ks in kernels:
        y0 = y
        for k in ks:
            y -= H
            P.append(rf"\node[anchor=east, inner sep=1pt] at ({G['CAT_W'] + G['NAME_W'] - 2:.2f},{y + H / 2:.2f}) {{{PAL.tex_escape(k)}}};")
            for ag, short, cx in xs:
                r = cells[(ag, k)]
                cell(cx, y, val(r))
                if r["state"] != "pass":
                    marks.append(rf"\fill[black] ({cx:.2f},{y:.2f}) -- ({cx + 3.6:.2f},{y:.2f}) -- ({cx:.2f},{y + 3.6:.2f}) -- cycle;")
                elif r.get("bug_marked") == "True":
                    marks.append(rf"\draw[black, line width=0.8pt, dash pattern=on 1.6pt off 1.0pt] ({cx + 0.6:.2f},{y + 0.6:.2f}) rectangle ({cx + W - 0.6:.2f},{y + H - 0.6:.2f});")
        P.append(rf"\node[rotate=90, inner sep=0pt] at ({G['CAT_W'] / 2:.2f},{(y0 + y) / 2:.2f}) {{{lab}}};")
        P.append(rf"\draw[line width=0.3pt] (0,{y:.2f}) -- ({width:.2f},{y:.2f});")
    y_grid = y
    y -= 2.0
    for name, fn in (("Geomean", lambda v: math.exp(sum(map(math.log, v)) / len(v))), ("Mean", lambda v: sum(v) / len(v))):
        y -= H
        P.append(rf"\node[anchor=east, inner sep=1pt] at ({G['CAT_W'] + G['NAME_W'] - 2:.2f},{y + H / 2:.2f}) {{\textit{{{name}}}}};")
        for ag, short, cx in xs:
            vals = [val(cells[(ag, k)]) for _, _, ks in kernels for k in ks]
            vals = [v for v in vals if v is not None and v > 0]
            cell(cx, y, fn(vals) if vals else None, bold=True)
    for ag, short, cx in xs:
        P.append(rf"\draw[white, line width=0.5pt] ({cx:.2f},{y:.2f}) -- ({cx:.2f},{y_top:.2f});")
    P.extend(marks)
    # legend: colour bar, failed or incorrect, bug marked, (no list price)
    yl = y - 16
    bx, bw = G["CAT_W"] + G["NAME_W"], 150.0
    steps = 60
    for i in range(steps):
        v = 10 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * (i + 0.5) / steps)
        col, _ = _ramp(v, lo, hi, ramp)
        P.append(rf"\fill[fill={PAL.tikz_rgb(col)}] ({bx + bw * i / steps:.2f},{yl:.2f}) rectangle ({bx + bw * (i + 1) / steps + 0.05:.2f},{yl + 6:.2f});")
    P.append(rf"\draw[line width=0.3pt] ({bx:.2f},{yl:.2f}) rectangle ({bx + bw:.2f},{yl + 6:.2f});")
    for tv in ticks:
        tx = bx + bw * (math.log10(tv) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        P.append(rf"\draw[line width=0.3pt] ({tx:.2f},{yl:.2f}) -- ({tx:.2f},{yl - 1.5:.2f});")
        P.append(rf"\node[anchor=north, inner sep=1pt] at ({tx:.2f},{yl - 1.5:.2f}) {{{tv:g}}};")
    P.append(rf"\node[anchor=east, inner sep=2pt] at ({bx:.2f},{yl + 3:.2f}) {{{unit}}};")
    lx = bx + bw + 20
    lw, lh = 12.0, 8.0
    ly = yl - 1
    P.append(rf"\fill[fill={PAL.tikz_rgb(ramp[1])}] ({lx:.2f},{ly:.2f}) rectangle ({lx + lw:.2f},{ly + lh:.2f});")
    P.append(rf"\fill[black] ({lx:.2f},{ly:.2f}) -- ({lx + 3.6:.2f},{ly:.2f}) -- ({lx:.2f},{ly + 3.6:.2f}) -- cycle;")
    P.append(rf"\node[anchor=west, inner sep=2pt] at ({lx + lw:.2f},{ly + lh / 2:.2f}) {{Failed or incorrect}};")
    lx += 82
    P.append(rf"\fill[fill={PAL.tikz_rgb(ramp[1])}] ({lx:.2f},{ly:.2f}) rectangle ({lx + lw:.2f},{ly + lh:.2f});")
    P.append(rf"\draw[black, line width=0.8pt, dash pattern=on 1.6pt off 1.0pt] ({lx + 0.6:.2f},{ly + 0.6:.2f}) rectangle ({lx + lw - 0.6:.2f},{ly + lh - 0.6:.2f});")
    P.append(rf"\node[anchor=west, inner sep=2pt] at ({lx + lw:.2f},{ly + lh / 2:.2f}) {{Bug marked}};")
    lx += 62
    if unpriced_legend:
        P.append(rf"\fill[fill={PAL.tikz_rgb(GREY_NA)}] ({lx:.2f},{ly:.2f}) rectangle ({lx + lw:.2f},{ly + lh:.2f});")
        P.append(rf"\node[anchor=west, inner sep=2pt] at ({lx + lw:.2f},{ly + lh / 2:.2f}) {{No list price}};")
    render_tikz_svg(P, os.path.join(ASSET_DIR, out))



def main() -> None:
    rows = build_rows()
    heatmap(rows, "input_tokens_total", 0.05, 1000.0, "Input tokens (M)", "input_tokens_heatmap.svg",
            [0.1, 1, 10, 100, 1000], PAL.INPUT_TOKEN_RAMP, scale=1e-6)
    heatmap(rows, "output_tokens", 1.0, 3000.0, "Output tokens (K)", "output_tokens_heatmap.svg", [1, 10, 100, 1000],
            PAL.OUTPUT_TOKEN_RAMP, scale=1e-3)
    heatmap(rows, "active_min", 1.0, 1000.0, "Active minutes", "time_heatmap.svg", [1, 10, 100, 1000], PAL.TIME_RAMP)
    heatmap(rows, "usd", 0.01, 100.0, "USD per session", "cost_heatmap.svg", [0.01, 0.1, 1, 10, 100],
            PAL.EXPENSE_RAMP, unpriced_legend=True)
    tot = sum(float(r["usd"]) for r in rows if r["usd"])
    print(f"{len(rows)} sessions, {tot:.0f} USD, {sum(float(r['active_min']) for r in rows) / 60:.0f} h")


if __name__ == "__main__":
    main()
