#!/usr/bin/env python3
"""
Per-kernel roofline model for the Roofline Model page (data/roofline.csv, includes/roofline-table.md).

Final model: the trial records' scored work models (notes/2026-10-02-round-011b-second-operator/workmodel-scored.json):
scored dimensions and the operation and byte formulas after the D1/D2 revisions. The AI model's formulas for the nine
revised kernels are the "before" side of the paper's Table (tables/perf-model.tex); all other kernels keep them.
Peaks as in the paper: FP32 75 TFLOP/s, FP64 1.2 TFLOP/s (adi), INT32 37.5 Top/s (floyd-warshall, nussinov),
HBM 7.7 TB/s. Bound = max(Ops / compute peak, Bytes / HBM bandwidth).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as PAL
from build_bug_analysis import render_tikz_svg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
WM = os.path.join(WS, "Agent_Bench_Artifact", "notes", "2026-10-02-round-011b-second-operator", "workmodel-scored.json")
ATT = os.path.join(WS, "icpe27", "figures", "attainment", "attainment.csv")
PEAK = {"fp32": 75e12, "fp64": 1.2e12, "int32": 37.5e12}
HBM = 7.7e12
PEAK_OF = {"adi": "fp64", "floyd-warshall": "int32", "nussinov": "int32"}
CATS = [("linear-algebra/blas", "BLAS"), ("linear-algebra/kernels", "Kernels"), ("linear-algebra/solvers", "Solvers"),
        ("datamining", "Data mining"), ("medley", "Medley"), ("stencils", "Stencils")]

# AI model formulas of the revised kernels (tables/perf-model.tex, before the arrows); rule D1 = bytes, D2 = ops
AI = {
    "2mm": ("D2", lambda e, f, b: (2 * e["ni"] * e["nj"] * (e["nk"] + e["nl"]), b)),
    "3mm": ("D2", lambda e, f, b: (2 * e["nj"] * (e["ni"] * e["nk"] + e["ni"] * e["nl"] + e["nl"] * e["nm"]), b)),
    "gemver": ("D2", lambda e, f, b: (8 * e["n"] ** 2, b)),
    "adi": ("D1, D2", lambda e, f, b: (30 * e["tsteps"] * (e["n"] - 2) ** 2, 16 * e["n"] ** 2)),
    "heat-3d": ("D1, D2", lambda e, f, b: (30 * e["tsteps"] * (e["n"] - 2) ** 3, 12 * e["n"] ** 3)),
    "jacobi-1d": ("D1", lambda e, f, b: (f, 12 * e["n"])),
    "jacobi-2d": ("D1", lambda e, f, b: (f, 12 * e["n"] ** 2)),
    "seidel-2d": ("D1", lambda e, f, b: (f, 8 * e["n"] ** 2)),
    "fdtd-2d": ("D1", lambda e, f, b: (f, 4 * (e["tmax"] + 6 * e["nx"] * e["ny"]))),
}


# the paper's formulas (tables/perf-model.tex), AI model -> revised, for the nine revised kernels
REVISED_TEX = {
    "2mm": (r"$2 n_i n_j (n_k + n_l)$", r"$2\min$ over the 2 evaluation orders", "", ""),
    "3mm": (r"$2 n_j (n_i n_k + n_i n_l + n_l n_m)$", r"$2\min$ over the 5 evaluation orders", "", ""),
    "gemver": (r"$8 n^2$", r"$4 n^2$", "", ""),
    "adi": (r"$30 T (n-2)^2$", r"$18 T (n-2)^2$", r"$16 n^2$", r"$32 T n^2$"),
    "heat-3d": (r"$30 T (n-2)^3$", r"$16 T (n-2)^3$", r"$12 n^3$", r"$16 T n^3$"),
    "jacobi-1d": ("", "", r"$12 n$", r"$16 T n$"),
    "jacobi-2d": ("", "", r"$12 n^2$", r"$16 T n^2$"),
    "seidel-2d": ("", "", r"$8 n^2$", r"$8 T n^2$"),
    "fdtd-2d": ("", "", r"$4 (T + 6 n_x n_y)$", r"$24 T n_x n_y$"),
}


def roof_chart(rows, out, roofs, label_groups=True, group_ratio=1.7, skip_label=(), zoom=None, render=True):
    """A roofline chart: attainable performance against arithmetic intensity, log-log, for the given kernels. Each kernel
    sits on its roof; a revised kernel shows the AI model (open circle) and the reviewed model (filled), joined by an
    arrow. Kernels at nearly the same intensity on the same roof are labelled together."""
    X0, W, H = 30.0, 360.0, 230.0
    xlo, xhi, ylo, yhi = 0.1, 1e5, 0.5, 200.0
    X = lambda v: X0 + W * (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo))
    Y = lambda v: H * (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))
    att = lambda pk, I: min(PEAK[pk], I * HBM) / 1e12
    P = [r"\usetikzlibrary{arrows.meta}"]
    grey = PAL.tikz_rgb(PAL.GREY_LIGHT)
    ink = PAL.tikz_rgb(PAL.GREY_DARK)
    for v in [0.1, 1, 10, 100, 1000, 1e4, 1e5]:
        P.append(rf"\draw[draw={grey}, line width=0.3pt] ({X(v):.2f},0) -- ({X(v):.2f},{H:.2f});")
        P.append(rf"\node[anchor=north, inner sep=1.5pt] at ({X(v):.2f},0) {{{v:g}}};")
    for v in [1, 10, 100]:
        P.append(rf"\draw[draw={grey}, line width=0.3pt] ({X0:.2f},{Y(v):.2f}) -- ({X0 + W:.2f},{Y(v):.2f});")
        P.append(rf"\node[anchor=east, inner sep=1.5pt] at ({X0:.2f},{Y(v):.2f}) {{{v:g}}};")
    P.append(rf"\draw[line width=0.4pt] ({X0:.2f},0) rectangle ({X0 + W:.2f},{H:.2f});")
    P.append(rf"\node[anchor=north] at ({X0 + W / 2:.2f},-10) {{Arithmetic intensity (operations per byte, log scale)}};")
    P.append(rf"\node[rotate=90, anchor=south] at ({X0 - 16:.2f},{H / 2:.2f}) {{Attainable performance (Top/s, log scale)}};")
    names = {"fp32": "FP32 peak, 75 TFLOP/s", "int32": "INT32 peak, 37.5 Top/s", "fp64": "FP64 peak, 1.2 TFLOP/s"}
    for pk in roofs:
        ridge = PEAK[pk] / HBM
        P.append(rf"\draw[line width=1.0pt, draw={ink}] ({X(ridge):.2f},{Y(PEAK[pk] / 1e12):.2f}) -- ({X(xhi):.2f},{Y(PEAK[pk] / 1e12):.2f});")
        P.append(rf"\node[anchor=south east, inner sep=1.5pt, font=\fontsize{{6.5}}{{7}}\selectfont] at ({X(xhi):.2f},{Y(PEAK[pk] / 1e12):.2f}) {{{names[pk]}}};")
    top = max(PEAK[pk] for pk in roofs)
    lo_I = ylo * 1e12 / HBM
    P.append(rf"\draw[line width=1.0pt, draw={ink}] ({X(max(xlo, lo_I)):.2f},{Y(max(xlo, lo_I) * HBM / 1e12):.2f}) -- ({X(top / HBM):.2f},{Y(top / 1e12):.2f});")
    ang = math.degrees(math.atan2(Y(10) - Y(1), X(10 / 7.7) - X(1 / 7.7)))
    P.append(rf"\node[rotate={ang:.1f}, anchor=south, inner sep=3pt, font=\fontsize{{6.5}}{{7}}\selectfont] at ({X(6):.2f},{Y(6 * 7.7):.2f}) {{HBM, 7.7 TB/s}};")
    col = {"compute": PAL.THEME_DARK, "memory": "1F4E8C"}
    arrows, dots = [], []
    for r in rows:
        pk = r["peak"].lower(); I = r["intensity"]
        x, y = X(I), Y(att(pk, I))
        if r["rule"]:
            Ia = r["ai_intensity"]; xa, ya = X(Ia), Y(att(pk, Ia))
            if abs(x - xa) + abs(y - ya) > 3:
                arrows.append(rf"\draw[-{{Stealth[length=7pt, width=6pt]}}, line width=1.4pt, draw={PAL.tikz_rgb(PAL.AMBER)}, shorten >=2.6pt, shorten <=2.4pt] ({xa:.2f},{ya:.2f}) -- ({x:.2f},{y:.2f});")
                dots.append(rf"\fill[fill=white, draw={PAL.tikz_rgb(col[r['ai_binding']])}, line width=0.9pt] ({xa:.2f},{ya:.2f}) circle (2.6pt);")
        dots.append(rf"\fill[fill={PAL.tikz_rgb(col[r['binding']])}] ({x:.2f},{y:.2f}) circle (2.6pt);")
    P += arrows + dots
    # labels: group kernels at nearly the same point (same roof, intensity within a factor of 1.7)
    if zoom:
        (zl, zh, pkz) = zoom
        yz = Y(PEAK[pkz] / 1e12)
        P.append(rf"\draw[draw={PAL.tikz_rgb(PAL.AMBER)}, line width=0.8pt, dashed] ({X(zl):.2f},{yz - 7:.2f}) rectangle ({X(zh):.2f},{yz + 7:.2f});")
        P.append(rf"\node[anchor=south, inner sep=1pt, font=\bfseries\fontsize{{6.5}}{{7}}\selectfont, text={PAL.tikz_rgb(PAL.AMBER)}] at ({(X(zl) + X(zh)) / 2:.2f},{yz + 7:.2f}) {{2mm and 3mm, magnified in (b)}};")
    pts = sorted([r for r in rows if r["kernel"] not in skip_label], key=lambda r: (r["peak"], r["binding"], r["intensity"]))
    groups = []
    for r in pts:
        g_ = groups[-1] if groups else None
        if g_ and g_[0]["peak"] == r["peak"] and g_[0]["binding"] == r["binding"] and r["intensity"] / g_[-1]["intensity"] < group_ratio:
            g_.append(r)
        else:
            groups.append([r])
    for gi, grp in enumerate(groups):
        pk = grp[0]["peak"].lower()
        I = math.exp(sum(math.log(r["intensity"]) for r in grp) / len(grp))
        x, y = X(I), Y(att(pk, I))
        names_ = [r["kernel"] for r in grp]
        off = label_groups(grp, x, y) if callable(label_groups) else None
        dx, dy, an = off if off else ((4, -6, "north west") if grp[0]["binding"] == "compute" else (5, -2, "west"))
        if len(grp) > 3:
            lines = [", ".join(names_[k:k + 3]) for k in range(0, len(names_), 3)]
            txt = r"\\".join(lines)
            P.append(rf"\draw[draw={ink}, line width=0.3pt] ({x:.2f},{y - 3:.2f}) -- ({x + dx:.2f},{y + dy:.2f});")
            P.append(rf"\node[anchor={an}, align=left, inner sep=1pt, font=\ttfamily\fontsize{{5.8}}{{6.8}}\selectfont] at ({x + dx:.2f},{y + dy:.2f}) {{{txt}}};")
        else:
            P.append(rf"\node[anchor={an}, align=left, inner sep=1pt, font=\ttfamily\fontsize{{6}}{{7}}\selectfont] at ({x + dx:.2f},{y + dy:.2f}) {{{', '.join(names_)}}};")
    ly = H + 10
    P.append(rf"\fill[fill=white, draw={ink}, line width=0.9pt] ({X0 + 4:.2f},{ly:.2f}) circle (2.6pt);")
    P.append(rf"\node[anchor=west] at ({X0 + 8:.2f},{ly:.2f}) {{AI model}};")
    P.append(rf"\draw[-{{Stealth[length=7pt, width=6pt]}}, line width=1.4pt, draw={PAL.tikz_rgb(PAL.AMBER)}] ({X0 + 52:.2f},{ly:.2f}) -- ({X0 + 74:.2f},{ly:.2f});")
    P.append(rf"\node[anchor=west] at ({X0 + 76:.2f},{ly:.2f}) {{review}};")
    P.append(rf"\fill[fill={PAL.tikz_rgb(PAL.THEME_DARK)}] ({X0 + 114:.2f},{ly:.2f}) circle (2.6pt);")
    P.append(rf"\node[anchor=west] at ({X0 + 118:.2f},{ly:.2f}) {{compute-bound}};")
    P.append(rf"\fill[fill={PAL.tikz_rgb('1F4E8C')}] ({X0 + 178:.2f},{ly:.2f}) circle (2.6pt);")
    P.append(rf"\node[anchor=west] at ({X0 + 182:.2f},{ly:.2f}) {{memory-bound}};")
    if not render:
        return P
    render_tikz_svg(P, os.path.join(ROOT, "docs", "assets", out))


def zoom_panel(rows, y0, lo=1100.0, hi=2000.0, pk="fp32"):
    """Panel (b): the FP32 roof between lo and hi Op/B, magnified; kernels at the same intensity share a label;
    revised kernels show AI model -> reviewed with the share of work removed."""
    X0, W = 30.0, 360.0
    X = lambda v: X0 + W * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    ink, grey = PAL.tikz_rgb(PAL.GREY_DARK), PAL.tikz_rgb(PAL.GREY_LIGHT)
    yr = y0 - 70
    P = []
    P.append(rf"\node[anchor=west, font=\bfseries] at ({X0 - 26:.2f},{y0:.2f}) {{(b)}};")
    P.append(rf"\node[anchor=west] at ({X0 - 10:.2f},{y0:.2f}) {{The FP32 roof from {lo:,.0f} to {hi:,.0f} operations per byte, magnified {(6 / math.log10(hi / lo)):.0f} times}};")
    for v in [t for t in (1400, 1500, 1600, 1700, 1800) if lo <= t <= hi]:
        P.append(rf"\draw[draw={grey}, line width=0.3pt] ({X(v):.2f},{yr - 45:.2f}) -- ({X(v):.2f},{yr + 45:.2f});")
        P.append(rf"\node[anchor=north, inner sep=1.5pt] at ({X(v):.2f},{yr - 45:.2f}) {{{v:,}}};")
    P.append(rf"\node[anchor=north] at ({X0 + W / 2:.2f},{yr - 55:.2f}) {{Arithmetic intensity (operations per byte, log scale)}};")
    P.append(rf"\draw[line width=1.0pt, draw={ink}] ({X0:.2f},{yr:.2f}) -- ({X0 + W:.2f},{yr:.2f});")
    P.append(rf"\node[anchor=south west, inner sep=1.5pt, font=\fontsize{{6.5}}{{7}}\selectfont] at ({X0:.2f},{yr + 30:.2f}) {{FP32 peak, 75 TFLOP/s}};")
    R = sorted([r for r in rows if r["peak"].lower() == pk and lo <= r["intensity"] <= hi], key=lambda r: r["intensity"])
    groups = []
    for r in R:
        if groups and r["intensity"] / groups[-1][-1]["intensity"] < 1.004:
            groups[-1].append(r)
        else:
            groups.append([r])
    col = PAL.THEME_DARK
    for i, grp in enumerate(groups):
        x = X(grp[0]["intensity"])
        for r in grp:
            if r["rule"]:
                xa = X(r["ai_intensity"])
                P.append(rf"\draw[-{{Stealth[length=6pt, width=5pt]}}, line width=1.4pt, draw={PAL.tikz_rgb(PAL.AMBER)}, shorten >=2.8pt, shorten <=2.6pt] ({xa:.2f},{yr:.2f}) -- ({X(r['intensity']):.2f},{yr:.2f});")
                P.append(rf"\fill[fill=white, draw={PAL.tikz_rgb(col)}, line width=0.9pt] ({xa:.2f},{yr:.2f}) circle (2.8pt);")
                cut = 100 * (1 - r["ops"] / r["ai_ops"])
                P.append(rf"\node[anchor=north, align=center, inner sep=1pt, font=\fontsize{{6}}{{7}}\selectfont, text={PAL.tikz_rgb(PAL.AMBER)}] at ({(xa + X(r['intensity'])) / 2:.2f},{yr - 9:.2f}) {{D2 removes {cut:.1f}\% of the work}};")
            P.append(rf"\fill[fill={PAL.tikz_rgb(col)}] ({X(r['intensity']):.2f},{yr:.2f}) circle (2.8pt);")
        name = ", ".join(r["kernel"] for r in grp)
        P.append(rf"\node[anchor=south, inner sep=2pt, font=\ttfamily\fontsize{{7}}{{8}}\selectfont] at ({x:.2f},{yr + 4:.2f}) {{{name}}};")
    return P


def intensity_strip(rows, out):
    """One row per kernel, sorted by arithmetic intensity (log x); ridge points of the FP32 and INT32 roofs as vertical
    lines; dot coloured by binding; revised kernels show the AI model (open) -> reviewed (filled) with an arrow."""
    R = sorted(rows, key=lambda r: r["intensity"])
    NAME_W, W, RH, RIGHT = 74.0, 290.0, 10.5, 70.0
    xlo, xhi = 0.2, 5e4
    X = lambda v: NAME_W + W * (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo))
    H = len(R) * RH
    ink, grey = PAL.tikz_rgb(PAL.GREY_DARK), PAL.tikz_rgb(PAL.GREY_LIGHT)
    col = {"compute": PAL.THEME_DARK, "memory": "1F4E8C"}
    P = [r"\usetikzlibrary{arrows.meta}"]
    r32 = PEAK["fp32"] / HBM
    P.append(rf"\fill[fill={PAL.tikz_rgb('EEF4FB')}] ({X(xlo):.2f},0) rectangle ({X(r32):.2f},{H:.2f});")
    P.append(rf"\fill[fill={PAL.tikz_rgb('EDF7F2')}] ({X(r32):.2f},0) rectangle ({X(xhi):.2f},{H:.2f});")
    for v in [1, 10, 100, 1000, 1e4]:
        P.append(rf"\draw[draw={grey}, line width=0.3pt] ({X(v):.2f},0) -- ({X(v):.2f},{H:.2f});")
    for v in [0.2, 1, 10, 100, 1000, 1e4, 5e4]:
        P.append(rf"\node[anchor=north, inner sep=1.5pt] at ({X(v):.2f},0) {{{v:g}}};")
    P.append(rf"\node[anchor=north] at ({NAME_W + W / 2:.2f},-10) {{Arithmetic intensity (operations per byte, log scale)}};")
    for pk, lab, dash in (("fp32", "FP32 ridge, 9.7", "solid"), ("int32", "INT32 ridge, 4.9", "dashed")):
        x = X(PEAK[pk] / HBM)
        P.append(rf"\draw[draw={ink}, line width=0.7pt, {dash}] ({x:.2f},0) -- ({x:.2f},{H + 3:.2f});")
        P.append(rf"\node[anchor={'south west' if pk == 'fp32' else 'south east'}, inner sep=1pt, font=\fontsize{{6}}{{7}}\selectfont] at ({x + (1 if pk == 'fp32' else -1):.2f},{H + 1:.2f}) {{{lab}}};")
    P.append(rf"\node[anchor=south west, font=\bfseries\fontsize{{6.5}}{{7}}\selectfont] at ({X(xhi) + 4:.2f},{H + 3:.2f}) {{Attainable}};")
    for i, r in enumerate(R):
        y = H - (i + 0.5) * RH
        if i % 2 == 0:
            P.append(rf"\fill[fill=white, fill opacity=0.45] ({NAME_W - 72:.2f},{y - RH / 2:.2f}) rectangle ({X(xhi) + RIGHT:.2f},{y + RH / 2:.2f});")
        nm = r["kernel"] + ("*" if r["rule"] else "")
        P.append(rf"\node[anchor=east, inner sep=1pt, font=\ttfamily\fontsize{{6.3}}{{7}}\selectfont] at ({NAME_W - 3:.2f},{y:.2f}) {{{nm}}};")
        x = X(r["intensity"])
        if r["rule"]:
            xa = X(r["ai_intensity"])
            if abs(xa - x) > 3:
                P.append(rf"\draw[-{{Stealth[length=6pt, width=5pt]}}, line width=1.3pt, draw={PAL.tikz_rgb(PAL.AMBER)}, shorten >=2.6pt, shorten <=2.4pt] ({xa:.2f},{y:.2f}) -- ({x:.2f},{y:.2f});")
            P.append(rf"\fill[fill=white, draw={PAL.tikz_rgb(col[r['ai_binding']])}, line width=0.9pt] ({xa:.2f},{y:.2f}) circle (2.5pt);")
        P.append(rf"\fill[fill={PAL.tikz_rgb(col[r['binding']])}] ({x:.2f},{y:.2f}) circle (2.6pt);")
        unit = "TFLOP/s" if r["peak"] != "INT32" else "Top/s"
        P.append(rf"\node[anchor=west, inner sep=1pt, font=\fontsize{{6.3}}{{7}}\selectfont] at ({X(xhi) + 4:.2f},{y:.2f}) {{{g(r['attainable_tops'])} {unit}{' (INT32)' if r['peak'] == 'INT32' else ''}}};")
    P.append(rf"\draw[line width=0.4pt] ({X(xlo):.2f},0) rectangle ({X(xhi):.2f},{H:.2f});")
    ly = H + 20
    P.append(rf"\fill[fill=white, draw={ink}, line width=0.9pt] ({NAME_W:.2f},{ly:.2f}) circle (2.5pt);")
    P.append(rf"\node[anchor=west] at ({NAME_W + 4:.2f},{ly:.2f}) {{AI model}};")
    P.append(rf"\draw[-{{Stealth[length=6pt, width=5pt]}}, line width=1.3pt, draw={PAL.tikz_rgb(PAL.AMBER)}] ({NAME_W + 48:.2f},{ly:.2f}) -- ({NAME_W + 68:.2f},{ly:.2f});")
    P.append(rf"\node[anchor=west] at ({NAME_W + 70:.2f},{ly:.2f}) {{review}};")
    P.append(rf"\fill[fill={PAL.tikz_rgb(PAL.THEME_DARK)}] ({NAME_W + 108:.2f},{ly:.2f}) circle (2.6pt);")
    P.append(rf"\node[anchor=west] at ({NAME_W + 112:.2f},{ly:.2f}) {{compute-bound}};")
    P.append(rf"\fill[fill={PAL.tikz_rgb('1F4E8C')}] ({NAME_W + 172:.2f},{ly:.2f}) circle (2.6pt);")
    P.append(rf"\node[anchor=west] at ({NAME_W + 176:.2f},{ly:.2f}) {{memory-bound}};")
    P.append(rf"\node[anchor=west, font=\fontsize{{6}}{{7}}\selectfont, text={ink}] at ({NAME_W + 232:.2f},{ly:.2f}) {{* revised by the review}};")
    render_tikz_svg(P, os.path.join(ROOT, "docs", "assets", out))


def binding(k, f, b):
    return "compute" if f / PEAK[PEAK_OF.get(k, "fp32")] >= b / HBM else "memory"


def perf(k, f, b):
    """Attainable performance in Top/s at intensity f/b."""
    return min(PEAK[PEAK_OF.get(k, "fp32")], f / b * HBM) / 1e12


def g(x):
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}" if x >= 1 else f"{x:.2f}"


def main():
    W = json.load(open(WM))["scored"]
    cat = {}
    for r in csv.DictReader(open(ATT)):
        cat.setdefault(r["kernel"], (r["category"], r["case"]))
    rows = []
    for c, lab in CATS:
        for k in sorted([k for k in W if cat[k][0] == c], key=lambda k: cat[k][1]):
            v = W[k]
            rule, fn = AI.get(k, ("", lambda e, f, b: (f, b)))
            fa, ba = fn(v["env"], v["flops"], v["bytes"])
            t, ta = max(v["flops"] / PEAK[PEAK_OF.get(k, "fp32")], v["bytes"] / HBM), max(fa / PEAK[PEAK_OF.get(k, "fp32")], ba / HBM)
            rows.append(dict(kernel=k, category=lab, scored=", ".join(f"{a} = {x:,}" for a, x in v["env"].items()),
                             ops_formula=v["flops_f"], bytes_formula=v["bytes_f"], ops=v["flops"], bytes=v["bytes"],
                             peak=PEAK_OF.get(k, "fp32").upper(), intensity=v["flops"] / v["bytes"],
                             attainable_tops=perf(k, v["flops"], v["bytes"]), binding=binding(k, v["flops"], v["bytes"]),
                             min_time_ms=t * 1e3, ai_ops=fa, ai_bytes=ba, ai_intensity=fa / ba,
                             ai_attainable_tops=perf(k, fa, ba), ai_binding=binding(k, fa, ba), ai_min_time_ms=ta * 1e3, rule=rule))
    with open(os.path.join(ROOT, "data", "roofline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ['<div class="njs-table" markdown="1">', "",
          "| Kernel | Category | Scored instance | Ops | Bytes | Intensity (Op/B) | Peak | Attainable (Top/s) | Bound by |",
          "| :--- | :--- | :--- | :--- | :--- | ---: | :---: | ---: | :---: |"]
    for r in rows:
        mark = " **✱**" if r["rule"] else ""
        md.append(f"| `{r['kernel']}`{mark} | {r['category']} | {r['scored']} | `{r['ops_formula']}` | `{r['bytes_formula']}` | "
                  f"{g(r['intensity'])} | {r['peak']} | {g(r['attainable_tops'])} | {r['binding']} |")
    md += ["", "</div>", ""]
    rv = ['<div class="njs-table" markdown="1">', "",
          "| Kernel | Rule | Ops: AI model → revised | Bytes: AI model → revised | Intensity (Op/B) | Attainable (Top/s) | Bound by |",
          "| :--- | :---: | :--- | :--- | :--- | :--- | :--- |"]
    order = ["jacobi-1d", "fdtd-2d", "jacobi-2d", "seidel-2d", "heat-3d", "adi", "2mm", "3mm", "gemver"]
    for k in order:
        r = next(x for x in rows if x["kernel"] == k)
        ao, ro, ab, rb = REVISED_TEX[k]
        bb = r["binding"] if r["ai_binding"] == r["binding"] else f"{r['ai_binding']} → **{r['binding']}**"
        rv.append(f"| `{k}` | {r['rule']} | {f'{ao} → {ro}' if ao else 'unchanged'} | {f'{ab} → {rb}' if ab else 'unchanged'} | "
                  f"{g(r['ai_intensity'])} → **{g(r['intensity'])}** | {g(r['ai_attainable_tops'])} → **{g(r['attainable_tops'])}** | {bb} |")
    rv += ["", "</div>", ""]
    os.makedirs(os.path.join(ROOT, "includes"), exist_ok=True)
    open(os.path.join(ROOT, "includes", "roofline-revisions.md"), "w").write("\n".join(rv))
    open(os.path.join(ROOT, "includes", "roofline-table.md"), "w").write("\n".join(md))
    stencil = [r for r in rows if r["category"] == "Stencils"]
    other = [r for r in rows if r["category"] != "Stencils"]
    STEN = {"jacobi-1d": (-5, -3, "east"), "fdtd-2d": (-5, 2, "east"), "jacobi-2d": (-5, 1, "east"), "heat-3d": (-5, -3, "east"),
            "seidel-2d": (-5, 4, "east"), "adi": (0, 6, "south")}
    roof_chart(stencil, "roofline_stencils.svg", ["fp32", "fp64"], lambda grp, x, y: STEN.get(grp[0]["kernel"]), group_ratio=1.0)
    OTH = {"deriche": (5, -3, "west"), "durbin": (0, -5, "north"), "doitgen": (0, -5, "north"),
           "nussinov": (0, 4, "south"), "floyd-warshall": (0, -5, "north"), "ludcmp": (0, -5, "north")}
    def oth(grp, x, y):
        if len(grp) == 1 and grp[0]["kernel"] in OTH:
            return OTH[grp[0]["kernel"]]
        if grp[0]["binding"] == "memory":
            return (8, -2, "west") if grp[0]["intensity"] > 0.8 else (-6, 3, "east")
        return None
    revised = [r for r in other if r["rule"]]
    Pa = roof_chart(revised, None, ["fp32"], lambda grp, x, y: (8, -2, "west") if grp[0]["binding"] == "memory" else None,
                    skip_label={"2mm", "3mm"}, zoom=(1380, 1880, "fp32"), render=False, group_ratio=1.01)
    Pa.insert(1, r"\node[anchor=west, font=\bfseries] at (4,262) {(a)};")
    render_tikz_svg(Pa + zoom_panel(revised, -40, 1380.0, 1880.0), os.path.join(ROOT, "docs", "assets", "roofline_others.svg"))

    print(len(rows), "kernels")


if __name__ == "__main__":
    main()
