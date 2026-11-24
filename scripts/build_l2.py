#!/usr/bin/env python3
"""
Session statistics and Level-2 Analyzer labels for the Session Behavior page.

Sources: the paper's figures/behavior/behavior.csv (deterministic preprocessing, one row per session, with the path
of the session's Level-2 report) and those 840 report.json files (goals v10, Opus 5.5; Fable 5.2 for Opus 5.5's
sessions) in the trial records. Writes data/l2.csv (one row per session, normalised labels) and
includes/session-behavior-tables.md (per-model tables, included by docs/posts/04-session-behavior.md).

Labels are normalised to each goal's label set (analysis-goals-v10): a value that starts with a known label takes
it, anything else is counted as "other". Nothing is re-judged.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bug_analysis import MODEL_DISPLAY
from build_cost import FRAMEWORKS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS = os.path.abspath(os.path.join(ROOT, ".."))
ARTIFACT = os.path.join(WS, "Agent_Bench_Artifact")
PAPER = os.path.join(WS, "icpe27")

ORDER = [a for _, ms in FRAMEWORKS for a, _ in ms]
NAME = {a: MODEL_DISPLAY[a][0] for a in ORDER}
FWOF = {a: fw for fw, ms in FRAMEWORKS for a, _ in ms}

L = {
    "judgement_right": ["yes", "partly", "no", "unknown"],
    "kind": ["quantitative", "qualitative", "none"],
    "bottleneck": ["compute", "hbm-bandwidth", "cache-bandwidth", "on-chip", "latency-occupancy", "launch-overhead",
                   "other", "none"],
    "basis": ["runtime", "static", "both", "none"],
    "yesno": ["yes", "no", "unknown", "n/a"],
    "stopping": ["justified", "reason-wrong", "no-reason", "continued-without-gain"],
    "g4": ["strategy", "mixed", "search", "none"],
    "focus": ["why-good", "why-weak", "both", "why-failed"],
    "g8": ["no-finding", "suspect", "confirmed-hack"],
    "g9": ["same-precision", "lower-accepted-within-tolerance", "loss-caught", "loss-not-caught", "higher-precision",
           "unknown"],
    "g11": ["guided", "mixed", "brute-force", "none"],
    "family": ["library call", "shared-memory tiling", "register reuse or streaming", "wide or coalesced loads",
               "buffers and storage layout", "fusion of passes or steps", "thread and block mapping",
               "synchronisation and contention", "reduction scheme", "algebraic or algorithmic rewrite",
               "precision and numerics", "hardware-specific instruction or feature", "launch overhead",
               "size-dependent parameter or path", "compiler-level code tuning"],
    "g5kind": ["compile or link", "tool or environment", "wrong result", "latent defect or edge case",
               "crash or runtime error", "timeout or hang", "kernel launch or resource limit", "slower or no gain",
               "measurement problem", "wrong belief or prediction"],
    "g5resp": ["diagnosed, then fixed", "changed without a stated diagnosis", "reverted or dropped",
               "worked around or accepted", "ignored or not noticed", "nothing needed"],
    "theme": ["performance model or bound", "measurement at the scored scale", "measurement discipline",
              "correctness testing", "library use", "kernel design", "acceptance and precision", "process and time",
              "device knowledge", "no optimisation attempted"],
    "g10basis": ["input values are bounded", "only part of the results is required as output",
                 "another stated condition", "code path our tests never ran"],
}


def lab(x, allowed: list[str]) -> str:
    """Normalise a label value: dicts give their 'label'; a string takes the first allowed label it starts with."""
    if isinstance(x, dict):
        x = x.get("label", x.get("verdict", ""))
    s = str(x if x is not None else "").strip().strip("`*").lower()
    for a in sorted(allowed, key=len, reverse=True):
        if s.startswith(a):
            return a
    return "other"


def sessions() -> list[dict]:
    rows = []
    for b in csv.DictReader(open(os.path.join(PAPER, "figures/behavior/behavior.csv"))):
        g = json.load(open(os.path.join(ARTIFACT, b["analyzer"])))["goals"]
        g3, pm = g["G3"], g["G3"]["performance_model"]
        fb = pm.get("final_bottleneck")
        d = {
            "agent": b["agent"], "model": NAME[b["agent"]], "kernel": b["kernel"], "state": b["state"],
            "g1_techniques": len(g["G1"]["techniques"]),
            "g1_kept_families": ";".join(sorted({lab(t.get("family"), L["family"]) for t in g["G1"]["techniques"]
                                                if isinstance(t, dict) and lab(t.get("kept"), ["yes", "partly"]) != "other"})),
            "g2_judgement_right": lab(g["G2"]["judgement_right"], L["judgement_right"]),
            "g3_kind": lab(pm.get("kind"), L["kind"]),
            "g3_bottleneck": lab(fb.get("class") if isinstance(fb, dict) else fb, L["bottleneck"]),
            "g3_basis_source": lab(pm.get("basis_source"), L["basis"]),
            "g3_helped": lab(g3["helped"], L["yesno"]),
            "g3_final_check_agrees": lab(g3["final_check"].get("agrees") if isinstance(g3["final_check"], dict)
                                         else g3["final_check"], L["yesno"]),
            "g3_continued_without_gain": lab(g3["continued_without_gain"], L["yesno"]),
            "g3_agent_stated_done": lab(g3["agent_judgement"].get("stated") if isinstance(g3["agent_judgement"], dict)
                                        else g3["agent_judgement"], L["yesno"]),
            "g3_room_left": lab(g3["room_left"], L["yesno"]),
            "g3_stopping": lab(g3["stopping"], L["stopping"]),
            "g4": lab(g["G4"]["label"], L["g4"]),
            "g5_errors": len(g["G5"]["errors"]),
            "g5_kinds": ";".join(lab(e.get("kind_label"), L["g5kind"]) for e in g["G5"]["errors"] if isinstance(e, dict)),
            "g5_responses": ";".join(lab(e.get("response_label"), L["g5resp"]) for e in g["G5"]["errors"] if isinstance(e, dict)),
            "g5_dead_ends": len(g["G5"]["dead_ends"]),
            "g6_strengths": ";".join(lab(e.get("theme"), L["theme"]) for e in g["G6"]["strengths"] if isinstance(e, dict)),
            "g6_weaknesses": ";".join(lab(e.get("theme"), L["theme"]) for e in g["G6"]["weaknesses"] if isinstance(e, dict)),
            "g7_focus": lab(g["G7"]["focus"], L["focus"]),
            "g7_catchable": lab(g["G7"]["catchable_by_agent"], L["yesno"]),
            "g8_verdict": lab(g["G8"]["verdict"], L["g8"]),
            "g9_acceptance": lab(g["G9"]["acceptance"], L["g9"]),
            "g10_changes": len(g["G10"]["changes"]),
            "g10_not_general": sum(1 for c in g["G10"]["changes"] if isinstance(c, dict) and lab(c.get("general"), L["yesno"]) == "no"),
            "g10_conditions": ";".join(lab(c.get("basis"), L["g10basis"]) for c in g["G10"]["task_conditions_relied_on"] if isinstance(c, dict)),
            "g10_bugs": len(g["G10"]["bugs_on_allowed_inputs"]),
            "g11_verdict": lab(g["G11"]["verdict"], L["g11"]),
        }
        cnt = g["G11"].get("counts") or {}
        for k in ("necessary", "narrowable", "avoidable"):
            try:
                d[f"g11_{k}"] = int(cnt.get(k, 0) or 0)
            except (TypeError, ValueError):
                d[f"g11_{k}"] = 0
        for k in ("steps_total", "steps_implement", "steps_debug", "steps_check", "steps_perf-measure",
                  "steps_perf-analysis", "steps_env-probe", "steps_read-task", "steps_wrap-up", "steps_other",
                  "shell_steps", "enum_steps", "compactions", "actors"):
            d[k] = int(float(b[k] or 0))
        for k in ("active_s", "llm_s", "tool_s"):
            d[k] = float(b[k] or 0)
        d["ident"] = b["ident"]
        rows.append(d)
    with open(os.path.join(ROOT, "data", "l2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def table(rows, cols, cell, caption, total=None, first="Model"):
    """cols: list of (header, key); cell(sub, key) -> str for a model's 30 sessions; total(rows, key) for the last row."""
    out = ['<div class="njs-table" markdown="1">', "",
           "| " + first + " | " + " | ".join(h for h, _ in cols) + " |",
           "| :--- | " + " | ".join("---:" for _ in cols) + " |"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["agent"]].append(r)
    for a in ORDER:
        out.append(f"| {NAME[a]} | " + " | ".join(cell(by[a], k) for _, k in cols) + " |")
    tf = total or cell
    out.append("| **All** | " + " | ".join(f"**{tf(rows, k)}**" for _, k in cols) + " |")
    out += ["", "</div>", "", f"*{caption}*", ""]
    return "\n".join(out)


def count(field, value):
    return lambda sub, _k=None: str(sum(1 for r in sub if r[field] == value) or "·")


def counter_cell(field):
    def f(sub, key):
        n = sum(r[field].split(";").count(key) for r in sub if r[field])
        return str(n or "·")
    return f


def mean(field, fmt="{:.0f}"):
    return lambda sub, _k=None: fmt.format(sum(r[field] for r in sub) / len(sub))


def write_matrix_data(rows: list[dict]) -> None:
    """docs/javascripts/session-matrix-data.js: one record per session for the interactive matrix."""
    att = {(r["agent"], r["kernel"]): r for r in csv.DictReader(open(os.path.join(PAPER, "figures/attainment/attainment.csv")))}
    cats = [("linear-algebra/blas", "BLAS"), ("linear-algebra/kernels", "Kernels"), ("linear-algebra/solvers", "Solvers"),
            ("datamining", "Data"), ("medley", "Medley"), ("stencils", "Stencils")]
    kern = {}
    for (a, k), r in att.items():
        kern.setdefault(k, (r["category"], r["case"]))
    kernels = []
    for c, lab_ in cats:
        for k in sorted([k for k, v in kern.items() if v[0] == c], key=lambda k: kern[k][1]):
            kernels.append([k, lab_])
    keep = ["g2_judgement_right", "g3_kind", "g3_bottleneck", "g3_helped", "g3_final_check_agrees", "g3_stopping",
            "g3_continued_without_gain", "g4", "g7_focus", "g8_verdict", "g9_acceptance", "g10_not_general", "g11_verdict",
            "g5_errors", "g5_dead_ends", "g1_techniques"]
    S = {}
    for r in rows:
        a_ = att[(r["agent"], r["kernel"])]
        d = {k: r[k] for k in keep}
        d.update(state=r["state"], att=round(float(a_["attainment"] or 0), 4), alg=a_["algorithm_changed"] == "True",
                 min=round(r["active_s"] / 60, 1), steps=r["steps_total"],
                 pa=round((r["steps_perf-analysis"] + r["steps_perf-measure"]) / max(1, r["steps_total"]), 3),
                 enum=round(r["enum_steps"] / r["shell_steps"], 3) if r["shell_steps"] else 0.0,
                 llm=round(r["llm_s"] / max(1, r["active_s"]), 3), compact=r["compactions"], ident=r["ident"] == "named")
        S[r["agent"] + "|" + r["kernel"]] = d
    data = {"frameworks": [[fw, [[a, NAME[a]] for a, _ in ms]] for fw, ms in FRAMEWORKS], "kernels": kernels, "s": S}
    open(os.path.join(ROOT, "docs", "javascripts", "session-matrix-data.js"), "w").write(
        "window.NJS_SESSIONS = " + json.dumps(data, separators=(",", ":")) + ";\n")


def write_errors_data(rows: list[dict]) -> None:
    """docs/javascripts/errors-matrix-data.js: per model, G5 error counts by kind and response."""
    out = {}
    for r in rows:
        ks = r["g5_kinds"].split(";") if r["g5_kinds"] else []
        rs = r["g5_responses"].split(";") if r["g5_responses"] else []
        m = out.setdefault(r["agent"], {"sessions": 0, "dead": 0, "c": {}})
        m["sessions"] += 1
        m["dead"] += r["g5_dead_ends"]
        for k, x in zip(ks, rs):
            m["c"].setdefault(k, {}).setdefault(x, 0)
            m["c"][k][x] += 1
    # bugs left at the end, per model: level-1 wrong output or crash, and Bug Verifier failures
    for r in csv.DictReader(open(os.path.join(PAPER, "figures/attainment/attainment.csv"))):
        if r["state"] == "pass" or r["agent"] not in out:
            continue
        f = out[r["agent"]].setdefault("final", {"wrong": 0, "crash": 0, "bv": 0})
        if r["status"].startswith("taken out"):
            f["bv"] += 1
        elif r["reason"].strip() == "wrong-output":
            f["wrong"] += 1
        elif r["reason"].strip() == "crash":
            f["crash"] += 1
    data = {"frameworks": [[fw, [[a, NAME[a]] for a, _ in ms]] for fw, ms in FRAMEWORKS],
            "kinds": L["g5kind"], "responses": L["g5resp"], "m": out}
    open(os.path.join(ROOT, "docs", "javascripts", "errors-matrix-data.js"), "w").write(
        "window.NJS_ERRORS = " + json.dumps(data, separators=(",", ":")) + ";\n")


SPEED_DIR = os.path.join(ARTIFACT, "notes", "2026-10-29-model-speed")
# which fit holds each model's trials of record (models-static.json source note): round 020 and Carbon, Fable 5.2,
# Argon, Kimi K3.1 Pro (round 022); all others the first passive run
SPEED_FILE = {"gemini-4-neon-xhigh-antigravity": "speed-passive-round020.json", "gemini-4-carbon-high-antigravity": "speed-passive-round020.json",
              "astra-6-2-codex-cli": "speed-passive-round020.json", "sol-6-2-codex-cli": "speed-passive-round020.json",
              "fable-5-2-claude-code": "speed-passive-fable52.json", "gemini-4-argon-high-antigravity": "speed-passive-gemini.json",
              "kimi-k3-1-pro-opencode": "speed-passive-round022.json"}


def speed_table() -> str:
    """includes/decoding-speed-table.md: the passive fit of each model (decode rate D with its 5-95% bootstrap range,
    fixed latency a, quality, requests used, plain output rate)."""
    out = ['<div class="njs-table" markdown="1">', "",
           "| Model | Framework | Decoding speed D (tok/s) | 90% range | Fixed latency a (s) | Plain output rate (tok/s) | Requests fitted | Fit quality |",
           "| :--- | :--- | ---: | :---: | ---: | ---: | ---: | :---: |"]
    for a in ORDER:
        d = json.load(open(os.path.join(SPEED_DIR, SPEED_FILE.get(a, "speed-passive.json"))))
        v = d.get("agents", d)[a]
        fit = v["fit"]; dec = fit["decode"]; p5, _, p95 = dec["rate_p5_p50_p95"]
        eff = v.get("effective_output_rate") or {}
        q = fit.get("quality") or v.get("quality") or ""
        out.append(f"| {NAME[a]} | {FWOF[a]} | {dec['rate_tok_s']:.0f} | {p5:.0f}–{p95:.0f} | {fit['a']['a_s']:.2f} | "
                   f"{eff.get('p50', 0):.0f} | {v['n_requests_trusted']:,} | {q} |")
    out += ["", "</div>", ""]
    return "\n".join(out)


def main() -> None:
    rows = sessions()
    md = []

    def H(t):
        md.append(f"\n### {t}\n")

    # ---- session statistics
    md.append("## Session Statistics\n")
    md.append("Measured by deterministic trace preprocessing, without an LLM. Each value is a per-session mean over the model's 30 sessions unless stated otherwise.\n")
    H("Steps by Activity")
    acts = [("implement", "Implement"), ("debug", "Debug"), ("check", "Check"), ("perf-measure", "Perf. measure"),
            ("perf-analysis", "Perf. analysis"), ("env-probe", "Env. probe"), ("read-task", "Read task"),
            ("wrap-up", "Wrap-up"), ("other", "Other")]
    def share(sub, k):
        tot = sum(r["steps_total"] for r in sub)
        return f"{100 * sum(r['steps_' + k] for r in sub) / tot:.0f}%" if tot else "·"
    cols = [("Steps", "_steps")] + [(h, k) for k, h in acts]
    md.append(table(rows, cols, lambda sub, k: mean("steps_total")(sub) if k == "_steps" else share(sub, k),
                    "Table 1. Mean steps per session and the share of steps by activity."))
    H("Time, Tools, and Context")
    def tcell(sub, k):
        n = len(sub)
        if k == "active":
            return f"{sum(r['active_s'] for r in sub) / n / 60:.0f}"
        if k == "llm":
            return f"{100 * sum(r['llm_s'] for r in sub) / max(1, sum(r['active_s'] for r in sub)):.0f}%"
        if k == "tool":
            return f"{100 * sum(r['tool_s'] for r in sub) / max(1, sum(r['active_s'] for r in sub)):.0f}%"
        if k == "shell":
            return f"{sum(r['shell_steps'] for r in sub) / n:.0f}"
        if k == "enum":
            s = sum(r["shell_steps"] for r in sub)
            return f"{100 * sum(r['enum_steps'] for r in sub) / s:.0f}%" if s else "·"
        if k == "compact":
            return str(sum(1 for r in sub if r["compactions"] > 0) or "·")
        if k == "ident":
            return str(sum(1 for r in sub if r["ident"] == "named") or "·")
    md.append(table(rows, [("Active (min)", "active"), ("Waiting on LLM", "llm"), ("Running tools", "tool"),
                           ("Shell steps", "shell"), ("Enumeration share", "enum"),
                           ("Sessions with compaction", "compact"), ("Sessions naming the kernel", "ident")],
                    tcell, "Table 2. Mean active time, its split between waiting on the model and running tools (the two can overlap), mean shell steps, the share of shell steps that enumerate parameters, and the number of sessions (of 30) with at least one context compaction and that name the PolyBench kernel."))

    # ---- Level-2 labels
    md.append("\n## Analyzer Labels by Goal\n")
    md.append("Counts of sessions (of 30 per model) with each label, from the Level-2 Analyzer's reports; the goals are defined on the [Evaluation Framework](../framework.md#analyzer-goals) page. A dot means zero.\n")
    def cat_table(field, labels, caption):
        return table(rows, [(l, l) for l in labels], lambda sub, k: count(field, k)(sub), caption)
    H("G2: Correctness Judgement")
    md.append(cat_table("g2_judgement_right", L["judgement_right"], "Table 3. Whether the agent's own correctness judgement was right."))
    H("G3: Performance Model")
    md.append(cat_table("g3_kind", L["kind"], "Table 4. Kind of performance model: quantitative only when the agent compares its numbers with an upper bound."))
    md.append(cat_table("g3_bottleneck", L["bottleneck"], "Table 5. The final bottleneck the agent names for its solution at the large sizes."))
    md.append(cat_table("g3_basis_source", L["basis"], "Table 6. Where a quantitative model's numbers come from: runtime measurements, static analysis, or both."))
    md.append(table(rows, [("Helped", "h"), ("Final check agrees", "a"), ("Final check disagrees", "d")],
                    lambda sub, k: count({"h": "g3_helped", "a": "g3_final_check_agrees", "d": "g3_final_check_agrees"}[k],
                                         {"h": "yes", "a": "yes", "d": "no"}[k])(sub),
                    "Table 7. Sessions where the bottleneck analysis drove a kept, measured gain, and where the final result reaches 50% of the named resource's peak (or 30% of the agent's latency model)."))
    H("G3: Stopping")
    md.append(table(rows, [("Said it was done", "s"), ("Continued without gain", "c"), ("Room left (Analyzer)", "r")] +
                    [(l, l) for l in L["stopping"]],
                    lambda sub, k: count({"s": "g3_agent_stated_done", "c": "g3_continued_without_gain", "r": "g3_room_left"}[k], "yes")(sub)
                    if k in ("s", "c", "r") else count("g3_stopping", k)(sub),
                    "Table 8. The agent's own stopping judgement, the tail without gain, the Analyzer's judgement of room left, and the stopping label: justified only when the agent says it is done with a reason that holds and did not keep trying without gain."))
    H("G4: Strategy")
    md.append(cat_table("g4", L["g4"], "Table 9. Strategy (hypothesis, change, measurement, decision), undirected search, a mix, or no optimization attempted."))
    H("G1: Optimization Techniques Kept")
    md.append(table(rows, [("Techniques tried", "_n")] + [(f, f) for f in L["family"]],
                    lambda sub, k: mean("g1_techniques")(sub) if k == "_n" else str(sum(1 for r in sub if k in r["g1_kept_families"].split(";")) or "·"),
                    "Table 10. Mean techniques tried per session, and the number of sessions whose final solution keeps at least one technique of each family."))
    H("G5: Errors and Dead Ends")
    md.append(table(rows, [("Errors", "_e"), ("Dead ends", "_d")] + [(k, k) for k in L["g5kind"]],
                    lambda sub, k: mean("g5_errors")(sub) if k == "_e" else mean("g5_dead_ends", "{:.1f}")(sub) if k == "_d" else counter_cell("g5_kinds")(sub, k),
                    "Table 11. Mean errors and dead ends per session, and the total number of errors of each kind."))
    md.append(table(rows, [(k, k) for k in L["g5resp"]], counter_cell("g5_responses"),
                    "Table 12. The agent's response to each error, totals."))
    H("G6: Strengths and Weaknesses")
    md.append(table(rows, [(k, k) for k in L["theme"]], counter_cell("g6_strengths"), "Table 13. Strengths by theme, totals."))
    md.append(table(rows, [(k, k) for k in L["theme"]], counter_cell("g6_weaknesses"), "Table 14. Weaknesses by theme, totals."))
    H("G7: Verification Result")
    md.append(table(rows, [(l, l) for l in L["focus"]] + [("Failure catchable by the agent", "_c")],
                    lambda sub, k: count("g7_catchable", "yes")(sub) if k == "_c" else count("g7_focus", k)(sub),
                    "Table 15. Focus of the explanation of the level-1 result (why the solution failed, is weak, is good, or both), and sessions where a check the agent ran could have caught the failure or weakness the Analyzer identifies."))
    H("G8 to G10: Legitimacy, Precision, and Generality")
    md.append(table(rows, [(l, l) for l in L["g8"]] + [(l, l) for l in L["g9"]],
                    lambda sub, k: count("g8_verdict", k)(sub) if k in L["g8"] else count("g9_acceptance", k)(sub),
                    "Table 16. G8 reward-hacking verdict and G9 precision acceptance."))
    md.append(table(rows, [("Algorithmic changes", "_c"), ("Not general", "_n"), ("Bugs on allowed inputs", "_b")] + [(k, k) for k in L["g10basis"]],
                    lambda sub, k: str(sum(r["g10_changes"] for r in sub) or "·") if k == "_c" else str(sum(r["g10_not_general"] for r in sub) or "·") if k == "_n"
                    else str(sum(r["g10_bugs"] for r in sub) or "·") if k == "_b" else counter_cell("g10_conditions")(sub, k),
                    "Table 17. G10 totals: algorithmic changes the Analyzer marks (any change of loop structure or algorithm, a broader notion than Analysis 2's verified algorithm changes), those judged not general, bugs claimed on allowed inputs, and task conditions relied on by kind."))
    H("G11: Enumeration")
    md.append(table(rows, [(l, l) for l in L["g11"]] + [("Necessary", "necessary"), ("Narrowable", "narrowable"), ("Avoidable", "avoidable")],
                    lambda sub, k: count("g11_verdict", k)(sub) if k in L["g11"] else str(sum(r[f"g11_{k}"] for r in sub) or "·"),
                    "Table 18. G11 verdict per session, and the total enumerations judged necessary, narrowable by an analysis, or avoidable."))
    write_matrix_data(rows)
    write_errors_data(rows)
    os.makedirs(os.path.join(ROOT, "includes"), exist_ok=True)
    open(os.path.join(ROOT, "includes", "session-behavior-tables.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(ROOT, "includes", "decoding-speed-table.md"), "w").write(speed_table())
    print(len(rows), "sessions")


if __name__ == "__main__":
    main()
