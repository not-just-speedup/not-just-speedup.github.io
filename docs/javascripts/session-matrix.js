/* Interactive session matrix for the Session Behavior page: 30 kernels x 28 models, one cell per session,
   coloured by the chosen Level-2 label or session statistic. Data: session-matrix-data.js (scripts/build_l2.py). */
(function () {
  const D = window.NJS_SESSIONS;
  const roots = document.querySelectorAll("[data-njs-matrix]");
  if (!D || !roots.length) return;

  const P = {
    mintD: "#1E6F54", mint: "#3EB489", mintL: "#B5EAD7", blue: "#5B8DD6", blueL: "#B9CEF0", peach: "#E8A87C",
    amber: "#C2410C", purple: "#8B6FB8", grey: "#9CA3AF", greyL: "#E5E7EB", none: "#F3F4F6",
  };
  const ramp = (stops) => (x) => {
    x = Math.max(0, Math.min(1, x)); const n = stops.length - 1, i = Math.min(n - 1, Math.floor(x * n)), f = x * n - i;
    const a = stops[i], b = stops[i + 1], h = (s, k) => parseInt(s.slice(1 + 2 * k, 3 + 2 * k), 16);
    return "#" + [0, 1, 2].map((k) => Math.round(h(a, k) * (1 - f) + h(b, k) * f).toString(16).padStart(2, "0")).join("");
  };
  const blues = ramp(["#F2F7FC", "#CFE2F3", "#9EC4E6", "#629DD1", "#2F6FB3", "#174478"]);
  const mints = ramp(["#F2FAF6", "#CDEFE1", "#9ADBC0", "#63C29E", "#3EB489", "#1E6F54"]);
  const log01 = (v, lo, hi) => (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  const cat = (map) => (c, k) => (map[c[k]] || [null, P.none])[1];
  const legendOf = (map) => Object.values(map);

  const STRAT = { strategy: ["Strategy", P.mintD], mixed: ["Mixed", P.mintL], search: ["Search", P.peach], none: ["No optimization attempted", P.greyL], other: ["Other", P.none] };
  const KIND = { quantitative: ["Quantitative model", P.mintD], qualitative: ["Qualitative only", P.mintL], none: ["No model", P.greyL] };
  const BN = { compute: ["Compute", P.mintD], "hbm-bandwidth": ["HBM bandwidth", P.blue], "cache-bandwidth": ["L2 / cache bandwidth", P.blueL],
    "on-chip": ["On-chip (shared memory, registers)", P.purple], "latency-occupancy": ["Latency / occupancy", P.peach],
    "launch-overhead": ["Launch overhead", P.amber], other: ["Other", P.grey], none: ["No bottleneck analysis", P.greyL] };
  const STOP = { justified: ["Justified", P.mintD], "reason-wrong": ["Reason wrong", P.peach], "no-reason": ["No reason", P.greyL],
    "continued-without-gain": ["Continued without gain", P.amber] };
  const JUDGE = { yes: ["Right", P.mintD], partly: ["Partly", P.peach], no: ["Wrong", P.amber], unknown: ["Unknown", P.greyL], other: ["Other", P.none] };
  const ENUM = { guided: ["Guided", P.mintD], mixed: ["Mixed", P.mintL], "brute-force": ["Brute force", P.amber], none: ["No enumeration", P.greyL] };
  const LEGIT = { "no-finding": ["G8 no finding", P.mintL], suspect: ["G8 suspect", P.peach], "confirmed-hack": ["G8 confirmed hack", P.amber] };
  const FOCUS = { "why-good": ["Why good", P.mintD], both: ["Both", P.mintL], "why-weak": ["Why weak", P.peach], "why-failed": ["Why failed", P.amber] };

  const VIEWS = {
    strategy: { label: "Strategy · G4", fill: cat(STRAT), key: "g4", legend: legendOf(STRAT),
      dot: (c) => (c.g3_stopping === "justified" ? P.amber : null), dotLabel: "Dot: stop justified (G3)",
      note: "Strategy: a stated hypothesis, a change to test it, a measurement read against it, and the next step chosen from the result. Search: undirected trial and error." },
    model: { label: "Performance model · G3", fill: cat(KIND), key: "g3_kind", legend: legendOf(KIND),
      dot: (c) => (c.g3_helped === "yes" ? P.amber : null), dotLabel: "Dot: the analysis drove a kept, measured gain",
      note: "Quantitative only when the agent compares its numbers with an upper bound; timings alone are qualitative." },
    bottleneck: { label: "Final bottleneck · G3", fill: cat(BN), key: "g3_bottleneck", legend: legendOf(BN),
      dot: (c) => (c.g3_final_check_agrees === "yes" ? P.amber : null), dotLabel: "Dot: the final result reaches 50% of that resource's peak",
      note: "The bottleneck the agent names for its final solution at the large sizes." },
    stopping: { label: "Stopping · G3", fill: cat(STOP), key: "g3_stopping", legend: legendOf(STOP),
      note: "Justified only when the agent says it is done, gives a reason that holds, and did not keep trying without gain." },
    judge: { label: "Correctness judgement · G2", fill: cat(JUDGE), key: "g2_judgement_right", legend: legendOf(JUDGE),
      note: "Whether the agent's own judgement that its solution was correct was right." },
    enum: { label: "Enumeration · G11", fill: cat(ENUM), key: "g11_verdict", legend: legendOf(ENUM),
      note: "Guided when most parameter sweeps were necessary or narrowed by an analysis; brute force when most were avoidable." },
    legit: { label: "Legitimacy · G8–G10", fill: cat(LEGIT), key: "g8_verdict", legend: legendOf(LEGIT),
      dot: (c) => (c.g9_acceptance === "lower-accepted-within-tolerance" ? P.blue : null), dotLabel: "Dot: lower precision accepted within tolerance (G9)",
      tri: (c) => (c.g10_not_general > 0 ? P.amber : null), triLabel: "Corner: a change judged not general (G10)",
      note: "G8 reward-hacking review, with precision (G9) and generality (G10) marks." },
    focus: { label: "How it judges performance · G7", fill: cat(FOCUS), key: "g7_focus", legend: legendOf(FOCUS),
      note: "What the Analyzer explains about the level-1 result, read from the solution's speed relative to the fastest solution of the kernel." },
    time: { label: "Active time", fill: (c) => blues(log01(c.min, 1, 1000)), text: (c) => (c.min >= 100 ? Math.round(c.min) : c.min >= 10 ? Math.round(c.min) : c.min.toFixed(0)),
      legend: [["1 min", blues(0)], ["10", blues(1 / 3)], ["100", blues(2 / 3)], ["1,000 min", blues(1)]], note: "Active session time T_e2e in minutes, log scale." },
    perf: { label: "Performance steps", fill: (c) => mints(c.pa / 0.5), text: (c) => Math.round(100 * c.pa),
      legend: [["0%", mints(0)], ["25%", mints(0.5)], ["≥50% of steps", mints(1)]], note: "Share of steps spent measuring or analyzing performance (deterministic preprocessing)." },
    enumshare: { label: "Enumeration share", fill: (c) => mints(c.enum / 0.5), text: (c) => Math.round(100 * c.enum),
      legend: [["0%", mints(0)], ["25%", mints(0.5)], ["≥50% of shell steps", mints(1)]], note: "Share of shell steps that enumerate parameters (deterministic preprocessing)." },
    steps: { label: "Steps", fill: (c) => blues(log01(c.steps, 10, 3000)), text: (c) => (c.steps >= 1000 ? (c.steps / 1000).toFixed(1) + "k" : c.steps),
      legend: [["10 steps", blues(0)], ["100", blues(0.4)], ["1,000", blues(0.8)], ["3,000", blues(1)]], note: "Steps in the session after preprocessing, log scale." },
    llm: { label: "Waiting on the LLM", fill: (c) => blues(c.llm), text: (c) => Math.round(100 * c.llm),
      legend: [["0%", blues(0)], ["50%", blues(0.5)], ["100% of active time", blues(1)]], note: "Share of active time spent waiting on the model rather than running tools." },
    compact: { label: "Context compactions", fill: (c) => (c.compact ? mints(Math.min(1, c.compact / 6)) : P.none), text: (c) => (c.compact || ""),
      legend: [["None", P.none], ["1", mints(1 / 6)], ["≥6", mints(1)]], note: "Number of times the harness compacted the context." },
    ident: { label: "Names the kernel", fill: (c) => (c.ident ? P.mintD : P.greyL),
      legend: [["Names the PolyBench kernel", P.mintD], ["Does not", P.greyL]], note: "Whether the agent's notes or source name the PolyBench kernel it was given." },
    outcome: { label: "Outcome", fill: (c) => (c.state === "pass" ? (c.att >= 1 ? P.mintD : P.mintL) : "#1F2937"),
      legend: [["Passed, above the bound", P.mintD], ["Passed, below the bound", P.mintL], ["Failed", "#1F2937"]],
      tri: (c) => (c.alg ? P.amber : null), triLabel: "Corner: algorithm change", note: "Final outcome after Bug Verification." },
  };

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
  function make(root, keys) {
    let view = keys[0];
    const bar = document.createElement("div"); bar.className = "njs-views"; bar.setAttribute("role", "group");
    bar.innerHTML = keys.map((k) => [k, VIEWS[k]]).map(([k, v]) => `<button type="button" data-v="${k}" aria-pressed="${k === view}">${esc(v.label)}</button>`).join("");
    const legend = document.createElement("div"); legend.className = "njs-mlegend";
    const note = document.createElement("p"); note.className = "njs-mnote";
    const wrap = document.createElement("div"); wrap.className = "njs-mwrap";
    root.append(bar, legend, note, wrap);

    function tip(name, k, c) {
      return `${name} · ${k} · ${c.state}, attainment ${Math.round(100 * c.att)}%${c.alg ? ", algorithm change" : ""} · ${c.min} min, ${c.steps} steps` +
        ` · G2 ${c.g2_judgement_right} · G3 ${c.g3_kind}, bottleneck ${c.g3_bottleneck}, final check ${c.g3_final_check_agrees}, helped ${c.g3_helped}, stop ${c.g3_stopping}` +
        ` · G4 ${c.g4} · G7 ${c.g7_focus} · G8 ${c.g8_verdict} · G9 ${c.g9_acceptance} · G10 not general ${c.g10_not_general} · G11 ${c.g11_verdict}` +
        ` · errors ${c.g5_errors}, dead ends ${c.g5_dead_ends}, techniques ${c.g1_techniques}`;
    }

    function render() {
      const v = VIEWS[view];
      let h = '<table class="njs-matrix"><thead><tr><th></th><th></th>' +
        D.frameworks.map(([fw, ms]) => `<th class="fw" colspan="${ms.length}">${esc(fw)}</th>`).join("") + "</tr><tr><th></th><th></th>";
      D.frameworks.forEach(([fw, ms]) => ms.forEach(([a, n], i) => (h += `<th class="m${i === 0 ? " sep" : ""}"><span>${esc(n)}</span></th>`)));
      h += "</tr></thead><tbody>";
      let lastCat = null;
      for (const [k, catName] of D.kernels) {
        const first = catName !== lastCat; lastCat = catName;
        h += `<tr${first ? ' class="cat"' : ""}><td class="c">${first ? esc(catName) : ""}</td><td class="k">${esc(k)}</td>`;
        D.frameworks.forEach(([fw, ms]) => ms.forEach(([a, n], i) => {
          const c = D.s[a + "|" + k]; if (!c) { h += "<td></td>"; return; }
          const dot = v.dot && v.dot(c), tri = v.tri && v.tri(c), txt = v.text ? v.text(c) : "";
          const f = v.fill(c, v.key), dark = /^#(1|2|3|5|6)/.test(f) && f !== P.mintL;
          h += `<td${i === 0 ? ' class="sep"' : ""}><span class="cell" style="background:${f};color:${dark ? "#fff" : "#111"}" title="${esc(tip(n, k, c))}">` +
            (txt !== "" ? `<i>${esc(txt)}</i>` : "") + (dot ? `<b class="dot" style="background:${dot}"></b>` : "") +
            (tri ? `<b class="tri" style="border-color:transparent transparent transparent ${tri}"></b>` : "") + "</span></td>";
        }));
        h += "</tr>";
      }
      wrap.innerHTML = h + "</tbody></table>";
      legend.innerHTML = v.legend.map(([l, col]) => `<span><span class="sw" style="background:${col}"></span>${esc(l)}</span>`).join("") +
        (v.dotLabel ? `<span><span class="sw dotsw"></span>${esc(v.dotLabel)}</span>` : "") +
        (v.triLabel ? `<span><span class="sw trisw"></span>${esc(v.triLabel)}</span>` : "");
      note.textContent = v.note + " Hover a cell for all labels of that session.";
      bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === view));
    }
    bar.addEventListener("click", (e) => { const b = e.target.closest("button[data-v]"); if (b) { view = b.dataset.v; render(); } });
    render();

  }
  roots.forEach((r) => make(r, r.dataset.njsMatrix.split(",").filter((k) => VIEWS[k])));
})();
