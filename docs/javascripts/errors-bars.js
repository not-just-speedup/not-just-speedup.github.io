/* Per-model stacked bars of G5 errors: bar length = errors per session, segments = the agent's response.
   Used on two pages with different error kinds: <div data-njs-errors="kind1,kind2,..." data-final="1">.
   Data: errors-matrix-data.js (scripts/build_l2.py). */
(function () {
  const D = window.NJS_ERRORS;
  const roots = document.querySelectorAll("[data-njs-errors]");
  if (!D || !roots.length) return;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

  // plain-language names for the Analyzer's labels (analysis-goals-v10, G5)
  const KIND = {
    "wrong result": "Wrong result", "latent defect or edge case": "Hidden defect or edge case", "crash or runtime error": "Crash",
    "timeout or hang": "Hang or timeout", "kernel launch or resource limit": "Launch or resource limit",
    "compile or link": "Compile or link error", "tool or environment": "Tool or session failure", "slower or no gain": "Slower or no gain",
    "measurement problem": "Invalid own measurement", "wrong belief or prediction": "Hypothesis refuted by own data",
  };
  const RESP = [
    ["diagnosed, then fixed", "Fixed after diagnosis", "#1E6F54"],
    ["changed without a stated diagnosis", "Changed without diagnosis", "#7BCFAE"],
    ["reverted or dropped", "Reverted or dropped", "#9EC4E6"],
    ["worked around or accepted", "Worked around or accepted", "#9CA3AF"],
    ["nothing needed", "No action needed", "#E5E7EB"],
    ["ignored or not noticed", "Ignored or not noticed", "#E8875A"],
  ];

  function make(root, kinds, withFinal) {
    let sel = "_all";
    const bar = document.createElement("div"); bar.className = "njs-views";
    bar.innerHTML = [["_all", "All of these"]].concat(kinds.map((k) => [k, KIND[k] || k]))
      .map(([k, l]) => `<button type="button" data-v="${esc(k)}" aria-pressed="${k === sel}">${esc(l)}</button>`).join("");
    const legend = document.createElement("div"); legend.className = "njs-mlegend";
    legend.innerHTML = RESP.map(([, l, c]) => `<span><span class="sw" style="background:${c}"></span>${esc(l)}</span>`).join("") +
      '<span><span class="sw" style="background:#F3F4F6"></span>Other response</span>';
    const box = document.createElement("div"); box.className = "njs-ebars";
    root.append(bar, legend, box);

    function counts(c, ks) {
      const out = {}; let n = 0;
      ks.forEach((k) => Object.entries(c[k] || {}).forEach(([r, v]) => { out[r] = (out[r] || 0) + v; n += v; }));
      return [out, n];
    }
    function render() {
      const ks = sel === "_all" ? kinds : [sel];
      const rows = []; let max = 0;
      const tot = {}; let totN = 0, totS = 0;
      D.frameworks.forEach(([fw, ms]) => ms.forEach(([a, name], i) => {
        const m = D.m[a]; if (!m) return;
        const [cnt, n] = counts(m.c, ks); const per = n / m.sessions; max = Math.max(max, per);
        Object.entries(cnt).forEach(([r, v]) => (tot[r] = (tot[r] || 0) + v)); totN += n; totS += m.sessions;
        rows.push({ fw: i === 0 ? fw : "", name, cnt, n, per, final: m.final || { wrong: 0, crash: 0, bv: 0 } });
      }));
      const head = `<div class="eh"><span></span><span>Per session, split by the agent's response</span>${withFinal ? "<span>Left at the end</span>" : ""}</div>`;
      const seg = (cnt, n, per) => {
        let x = "", used = 0;
        RESP.forEach(([r, l, c]) => { const v = cnt[r] || 0; if (!v) return; used += v;
          x += `<i style="width:${(100 * per * v / n / max).toFixed(2)}%;background:${c}" title="${esc(l)}: ${v} (${Math.round(100 * v / n)}%)"></i>`; });
        const o = n - used; if (o > 0) x += `<i style="width:${(100 * per * o / n / max).toFixed(2)}%;background:#F3F4F6" title="Other response: ${o}"></i>`;
        return x;
      };
      let h = head;
      rows.forEach((r) => {
        const fin = r.final.wrong + r.final.crash + r.final.bv;
        h += (r.fw ? `<div class="efw">${esc(r.fw)}</div>` : "") +
          `<div class="er"><span class="en">${esc(r.name)}</span><span class="eb" title="${esc(r.name)}: ${r.n} errors in 30 sessions">${seg(r.cnt, r.n, r.per)}<b>${r.per.toFixed(1)}</b></span>` +
          (withFinal ? `<span class="ef" title="Level-1 wrong output ${r.final.wrong}, crash ${r.final.crash}; Bug Verifier ${r.final.bv}">${fin ? "■".repeat(fin) : "·"}</span>` : "") + "</div>";
      });
      const allFinal = rows.reduce((s, r) => s + r.final.wrong + r.final.crash + r.final.bv, 0);
      h += `<div class="er et"><span class="en">All models</span><span class="eb" title="${totN} errors in ${totS} sessions">${seg(tot, totN, Math.min(max, totN / totS))}<b>${(totN / totS).toFixed(1)}</b></span>` +
        (withFinal ? `<span class="ef">${allFinal} in total</span>` : "") + "</div>";
      box.innerHTML = h;
      bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === sel));
    }
    bar.addEventListener("click", (e) => { const b = e.target.closest("button[data-v]"); if (b) { sel = b.dataset.v; render(); } });
    render();
  }
  roots.forEach((r) => make(r, r.dataset.njsErrors.split(","), r.dataset.final === "1"));
})();
