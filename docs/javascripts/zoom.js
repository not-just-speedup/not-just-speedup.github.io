/* Fullscreen vector-zoom lightbox for Not-Just-Speedup figures.
   Pure white backdrop, zero external dependencies, compatible with MkDocs instant navigation. */
(function () {
  let overlay = null;
  let imgEl = null;
  let tableEl = null;
  let mode = "img";
  let openLink = null;
  let scaleLabel = null;
  let scale = 1;

  function ensureOverlay() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.className = "njs-lightbox";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.innerHTML =
      '<div class="njs-lb-bar">' +
      '  <span class="njs-lb-title"></span>' +
      '  <div class="njs-lb-actions">' +
      '    <button type="button" data-act="out" aria-label="Zoom out">-</button>' +
      '    <span class="njs-lb-scale">100%</span>' +
      '    <button type="button" data-act="in" aria-label="Zoom in">+</button>' +
      '    <button type="button" data-act="reset">Reset</button>' +
      '    <a class="njs-lb-open" target="_blank" rel="noopener">Open SVG</a>' +
      '    <button type="button" data-act="close" aria-label="Close">Close</button>' +
      "  </div>" +
      "</div>" +
      '<div class="njs-lb-stage"><img alt=""><div class="njs-lb-table md-typeset"></div></div>';
    document.body.appendChild(overlay);
    imgEl = overlay.querySelector(".njs-lb-stage img");
    tableEl = overlay.querySelector(".njs-lb-table");
    openLink = overlay.querySelector(".njs-lb-open");
    scaleLabel = overlay.querySelector(".njs-lb-scale");

    overlay.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-act]");
      if (btn) {
        const act = btn.getAttribute("data-act");
        if (act === "in") setScale(scale + 0.25);
        else if (act === "out") setScale(scale - 0.25);
        else if (act === "reset") setScale(1);
        else if (act === "close") closeBox();
        return;
      }
      if (e.target === overlay || e.target.classList.contains("njs-lb-stage")) {
        closeBox();
      }
    });

    document.addEventListener("keydown", (e) => {
      if (!overlay.classList.contains("is-open")) return;
      if (e.key === "Escape") closeBox();
      else if (e.key === "+" || e.key === "=") setScale(scale + 0.25);
      else if (e.key === "-") setScale(scale - 0.25);
      else if (e.key === "0") setScale(1);
    });
  }

  function setScale(next) {
    scale = Math.max(0.5, Math.min(3.5, Math.round(next * 100) / 100));
    if (mode === "table") {
      tableEl.style.fontSize = `${(0.9 * scale).toFixed(2)}rem`;
    } else {
      imgEl.style.width = `${Math.round(scale * 94)}vw`;
      imgEl.style.maxWidth = scale === 1 ? "1600px" : "none";
    }
    scaleLabel.textContent = `${Math.round(scale * 100)}%`;
  }

  function openTable(table, title) {
    ensureOverlay();
    mode = "table";
    imgEl.style.display = "none";
    openLink.style.display = "none";
    tableEl.style.display = "block";
    tableEl.innerHTML = "";
    tableEl.appendChild(table.cloneNode(true));
    overlay.querySelector(".njs-lb-title").textContent = title || "";
    setScale(1);
    overlay.classList.add("is-open");
    document.body.style.overflow = "hidden";
  }

  function openBox(src, alt) {
    ensureOverlay();
    mode = "img";
    imgEl.style.display = "";
    openLink.style.display = "";
    tableEl.style.display = "none";
    imgEl.src = src;
    imgEl.alt = alt || "";
    openLink.href = src;
    overlay.querySelector(".njs-lb-title").textContent = alt || "";
    setScale(1);
    overlay.classList.add("is-open");
    document.body.style.overflow = "hidden";
  }

  function closeBox() {
    if (!overlay) return;
    overlay.classList.remove("is-open");
    document.body.style.overflow = "";
  }

  function bindFigures() {
    document.querySelectorAll("figure.njs-fig").forEach((fig) => {
      const img = fig.querySelector("img");
      if (!img || fig.dataset.njsBound) return;
      fig.dataset.njsBound = "1";
      img.addEventListener("click", () => openBox(img.src, img.alt));

      const tools = document.createElement("div");
      tools.className = "njs-fig-tools";
      tools.innerHTML =
        '<button type="button" class="njs-fig-btn">Expand figure</button>' +
        `<a class="njs-fig-btn" href="${img.src}" target="_blank" rel="noopener">Open vector SVG</a>`;
      tools.querySelector("button").addEventListener("click", () => openBox(img.src, img.alt));
      const cap = fig.querySelector("figcaption");
      if (cap) fig.insertBefore(tools, cap);
      else fig.appendChild(tools);
    });
  }

  function bindTables() {
    document.querySelectorAll("div.njs-table").forEach((box) => {
      const table = box.querySelector("table");
      if (!table || box.dataset.njsBound) return;
      box.dataset.njsBound = "1";
      let cap = box.nextElementSibling;
      const title = cap && cap.textContent.startsWith("Table") ? cap.textContent.split(".").slice(0, 2).join(".") : "";
      const tools = document.createElement("div");
      tools.className = "njs-fig-tools";
      tools.innerHTML = '<button type="button" class="njs-fig-btn">Expand table</button>';
      tools.querySelector("button").addEventListener("click", () => openTable(table, title));
      box.appendChild(tools);
    });
  }

  function bindAll() {
    bindFigures();
    bindTables();
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(bindAll);
  } else {
    document.addEventListener("DOMContentLoaded", bindAll);
  }
})();
