// OQVAR web app: drawing register, the sheet with takeoff marks, the check sheet, linked 3D.
const $ = (s, r = document) => r.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
};
const s = (tag, attrs = {}) => { const el = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); return el; };
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (n) => Number(n).toLocaleString("en-US");
const pad2 = (n) => String(n).padStart(2, "0");
// line icons in the spirit of SF Symbols (20 x 20, round caps); painted into [data-icon] slots and used by the renderers
const ICONS = {
  sidebar: "M4.5 3.5h11a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2ZM7.5 3.5v13",
  search: "M8.7 14.4a5.7 5.7 0 1 0 0-11.4 5.7 5.7 0 0 0 0 11.4ZM13 13l4 4",
  doc: "M5.5 2.5h6l3 3v12h-9zM11.5 2.5v3h3M8 9.5h4M8 12.5h4",
  cube: "M10 2.5l6.5 3.5v8L10 17.5 3.5 14V6zM3.5 6 10 9.5 16.5 6M10 9.5v8",
  graph: "M5 15.5a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM15 15.5a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM10 7.5a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM8.9 7.2 6.1 11.8M11.1 7.2l2.8 4.6M7 13.5h6",
  shield: "M10 2.5 16 5v4.8c0 3.6-2.5 6.4-6 7.7-3.5-1.3-6-4.1-6-7.7V5zM7.3 10l2 2 3.6-3.8",
  split: "M4.5 3.5h11a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2ZM10 3.5v13",
  chart: "M4 16.5v-5M8 16.5V4M12 16.5V8.5M16 16.5v-9",
  "chev-l": "M12.5 4.5 7 10l5.5 5.5", "chev-r": "M7.5 4.5 13 10l-5.5 5.5",
  plus: "M10 4.5v11M4.5 10h11", minus: "M4.5 10h11",
  fit: "M12 3.5h4.5V8M8 16.5H3.5V12M16.5 3.5 11.5 8.5M3.5 16.5l5-5",
  info: "M10 17.5a7.5 7.5 0 1 0 0-15 7.5 7.5 0 0 0 0 15ZM10 9v5M10 6.3v.2",
  mic: "M10 12.5a2.5 2.5 0 0 0 2.5-2.5V5a2.5 2.5 0 0 0-5 0v5a2.5 2.5 0 0 0 2.5 2.5ZM5 9.5a5 5 0 0 0 10 0M10 14.5v3",
  send: "M10 15.5v-11M5.5 9 10 4.5 14.5 9",
  upload: "M10 12.5v-9M6.5 7 10 3.5 13.5 7M4 11.5v4a1.5 1.5 0 0 0 1.5 1.5h9a1.5 1.5 0 0 0 1.5-1.5v-4",
  x: "M5.5 5.5l9 9M14.5 5.5l-9 9",
  check: "M5 10.5l3.2 3.2L15 6.8", approx: "M4.5 8.2c1.8-1.6 3.4-1.6 5.5 0s3.7 1.6 5.5 0M4.5 12.8c1.8-1.6 3.4-1.6 5.5 0s3.7 1.6 5.5 0",
  sum: "M14.5 4.5h-9L10 10l-4.5 5.5h9", dash: "M6 10h8",
  bubble: "M10 3.5c4.1 0 7 2.5 7 5.6s-2.9 5.6-7 5.6c-.9 0-1.8-.1-2.6-.4L4 16l.9-3.1C3.7 11.9 3 10.6 3 9.1 3 6 5.9 3.5 10 3.5Z",
  photo: "M4 4.5h12a1.5 1.5 0 0 1 1.5 1.5v8a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 14V6A1.5 1.5 0 0 1 4 4.5ZM2.5 12.5l4-3.5 3 2.5 3-3 4.5 4M13 8a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1Z",
  table: "M4 3.5h12A1.5 1.5 0 0 1 17.5 5v10a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 15V5A1.5 1.5 0 0 1 4 3.5ZM2.5 7.5h15M2.5 11.5h15M8 7.5v9",
  scan: "M3 7V4.5A1.5 1.5 0 0 1 4.5 3H7M13 3h2.5A1.5 1.5 0 0 1 17 4.5V7M17 13v2.5a1.5 1.5 0 0 1-1.5 1.5H13M7 17H4.5A1.5 1.5 0 0 1 3 15.5V13M6.5 10h7",
  plan: "M3 3.5h14v13H3zM3 9.5h6M9 3.5v9M13 9.5v7M13 12.5h4",
  house: "M3.5 9 10 3.5 16.5 9v7.5H12v-5H8v5H3.5z",
  return: "M16 5v4.5a2 2 0 0 1-2 2H4.5M7.5 8.5l-3 3 3 3",
  history: "M3.5 10a6.5 6.5 0 1 0 1.9-4.6M3.5 3.5v3.2h3.2M10 6.5V10l2.5 1.6",
  "sidebar-r": "M4.5 3.5h11a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2ZM12.5 3.5v13M14.6 7h.8M14.6 9.5h.8",
  gear: "M16.02 8.14L18.18 8.60L18.18 11.40L16.02 11.86L15.57 12.94L16.78 14.79L14.79 16.78L12.94 15.57L11.86 16.02L11.40 18.18L8.60 18.18L8.14 16.02L7.06 15.57L5.21 16.78L3.22 14.79L4.43 12.94L3.98 11.86L1.82 11.40L1.82 8.60L3.98 8.14L4.43 7.06L3.22 5.21L5.21 3.22L7.06 4.43L8.14 3.98L8.60 1.82L11.40 1.82L11.86 3.98L12.94 4.43L14.79 3.22L16.78 5.21L15.57 7.06ZM12.6 10a2.6 2.6 0 1 1-5.2 0 2.6 2.6 0 0 1 5.2 0Z",
};
const icon = (name) => { const el = s("svg", { class: "i", viewBox: "0 0 20 20", "aria-hidden": "true" }); el.append(s("path", { d: ICONS[name] || "" })); return el; };
const paintIcons = (root = document) => root.querySelectorAll("[data-icon]").forEach((el) => { if (!el.querySelector(":scope > svg.i")) el.prepend(icon(el.dataset.icon)); });
const api = {
  async get(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${r.status} ${url}`); return r.json(); },
  async post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${url}`);
    return r.json();
  },
};

const DEMOS = [
  ["CAD", "data/swiss_dwellings/site1047_bldg2154_floor9740.pdf"],
  ["BIM", "data/ifc_models/kit-fzk-haus_ac20-fzk-haus.ifc"],
  ["Raster", "data/cubicasa5k/colorful-11260.png"],
  ["Schedule", "data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf"],
  ["Scan", "data/finna/hka.139877668752900_168422869617700.jpg"],
];
const SHORT = { swiss_dwellings: "Swiss CAD", cubicasa5k: "CubiCasa", opening_design: "Opening Dsn" };
const ENGINE = { geometry: "CAD geometry", bim: "BIM model", detection: "Detectors + vision", planner: "Planned query", graph: "Room graph", mep: "MEP model",
  document: "Sheet text", vision: "Vision model", none: "—" };

const state = {
  source: null, q: "", onlyQ: true, offset: 0, total: 0, busy: false, tab: "stoa",
  drawing: null, sheet: null, view: "sheet", cards: [], active: null, focus: 0, threads: new Map(), seq: 0,
  session: { n: 0, checked: 0, right: 0, tokens: 0, cost: 0 }, pending3d: null, plan: null, planF: 1, items: [],
  graphOn: false, graphs: new Map(), heatOn: false, split: null, insp: false,
};

// ------------------------------------------------------------------ pan / zoom stage
const stage = {
  root: $("#stage"), canvas: $("#canvas"), x: 0, y: 0, k: 1, W: 1, H: 1,
  set(el, W, H) { this.W = W; this.H = H; this.canvas.replaceChildren(el); this.fit(false); },
  free() {  // the part of the stage the floating controls leave open: the view capsule above, the ask dock below
    const r = this.root.getBoundingClientRect(), cap = $("#vcap"), dock = $("#dock");
    let top = 16, bottom = 16;
    if (cap?.offsetParent) top = Math.max(top, cap.getBoundingClientRect().bottom - r.top + 10);
    if (dock?.offsetParent) bottom = Math.max(bottom, r.bottom - dock.getBoundingClientRect().top + 10);
    if (top + bottom > r.height * 0.6) { top = 16; bottom = 16; }  // a short window: use all of it
    return { x: 0, y: top, w: r.width, h: r.height - top - bottom };
  },
  fitK() {  // never <= 0: a hidden or collapsed stage (0 px tall) would otherwise mirror the sheet; the ResizeObserver refits later
    const f = this.free();
    return Math.max(Math.min((f.w - 64) / this.W, (f.h - 24) / this.H), 0.01);
  },
  apply(anim) {
    this.canvas.classList.toggle("anim", !!anim);
    clearTimeout(this.animT); if (anim) this.animT = setTimeout(() => this.canvas.classList.remove("anim"), 460);
    this.canvas.style.transform = `translate(${this.x}px, ${this.y}px) scale(${this.k})`;
    $("#z-val").textContent = `${Math.round((this.k / this.fitK()) * 100)}%`;
    scaleOverlay();
  },
  fit(anim = true) {
    const f = this.free();
    this.k = this.fitK();
    this.x = f.x + (f.w - this.W * this.k) / 2; this.y = f.y + (f.h - this.H * this.k) / 2;
    this.apply(anim);
  },
  zoomAt(cx, cy, f, anim = false) {
    const k2 = Math.min(Math.max(this.k * f, this.fitK() * 0.4), this.fitK() * 40);
    this.x = cx - (cx - this.x) * (k2 / this.k); this.y = cy - (cy - this.y) * (k2 / this.k); this.k = k2;
    this.apply(anim);
  },
  zoomTo([x0, y0, x1, y1], anim = true) {
    const f = this.free(), fitK = this.fitK();
    const bw = Math.max(x1 - x0, 1), bh = Math.max(y1 - y0, 1);
    const cap = state.sheet?.kind === "svg" ? fitK * 16 : Math.max(1.6, fitK * 2.6);  // raster: stay near native pixels
    this.k = Math.max(fitK, Math.min(f.w / (bw * (bw > 5 * bh ? 1.25 : 3.2)), f.h / (bh * 3.2), cap));
    this.x = f.x + f.w / 2 - ((x0 + x1) / 2) * this.k; this.y = f.y + f.h / 2 - ((y0 + y1) / 2) * this.k;
    this.apply(anim);
  },
  init() {
    let drag = null;
    this.root.addEventListener("wheel", (e) => {
      if (state.view !== "sheet") return;
      e.preventDefault();
      const r = this.root.getBoundingClientRect();
      this.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0016));
    }, { passive: false });
    this.root.addEventListener("pointerdown", (e) => {
      if (state.view !== "sheet" || e.button !== 0 || e.target.closest(".hud, .start")) return;
      drag = { x: e.clientX, y: e.clientY, ox: this.x, oy: this.y, moved: false };
    });
    window.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 4) return;
      drag.moved = true; this.root.classList.add("drag");
      this.x = drag.ox + dx; this.y = drag.oy + dy; this.apply(false);
    });
    window.addEventListener("pointerup", () => { if (drag?.moved) this.dragEnd = Date.now(); drag = null; this.root.classList.remove("drag"); });
    this.root.addEventListener("dblclick", (e) => {
      if (state.view !== "sheet") return;
      const r = this.root.getBoundingClientRect();
      this.zoomAt(e.clientX - r.left, e.clientY - r.top, 2, true);
    });
    const centre = (f) => { const r = this.root.getBoundingClientRect(); this.zoomAt(r.width / 2, r.height / 2, f, true); };
    $("#z-in").onclick = () => centre(1.5); $("#z-out").onclick = () => centre(1 / 1.5); $("#z-fit").onclick = () => this.fit();
    const dock = $("#dock");
    new ResizeObserver(() => $("#board").style.setProperty("--dock-h", `${dock.offsetHeight ? dock.offsetHeight + 24 : 0}px`)).observe(dock);
    new ResizeObserver(() => {  // a side panel opened or the window changed: keep the focused mark (or the whole sheet) in view
      if (!state.sheet || this.canvas.classList.contains("anim")) return;
      const it = state.view === "sheet" && state.items[state.focus - 1];
      if (it) this.zoomTo(it.bbox, false); else this.fit(false);
    }).observe(this.root);
  },
};
const wasDrag = () => Date.now() - (stage.dragEnd || 0) < 250;

// ------------------------------------------------------------------ register
const EXAMPLES = { CAD: ["CAD floor plan", "plan"], BIM: ["BIM model", "cube"], Raster: ["Raster plan", "photo"], Schedule: ["Door schedule", "table"], Scan: ["Archive scan", "scan"] };
function renderExamples() {
  $("#demo").replaceChildren(...DEMOS.map(([k, id]) => h("button", { class: "ex", "data-id": id, onclick: () => openDrawing(id) },
    icon(EXAMPLES[k][1]), EXAMPLES[k][0], h("small", {}, k))));
}

// the start screen: nothing open yet, so the one thing to do is upload a plan (or pick an example)
function showStart() {
  if (state.drawing) state.threads.set(state.drawing.id, state.cards);
  state.drawing = null; state.sheet = null; state.cards = []; state.active = null; state.items = [];
  document.body.classList.add("no-drawing");
  $("#start").hidden = false;
  stage.canvas.replaceChildren(); $("#three").classList.add("hidden"); $("#canvas").classList.remove("hidden");
  state.insp = false; showHistory(false); setMsg("");
  $("#d-title").textContent = "Explore"; $("#d-meta").textContent = "Upload a floor plan, or open one from the library";
  $("#q").disabled = true; $("#send").disabled = true; $("#q").placeholder = "Open a drawing to ask about it";
  document.querySelectorAll(".row, .ex").forEach((el) => el.classList.remove("on"));
  history.replaceState(null, "", location.pathname);
  renderThread();
}

async function loadList(reset = true) {
  if (state.busy) return;
  state.busy = true;
  if (reset) { state.offset = 0; $("#list").replaceChildren(); }
  const p = new URLSearchParams({ offset: state.offset, limit: state.offset ? 30 : 15, only_questions: state.onlyQ });
  if (state.source) p.set("source", state.source);
  if (state.q) p.set("q", state.q);
  let r;
  try { r = await api.get(`/api/drawings?${p}`); } catch (e) { state.busy = false; $("#list").replaceChildren(h("div", { class: "more" }, `Register unavailable: ${e.message}`)); return; }
  state.total = r.total;
  $("#reg-count").textContent = fmt(r.total);
  if (reset) {
    $("#sources").replaceChildren(h("option", { value: "" }, "All sources"),
      ...r.facets.map((f) => h("option", { value: f.source, title: `${f.label} — ${f.kind}` }, `${SHORT[f.source] || f.label}  ·  ${fmt(f.count)}`)));
    $("#sources").value = state.source || "";
  }
  $("#list .more")?.remove();
  for (const d of r.items) $("#list").append(rowEl(d));
  state.offset += r.items.length;
  if (state.offset < r.total) {
    $("#list").append(h("button", { class: "more more-btn", onclick: () => loadList(false) }, `Show more`, h("span", {}, `${fmt(r.total - state.offset)} left`)));
  } else if (!r.total) $("#list").append(h("div", { class: "more" }, "No sheets match."));
  state.busy = false;
}

function rowEl(d) {
  return h("button", { class: "row" + (state.drawing?.id === d.id ? " on" : ""), "data-id": d.id, onclick: () => openDrawing(d.id) },
    h("img", { loading: "lazy", src: `/api/thumb?id=${encodeURIComponent(d.id)}&w=160`, alt: "" }),
    h("div", {}, h("b", { title: d.title }, d.title),
      h("span", {}, d.formats.slice(0, 3).join(" · ").toUpperCase(), d.has_3d ? [" ·", icon("cube"), "3D"] : null)));
}

// ------------------------------------------------------------------ drawing + sheets
async function openDrawing(id) {
  if (state.drawing) state.threads.set(state.drawing.id, state.cards);
  document.body.classList.remove("no-drawing"); $("#start").hidden = true;
  $("#q").disabled = false; $("#send").disabled = false; $("#q").placeholder = "Ask about this drawing…";
  setMsg("Opening the sheet", true);
  let d;
  try { d = await api.get(`/api/drawing?id=${encodeURIComponent(id)}`); } catch (e) { setMsg(`<b>Could not open this sheet</b>${esc(e.message)}`); return; }
  state.drawing = d; state.cards = state.threads.get(d.id) || []; state.active = state.cards[0] || null; state.focus = 0; state.items = [];
  history.replaceState(null, "", `#${encodeURIComponent(d.id)}`);
  document.querySelectorAll(".row, .ex").forEach((el) => el.classList.toggle("on", el.dataset.id === d.id));
  $("#d-title").textContent = d.title; $("#d-title").title = d.id;
  const tag3d = !d.has_3d ? "" : d.model_kind === "raster" ? " · 3D (approximate)" : d.model_note ? " · 3D (project model)" : " · 3D";
  $("#d-meta").replaceChildren(`${d.source_info.label || d.source} · ${d.formats.join(" · ")}`,
    d.model_note ? h("span", { title: d.model_note }, tag3d) : tag3d,
    d.question_count ? h("i", {}, ` · ${d.question_count} questions`) : "");
  const c = d.credit || {};
  $("#credit").innerHTML = [c.title && `<b>${esc(c.title)}</b>`, c.credit && esc(c.credit), c.licence && esc(c.licence),
    c.url && `<a href="${esc(c.url)}" target="_blank" rel="noopener">source</a>`].filter(Boolean).join(" · ") || "—";
  const sel = $("#sheet-select");
  sel.replaceChildren(...d.sheets.map((sh, i) => h("option", { value: i }, sh.label)));
  sel.classList.toggle("hidden", d.sheets.length < 2);
  $("#btn-3d").disabled = !d.has_3d;
  $("#btn-graph").disabled = d.source !== "swiss_dwellings";  // IFC graphs answer questions but have no sheet coordinates to draw in
  $("#btn-graph").title = d.source === "swiss_dwellings" ? "Show how the rooms connect: doors, open connections and shared walls"
    : d.source.startsWith("ifc") ? "Room graph questions work on this model; the graph is drawn on Swiss CAD sheets only" : "No room graph for raster or scanned sheets";
  $("#three").src = "about:blank"; state.sceneUrl = null;
  renderThread();
  if (d.sheets.length) { setView("sheet"); await showSheet(0); }
  else if (d.has_3d) { stage.canvas.replaceChildren(); state.sheet = null; setView("3d"); }
  else { stage.canvas.replaceChildren(); state.sheet = null; setMsg(`<b>No preview for this file type</b>${esc(d.formats.join(", ").toUpperCase())} files are listed in the register but cannot be drawn yet.`); }
}

async function showSheet(i) {
  const sh = state.drawing.sheets[i];
  state.sheet = sh; state.sheetIndex = i; state.plan = null; state.items = []; $("#sheet-select").value = i;
  setMsg("Opening the sheet", true);
  try {
    if (sh.kind === "svg") {
      const text = await (await fetch(sh.src)).text();
      const plan = document.importNode(new DOMParser().parseFromString(text, "image/svg+xml").documentElement, true);
      plan.classList.add("plan"); plan.removeAttribute("width"); plan.removeAttribute("height");
      const vb = (plan.getAttribute("viewBox") || `0 0 ${sh.width} ${sh.height}`).split(/[\s,]+/).map(Number);
      state.planF = 6; state.planVB = vb; state.plan = plan;
      plan.append(s("g", { class: "tg" }));
      plan.addEventListener("click", (e) => {
        if (wasDrag()) return;
        const g = e.target.closest("[data-guid]");
        if (g) tip(`${g.getAttribute("class").replace("cut ", "").replace("pl-ev", "").replace("focus", "").trim()} · ${g.dataset.name || ""}`, e);
      });
      stage.set(h("div", { class: "sheet", style: { width: `${vb[2] * state.planF}px`, height: `${vb[3] * state.planF}px` } }, plan), vb[2] * state.planF, vb[3] * state.planF);
    } else {
      const img = h("img", { src: sh.src, alt: state.drawing.title, draggable: "false" });
      const vb = `0 0 ${sh.width} ${sh.height}`;
      stage.set(h("div", { class: "sheet", style: { width: `${sh.width}px`, height: `${sh.height}px` } },
        img, s("svg", { class: "ov gr", viewBox: vb }), s("svg", { class: "ov hl", viewBox: vb }), s("svg", { class: "ov tg", viewBox: vb })), sh.width, sh.height);
      await new Promise((res, rej) => { if (img.complete) res(); else { img.onload = res; img.onerror = () => rej(new Error("the image did not load")); } });
    }
  } catch (e) { setMsg(`<b>Could not open this sheet</b>${esc(e.message)}`); return; }
  setMsg("");
  drawOverlay(); drawGraph();
}

function setMsg(html, busy = false) {
  $("#stage-msg").innerHTML = !html ? "" : busy ? `<div><span class="mark"></span><b>${html}</b></div>` : `<div>${html}</div>`;
}

function setView(v) {
  if (v === "3d" && !state.drawing?.has_3d) return;  // nothing loaded yet, or no model for this drawing
  state.view = v;
  document.querySelectorAll("#view-seg button").forEach((b) => { b.classList.toggle("on", b.dataset.view === v); b.setAttribute("aria-pressed", b.dataset.view === v); });
  $("#three").classList.toggle("hidden", v !== "3d"); $("#canvas").classList.toggle("hidden", v === "3d");
  document.querySelector(".zoom").classList.toggle("hidden", v === "3d");
  state.focus = 0;
  if (v === "3d") open3D();
  else {  // back on the sheet: draw the active answer's marks again (the 3D view does not), so its takeoff tiles work
    setMsg("");
    if (state.active?.res && state.sheet) drawOverlay();
    updateStepper();
  }
  renderInspector();
}

async function open3D() {
  state.focus = 0; updateStepper();
  if (!state.sceneUrl) {
    const from = { ifc: "the IFC model", dxf: "the CAD geometry", mesh: "the 3D mesh", raster: "the scan (approximate)" };
    setMsg(state.drawing.upload ? "Tracing walls, rooms, doors and windows from your plan (up to half a minute the first time)"
      : `Building the 3D model from ${from[state.drawing.model_kind] || "the model"}`, true);
    try {
      const r = await api.post("/api/scene", { id: state.drawing.id, question: "" });
      state.sceneUrl = r.url; state.pending3d = state.active?.res?.ids || [];
      $("#three").src = r.url;
    } catch (e) { setMsg(`<b>3D view unavailable</b>${esc(e.message)}`); return; }
  } else highlight3D(state.active?.res?.ids || []);
  setMsg("");
}

function highlight3D(ids) {
  state.pending3d = ids; state.focus = 0; updateStepper();
  $("#three").contentWindow?.postMessage({ type: "oqvar:highlight", ids }, location.origin);
}
window.addEventListener("message", (e) => {
  if (e.origin !== location.origin || e.data?.type !== "oqvar:ready") return;
  const d = state.drawing, c = d?.credit || {};
  if (d) $("#three").contentWindow?.postMessage({ type: "oqvar:meta", title: d.title,  // names the drawing in the 3D view's film
    credit: [c.credit, c.licence].filter(Boolean).join(" · ") }, location.origin);
  if (state.pending3d) highlight3D(state.pending3d);
});

// ------------------------------------------------------------------ takeoff marks on the sheet
// raster sheets: one multiply-blended layer for highlighter shapes, one normal layer for balloons;
// IFC plans: the plan's own elements are recoloured, balloons go in a group inside the plan
const layers = () => (state.sheet?.kind === "svg"
  ? { shapes: null, tags: state.plan?.querySelector("g.tg") }
  : { shapes: $("#canvas svg.ov.hl"), tags: $("#canvas svg.ov.tg") });

function scaleOverlay() {
  const { shapes, tags } = layers(); if (!tags) return;
  const px = 1 / (stage.k * (state.sheet.kind === "svg" ? state.planF : 1));  // one screen pixel in overlay units
  shapes?.style.setProperty("--sw", `${2 * px}px`);
  $("#canvas svg.ov.gr")?.style.setProperty("--sw", `${px}px`);
  for (const c of document.querySelectorAll("#canvas svg.ov.gr .n")) c.setAttribute("r", 4.5 * px);
  for (const c of document.querySelectorAll("#canvas svg.ov.gr .dn")) c.setAttribute("r", 2.6 * px);
  for (const c of document.querySelectorAll("#canvas .route-end")) c.setAttribute("r", 6 * px);
  for (const it of state.items) it.balloon?.setAttribute("transform", `translate(${it.bx} ${it.by}) scale(${px})`);
}

function drawOverlay() {
  const { shapes, tags } = layers(); if (!tags) return;
  shapes?.replaceChildren(); tags.replaceChildren();
  state.plan?.querySelectorAll(".pl-ev").forEach((e) => e.classList.remove("pl-ev", "focus", "b-direct", "b-inferred", "b-review"));
  const card = state.active; state.items = [];
  if (!card?.res || !state.sheet) { updateStepper(); return; }
  const ov = card.res.overlay;
  if (ov.space === "guid") {
    if (state.sheet.kind !== "svg") { updateStepper(); return; }
    for (const it of ov.items) {
      const el = state.plan.querySelector(`[data-guid="${CSS.escape(it.guid)}"]`);
      if (!el) continue;
      el.classList.add("pl-ev", `b-${basisOf(it)}`);
      let b; try { b = el.getBBox(); } catch { continue; }
      const vb = state.planVB, F = state.planF;
      state.items.push({ ...it, el, bx: b.x + b.width / 2, by: b.y + b.height / 2, pb: [b.x, b.y, b.x + b.width, b.y + b.height],
        bbox: [(b.x - vb[0]) * F, (b.y - vb[1]) * F, (b.x + b.width - vb[0]) * F, (b.y + b.height - vb[1]) * F] });
    }
  } else if (state.sheet.kind === "image" && state.sheetIndex === 0) {
    if (ov.path?.length > 1) {  // graph route: the walked line, start to finish
      shapes.append(s("polyline", { class: "route", points: ov.path.map((p) => p.join(",")).join(" ") }));
      const [ex, ey] = ov.path[ov.path.length - 1];
      tags.append(s("circle", { class: "route-end", cx: ex, cy: ey, r: 6 }));
    }
    ov.items.forEach((it, i) => {
      const cls = `shape b-${basisOf(it)}`;
      const shape = it.kind === "point" ? s("circle", { class: cls, cx: it.pts[0][0], cy: it.pts[0][1], r: 26 })
        : s("polygon", { class: cls, points: it.pts.map((p) => p.join(",")).join(" ") });
      shape.style.animationDelay = `${Math.min(i * 14, 420)}ms`;
      shapes.append(shape);
      const [x0, y0, x1, y1] = it.bbox, row = x1 - x0 > 5 * (y1 - y0);
      state.items.push({ ...it, el: shape, row, bx: row ? x0 : (x0 + x1) / 2, by: (y0 + y1) / 2 });  // table rows: balloon at the row start
    });
  }
  state.items.forEach((it, i) => {
    it.i = i + 1;
    // a callout: leader line from the item to a numbered balloon beside it (left of a table row)
    const balloon = s("g", { class: `balloon b-${basisOf(it)}` }), [cx, cy] = it.row ? [-13, 0] : [17, -17];
    if (!it.row) balloon.append(s("line", { x1: 0, y1: 0, x2: 10, y2: -10 }), s("circle", { class: "dot", r: 2 }));
    balloon.append(s("circle", { cx, cy, r: 9.5 }), Object.assign(s("text", { x: cx, y: cy }), { textContent: it.n }));
    tags.append(balloon); it.balloon = balloon;
    const label = `${it.n} · ${it.label || "item"}${it.value != null ? ` · ${it.value}` : ""}`;
    for (const el of [it.el, balloon]) {
      el.addEventListener("click", (e) => { if (wasDrag()) return; e.stopPropagation(); focusItem(it.i); });
      el.addEventListener("mousemove", (e) => tip(label, e)); el.addEventListener("mouseleave", () => tip(null));
    }
  });
  scaleOverlay(); applyFocus(false); updateStepper(); drawHeatKey();
}

// ------------------------------------------------------------------ room graph layer (stoa_qa.graph via /api/graph)
async function toggleGraph() {
  state.graphOn = !state.graphOn;
  $("#btn-graph").setAttribute("aria-pressed", state.graphOn);
  await drawGraph();
}

async function drawGraph() {
  const layer = $("#canvas svg.ov.gr"); if (!layer) return;
  layer.replaceChildren();
  const d = state.drawing;
  if (!state.graphOn || !d || state.sheet?.kind !== "image" || state.sheetIndex !== 0) return;
  let g = state.graphs.get(d.id);
  if (!g) {
    try { g = await api.get(`/api/graph?id=${encodeURIComponent(d.id)}`); } catch (e) { g = { error: e.message }; }
    state.graphs.set(d.id, g);
  }
  if (g.error || g.space !== "px") { setMsg(`<b>No room graph on this sheet</b>${esc(g.error || "it can be queried, but not drawn on this kind of sheet")}`); setTimeout(() => setMsg(""), 2600); return; }
  if (state.drawing !== d) return;
  const at = new Map(g.nodes.map((n) => [n.id, n]));
  for (const e of g.edges) {
    const a = at.get(e.a), b = at.get(e.b);
    const pts = [a?.px, e.px, b?.px].filter(Boolean);
    if (pts.length < 2) continue;
    const line = s("polyline", { class: `e ${e.kind}`, points: pts.map((p) => p.join(",")).join(" ") });
    layer.append(line);
    if (e.kind === "door" && e.px) layer.append(s("circle", { class: "dn", cx: e.px[0], cy: e.px[1], r: 2.6 }));
  }
  for (const n of g.nodes) {
    if (!n.px) continue;
    const links = g.edges.filter((e) => e.a === n.id || e.b === n.id);
    const doors = links.filter((e) => e.kind === "door").length, open = links.filter((e) => e.kind === "open").length;
    const text = `${n.name}${n.area_m2 ? ` · ${n.area_m2} m²` : ""} · ${doors} door${doors === 1 ? "" : "s"}${open ? ` · ${open} open` : ""} · ${n.windows} window${n.windows === 1 ? "" : "s"}`;
    const c = s("circle", { class: `n${n.kind === "outside" ? " out" : ""}`, cx: n.px[0], cy: n.px[1], r: 4.5 });
    c.addEventListener("mousemove", (ev) => tip(text, ev)); c.addEventListener("mouseleave", () => tip(null));
    layer.append(c);
  }
  scaleOverlay();
}

function applyFocus(zoom = true) {
  const { shapes, tags } = layers(); if (!tags) return;
  const f = state.focus;
  shapes?.classList.toggle("dim", f > 0); tags.classList.toggle("dim", f > 0);
  for (const it of state.items) {
    const on = it.i === f;
    it.el.classList.toggle("focus", on); it.balloon.classList.toggle("focus", on);
    if (on) { if (it.el.parentNode === shapes) shapes.append(it.el); tags.append(it.balloon); }
  }
  const n = state.items[f - 1]?.n ?? -1;
  document.querySelectorAll("#insp-body [data-n]").forEach((el) => {
    const on = +el.dataset.n === n;
    el.classList.toggle("focus", on);
    if (on) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
  tags.querySelectorAll(".pulse").forEach((e) => e.remove());
  const fit = state.items[f - 1];
  if (fit) {  // a ring that pulses around the focused item, so the eye finds it at any zoom
    const [x0, y0, x1, y1] = fit.pb || fit.bbox, m = Math.max(x1 - x0, y1 - y0) * 0.12 + (fit.pb ? 0.3 : 4);
    for (const cls of ["pulse ring", "pulse wave"])
      tags.prepend(s("rect", { class: cls, x: x0 - m, y: y0 - m, width: x1 - x0 + 2 * m, height: y1 - y0 + 2 * m, rx: Math.min(m, fit.pb ? 0.5 : 8), "vector-effect": "non-scaling-stroke" }));
  }
  updateLegend();
  if (zoom) { const it = state.items[f - 1]; if (it) stage.zoomTo(it.bbox); else stage.fit(); }
}

// In the 3D view the same stepper flies the camera to each marked item (blueprint3d/viewer: oqvar:focus)
const items3d = () => (state.active?.res?.overlay?.items || []).filter((it) => it.id != null);
const stepItems = () => (state.view === "3d" ? items3d() : state.items);

function focus3D() {
  const it = items3d()[state.focus - 1];
  $("#three").contentWindow?.postMessage({ type: "oqvar:focus", id: it ? it.id : null }, location.origin);
  document.querySelectorAll("#insp-body [data-n]").forEach((el) => {
    const on = !!it && +el.dataset.n === it.n;
    el.classList.toggle("focus", on);
    if (on) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
}

function focusItem(i) {
  if (state.view === "3d") {
    const n = items3d().length; if (!n) return;
    state.focus = i < 1 ? n : i > n ? 1 : i;
    focus3D(); updateStepper(); return;
  }
  if (state.view !== "sheet") setView("sheet");
  const n = state.items.length; if (!n) return;
  state.focus = i < 1 ? n : i > n ? 1 : i;
  applyFocus(); updateStepper();
}

function showAll() {
  state.focus = 0;
  if (state.view === "3d") focus3D(); else applyFocus();
  updateStepper();
}

function updateStepper() {
  const row = $("#c-step"); if (!row) return;
  const items = stepItems(), n = items.length;
  row.classList.toggle("hidden", !n);
  if (!n) return;
  const it = items[state.focus - 1];
  $("#st-label").innerHTML = it ? `<b>${state.focus}</b> <i>of ${n}</i> · ${esc(it.label || "item")}${it.value != null ? ` <em>· ${esc(it.value)}</em>` : ""}`
    : `<b>${n}</b> marked ${state.view === "3d" ? "in the model" : "on the sheet"} <em class="hint">· step through to check each</em>`;
  $("#st-all").classList.toggle("hidden", !it);
}

function tip(text, e) {
  const t = $("#tip");
  if (!text) { t.classList.add("hidden"); return; }
  t.textContent = text; t.classList.remove("hidden");
  t.style.left = `${Math.min(e.clientX + 14, innerWidth - 350)}px`; t.style.top = `${e.clientY + 16}px`;
  clearTimeout(tip.timer); tip.timer = setTimeout(() => t.classList.add("hidden"), 2600);
}

// ------------------------------------------------------------------ check sheet: questions
async function ask(question) {
  question = (question || "").trim();
  if (!question || !state.drawing || state.asking) return;
  state.asking = true; $("#send").disabled = true; $("#q").value = "";
  const now = new Date();
  const card = { question, res: null, drawing: state.drawing.id, no: ++state.seq, at: `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}` };
  state.cards.unshift(card); state.active = card; state.focus = 0; state.items = [];
  showHistory(false); renderThread(); drawOverlay();
  try { card.res = await api.post("/api/ask", { id: state.drawing.id, question, sheet_path: state.sheet?.path || null, sheet_page: state.sheet?.page || 0 }); }
  catch (e) { card.error = e.message; }
  state.asking = false; $("#send").disabled = false;
  if (card.drawing !== state.drawing.id) return;
  card.fresh = true;
  await activate(card);
  card.fresh = false;
}

async function activate(card) {
  state.active = card; state.focus = 0;
  const ov = card.res?.overlay;
  if (ov?.space === "guid" && ov.sheets?.length && state.drawing.sheets.length) {
    const cur = ov.sheets.find((x) => x.path === state.sheet?.path), best = ov.sheets[0];
    if (!cur || cur.hits < best.hits) {
      const i = state.drawing.sheets.findIndex((x) => x.path === best.path);
      if (i >= 0 && i !== state.sheetIndex) { renderThread(); await showSheet(i); stage.fit(); if (state.view === "3d") highlight3D(card.res.ids); return; }
    }
  }
  renderThread();
  if (state.view === "sheet") { drawOverlay(); stage.fit(); } else highlight3D(card.res?.ids || []);
}

function dismiss() {  // close the answer card: its marks leave the sheet, the answer stays in the history
  state.active = null; state.focus = 0;
  renderThread(); drawOverlay();
  if (state.view === "3d") highlight3D([]); else stage.fit();
}

// ------------------------------------------------------------------ check sheet: records
function stampOf(r) {
  if (r.expected != null && r.correct === true) return ["ok", "Matches Stoa", `Stoa's answer is ${r.expected}`];
  if (r.expected != null && r.correct === false) return ["bad", `Stoa says ${r.expected}`, "Differs from Stoa's answer"];
  if (r.value == null) return ["none", "No answer", "Not guessed"];
  const k = r.engine.kind;
  if (k === "vision") return ["est", "Estimate", "Read by a vision model, check it yourself"];
  if (k === "detection") return r.confidence >= 0.85 ? ["calc", "Cross-checked", "Two sources agree"] : ["est", "Unconfirmed", "The sources differ"];
  if (k === "document") return ["calc", "Counted", "Rows on the sheet"];
  if (k === "planner") return ["calc", "Computed", "AI-planned query"];
  if (k === "graph") return ["calc", "Traced", "Through the room graph"];
  if (k === "mep") return ["calc", "Computed", "From the MEP model"];
  return ["calc", "Computed", k === "bim" ? "From the model" : "From the geometry"];
}
const STATUS_ICON = { ok: "check", bad: "x", est: "approx", calc: "sum", none: "dash" };

function summaryEl(r, sub) {  // one quiet line: how it was found, how long, whether AI was used
  const e = r.engine, time = r.latency_ms < 1000 ? `${Math.max(1, Math.round(r.latency_ms))} ms` : `${(r.latency_ms / 1000).toFixed(1)} s`;
  return h("p", { class: "summary" }, h("b", {}, sub), ` · ${time}${e.llm_tokens ? "" : " · no AI"}`);
}

// a list answer ("find all the windows"): its lines, the top-level ones, and what they are when that can be named
function listOf(r) {
  if (typeof r.value !== "string" || !r.value.includes("\n")) return null;
  const all = r.value.split("\n").filter((l) => l.trim()), top = all.filter((l) => !l.startsWith("  "));
  if (top.length < 2) return null;
  const planned = /^(.+?)\s*→\s*list\b/i.exec(r.method || "");                       // the planner says "windows → list"
  const first = top.map((l) => (/^([^\W\d_]+)\s+\d/u.exec(l.trim()) || [])[1]?.toLowerCase());  // or every line reads "window 3 — …"
  const noun = planned ? planned[1].trim() : first.every((w) => w && w === first[0]) ? (first[0].endsWith("s") ? first[0] : `${first[0]}s`) : null;
  const preview = all.slice(0, 3).some((l) => l.length > 60) ? 2 : 3;  // unnamed lists: a short preview, the rest on request
  return { all, top, noun, preview };
}

function linesEl(lines) {
  return h("ol", { class: "lines" }, ...lines.map((ln) => {
    const sub = ln.startsWith("  "), t = ln.trim();
    const cut = t.includes(" — ") ? t.indexOf(" — ") : t.includes(": ") ? t.indexOf(": ") : -1;
    const detail = cut < 0 ? null : t.slice(cut).replace(/^( — |: )/, "");
    return h("li", { class: [sub ? "sub" : "", detail?.length > 30 ? "long" : ""].join(" ").trim() || null },  // long details wrap under their label
      h("span", {}, cut < 0 ? t : t.slice(0, cut).replace(/:$/, "")), detail ? h("em", {}, detail) : null);
  }));
}

function valueEl(r) {
  if (r.value == null) return h("div", { class: "val no" }, r.text);
  if (typeof r.value === "number") {
    const m = r.text.match(/^(-?[\d.,]+)\s*(.*)$/);
    return h("div", { class: "val" }, m ? m[1] : r.text, m && m[2] ? h("small", {}, m[2]) : null);
  }
  const lines = String(r.value).split("\n");
  if (lines.length === 1 && lines[0].length <= 22) return h("div", { class: "val word" }, lines[0]);
  return linesEl(lines);
}

function specEl(r) {
  const e = r.engine, c = r.confidence >= 0.85 ? [3, "High"] : r.confidence >= 0.6 ? [2, "Medium"] : [1, "Low"];
  const time = r.latency_ms < 1000 ? `${Math.max(1, Math.round(r.latency_ms))} ms` : `${(r.latency_ms / 1000).toFixed(1)} s`;
  const method = e.kind === "planner" ? h("dd", { class: "wide" }, h("code", {}, r.method)) : h("dd", { class: "wide" }, r.method);
  return h("dl", { class: "spec" },
    h("dt", {}, e.kind === "planner" ? "Query" : "Method"), method,
    h("dt", {}, "Engine"), h("dd", { title: e.note }, ENGINE[e.kind] || e.kind), h("dt", {}, "Time"), h("dd", {}, time),
    h("dt", {}, "Cost"), h("dd", { title: e.models.join(", ") }, e.llm_tokens ? `${fmt(e.llm_tokens)} tokens · $${e.cost_usd.toFixed(5)}` : h("span", { class: "zero" }, "0 tokens · $0")),
    h("dt", {}, "Trust"), r.value == null ? h("dd", {}, "—")
      : h("dd", {}, h("span", { class: "conf", "aria-hidden": "true" }, ...[1, 2, 3].map((i) => h("i", { class: i <= c[0] ? "f" : null }))), c[1]));
}

// the crew: who the general contractor asked, in order; then the auditor's independent checks
// ------------------------------------------------------------------ cross-sheet references (stoa_qa.xref)
// each pair: the callout on this sheet beside the detail it points at on the other sheet; "Open" jumps there
function refsEl(r) {
  const refs = r.references || [];
  if (!refs.length) return null;
  const crop = (p, size) => `/api/crop?path=${encodeURIComponent(p.path)}&page=${p.page}&kind=pdf&size=${size}` +
    `&x0=${p.box[0]}&y0=${p.box[1]}&x1=${p.box[2]}&y1=${p.box[3]}`;
  return h("div", { class: "xrefs" },
    h("span", { class: "lbl" }, "References · ", h("b", {}, `${refs.length} followed`)),
    ...refs.slice(0, 12).map((x) => h("div", { class: "xref" },
      x.from.box ? h("figure", {}, h("img", { src: crop(x.from, 180), alt: `callout ${x.label}` }), h("figcaption", {}, `callout ${x.from.text || x.label}`))
        : h("figure", { class: "none" }, h("figcaption", {}, "asked directly")),
      h("span", { class: "arrow", "aria-hidden": "true" }, "→"),
      h("figure", {}, h("img", { src: crop(x.to, 420), alt: `${x.label}: ${x.to.title || x.to.sheet}` }),
        h("figcaption", {}, h("b", {}, x.label), x.to.title ? ` · ${x.to.title}` : "", h("br"), h("i", {}, `found by ${x.to.how}`))),
      h("button", { class: "btn mini", title: `Open sheet ${x.to.sheet} at this detail`, onclick: (ev) => { ev.stopPropagation(); openReference(x.to); } }, `Open ${x.to.sheet}`))));
}

// ------------------------------------------------------------------ when nothing could answer (stoa_qa.explain)
// the reason in plain words, what the drawing does hold, who was asked, and questions that will work here
function whyEl(r) {
  const w = r.unanswered;
  if (!w) return null;
  const box = h("div", { class: `why ${w.category}` },
    w.detail ? h("p", { class: "why-detail" }, w.detail) : null,
    h("p", { class: "why-next" }, w.next));
  if (w.alternatives?.length) box.append(h("div", { class: "why-alts" }, h("span", { class: "lbl" }, "Try instead"),
    ...w.alternatives.map((q) => h("button", { class: "alt", title: "Ask this", onclick: (ev) => { ev.stopPropagation(); ask(q); } }, q))));
  if (w.checked?.length) box.append(h("details", { class: "why-checked", onclick: (ev) => ev.stopPropagation() },
    h("summary", {}, `What I checked (${w.checked.length})`), h("ul", {}, ...w.checked.map((c) => h("li", {}, c)))));
  return box;
}

async function openReference(to) {
  if (state.drawing?.id !== to.path && !state.drawing?.files?.includes(to.path)) await openDrawing(to.path);
  const i = state.drawing.sheets.findIndex((s) => s.path === to.path && (s.page || 0) === (to.page || 0));
  if (i >= 0 && i !== state.sheetIndex) await showSheet(i);
  if (state.view !== "sheet") setView("sheet");
  stage.zoomTo(to.box);
}

function crewEl(r) {
  const crew = r.crew || [], checks = r.audit || [];
  if (!crew.length && !checks.length) return null;
  const MARK = { pass: "✓", warn: "!", fail: "✗", info: "i" };
  const box = h("div", { class: "crew" });
  if (crew.length) box.append(h("ol", { class: "crew-run", "aria-label": "Crew" }, ...crew.map((c) =>
    h("li", { class: c.outcome, title: `${c.role}${c.note ? ` — ${c.note}` : ""} · ${c.ms} ms${c.tokens ? ` · ${c.tokens} tokens` : ""}` },
      h("b", {}, c.agent), h("i", {}, c.outcome)))));
  if (checks.length) box.append(h("ul", { class: "audit", "aria-label": "Auditor checks" }, ...checks.map((c) =>
    h("li", { class: c.status }, h("span", { class: "mk", "aria-hidden": "true" }, MARK[c.status] || "·"),
      h("span", {}, h("b", {}, c.name), " ", c.detail)))));
  return box;
}

function takeoffEl(card, on) {
  const r = card.res, items = r.overlay.items, sh = state.drawing.sheets[0];
  const go = (n) => (ev) => {
    ev.stopPropagation();
    const jump = async () => {
      if (state.view === "3d") {           // the 3D view: the camera flies to the item
        const k = items3d().findIndex((x) => x.n === n);
        if (k >= 0) { focusItem(k + 1); return; }
        setView("sheet");                  // an item with no 3D element: show it on the sheet instead
      }
      let i = state.items.findIndex((x) => x.n === n);
      if (i < 0) {  // not on the sheet in view: open the sheet that holds it, then focus it there
        const j = sheetOf(r, n);
        if (j >= 0 && j !== state.sheetIndex) { await showSheet(j); renderThread(); i = state.items.findIndex((x) => x.n === n); }
      }
      if (i >= 0) focusItem(i + 1);
    };
    if (!on) activate(card).then(jump); else jump();
  };
  const box = h("div", { class: "take" }, h("div", { class: "take-head" },
    h("span", { class: "lbl" }, "Evidence ", h("b", {}, `· ${items.length} marked`)),
    h("div", {},
      state.drawing.has_3d && r.ids.length && state.view !== "3d" ? h("button", { class: "btn mini", title: "Show them in the 3D model", onclick: (ev) => { ev.stopPropagation(); state.active = card; setView("3d"); renderThread(); } }, icon("cube"), "3D") : null)));
  const px = r.overlay.space === "px" && sh;
  const wide = px && items.filter((it) => it.bbox[2] - it.bbox[0] > 5 * (it.bbox[3] - it.bbox[1])).length > items.length / 2;
  if (wide) {  // table rows: each one as a readable strip of the sheet
    box.append(h("div", { class: "strips" }, ...items.slice(0, 80).map((it) => {
      const [x0, y0, x1, y1] = it.bbox, k = 22 / Math.max(y1 - y0, 4);
      return h("button", { class: "st", "data-n": it.n, title: it.label || "", onclick: go(it.n),
        style: { width: `${Math.max((x1 - x0) * k + 24, 120)}px`, backgroundImage: `url("${sh.src}")`, backgroundSize: `${sh.width * k}px ${sh.height * k}px`,
          backgroundPosition: `${24 - x0 * k}px ${-y0 * k}px` } }, h("span", { class: "tag" }, it.n));
    })));
  } else if (px) {  // a crop of the sheet around every counted item, with the item highlighted
    box.append(h("div", { class: "grid" }, ...items.slice(0, 96).map((it) => {
      const [x0, y0, x1, y1] = it.bbox, side = Math.min(Math.max(x1 - x0, y1 - y0) * 1.9 + 30, sh.width, sh.height);
      const sx = Math.min(Math.max((x0 + x1) / 2 - side / 2, 0), sh.width - side), sy = Math.min(Math.max((y0 + y1) / 2 - side / 2, 0), sh.height - side);
      const pc = (v) => `${(v / side) * 100}%`;
      return h("button", { class: "th", "data-n": it.n, title: `${it.label || ""}${it.value != null ? ` · ${it.value}` : ""} · click to go there`, onclick: go(it.n),
        style: { backgroundImage: `url("${sh.src}")`, backgroundSize: `${(sh.width / side) * 100}% ${(sh.height / side) * 100}%`,
          backgroundPosition: `${sh.width > side ? (sx / (sh.width - side)) * 100 : 0}% ${sh.height > side ? (sy / (sh.height - side)) * 100 : 0}%` } },
        h("u", { style: { left: pc(x0 - sx), top: pc(y0 - sy), width: pc(x1 - x0), height: pc(y1 - y0) } }), h("span", { class: "tag" }, it.n));
    })));
  } else {
    box.append(h("div", { class: "elist" }, ...items.slice(0, 150).map((it) =>
      h("button", { class: "el", "data-n": it.n, onclick: go(it.n) }, h("span", { class: "tag" }, it.n), it.label || "element",
        it.value != null ? h("em", {}, typeof it.value === "number" ? it.value.toFixed(2) : it.value) : null))));
  }
  if (on && r.overlay.space === "guid" && state.sheet?.kind === "svg" && state.items.length < items.length)
    box.append(h("p", { class: "note" }, `${state.items.length} of ${items.length} are on this sheet. Pick another storey from the sheet menu at the top left, or open the 3D model.`));
  if (on && r.overlay.space === "guid" && !state.drawing.sheets.length)
    box.append(h("p", { class: "note" }, "This model has no storey plan in the dataset: the elements are highlighted in the 3D model."));
  return box;
}

// ------------------------------------------------------------------ the answer card, the history menu, the details panel
function renderThread() { renderCard(); renderHistory(); renderInspector(); }

function cardValue(r) {  // the answer at a glance: a number, a word, or the first lines of a longer text
  if (r.value == null) return h("div", { class: "val word" }, "No answer");
  const list = listOf(r);
  if (list?.noun) return h("div", { class: "val" }, fmt(list.top.length), h("small", {}, list.noun));
  if (list) return linesEl(list.all.slice(0, 2));
  if (typeof r.value === "number") return valueEl(r);
  const t = String(r.value);
  return !t.includes("\n") && t.length <= 22 ? h("div", { class: "val word" }, t) : h("p", { class: "c-text" }, t);
}

function renderCard() {
  const box = $("#card"), card = state.active;
  box.classList.toggle("hidden", !card || !state.drawing);
  if (!card || !state.drawing) { box.replaceChildren(); return; }
  const r = card.res;
  const kids = [h("div", { class: "c-top" },
    h("p", { class: "c-q", title: card.question }, card.question),
    r ? h("button", { class: "c-btn", id: "btn-details", "aria-controls": "inspector", "aria-expanded": String(inspOpen()), title: "Evidence, method and checks (⌘I)",
      onclick: () => setInspector(!inspOpen()) }, icon("sidebar-r"), "Details") : null,
    h("button", { class: "icon-btn c-x", "aria-label": "Close this answer", title: "Close (it stays in the history)", onclick: dismiss }, icon("x")))];
  if (card.error) kids.push(h("p", { class: "c-text muted" }, `The request failed: ${card.error}`));
  else if (!r) kids.push(h("div", { class: "wait" }, "Reading the drawing…"));
  else {
    const [cls, word, sub] = stampOf(r), list = listOf(r);
    const pill = h("span", { class: `pill ${cls}`, title: sub }, icon(STATUS_ICON[cls]), cls === "none" ? sub : word);
    if (!card.fresh) pill.style.animation = "none";
    kids.push(h("div", { class: "c-a" }, cardValue(r), pill));
    if (list && !list.noun && list.all.length > 2) kids.push(h("button", { class: "c-link", onclick: () => setInspector(true) }, `${list.all.length - 2} more in Details`));
    const w = r.unanswered;
    if (w) kids.push(h("div", { class: "c-why" }, h("p", {}, w.next || w.detail || r.text),
      w.alternatives?.length ? h("div", { class: "c-alts" }, ...w.alternatives.slice(0, 3).map((q) => h("button", { class: "alt", title: "Ask this", onclick: () => ask(q) }, q))) : null));
    else if (r.value == null) kids.push(h("p", { class: "c-text muted" }, r.text));
    kids.push(h("div", { class: "c-step hidden", id: "c-step" },
      h("button", { class: "icon-btn", id: "st-prev", "aria-label": "Previous mark", title: "Previous (←)", onclick: () => focusItem(state.focus - 1) }, icon("chev-l")),
      h("button", { class: "icon-btn", id: "st-next", "aria-label": "Next mark", title: "Next (→)", onclick: () => focusItem(state.focus + 1) }, icon("chev-r")),
      h("output", { id: "st-label" }),
      h("button", { class: "st-all hidden", id: "st-all", title: "Show every mark (esc)", onclick: showAll }, "Show all")));
  }
  box.replaceChildren(...kids);
  updateStepper();
}

function miniOf(r) {  // the answer in a few characters, for the one-line history
  if (r.value == null) return "—";
  const list = listOf(r);
  if (list) return `${list.top.length} ${list.noun || "lines"}`;
  const t = typeof r.value === "number" ? r.text : String(r.value).split("\n")[0];
  return t.length > 18 ? `${t.slice(0, 17)}…` : t;
}

function showHistory(open) {
  const box = $("#hist"), btn = $("#btn-hist");
  box.hidden = !open; btn.setAttribute("aria-expanded", String(open));
  if (open) renderHistory();
}

function renderHistory() {
  const n = state.cards.length, box = $("#hist");
  $("#hist-n").textContent = n > 1 ? n : "";
  $("#btn-hist").disabled = !n;
  if (!n && !box.hidden) showHistory(false);
  if (box.hidden) return;
  box.replaceChildren(h("p", { class: "hist-head" }, "Asked about this drawing"), ...state.cards.map((c) => {
    const [cls] = c.res ? stampOf(c.res) : [c.error ? "bad" : "none"];
    return h("button", { class: `h-row${c === state.active ? " on" : ""}`, role: "menuitem", title: c.question,
      onclick: () => { showHistory(false); activate(c); } },
      h("i", { class: `dot ${cls}` }), h("span", {}, c.question), h("b", {}, c.error ? "failed" : c.res ? miniOf(c.res) : "…"));
  }));
}

const inspOpen = () => state.insp && !!state.active?.res && !!state.drawing;
function setInspector(on) { state.insp = !!on; renderInspector(); }

function renderInspector() {
  const open = inspOpen(), was = document.body.classList.contains("insp-on");
  document.body.classList.toggle("insp-on", open);
  $("#btn-details")?.setAttribute("aria-expanded", String(open));
  if (!open) return;  // the content stays while the panel slides away
  const card = state.active, r = card.res, [cls, word, sub] = stampOf(r), list = listOf(r);
  const sec = (title, ...kids) => h("section", { class: "i-sec" }, title ? h("h3", {}, title) : null, ...kids);
  const out = [sec(null, h("p", { class: "i-q" }, card.question),
    h("div", { class: "rec-a" }, list ? (list.noun ? h("div", { class: "val" }, fmt(list.top.length), h("small", {}, list.noun)) : null) : valueEl(r),
      h("span", { class: `pill ${cls}`, style: { animation: "none" } }, icon(STATUS_ICON[cls]), cls === "none" ? sub : word)),
    summaryEl(r, sub), list && !list.noun ? h("div", { class: "i-lines" }, linesEl(list.all)) : null)];
  if (list?.noun) out.push(sec(`All ${list.top.length} ${list.noun}`, linesEl(list.all)));
  const why = whyEl(r); if (why) out.push(sec("Why there is no answer", why));
  if (r.overlay.items.length) out.push(sec(null, takeoffEl(card, true)));
  else if (r.value != null && !["planner", "none"].includes(r.engine.kind) && !r.overlay.legends?.length)
    out.push(sec(null, h("p", { class: "note" }, "Nothing on the sheet could be marked for this answer, so treat it as an estimate.")));
  out.push(sec("How it was found", specEl(r), crewEl(r)));
  const refs = refsEl(r); if (refs) out.push(sec(null, refs));
  if (r.overlay.legends?.length) { const lg = legendEl(card); if (lg) out.push(sec(null, lg)); }
  if (r.expected != null && r.correct == null)
    out.push(sec("Stoa's reference answer", h("p", { class: "note" }, "Free text, not scored"), h("pre", { class: "ref-text" }, String(r.expected))));
  const body = $("#insp-body"), keep = was && body.dataset.no === String(card.no) ? body.scrollTop : 0;
  body.replaceChildren(...out); body.dataset.no = card.no; body.scrollTop = keep;
  if (state.focus) (state.view === "3d" ? focus3D : () => applyFocus(false))();
}

// ------------------------------------------------------------------ X-ray view: how each mark was established
// basis from pipeline.overlay(): direct (geometry, BIM, sheet text), inferred (detectors), review (vision only).
// The auditor can overrule: a failed check, or a vision-only answer, puts every mark up for sign-off.
const BASIS = { direct: ["Direct", "read from geometry, the model or the sheet text"], inferred: ["Inferred", "detected from symbols / pixels"],
  review: ["Sign-off", "vision only or an auditor fail: a person must check it"] };
function needsSignOff(r) { return !!r && (r.engine?.kind === "vision" || (r.audit || []).some((c) => c.status === "fail")); }
function basisOf(it) {
  const r = state.active?.res;
  return needsSignOff(r) ? "review" : BASIS[it.basis] ? it.basis : "inferred";
}

function toggleHeat() {
  state.heatOn = !state.heatOn;
  $("#btn-heat").setAttribute("aria-pressed", state.heatOn);
  $("#stage").classList.toggle("heat", state.heatOn);
  drawHeatKey();
}

function drawHeatKey() {
  let key = $("#heat-key");
  if (!key) { key = h("div", { class: "heat-key", id: "heat-key", role: "note" }); $("#stage").append(key); }
  key.classList.toggle("hidden", !state.heatOn);
  if (!state.heatOn) return;
  const n = { direct: 0, inferred: 0, review: 0 };
  for (const it of state.items) n[basisOf(it)]++;
  const r = state.active?.res, why = needsSignOff(r) ? (r.engine.kind === "vision" ? "Vision-only answer" : "Auditor check failed") : null;
  key.replaceChildren(h("b", {}, "Confidence"),
    ...Object.entries(BASIS).map(([k, [label, sub]]) => h("span", { class: `k ${k}`, title: sub }, h("i", {}), label, h("em", {}, n[k]))),
    ...[why ? h("small", {}, `${why}: every mark needs sign-off`) : !state.items.length ? h("small", {}, "No marks on this sheet") : null].filter(Boolean));
}

function sheetOf(r, n) {  // index of the sheet that shows evidence item n
  const it = r.overlay.items.find((x) => x.n === n); if (!it) return -1;
  if (r.overlay.space === "guid") return it.sheet ? state.drawing.sheets.findIndex((x) => x.kind === "svg" && x.path === it.sheet) : -1;
  return state.drawing.sheets.findIndex((x) => x.kind === "image");  // pixel marks are drawn on the first image sheet
}

// the legend that defines the focused mark (opening_design sheets print one; other sources have none, so nothing is shown)
function legendPick(r) {
  const f = state.active?.res === r ? state.items[state.focus - 1] : null;
  return f?.legend || r.overlay.legend || [0, -1];
}

function legendEl(card) {
  const r = card.res, [li, ri] = legendPick(r), lg = r.overlay.legends[li];
  if (!lg) return null;
  const [x0, y0, x1, y1] = lg.bbox, row = lg.rows[ri], W = 300, k = W / Math.max(x1 - x0, 1);
  const viewH = Math.min(y1 - y0, 210 / k);
  const wy0 = row ? Math.min(Math.max(row.bbox[1] - viewH / 2, y0), y1 - viewH) : y0;
  const where = lg.own ? "this sheet" : lg.sheet;
  const thumb = h("button", { class: "lg-thumb", title: "Open the legend beside the drawing",
    style: { height: `${viewH * k}px`, backgroundImage: `url("${lg.src}")`, backgroundSize: `${lg.width * k}px ${lg.height * k}px`,
      backgroundPosition: `${-x0 * k}px ${-wy0 * k}px` },
    onclick: (ev) => { ev.stopPropagation(); openSplit({ legend: lg, row }); } },
    row ? h("u", { style: { left: `${(row.bbox[0] - x0) * k - 4}px`, top: `${(row.bbox[1] - wy0) * k - 3}px`,
      width: `${(row.bbox[2] - row.bbox[0]) * k + 8}px`, height: `${(row.bbox[3] - row.bbox[1]) * k + 6}px` } }) : null);
  return h("div", { class: "legend" },
    h("div", { class: "take-head" }, h("span", { class: "lbl" }, "Legend ", h("b", {}, `· ${lg.title} · ${where}`)),
      h("button", { class: "btn mini", onclick: (ev) => { ev.stopPropagation(); openSplit({ legend: lg, row }); } }, icon("split"), "Beside")),
    thumb, h("p", { class: "note" }, row ? `Defines ${row.text}` : `${lg.rows.length} entries — no entry names this mark; the whole legend is shown.`));
}

function updateLegend() {
  const box = document.querySelector("#insp-body .legend"), card = state.active;
  if (!box || !card?.res?.overlay.legends?.length) return;
  const el = legendEl(card); if (el) box.replaceWith(el);
}

// split screen: a second sheet beside the drawing (another sheet of the set, a storey, or the legend sheet)
const v2 = {
  x: 0, y: 0, k: 1, W: 1, H: 1,
  root: null, canvas: null,
  fitK() { const r = this.root.getBoundingClientRect(); return Math.min((r.width - 40) / this.W, (r.height - 40) / this.H); },
  apply(anim) { this.canvas.classList.toggle("anim", !!anim); this.canvas.style.transform = `translate(${this.x}px, ${this.y}px) scale(${this.k})`; },
  fit(anim = false) { const r = this.root.getBoundingClientRect(); this.k = this.fitK(); this.x = (r.width - this.W * this.k) / 2; this.y = (r.height - this.H * this.k) / 2; this.apply(anim); },
  zoomTo([x0, y0, x1, y1]) {
    const r = this.root.getBoundingClientRect();
    this.k = Math.max(this.fitK(), Math.min(r.width / ((x1 - x0) * 1.15), r.height / ((y1 - y0) * 1.15), 3));
    this.x = r.width / 2 - ((x0 + x1) / 2) * this.k; this.y = r.height / 2 - ((y0 + y1) / 2) * this.k; this.apply(true);
  },
};

function initSplit() {
  const st = $("#stage"), wrap = h("div", { class: "stages" });
  st.replaceWith(wrap); wrap.append(st);
  const pane = h("section", { class: "stage2 hidden", id: "stage2", "aria-label": "Second sheet" },
    h("header", {}, h("span", { class: "lbl" }, "Beside"), h("select", { id: "s2-select", "aria-label": "Sheet shown beside the drawing" }),
      h("button", { class: "btn mini", id: "s2-fit" }, "Fit"), h("button", { class: "btn mini", id: "s2-close", "aria-label": "Close the second sheet" }, "×")),
    h("div", { class: "s2-view", id: "s2-view" }, h("div", { class: "canvas", id: "s2-canvas" })));
  wrap.append(pane);
  v2.root = $("#s2-view"); v2.canvas = $("#s2-canvas");
  $("#s2-close").onclick = closeSplit; $("#s2-fit").onclick = () => v2.fit(true);
  $("#s2-select").addEventListener("change", (e) => showSplit(state.split.options[+e.target.value]));
  let drag = null;
  v2.root.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = v2.root.getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
    const k2 = Math.min(Math.max(v2.k * Math.exp(-e.deltaY * 0.0016), v2.fitK() * 0.4), v2.fitK() * 30);
    v2.x = cx - (cx - v2.x) * (k2 / v2.k); v2.y = cy - (cy - v2.y) * (k2 / v2.k); v2.k = k2; v2.apply(false);
  }, { passive: false });
  v2.root.addEventListener("pointerdown", (e) => { if (e.button === 0) drag = { x: e.clientX, y: e.clientY, ox: v2.x, oy: v2.y }; });
  window.addEventListener("pointermove", (e) => { if (drag) { v2.x = drag.ox + e.clientX - drag.x; v2.y = drag.oy + e.clientY - drag.y; v2.apply(false); } });
  window.addEventListener("pointerup", () => { drag = null; });
  new ResizeObserver(() => { if (state.split?.shown) v2.fit(false); }).observe(v2.root);
}

async function splitOptions() {
  const d = state.drawing, out = [];
  for (const lg of state.active?.res?.overlay.legends || [])
    out.push({ label: `Legend · ${lg.title} · ${lg.own ? "this sheet" : lg.sheet}`, src: lg.src, width: lg.width, height: lg.height, legend: lg });
  d.sheets.forEach((sh, i) => { if (i !== state.sheetIndex && sh.kind === "image") out.push({ label: `This drawing · ${sh.label}`, ...sh }); });
  // the rest of the set: same folder (real drawing sets) or the other floors of the same building (Swiss)
  const m = d.id.match(/^(data\/swiss_dwellings\/site\d+_bldg\d+)_/), folder = d.id.split("/").slice(0, -1).join("/");
  const q = m ? m[1].split("/").pop() : d.source === "opening_design" ? folder.split("/").slice(2).join(" ") : null;
  if (q) {
    try {
      const r = await api.get(`/api/drawings?${new URLSearchParams({ source: d.source, q, limit: 60 })}`);
      for (const it of r.items) if (it.id !== d.id && it.has_sheet) out.push({ label: `Set · ${it.title}`, drawing: it.id });
    } catch { /* the set list is optional */ }
  }
  return out;
}

async function openSplit(want = {}) {
  if (!state.drawing) return;
  const options = await splitOptions();
  if (!options.length) { setMsg("<b>Nothing to put beside this sheet</b>This drawing has no other sheets, set or legend."); setTimeout(() => setMsg(""), 2400); return; }
  state.split = { options, shown: null };
  const ov = state.active?.res?.overlay;
  if (!want.legend && ov?.legends?.length) {  // default: the legend entry that defines the answer's mark
    const [li, ri] = legendPick(state.active.res);
    want = { legend: ov.legends[li], row: ov.legends[li]?.rows[ri] };
  }
  $("#s2-select").replaceChildren(...options.map((o, i) => h("option", { value: i }, o.label)));
  $("#stage2").classList.remove("hidden"); $("#btn-split").setAttribute("aria-pressed", "true"); $("#board").classList.add("split");
  const i = Math.max(0, want.legend ? options.findIndex((o) => o.legend && o.legend.path === want.legend.path && o.legend.title === want.legend.title) : 0);
  $("#s2-select").value = i;
  await showSplit(options[i], want.row);
}

function closeSplit() {
  state.split = null; $("#stage2").classList.add("hidden"); $("#btn-split").setAttribute("aria-pressed", "false"); $("#board").classList.remove("split");
}

async function showSplit(opt, row) {
  if (!opt || !state.split) return;
  if (opt.drawing && !opt.src) {  // a sheet of another drawing in the set: fetch its first sheet
    try { const d = await api.get(`/api/drawing?id=${encodeURIComponent(opt.drawing)}`); Object.assign(opt, d.sheets.find((x) => x.kind === "image") || {}); } catch { return; }
    if (!opt.src) return;
  }
  state.split.shown = opt; v2.W = opt.width; v2.H = opt.height;
  const vb = `0 0 ${opt.width} ${opt.height}`, mark = s("svg", { class: "ov tg", viewBox: vb });
  const lg = opt.legend;
  if (lg) {
    const [x0, y0, x1, y1] = lg.bbox;
    mark.append(s("rect", { class: "lg-box", x: x0, y: y0, width: x1 - x0, height: y1 - y0, "vector-effect": "non-scaling-stroke" }));
    if (row) {
      const [a, b, c, d] = row.bbox;
      for (const cls of ["pulse ring", "pulse wave"]) mark.append(s("rect", { class: cls, x: a - 6, y: b - 5, width: c - a + 12, height: d - b + 10, rx: 4, "vector-effect": "non-scaling-stroke" }));
    }
  }
  v2.canvas.replaceChildren(h("div", { class: "sheet", style: { width: `${opt.width}px`, height: `${opt.height}px` } },
    h("img", { src: opt.src, alt: opt.label, draggable: "false" }), mark));
  await new Promise((r) => setTimeout(r, 40));  // let the pane lay out before measuring it
  v2.fit(false);
  if (lg) requestAnimationFrame(() => v2.zoomTo(row ? [Math.min(row.bbox[0], lg.bbox[0]), row.bbox[1] - 60, Math.max(row.bbox[2], lg.bbox[2]), row.bbox[3] + 60] : lg.bbox));
}

// ------------------------------------------------------------------ test report
async function openReport() {
  const dlg = $("#bench"), body = $("#bench-body");
  body.replaceChildren(h("div", { class: "wait" }, "Loading the report"));
  dlg.showModal();
  let b;
  try { b = await api.get("/api/benchmark"); } catch (e) { body.replaceChildren(h("div", { class: "group" }, h("p", {}, `The report is unavailable: ${e.message}`))); return; }
  body.replaceChildren(
    h("div", { class: "kpis" }, ...(b.kpis || []).map((k) => h("div", { class: "kpi" }, h("b", {}, k.value), h("span", {}, k.label)))),
    ...(b.groups || []).map((g) => h("section", { class: "group" }, h("h3", {}, g.title), g.note && h("p", {}, g.note),
      ...g.rows.map((r) => h("div", { class: "brow" }, h("div", {}, h("b", {}, r.label), h("small", {}, r.sub || "")),
        h("div", { class: `bar ${r.kind === "ev" ? "ev" : r.kind === "base" ? "" : "top"}`, role: "img", "aria-label": `${r.pct.toFixed(1)} percent` }, h("i", { style: { width: `${r.pct}%` } })),
        h("output", {}, `${r.pct.toFixed(1)}%`))))),
    h("div", { class: "fine" }, ...(b.notes || []).map((n) => h("p", {}, n)), b.generated && h("p", {}, `Generated ${b.generated} from the evaluation runs in this repository.`)));
}

// ------------------------------------------------------------------ wiring
function initVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition, mic = $("#mic");
  if (!SR) { mic.title = "Voice input needs Chrome or Edge"; mic.disabled = true; return; }
  const rec = new SR(); rec.interimResults = true; rec.lang = navigator.language || "en-US";
  let live = false;
  rec.onresult = (e) => { $("#q").value = [...e.results].map((r) => r[0].transcript).join(""); if (e.results[e.results.length - 1].isFinal) { rec.stop(); ask($("#q").value); } };
  rec.onend = () => { live = false; mic.classList.remove("live"); };
  rec.onerror = () => { live = false; mic.classList.remove("live"); };
  mic.onclick = () => { if (live) rec.stop(); else { live = true; mic.classList.add("live"); rec.start(); } };
}

// ------------------------------------------------------------------ upload a plan of your own
async function uploadPlan(file) {
  const card = $("#up-msg"), onStart = !$("#start").hidden;
  const msg = { set textContent(t) { card.textContent = t; if (!onStart) { setMsg(esc(t)); clearTimeout(uploadPlan.timer); uploadPlan.timer = setTimeout(() => setMsg(""), 4000); } } };
  if (!file) return;
  if (!/^image\/(png|jpeg|webp|tiff|bmp)$/.test(file.type)) { msg.textContent = "Use a PNG, JPEG, WebP, TIFF or BMP image of a drawing."; return; }
  card.textContent = `Uploading ${file.name}…`; setMsg(`Uploading ${esc(file.name)}`, true);
  let d;
  try {
    const r = await fetch("/api/upload", { method: "POST", headers: { "content-type": file.type }, body: file });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status}`);
    d = await r.json();
  } catch (e) { setMsg(""); msg.textContent = `Upload failed: ${e.message}`; return; }
  const det = d.detector || {};
  const available = det.upload_available?.length ? det.upload_available : det.available;
  const sheet = d.sheets?.[0] || {};
  const scaled = d.scaled_from ? ` · scaled from ${d.scaled_from[0]}×${d.scaled_from[1]} to ${sheet.width}×${sheet.height} px for reading` : "";
  card.textContent = (available?.length ? `${file.name} · detectors: ${available.join(" + ")}`
    : `${file.name} · no detector installed on this server: counts fall back to a vision model or are refused`) + scaled;
  await openDrawing(d.id);
}

function initUpload() {
  $("#up-file").addEventListener("change", (e) => { uploadPlan(e.target.files[0]); e.target.value = ""; });
  $("#start-choose").onclick = () => $("#up-file").click();
  $("#btn-upload").onclick = () => $("#up-file").click();
  const root = stage.root, files = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  root.addEventListener("dragover", (e) => { if (!files(e)) return; e.preventDefault(); root.classList.add("drop"); });
  root.addEventListener("dragleave", () => root.classList.remove("drop"));
  root.addEventListener("drop", (e) => { if (!files(e)) return; e.preventDefault(); root.classList.remove("drop"); uploadPlan(e.dataTransfer.files[0]); });
}

// ------------------------------------------------------------------ library toggle (⌘B) and the command palette (⌘K)
function setExamples(on) {  // the example drawings library: off on every load, until turned on
  document.body.classList.toggle("lib-off", !on);
  $("#set-examples").checked = on;
}
const examplesOn = () => !document.body.classList.contains("lib-off");

function initSettings() {
  const btn = $("#btn-settings"), menu = $("#settings");
  const show = (open) => { menu.hidden = !open; btn.setAttribute("aria-expanded", open); };
  btn.onclick = (e) => { e.stopPropagation(); show(menu.hidden); };
  document.addEventListener("click", (e) => { if (!menu.hidden && !menu.contains(e.target)) show(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { show(false); btn.focus(); } });
  $("#set-examples").addEventListener("change", (e) => setExamples(e.target.checked));
  $("#open-bench").addEventListener("click", () => show(false));
  setExamples(false);
  $("#start-examples").onclick = () => { setExamples(true); $("#grp-ex").open = true; };
  $("#lib-close").onclick = () => { setExamples(false); showStart(); };
  // "All drawings" fetches its list the first time it is opened
  $("#grp-all").addEventListener("toggle", (e) => { if (e.target.open && !state.listLoaded) { state.listLoaded = true; loadList(); } });
}

const palette = { items: [], sel: 0, timer: null, seq: 0 };
function paletteCommands() {
  const d = state.drawing;
  return [
    { label: "Show the sheet", icon: "doc", when: () => d?.sheets.length && state.view !== "sheet", run: () => setView("sheet") },
    { label: "Show the 3D model", icon: "cube", when: () => d?.has_3d && state.view !== "3d", run: () => setView("3d") },
    { label: state.graphOn ? "Hide the room graph" : "Show the room graph", icon: "graph", when: () => !$("#btn-graph").disabled, run: toggleGraph },
    { label: state.heatOn ? "Hide confidence colours" : "Colour marks by confidence", icon: "shield", run: toggleHeat },
    { label: state.split ? "Close split view" : "Put another sheet beside this one", icon: "split", run: () => (state.split ? closeSplit() : openSplit()) },
    { label: "Fit the sheet to the window", icon: "fit", when: () => state.view === "sheet", run: () => stage.fit() },
    { label: examplesOn() ? "Hide the example drawings" : "Show the example drawings", icon: "sidebar", keys: "⌘B", run: () => setExamples(!examplesOn()) },
    { label: "Upload a plan image", icon: "upload", run: () => $("#up-file").click() },
    { label: "Test report — accuracy on Stoa's questions", icon: "chart", run: openReport },
    { label: "OQVAR home page", icon: "house", run: () => { location.href = "/"; } },
  ].filter((c) => !c.when || c.when());
}

function openPalette() {
  const dlg = $("#palette");
  if (dlg.open) { $("#pal-q").select(); return; }
  $("#pal-q").value = ""; dlg.showModal(); renderPalette([]);
}

function renderPalette(drawings) {
  const q = $("#pal-q").value.trim(), ql = q.toLowerCase(), items = [];
  if (q && state.drawing) items.push({ label: `Ask “${q}”`, sub: state.drawing.title, icon: "bubble", run: () => ask(q) });
  const cmds = paletteCommands().filter((c) => !ql || c.label.toLowerCase().includes(ql));
  const sects = [];
  if (items.length) sects.push(["", items.slice()]);
  if (cmds.length) sects.push(["Actions", cmds]);
  if (!q) sects.push(["Examples", DEMOS.map(([k, id]) => ({ label: EXAMPLES[k][0], sub: k, icon: EXAMPLES[k][1], run: () => openDrawing(id) }))]);
  if (drawings.length) sects.push(["Drawings", drawings.map((d) => ({ label: d.title, sub: d.formats.slice(0, 3).join(" · ").toUpperCase(), thumb: d.id, run: () => openDrawing(d.id) }))]);
  palette.items = sects.flatMap(([, list]) => list); palette.sel = 0;
  const list = $("#pal-list");
  list.replaceChildren(...sects.flatMap(([title, entries]) => [title ? h("p", { class: "pal-sect" }, title) : null,
    ...entries.map((it) => h("button", { class: "pal-item", role: "option", "aria-selected": String(it === palette.items[0]),
      onmousemove: () => selectPalette(palette.items.indexOf(it)), onclick: () => runPalette(it) },
      it.thumb ? h("img", { loading: "lazy", alt: "", src: `/api/thumb?id=${encodeURIComponent(it.thumb)}&w=120` }) : icon(it.icon),
      h("span", { class: "pi-t" }, it.label), it.sub ? h("span", { class: "pi-sub" }, it.sub) : null, it.keys ? h("kbd", {}, it.keys) : null))]).filter(Boolean));
  if (!palette.items.length) list.append(h("p", { class: "pal-empty" }, q ? "Nothing matches. Try another word." : "Type to search"));
}

function selectPalette(i) {
  const n = palette.items.length; if (!n) return;
  palette.sel = (i + n) % n;
  const btns = [...document.querySelectorAll("#pal-list .pal-item")];
  btns.forEach((b, j) => b.setAttribute("aria-selected", j === palette.sel));
  btns[palette.sel]?.scrollIntoView({ block: "nearest" });
}

function runPalette(it) { $("#palette").close(); it?.run(); }

function initPalette() {
  const input = $("#pal-q"), dlg = $("#palette");
  input.addEventListener("input", () => {
    renderPalette([]);
    clearTimeout(palette.timer);
    const q = input.value.trim(); if (q.length < 2) return;
    const seq = ++palette.seq;
    palette.timer = setTimeout(async () => {
      try {
        const r = await api.get(`/api/drawings?${new URLSearchParams({ q, limit: 8, only_questions: false })}`);
        if (seq === palette.seq && dlg.open) renderPalette(r.items);
      } catch { /* searching drawings is optional; the actions still work */ }
    }, 160);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); selectPalette(palette.sel + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); selectPalette(palette.sel - 1); }
    else if (e.key === "Enter") { e.preventDefault(); runPalette(palette.items[palette.sel]); }
  });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  $("#open-palette").onclick = openPalette;
}

function init() {
  paintIcons(); renderExamples(); initPalette();
  stage.init(); initVoice(); initUpload(); renderThread();
  initSettings();
  $("#sources").addEventListener("change", (e) => { state.source = e.target.value || null; loadList(); });
  const q = $("#q"), grow = () => { q.style.height = "auto"; q.style.height = `${Math.min(q.scrollHeight, 120)}px`; };
  q.addEventListener("input", grow);
  let timer;
  $("#search").addEventListener("input", (e) => { clearTimeout(timer); timer = setTimeout(() => { state.q = e.target.value.trim(); loadList(); }, 220); });
  $("#only-q").addEventListener("change", (e) => { state.onlyQ = e.target.checked; loadList(); });
  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); ask($("#q").value); });
  $("#q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("#q").value); }
    else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && !q.value && state.active) { e.preventDefault(); focusItem(state.focus + (e.key === "ArrowRight" ? 1 : -1)); }
    else if (e.key === "Escape" && !$("#hist").hidden) { e.stopPropagation(); showHistory(false); }
  });
  $("#sheet-select").addEventListener("change", (e) => { state.focus = 0; showSheet(+e.target.value).then(renderThread); });
  document.querySelectorAll("#view-seg button").forEach((b) => b.addEventListener("click", () => { if (!b.disabled) setView(b.dataset.view); }));
  $("#btn-graph").addEventListener("click", () => { if (!$("#btn-graph").disabled) toggleGraph(); });
  $("#btn-heat").addEventListener("click", toggleHeat);
  $("#btn-split").addEventListener("click", () => (state.split ? closeSplit() : openSplit()));
  initSplit();
  $("#insp-close").onclick = () => setInspector(false);
  $("#btn-hist").onclick = (e) => { e.stopPropagation(); showHistory($("#hist").hidden); };
  document.addEventListener("click", (e) => { if (!$("#hist").hidden && !e.target.closest("#hist, #btn-hist")) showHistory(false); });
  $("#open-bench").onclick = openReport; $("#close-bench").onclick = () => $("#bench").close();
  $("#bench").addEventListener("click", (e) => { if (e.target.id === "bench") $("#bench").close(); });
  window.addEventListener("keydown", (e) => {
    const cmd = (e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey;
    if (cmd && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); return; }
    if (cmd && e.key.toLowerCase() === "b") { e.preventDefault(); setExamples(!examplesOn()); return; }
    if (cmd && e.key.toLowerCase() === "i") { e.preventDefault(); if (state.active?.res) setInspector(!inspOpen()); return; }
    if ($("#bench").open || $("#palette").open) return;
    if (e.target.matches("textarea, select, input:not([type=checkbox])") && !(e.target.id === "q" && !e.target.value && e.key === "Escape")) {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.key === "/") { e.preventDefault(); $("#q").focus(); }
    else if (e.key === "ArrowRight") focusItem(state.focus + 1);
    else if (e.key === "ArrowLeft") focusItem(state.focus - 1);
    else if (e.key === "Escape") {  // one step back at a time: the history menu, the focused mark, the details panel
      if (!$("#hist").hidden) showHistory(false);
      else if (state.focus) showAll();
      else if (inspOpen()) setInspector(false);
    }
  });
  const fromHash = () => (location.hash.length > 1 ? decodeURIComponent(location.hash.slice(1)) : null);
  window.addEventListener("hashchange", () => { const id = fromHash(); if (id && id !== state.drawing?.id) openDrawing(id); });
  const first = fromHash();
  if (first) openDrawing(first); else showStart();
}
init();
