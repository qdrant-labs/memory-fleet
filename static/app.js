/* Fleet Memory — Qdrant Edge showcase UI. Vanilla JS, one WebSocket. */
"use strict";

const $ = (id) => document.getElementById(id);
const ws = new WebSocket(`ws://${location.host}/ws`);
const send = (m) => ws.readyState === 1 && ws.send(JSON.stringify(m));

// ---------- state ----------
const S = {
  tracks: new Map(),     // tid -> {state, label, object_id, score, epoch}
  boxes: [],
  bursts: new Map(),
  frame: null,
  latencies: [],
  memories: 0,
  inventory: [],
  archived: new Map(),   // "tid:epoch" -> {tid, epoch, thumb, t}
  drawerMode: null,
  pop: null,
  selected: new Set(),
  expanded: null,
  fleetOn: false,
  cameraOn: true,
  searchHits: null,
  searchResults: null,
  mapPoints: [],
  scaleOn: false,
  t0: Date.now(),
};

// ---------- websocket ----------
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  const h = handlers[m.type];
  if (h) h(m);
};
ws.onclose = () => toast("connection lost — reload");

const handlers = {
  hello(m) {
    $("device-name").textContent = (m.device || "unit").toUpperCase();
    setFleet(m.fleet);
    setCamera(m.camera !== false);
    S.memories = m.memories;
    bumpCounts();
    initSliders(m.thresholds, m.detector_conf, m.detector_max_area ?? 0.2);
    send({ cmd: "inventory" });
    send({ cmd: "map" });
  },
  frame(m) {
    const img = new Image();
    img.onload = () => { S.frame = img; draw(); };
    img.src = "data:image/jpeg;base64," + m.jpg;
    S.boxes = m.boxes;
    $("t-detect").textContent = m.detect_ms + " ms";
    for (const b of m.boxes) {
      const t = S.tracks.get(b.tid);
      if (t && t.epoch !== b.epoch) S.tracks.delete(b.tid);
    }
    renderUnknownCount();
    if (S.drawerMode === "unknowns") renderUnknowns();
  },
  perf(m) {
    $("t-fps").textContent = m.fps.toFixed(1);
    $("m-detect").textContent = m.detect_ms;
    if (m.embed_ms) $("m-embed").textContent = m.embed_ms.toFixed(1);
  },
  query(m) {
    S.latencies.push(m.ms);
    if (S.latencies.length > 140) S.latencies.shift();
    S.memories = m.searched;
    $("hud-ms").textContent = fmtMs(m.ms);
    $("t-query").textContent = fmtMs(m.ms);
    bumpCounts();
    drawSpark();
  },
  track_update(m) {
    const prev = S.tracks.get(m.tid);
    S.tracks.set(m.tid, m);
    if (m.state !== "capturing") S.bursts.delete(m.tid);
    if (m.state === "recognized" && (!prev || prev.object_id !== m.object_id)) {
      pulseMapNode(m.object_id);
    }
    if (S.pop && S.pop.tid === m.tid && m.state === "recognized") hidePop();
    if (S.drawerMode === "unknowns") renderUnknowns();
  },
  burst_progress(m) { S.bursts.set(m.tid, m); },
  stats(m) { S.memories = m.memories; setFleet(m.fleet); bumpCounts(); },
  camera(m) { setCamera(m.on); },
  object_created(m) {
    toast(`taught «${m.label}» — it will remember`);
    memPulse();
    refreshData();
  },
  object_updated(m) { if (m.views) memPulse(); refreshData(); },
  object_deleted() { refreshData(); },
  objects_merged() { toast("two memories merged into one"); refreshData(); },
  rename_conflict(m) {
    if (confirm(`«${m.label}» already exists — merge into it?`))
      send({ cmd: "merge", keep_id: m.existing_id, fold_id: m.object_id });
  },
  unknown_archived(m) {
    S.archived.set(`${m.tid}:${m.epoch}`, m);
    renderUnknownCount();
    if (S.drawerMode === "unknowns") renderUnknowns();
  },
  unknown_removed(m) {
    S.archived.delete(`${m.tid}:${m.epoch}`);
    renderUnknownCount();
    if (S.drawerMode === "unknowns") renderUnknowns();
  },
  inventory(m) {
    S.inventory = m.items;
    bumpCounts();
    renderMemories();
    if (S.drawerMode === "inventory") renderInventory();
  },
  thresholds() {},
  pull_applied(m) {
    toast(`fleet pull applied${m.deduped ? ` · ${m.deduped} local deduped` : ""}`);
    refreshData();
  },
  push_done(m) {
    toast(m.count ? `${m.count} pushed — every unit now knows` : "nothing to push");
    refreshData();
  },
  fleet_error(m) { toast(`fleet: ${m.message}`); },
  search_results(m) {
    S.searchResults = m;
    S.searchHits = m.text ? new Set(m.hits.map((h) => h.object_id)) : null;
    renderSearchResults();
    if (S.drawerMode === "inventory") renderInventory();
    drawMap();
  },
  map(m) { S.mapPoints = m.points; drawMap(); },
  scale(m) {
    S.scaleOn = m.on;
    $("scale-banner").classList.toggle("hidden", !m.on);
    toast(m.on ? "⚡ 100k memories attached — watch the latency" : "stunt shard detached");
  },
  error(m) { toast(m.message); },
};

function refreshData() { send({ cmd: "inventory" }); send({ cmd: "map" }); }

function bumpCounts() {
  $("m-memories").textContent = S.memories.toLocaleString();
  $("t-count").textContent = S.memories.toLocaleString();
  $("m-objects").textContent = S.inventory.filter((o) => !o.ignored).length;
}

// ---------- clock ----------
setInterval(() => {
  const s = Math.floor((Date.now() - S.t0) / 1000);
  $("clock").textContent = `T+${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}, 1000);

// ---------- live feed canvas ----------
const view = $("view"), ctx = view.getContext("2d");
const COLORS = { recognized: "#34f0b0", suggest: "#ffbe66", unknown: "#8aa4ff",
                 ignored: "rgba(128, 133, 171, .35)", capturing: "#dc244c", pending: "#8aa4ff" };

function fit() {
  view.width = view.clientWidth * devicePixelRatio;
  view.height = view.clientHeight * devicePixelRatio;
  fitMap();
}
addEventListener("resize", () => { fit(); draw(); drawMap(); });

function mapping() {
  if (!S.frame) return null;
  const s = Math.min(view.width / S.frame.width, view.height / S.frame.height);
  const w = S.frame.width * s, h = S.frame.height * s;
  return { x: (view.width - w) / 2, y: (view.height - h) / 2, w, h };
}

function draw() {
  ctx.clearRect(0, 0, view.width, view.height);
  const m = mapping();
  if (!m || !S.cameraOn) { $("no-feed").style.display = "flex"; return; }
  $("no-feed").style.display = "none";
  ctx.drawImage(S.frame, m.x, m.y, m.w, m.h);
  for (const b of S.boxes) {
    const t = S.tracks.get(b.tid) || { state: "pending" };
    const st = t.state === "capturing" && S.bursts.has(b.tid) ? "capturing" : t.state;
    const [x1, y1, x2, y2] = b.box;
    drawBox(m.x + x1 * m.w, m.y + y1 * m.h, (x2 - x1) * m.w, (y2 - y1) * m.h, st, t, b);
  }
}

function drawBox(x, y, w, h, state, t, b) {
  ctx.save();
  const c = COLORS[state] || COLORS.pending;
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.strokeStyle = c;
  if (state === "ignored") ctx.globalAlpha = 0.3;

  if (state === "recognized" || state === "capturing") {
    brackets(x, y, w, h);
  } else {
    ctx.setLineDash(state === "suggest" ? [10, 6] : [3, 6]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }

  const px = 12 * devicePixelRatio;
  ctx.font = `700 ${px}px "JetBrains Mono", ui-monospace, Menlo, monospace`;
  if (state === "recognized") {
    chip(x, y - 8 * devicePixelRatio, ` ${t.label} · ${t.score.toFixed(2)} `, c);
  } else if (state === "suggest") {
    chip(x, y - 8 * devicePixelRatio, ` ${t.label}? tap to answer `, c);
  } else if (state === "capturing") {
    const bp = S.bursts.get(b.tid) || { have: 0, want: 6 };
    burstRing(x + w / 2, y + h / 2, Math.min(w, h) * 0.28, bp.have / bp.want);
    chip(x, y - 8 * devicePixelRatio, ` learning ${t.label}… rotate it `, COLORS.capturing);
  } else if (state !== "ignored") {
    chip(x, y - 8 * devicePixelRatio, " ? ", COLORS.unknown);
  }
  ctx.restore();
}

function brackets(x, y, w, h) {
  const L = Math.min(w, h) * 0.22;
  ctx.beginPath();
  for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
    ctx.moveTo(cx + dx * L, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * L);
  }
  ctx.stroke();
}

function chip(x, y, text, color) {
  const pad = 5 * devicePixelRatio;
  const wt = ctx.measureText(text).width + pad * 2;
  const ht = 20 * devicePixelRatio;
  ctx.fillStyle = "rgba(5, 5, 11, .88)";
  ctx.strokeStyle = color;
  ctx.lineWidth = devicePixelRatio;
  ctx.beginPath();
  ctx.roundRect(x, y - ht, wt, ht, 4 * devicePixelRatio);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(text, x + pad, y - ht / 3.4);
}

function burstRing(cx, cy, r, frac) {
  ctx.beginPath();
  ctx.strokeStyle = "rgba(220, 36, 76, .3)";
  ctx.lineWidth = 4 * devicePixelRatio;
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = "#dc244c";
  ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
  ctx.stroke();
}

// ---------- sparkline ----------
const spark = $("spark"), sctx = spark.getContext("2d");
function drawSpark() {
  const w = spark.width, h = spark.height;
  sctx.clearRect(0, 0, w, h);
  if (S.latencies.length < 2) return;
  const max = Math.max(...S.latencies, 0.5);
  sctx.strokeStyle = "rgba(52, 240, 176, .25)";
  sctx.beginPath(); sctx.moveTo(0, h - 3); sctx.lineTo(w, h - 3); sctx.stroke();
  sctx.beginPath();
  sctx.strokeStyle = "#34f0b0";
  sctx.lineWidth = 1.5;
  S.latencies.forEach((v, i) => {
    const x = (i / (S.latencies.length - 1)) * w;
    const y = h - 4 - (v / max) * (h - 12);
    i ? sctx.lineTo(x, y) : sctx.moveTo(x, y);
  });
  sctx.stroke();
}

function fmtMs(ms) { return ms < 1 ? `${Math.round(ms * 1000)} µs` : `${ms.toFixed(1)} ms`; }

// ---------- latest memories ----------
function relTime(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function renderMemories() {
  const box = $("recent-memories");
  const objs = S.inventory.filter((o) => !o.ignored && o.created)
    .sort((a, b) => b.created - a.created).slice(0, 12);
  if (!objs.length) {
    box.innerHTML = `<div class="mem-none">nothing remembered yet — teach me something</div>`;
    return;
  }
  box.innerHTML = objs.map((o) => `
    <div class="mem-row" data-id="${o.object_id}" data-label="${esc(o.label)}">
      <img class="mem-thumb" src="${o.thumb ? "data:image/jpeg;base64," + o.thumb : ""}" alt="">
      <div class="mem-main">
        <div class="mem-label">${esc(o.label)}</div>
        <div class="mem-meta">${o.views.length} vector${o.views.length === 1 ? "" : "s"}
          · learned ${relTime(o.created)}${o.sightings ? ` · seen ${o.sightings}×` : ""}${o.local ? "" : " · fleet"}</div>
      </div>
    </div>`).join("");
  box.querySelectorAll(".mem-row").forEach((el) => {
    el.onclick = () => searchFor(el.dataset.label, el.dataset.id);
  });
}

function searchFor(label, oid) {
  $("search").value = label;
  send({ cmd: "search", text: label });
  if (oid) pulseMapNode(oid);
}

// ---------- memory flow particle ----------
function memPulse() {
  const from = $("feed-wrap").getBoundingClientRect();
  const to = $("m-memories").getBoundingClientRect();
  const dot = document.createElement("div");
  dot.className = "mem-pulse";
  dot.style.left = from.left + from.width / 2 + "px";
  dot.style.top = from.top + from.height / 2 + "px";
  document.body.appendChild(dot);
  requestAnimationFrame(() => {
    dot.style.left = to.left + to.width / 2 + "px";
    dot.style.top = to.top + to.height / 2 + "px";
    dot.style.opacity = "0";
    dot.style.transform = "scale(.4)";
  });
  setTimeout(() => dot.remove(), 900);
  $("m-memories").classList.add("flash");
  setTimeout(() => $("m-memories").classList.remove("flash"), 700);
}

// ---------- memory map (constellation) ----------
const mapC = $("map-canvas"), mctx = mapC.getContext("2d");
const mapPulses = new Map(); // object_id -> pulse start time
function fitMap() {
  mapC.width = mapC.clientWidth * devicePixelRatio;
  mapC.height = mapC.clientHeight * devicePixelRatio;
}
function pulseMapNode(oid) {
  mapPulses.set(oid, performance.now());
  drawMap();
}
function mapPts() {
  const w = mapC.width, h = mapC.height, pad = 22 * devicePixelRatio;
  return S.mapPoints.map((p) => ({
    ...p, px: pad + p.x * (w - pad * 2), py: pad + p.y * (h - pad * 2),
  }));
}

function drawMap() {
  const w = mapC.width, h = mapC.height;
  if (!w) return;
  mctx.clearRect(0, 0, w, h);
  const pts = mapPts();
  const now = performance.now();
  let livePulse = false;
  mctx.font = `${9.5 * devicePixelRatio}px "JetBrains Mono", ui-monospace, monospace`;
  for (const p of pts) {
    const hit = !S.searchHits || S.searchHits.has(p.object_id);
    const color = p.local ? "#dc244c" : "#34f0b0";
    mctx.globalAlpha = hit ? 1 : 0.15;
    const pulse = mapPulses.get(p.object_id);
    if (pulse !== undefined) {
      const age = (now - pulse) / 700;
      if (age < 1) {
        livePulse = true;
        mctx.beginPath();
        mctx.strokeStyle = color;
        mctx.globalAlpha = (1 - age) * (hit ? 1 : 0.15);
        mctx.arc(p.px, p.py, (4 + age * 14) * devicePixelRatio, 0, Math.PI * 2);
        mctx.stroke();
        mctx.globalAlpha = hit ? 1 : 0.15;
      } else mapPulses.delete(p.object_id);
    }
    mctx.fillStyle = color;
    mctx.shadowColor = color;
    mctx.shadowBlur = 7 * devicePixelRatio;
    mctx.beginPath();
    mctx.arc(p.px, p.py, (S.searchHits && hit ? 5 : 3.5) * devicePixelRatio, 0, Math.PI * 2);
    mctx.fill();
    mctx.shadowBlur = 0;
    mctx.fillStyle = "#b0b4d2";
    mctx.fillText(p.label.slice(0, 13), p.px + 8 * devicePixelRatio, p.py + 3 * devicePixelRatio);
  }
  mctx.globalAlpha = 1;
  if (livePulse) requestAnimationFrame(drawMap);
}

// map hover -> preview; click -> search that memory
function mapHit(e) {
  const r = 16 * devicePixelRatio;
  const mx = e.offsetX * devicePixelRatio, my = e.offsetY * devicePixelRatio;
  let best = null, bd = r * r;
  for (const p of mapPts()) {
    const d = (p.px - mx) ** 2 + (p.py - my) ** 2;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}
mapC.addEventListener("mousemove", (e) => {
  const p = mapHit(e);
  const tip = $("map-tip");
  if (!p) { tip.classList.add("hidden"); return; }
  const inv = S.inventory.find((o) => o.object_id === p.object_id) || {};
  tip.querySelector("img").src = inv.thumb ? "data:image/jpeg;base64," + inv.thumb : "";
  tip.querySelector("span").textContent = p.label;
  tip.style.left = Math.min(e.offsetX + 12, mapC.clientWidth - 150) + "px";
  tip.style.top = Math.max(e.offsetY - 46, 4) + "px";
  tip.classList.remove("hidden");
});
mapC.addEventListener("mouseleave", () => $("map-tip").classList.add("hidden"));
mapC.addEventListener("click", (e) => {
  const p = mapHit(e);
  if (p) searchFor(p.label, p.object_id);
});

// ---------- search ----------
let searchTimer = null;
$("search").oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const text = $("search").value.trim();
    if (!text) {
      S.searchHits = null; S.searchResults = null;
      renderSearchResults(); drawMap();
      if (S.drawerMode === "inventory") renderInventory();
      return;
    }
    send({ cmd: "search", text });
  }, 220);
};

function renderSearchResults() {
  const box = $("search-results");
  if (!S.searchResults || !S.searchResults.text) {
    box.innerHTML = "";
    $("search-ms").textContent = "miniCOIL + dense · RRF";
    return;
  }
  const { hits, text, ms } = S.searchResults;
  $("search-ms").textContent = `${hits.length} hit${hits.length === 1 ? "" : "s"} · ${fmtMs(ms || 0)} on-device`;
  if (!hits.length) {
    box.innerHTML = `<div class="result-none">no memory matches «${esc(text)}»</div>`;
    return;
  }
  const max = Math.max(...hits.map((hd) => hd.score), 0.01);
  box.innerHTML = hits.map((hd) => `
    <div class="result-row" data-id="${hd.object_id}">
      <img class="result-thumb" src="${hd.thumb ? "data:image/jpeg;base64," + hd.thumb : ""}" alt="">
      <div class="result-main">
        <div class="result-label">${esc(hd.label)} ${hd.similar ? '<span class="sim-tag">looks similar</span>' : ""}</div>
        <div class="result-meta">${hd.views} vector${hd.views === 1 ? "" : "s"}${hd.sightings ? ` · seen ${hd.sightings}×` : ""}${hd.last_seen ? ` · last ${relTime(hd.last_seen)}` : ""} · ${hd.local ? "this unit" : "fleet"}</div>
      </div>
      <span class="result-score"><i style="width:${Math.round((hd.score / max) * 100)}%"></i></span>
    </div>`).join("");
  box.querySelectorAll(".result-row").forEach((el) => {
    el.onclick = () => { pulseMapNode(el.dataset.id); openDrawer("inventory"); };
  });
}

// ---------- feed interactions ----------
view.addEventListener("click", (e) => {
  const m = mapping();
  if (!m) return;
  const px = e.offsetX * devicePixelRatio, py = e.offsetY * devicePixelRatio;
  let best = null, bestArea = Infinity;
  for (const b of S.boxes) {
    const [x1, y1, x2, y2] = b.box;
    const x = m.x + x1 * m.w, y = m.y + y1 * m.h, w = (x2 - x1) * m.w, h = (y2 - y1) * m.h;
    if (px >= x && px <= x + w && py >= y && py <= y + h && w * h < bestArea) { best = b; bestArea = w * h; }
  }
  if (best) showPop(best, e.offsetX, e.offsetY); else hidePop();
});

function showPop(b, cx, cy) {
  const t = S.tracks.get(b.tid) || { state: "unknown" };
  const pop = $("popover");
  S.pop = { tid: b.tid, epoch: b.epoch };
  let html = "";
  if (t.state === "recognized") {
    html = `<h4>${esc(t.label)} · ${t.score.toFixed(2)}</h4>
      <div class="pop-row">
        <button class="pop-btn ghost" data-act="reject">not «${esc(t.label)}»</button>
        <button class="pop-btn ghost" data-act="ignore">ignore</button>
      </div>`;
  } else if (t.state === "suggest") {
    html = `<h4>looks like «${esc(t.label)}» — same?</h4>
      <div class="pop-row">
        <button class="pop-btn ok" data-act="confirm">yes</button>
        <button class="pop-btn ghost" data-act="reject">no</button>
      </div>
      <input type="text" id="teach-name" placeholder="or teach a new name…">
      <div class="pop-row"><button class="pop-btn" data-act="teach">teach</button></div>`;
  } else {
    html = `<h4>unknown — teach me</h4>
      <input type="text" id="teach-name" placeholder="what is this?">
      <div class="pop-row">
        <button class="pop-btn" data-act="teach">teach</button>
        <button class="pop-btn ghost" data-act="ignore">ignore</button>
      </div>`;
  }
  pop.innerHTML = html;
  pop.classList.remove("hidden");
  pop.style.left = Math.min(cx, view.clientWidth - 260) + "px";
  pop.style.top = Math.min(cy, view.clientHeight - 170) + "px";
  const inp = $("teach-name");
  if (inp) { inp.focus(); inp.onkeydown = (ev) => { if (ev.key === "Enter") act("teach", t); }; }
  pop.onclick = (ev) => { const a = ev.target.dataset && ev.target.dataset.act; if (a) act(a, t); };
}

function act(a, t) {
  const { tid, epoch } = S.pop || {};
  if (tid === undefined) return;
  if (a === "teach") {
    const label = ($("teach-name") || {}).value?.trim();
    if (!label) return;
    send({ cmd: "teach", tid, epoch, label });
    toast(`learning «${label}» — rotate it slowly`);
  } else if (a === "confirm") send({ cmd: "confirm", tid, epoch, object_id: t.object_id });
  else if (a === "reject") send({ cmd: "reject", tid, epoch, object_id: t.object_id });
  else if (a === "ignore") send({ cmd: "ignore_track", tid, epoch });
  hidePop();
}
function hidePop() { $("popover").classList.add("hidden"); S.pop = null; }

// ---------- drawers ----------
$("btn-unknowns").onclick = () => openDrawer("unknowns");
$("btn-inventory").onclick = () => { openDrawer("inventory"); send({ cmd: "inventory" }); };
$("drawer-close").onclick = () => closeDrawer();
$("btn-tuning").onclick = () => $("tuning").classList.toggle("hidden");
$("tuning-close").onclick = () => $("tuning").classList.add("hidden");
$("btn-pull").onclick = () => { send({ cmd: "pull_now" }); addEvent("fleet", "pulling fleet memory…"); };
$("btn-camera").onclick = () => send({ cmd: "camera", on: !S.cameraOn });

function openDrawer(mode) {
  S.drawerMode = mode;
  $("drawer-title").textContent = mode === "inventory" ? "memory · curation" : "unknowns";
  $("drawer").classList.remove("hidden");
  $("drawer-foot").classList.toggle("hidden", mode !== "inventory");
  mode === "unknowns" ? renderUnknowns(true) : renderInventory();
}
function closeDrawer() { S.drawerMode = null; $("drawer").classList.add("hidden"); }

// ---------- unknowns (live + recently departed) ----------
function liveUnknowns() {
  return S.boxes
    .filter((b) => ["unknown", "pending"].includes((S.tracks.get(b.tid) || { state: "pending" }).state))
    .sort((a, b) => b.stability - a.stability);
}
function renderUnknownCount() {
  $("unknown-count").textContent = liveUnknowns().length + S.archived.size;
}

let unkSig = "";
function renderUnknowns(force) {
  const body = $("drawer-body");
  const live = liveUnknowns();
  // re-render only when the SET changes — a per-frame rebuild stole input
  // focus mid-word and ate clicks on freshly-replaced buttons
  const sig = live.map((b) => b.tid).join(",") + "|" + [...S.archived.keys()].join(",");
  if (!force && sig === unkSig) return;
  unkSig = sig;
  let html = "";
  if (live.length) {
    html += `<div class="section-head">in view — click to teach</div>`;
    html += live.map((b) => `
      <div class="unknown-item" data-tid="${b.tid}">
        <div class="inv-main">
          <div class="inv-label">track ${b.tid}</div>
          <div class="stab" style="width:${Math.min(b.stability * 8, 100)}%"></div>
        </div>
        <span class="inv-meta">teach →</span>
      </div>`).join("");
  }
  if (S.archived.size) {
    html += `<div class="section-head">recently seen — left the frame, still teachable</div>`;
    html += [...S.archived.values()].reverse().map((a) => `
      <div class="unknown-item archived" data-key="${a.tid}:${a.epoch}" style="cursor:default">
        <img class="unk-thumb" src="${a.thumb ? "data:image/jpeg;base64," + a.thumb : ""}" alt="">
        <div class="inv-main">
          <div class="teach-inline">
            <input type="text" placeholder="what was this?" data-tid="${a.tid}" data-epoch="${a.epoch}">
            <button class="mini-btn" data-act="teach">teach</button>
            <button class="mini-btn" data-act="dismiss">✕</button>
          </div>
        </div>
      </div>`).join("");
  }
  body.innerHTML = html || `<p style="color:var(--faint);padding:8px">nothing unknown — show me something new</p>`;

  body.querySelectorAll(".unknown-item:not(.archived)").forEach((el) => {
    el.onclick = () => {
      const b = S.boxes.find((x) => x.tid === +el.dataset.tid);
      if (b) {
        const m = mapping();
        showPop(b, (m.x + b.box[0] * m.w) / devicePixelRatio + 20,
                   (m.y + b.box[1] * m.h) / devicePixelRatio + 20);
      }
    };
  });
  body.querySelectorAll(".archived").forEach((el) => {
    const input = el.querySelector("input");
    const doTeach = () => {
      const label = input.value.trim();
      if (!label) return;
      send({ cmd: "teach", tid: +input.dataset.tid, epoch: +input.dataset.epoch, label });
    };
    input.onkeydown = (ev) => { if (ev.key === "Enter") doTeach(); };
    el.querySelector('[data-act="teach"]').onclick = doTeach;
    el.querySelector('[data-act="dismiss"]').onclick = () =>
      send({ cmd: "dismiss_unknown", tid: +input.dataset.tid, epoch: +input.dataset.epoch });
  });
}

// ---------- inventory + curation ----------
function renderInventory() {
  const body = $("drawer-body");
  const objects = S.inventory.filter((o) => !o.ignored);
  const ignored = S.inventory.filter((o) => o.ignored);
  let html = "";
  if (!objects.length && !ignored.length) {
    html = `<p style="color:var(--faint);padding:8px">no memories yet — teach something</p>`;
  }
  if (objects.length) {
    html += `<div class="section-head">objects — select to push or merge</div>`;
    html += objects.map((o) => invRow(o)).join("");
  }
  if (ignored.length) {
    html += `<div class="section-head">ignored — blocklisted looks, never tracked</div>`;
    html += ignored.map((o) => invRow(o)).join("");
  }
  body.innerHTML = html;

  body.querySelectorAll(".inv-item").forEach((el) => {
    el.onclick = () => {
      S.expanded = S.expanded === el.dataset.id ? null : el.dataset.id;
      renderInventory();
    };
  });
  body.querySelectorAll(".inv-check").forEach((cb) => {
    cb.onclick = (ev) => {
      ev.stopPropagation();
      cb.checked ? S.selected.add(cb.dataset.id) : S.selected.delete(cb.dataset.id);
      updateCuration();
    };
  });
  body.querySelectorAll(".view-x").forEach((btn) => {
    btn.onclick = (ev) => {
      ev.stopPropagation();
      send({ cmd: "prune", object_id: btn.dataset.oid, view_id: btn.dataset.vid });
    };
  });
  body.querySelectorAll(".mini-btn[data-act]").forEach((btn) => {
    btn.onclick = (ev) => {
      ev.stopPropagation();
      const id = btn.closest(".inv-item").dataset.id;
      const actn = btn.dataset.act;
      if (actn === "forget" && confirm("forget this object?")) send({ cmd: "forget", object_id: id });
      if (actn === "unignore") send({ cmd: "forget", object_id: id });
      if (actn === "ignore" && confirm("blocklist this object? it will never be tracked again"))
        send({ cmd: "ignore_object", object_id: id });
      if (actn === "rename") {
        const label = prompt("new name:");
        if (label) send({ cmd: "rename", object_id: id, label });
      }
    };
  });
  updateCuration();
}

function invRow(o) {
  const dimmed = S.searchHits && !S.searchHits.has(o.object_id);
  const badge = o.ignored
    ? `<span class="badge ignored">ignored</span>`
    : `<span class="badge ${o.local ? "" : "fleet"}">${o.local ? (o.pushed ? "pushed" : "local") : "fleet"}</span>`;
  const actions = o.ignored
    ? `<button class="mini-btn" data-act="unignore" title="track this again">unignore</button>`
    : o.local
      ? `<button class="mini-btn" data-act="rename" title="rename">✎</button>
         <button class="mini-btn" data-act="ignore" title="blocklist">⊘</button>
         <button class="mini-btn" data-act="forget" title="forget">✕</button>`
      : "";
  return `
    <div class="inv-item" data-id="${o.object_id}" style="${dimmed ? "opacity:.25" : ""}">
      ${o.local && !o.ignored ? `<input type="checkbox" class="inv-check" data-id="${o.object_id}"
        ${S.selected.has(o.object_id) ? "checked" : ""}>` : ""}
      <img class="inv-thumb" src="${o.thumb ? "data:image/jpeg;base64," + o.thumb : ""}" alt="">
      <div class="inv-main">
        <div class="inv-label">${esc(o.label || "(unnamed)")} ${badge}</div>
        <div class="inv-meta">${o.views.length} vector${o.views.length === 1 ? "" : "s"} · ${esc(o.device || "")}</div>
      </div>
      <div class="inv-actions">${actions}</div>
    </div>
    ${(o.local || o.ignored) && S.expanded === o.object_id ? viewsRow(o) : ""}`;
}

function viewsRow(o) {
  return `<div class="views-row">
    ${o.views.map((v) => `
      <span class="view-cell ${v.human ? "human" : ""}" title="${v.human ? "human-taught view" : "auto-captured view"}">
        <img src="/thumbs/${v.view_id}.jpg" alt="" onerror="this.style.opacity=.12">
        <button class="view-x" data-oid="${o.object_id}" data-vid="${v.view_id}" title="prune this vector">✕</button>
      </span>`).join("")}
  </div>`;
}

function updateCuration() {
  const n = S.selected.size;
  $("btn-push").disabled = n === 0 || !S.fleetOn;
  $("btn-push").textContent = n ? `⛟ PUSH ${n} TO FLEET` : "⛟ PUSH TO FLEET";
  $("btn-merge").disabled = n !== 2;
}
$("btn-push").onclick = () => {
  if (!S.selected.size) return;
  send({ cmd: "push", object_ids: [...S.selected] });
  addEvent("fleet", `pushing ${S.selected.size} to the fleet…`);
  S.selected.clear();
  updateCuration();
};
$("btn-merge").onclick = () => {
  const [a, b] = [...S.selected];
  if (b && confirm("merge the two selected objects? the first keeps its name"))
    send({ cmd: "merge", keep_id: a, fold_id: b });
  S.selected.clear();
};

// ---------- tuning ----------
function initSliders(t, conf, maxArea) {
  const wire = (id, vid, val, fn, fmt) => {
    const el = $(id);
    const show = fmt || ((v) => (+v).toFixed(2));
    el.value = val;
    $(vid).textContent = show(val);
    el.oninput = () => { $(vid).textContent = show(el.value); fn(); tierBand(); };
  };
  const sendT = () => send({ cmd: "thresholds", s_same: +$("s-same").value,
                             s_suggest: +$("s-suggest").value, s_ignore: +$("s-ignore").value });
  wire("s-same", "v-same", t.s_same, sendT);
  wire("s-suggest", "v-suggest", t.s_suggest, sendT);
  wire("s-ignore", "v-ignore", t.s_ignore, sendT);
  wire("s-conf", "v-conf", conf, () => send({ cmd: "conf", value: +$("s-conf").value }));
  wire("s-area", "v-area", maxArea, () => send({ cmd: "max_area", value: +$("s-area").value }),
       (v) => `${Math.round(v * 100)}% of frame`);
  tierBand();
}
function tierBand() {
  const sug = +$("s-suggest").value, same = +$("s-same").value;
  const bands = document.querySelectorAll("#tier-band .band");
  bands[0].style.width = sug * 100 + "%";
  bands[1].style.width = Math.max(same - sug, 0) * 100 + "%";
  bands[2].style.width = (1 - same) * 100 + "%";
}

// ---------- camera / fleet / scale ----------
function setCamera(on) {
  S.cameraOn = on;
  $("btn-camera").classList.toggle("cam-off", !on);
  $("btn-camera").textContent = on ? "⏻ CAMERA" : "⏻ CAMERA OFF";
  $("rec").style.display = on ? "flex" : "none";
  if (!on) { S.frame = null; S.boxes = []; draw(); }
}
function setFleet(on) {
  S.fleetOn = on;
  $("fleet-dot").className = "unit-dot " + (on ? "" : "off");
  $("fleet-label").textContent = on ? "fleet linked" : "fleet offline";
  $("btn-pull").classList.toggle("hidden", !on);
}
addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "s" && !e.metaKey && !e.ctrlKey
      && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    send({ cmd: "scale", on: !S.scaleOn });
  }
});

// ---------- misc ----------
function toast(msg) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 4200);
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

fit();
setInterval(renderUnknownCount, 1000);
