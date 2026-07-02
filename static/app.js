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
  openCats: new Set(),     // drawer category boxes open (label groups)
  openMemCats: new Set(),  // latest-memories category boxes open
  fleetOn: false,
  fleetConfigured: false,
  cameraOn: true,
  searchHits: null,
  searchResults: null,
  mapPoints: [],
  scaleOn: false,
  t0: Date.now(),
  frameSeq: 0,   // decode-order guard: a slow JPEG must not overwrite a newer one
  shownSeq: 0,
  dirty: true,   // the rAF loop repaints only when something changed
};

// ---------- websocket ----------
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  const h = handlers[m.type];
  if (h) h(m);
};
ws.onclose = () => {
  S.dead = true;
  $("unit-dot").className = "unit-dot off";
  $("no-feed").textContent = "connection lost · reload the page";
  $("no-feed").style.display = "flex";
  toast("connection lost · reload");
};

const handlers = {
  hello(m) {
    $("device-name").textContent = (m.device || "unit").toUpperCase();
    S.fleetConfigured = m.fleet;
    setFleet(!!m.fleet_online); // current sync state; fleet_status keeps it live
    setCamera(m.camera !== false);
    S.memories = m.memories;
    bumpCounts();
    initSliders(m.thresholds, m.detector_conf, m.detector_max_area ?? 0.2, m.target_fps ?? 8);
    send({ cmd: "inventory" });
    send({ cmd: "map" });
  },
  frame(m) {
    // video-only frames at ~24 fps (boxes arrive separately at ~8 Hz and the
    // rAF loop interpolates); a frame may still carry boxes, applied if present
    const seq = ++S.frameSeq;
    const img = new Image();
    img.onload = () => { if (seq >= S.shownSeq) { S.shownSeq = seq; S.frame = img; S.dirty = true; } };
    img.src = "data:image/jpeg;base64," + m.jpg;
    vfpsTick();
    if (m.boxes !== undefined) applyBoxes(m);
  },
  boxes(m) { applyBoxes(m); },
  perf(m) {
    $("m-detect").textContent = m.detect_ms;
    if (m.embed_ms) {
      $("m-embed").textContent = m.embed_ms.toFixed(1);
      $("t-embed").textContent = m.embed_ms.toFixed(1) + " ms";
    }
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
    S.dirty = true;
    if (m.state !== "capturing") S.bursts.delete(m.tid);
    if (m.state === "recognized" && (!prev || prev.object_id !== m.object_id)) {
      pulseMapNode(m.object_id);
    }
    if (S.pop && S.pop.tid === m.tid && m.state === "recognized") hidePop();
    if (S.drawerMode === "unknowns") renderUnknowns();
  },
  burst_progress(m) { S.bursts.set(m.tid, m); S.dirty = true; },
  stats(m) { S.memories = m.memories; bumpCounts(); },
  fleet_status(m) {
    setFleet(m.online);
    toast(m.online ? "fleet linked · memories syncing" : "fleet unreachable · running local");
  },
  camera(m) { setCamera(m.on); },
  object_created(m) {
    toast(`taught «${m.label}» · it will remember`);
    memPulse();
    refreshData();
  },
  object_updated(m) { if (m.views) memPulse(); refreshData(); },
  object_deleted() { refreshData(); },
  objects_merged() { toast("two memories merged into one"); refreshData(); },
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
    // prune the curation selection: forgotten/merged/pushed-away ids must not
    // linger invisibly and feed a stale merge/push
    const selectable = new Set(m.items.filter((o) => o.local).map((o) => o.object_id));
    for (const id of [...S.selected]) if (!selectable.has(id)) S.selected.delete(id);
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
    toast(m.count ? `${m.count} pushed · every unit now knows` : "nothing to push");
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
    toast(m.on ? "⚡ 300k memories attached · watch the latency" : "stunt shard detached");
  },
  error(m) { toast(m.message); },
};

function applyBoxes(m) {
  S.boxes = m.boxes;
  S.dirty = true;
  $("t-detect").textContent = m.detect_ms + " ms";
  const seen = new Set(m.boxes.map((b) => b.tid));
  for (const b of m.boxes) {
    const t = S.tracks.get(b.tid);
    if (t && t.epoch !== b.epoch) { S.tracks.delete(b.tid); S.bursts.delete(b.tid); }
  }
  for (const tid of [...S.bursts.keys()]) if (!seen.has(tid)) S.bursts.delete(tid);
  renderUnknownCount();
  if (S.drawerMode === "unknowns") renderUnknowns();
}

function refreshData() { send({ cmd: "inventory" }); send({ cmd: "map" }); }

function bumpCounts() {
  $("m-memories").textContent = S.memories.toLocaleString();
  $("t-count").textContent = S.memories.toLocaleString();
  $("m-objects").textContent = S.inventory.filter((o) => !o.ignored).length;
}

// ---------- clock ----------
setInterval(() => {
  if (S.dead) return;
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
  fitSpark();
}
addEventListener("resize", () => { fit(); draw(); drawMap(); drawSpark(); });

function mapping() {
  if (!S.frame) return null;
  const s = Math.min(view.width / S.frame.width, view.height / S.frame.height);
  const w = S.frame.width * s, h = S.frame.height * s;
  return { x: (view.width - w) / 2, y: (view.height - h) / 2, w, h };
}

// ---------- render loop ----------
// Boxes arrive at detect rate (~8 Hz) while video streams at ~24 fps; each
// displayed box eases toward its latest detection so motion reads smooth.
const disp = new Map(); // tid -> currently displayed [x1, y1, x2, y2]
let lastAnimT = performance.now();
function animate(t) {
  requestAnimationFrame(animate);
  const dt = Math.min(t - lastAnimT, 100);
  lastAnimT = t;
  let moving = false;
  const alive = new Set();
  for (const b of S.boxes) {
    alive.add(b.tid);
    const cur = disp.get(b.tid);
    if (!cur) { disp.set(b.tid, b.box.slice()); moving = true; continue; }
    const k = 1 - Math.exp(-dt / 60); // ~60 ms settle, frame-rate independent
    for (let i = 0; i < 4; i++) {
      const d = b.box[i] - cur[i];
      if (Math.abs(d) > 0.0004) { cur[i] += d * k; moving = true; }
      else cur[i] = b.box[i];
    }
  }
  for (const tid of [...disp.keys()]) if (!alive.has(tid)) disp.delete(tid);
  if (S.dirty || moving) { S.dirty = false; draw(); }
}
requestAnimationFrame(animate);

// displayed video rate (received frames, 2 s window)
const vTimes = [];
function vfpsTick() {
  const t = performance.now();
  vTimes.push(t);
  while (vTimes.length && t - vTimes[0] > 2000) vTimes.shift();
}
setInterval(() => { $("t-vfps").textContent = (vTimes.length / 2).toFixed(0); }, 1000);

let tickerRect = "";
function draw() {
  if (S.dead) return;
  ctx.clearRect(0, 0, view.width, view.height);
  const m = mapping();
  if (!m || !S.cameraOn) { $("no-feed").style.display = "flex"; return; }
  $("no-feed").style.display = "none";
  ctx.drawImage(S.frame, m.x, m.y, m.w, m.h);
  // keep the pipeline pill on the VIDEO, not floating over letterbox bars
  const rect = `${m.x}|${m.y + m.h}`;
  if (rect !== tickerRect) {
    tickerRect = rect;
    const t = $("pipeline-ticker");
    const pad = 14;
    t.style.left = m.x / devicePixelRatio + pad + "px";
    t.style.right = m.x / devicePixelRatio + pad + "px";
    t.style.bottom = (view.height - m.y - m.h) / devicePixelRatio + pad + "px";
  }
  for (const b of S.boxes) {
    const t = S.tracks.get(b.tid) || { state: "pending" };
    const st = t.state === "capturing" && S.bursts.has(b.tid) ? "capturing" : t.state;
    const [x1, y1, x2, y2] = disp.get(b.tid) || b.box; // eased position
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

// ---------- sparkline (rolling recognition-query latency) ----------
const spark = $("spark"), sctx = spark.getContext("2d");
function fitSpark() {
  spark.width = spark.clientWidth * devicePixelRatio;
  spark.height = spark.clientHeight * devicePixelRatio;
}
function drawSpark() {
  const w = spark.width, h = spark.height, dpr = devicePixelRatio;
  if (!w) return;
  sctx.clearRect(0, 0, w, h);
  if (S.latencies.length < 2) return;
  const max = Math.max(...S.latencies, 0.5);
  const n = S.latencies.length;
  const px = (i) => 3 * dpr + (i / (n - 1)) * (w - 10 * dpr);
  const py = (v) => h - 4 * dpr - (v / max) * (h - 14 * dpr);
  // recessive baseline
  sctx.strokeStyle = "rgba(58, 240, 180, .18)";
  sctx.lineWidth = 1;
  sctx.beginPath(); sctx.moveTo(0, h - 3 * dpr); sctx.lineTo(w, h - 3 * dpr); sctx.stroke();
  // area wash under the line
  sctx.beginPath();
  S.latencies.forEach((v, i) => (i ? sctx.lineTo(px(i), py(v)) : sctx.moveTo(px(0), py(v))));
  sctx.lineTo(px(n - 1), h - 3 * dpr); sctx.lineTo(px(0), h - 3 * dpr); sctx.closePath();
  sctx.fillStyle = "rgba(58, 240, 180, .09)";
  sctx.fill();
  // the line itself
  sctx.beginPath();
  S.latencies.forEach((v, i) => (i ? sctx.lineTo(px(i), py(v)) : sctx.moveTo(px(0), py(v))));
  sctx.strokeStyle = "#3af0b4";
  sctx.lineWidth = 2 * dpr;
  sctx.lineJoin = sctx.lineCap = "round";
  sctx.stroke();
  // current value: dot with a punched surface ring
  const lx = px(n - 1), ly = py(S.latencies[n - 1]);
  sctx.save();
  sctx.globalCompositeOperation = "destination-out";
  sctx.beginPath(); sctx.arc(lx, ly, 5 * dpr, 0, Math.PI * 2); sctx.fill();
  sctx.restore();
  sctx.beginPath(); sctx.arc(lx, ly, 3 * dpr, 0, Math.PI * 2);
  sctx.fillStyle = "#3af0b4"; sctx.fill();
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

function memRow(o, cls = "") {
  return `
    <div class="mem-row ${cls}" data-id="${o.object_id}" data-label="${esc(o.label)}">
      <img class="mem-thumb" src="${o.thumb ? "data:image/jpeg;base64," + o.thumb : ""}" alt="">
      <div class="mem-main">
        <div class="mem-label">${esc(o.label)}</div>
        <div class="mem-meta">${o.views.length} vector${o.views.length === 1 ? "" : "s"}
          · learned ${relTime(o.created)}${o.sightings ? ` · seen ${o.sightings}×` : ""}${o.local ? "" : ` · ${esc(o.device || "fleet")}`}</div>
      </div>
    </div>`;
}

function renderMemories() {
  const box = $("recent-memories");
  const objs = S.inventory.filter((o) => !o.ignored && o.created)
    .sort((a, b) => b.created - a.created);
  if (!objs.length) {
    box.innerHTML = `<div class="mem-none">nothing remembered yet · teach me something</div>`;
    return;
  }
  // category boxes: five hats are ONE "hat" row until opened
  const cats = new Map();
  for (const o of objs) {
    const k = (o.label || "").trim().toLowerCase();
    cats.has(k) ? cats.get(k).push(o) : cats.set(k, [o]);
  }
  box.innerHTML = [...cats.entries()].slice(0, 12).map(([k, g]) => {
    if (g.length === 1) return memRow(g[0]);
    const open = S.openMemCats.has(k);
    const vecs = g.reduce((n, o) => n + o.views.length, 0);
    return `
      <div class="mem-row cat" data-cat="${esc(k)}">
        <img class="mem-thumb" src="${g[0].thumb ? "data:image/jpeg;base64," + g[0].thumb : ""}" alt="">
        <div class="mem-main">
          <div class="mem-label">${esc(g[0].label)} <span class="cat-count">×${g.length}</span></div>
          <div class="mem-meta">${g.length} instances · ${vecs} vectors · newest ${relTime(g[0].created)}</div>
        </div>
        <span class="cat-caret">${open ? "▾" : "▸"}</span>
      </div>
      ${open ? g.map((o) => memRow(o, "in-cat")).join("") : ""}`;
  }).join("");
  box.querySelectorAll(".mem-row.cat").forEach((el) => {
    el.onclick = () => {
      const k = el.dataset.cat;
      S.openMemCats.has(k) ? S.openMemCats.delete(k) : S.openMemCats.add(k);
      renderMemories();
    };
  });
  box.querySelectorAll(".mem-row[data-id]").forEach((el) => {
    // inside a category the instance opens its views; a lone memory keeps
    // the search + map-pulse beat
    el.onclick = () => el.classList.contains("in-cat")
      ? openViews(el.dataset.id)
      : searchFor(el.dataset.label, el.dataset.id);
  });
}

function openViews(oid) {
  const o = S.inventory.find((x) => x.object_id === oid);
  S.expanded = oid;
  if (o) S.openCats.add(catKey(o));  // keep its category open behind it
  openDrawer("inventory");
  send({ cmd: "inventory" });
  renderInventory();
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
// Points only — labels live in the hover tip so a dense hive stays readable.
// Wheel zooms (cursor-anchored), drag pans, double-click or ⌂ resets.
const mapC = $("map-canvas"), mctx = mapC.getContext("2d");
const mapPulses = new Map(); // object_id -> pulse start time
const mapView = { k: 1, tx: 0, ty: 0 };
function fitMap() {
  mapC.width = mapC.clientWidth * devicePixelRatio;
  mapC.height = mapC.clientHeight * devicePixelRatio;
  clampMapView();
}
function pulseMapNode(oid) {
  mapPulses.set(oid, performance.now());
  drawMap();
}
function mapPts() {
  const w = mapC.width, h = mapC.height, pad = 22 * devicePixelRatio;
  return S.mapPoints.map((p) => ({
    ...p,
    px: (pad + p.x * (w - pad * 2)) * mapView.k + mapView.tx,
    py: (pad + p.y * (h - pad * 2)) * mapView.k + mapView.ty,
  }));
}

function drawMap() {
  const w = mapC.width, h = mapC.height;
  if (!w) return;
  mctx.clearRect(0, 0, w, h);
  const pts = mapPts();
  const now = performance.now();
  let livePulse = false;
  for (const p of pts) {
    if (p.px < -20 || p.px > w + 20 || p.py < -20 || p.py > h + 20) continue;
    const hit = !S.searchHits || S.searchHits.has(p.object_id);
    const color = p.local ? "#dc244c" : "#3af0b4";
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
  }
  mctx.globalAlpha = 1;
  if (livePulse) requestAnimationFrame(drawMap);
}

// zoom + pan
function clampMapView() {
  const w = mapC.width, h = mapC.height, k = mapView.k;
  mapView.tx = Math.min(0, Math.max(w * (1 - k), mapView.tx));
  mapView.ty = Math.min(0, Math.max(h * (1 - k), mapView.ty));
}
function setMapZoom(k, cx, cy) {
  const k0 = mapView.k;
  k = Math.min(16, Math.max(1, k));
  if (k === k0) return;
  mapView.tx = cx - ((cx - mapView.tx) * k) / k0;
  mapView.ty = cy - ((cy - mapView.ty) * k) / k0;
  mapView.k = k;
  if (k === 1) { mapView.tx = 0; mapView.ty = 0; }
  clampMapView();
  mapC.classList.toggle("pannable", k > 1);
  $("map-reset").classList.toggle("hidden", k === 1);
  drawMap();
}
function resetMapView() { setMapZoom(1, 0, 0); }
$("map-reset").onclick = resetMapView;
mapC.addEventListener("dblclick", resetMapView);
mapC.addEventListener("wheel", (e) => {
  e.preventDefault();
  const dpr = devicePixelRatio;
  setMapZoom(mapView.k * Math.exp(-e.deltaY * 0.0015), e.offsetX * dpr, e.offsetY * dpr);
}, { passive: false });

let mapDrag = null, mapDragged = false;
mapC.addEventListener("pointerdown", (e) => {
  if (mapView.k <= 1) return;
  mapDrag = { x: e.clientX, y: e.clientY };
  mapDragged = false;
  mapC.setPointerCapture(e.pointerId);
  mapC.classList.add("panning");
});
mapC.addEventListener("pointerup", (e) => {
  if (!mapDrag) return;
  mapDrag = null;
  mapC.classList.remove("panning");
  if (mapC.hasPointerCapture?.(e.pointerId)) mapC.releasePointerCapture(e.pointerId);
});

// hover -> preview; click -> search that memory
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
mapC.addEventListener("pointermove", (e) => {
  const tip = $("map-tip");
  if (mapDrag) {
    tip.classList.add("hidden");
    const dpr = devicePixelRatio;
    const dx = (e.clientX - mapDrag.x) * dpr, dy = (e.clientY - mapDrag.y) * dpr;
    if (Math.abs(dx) + Math.abs(dy) > 2) mapDragged = true;
    mapView.tx += dx; mapView.ty += dy;
    mapDrag = { x: e.clientX, y: e.clientY };
    clampMapView();
    drawMap();
    return;
  }
  const p = mapHit(e);
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
  if (mapDragged) { mapDragged = false; return; } // a pan is not a click
  const p = mapHit(e);
  if (p) searchFor(p.label, p.object_id);
});

// ---------- search ----------
let searchTimer = null;
$("search").oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const text = $("search").value.trim();
    if (!text) { clearSearch(); return; }
    send({ cmd: "search", text });
  }, 220);
};

function clearSearch() {
  $("search").value = "";
  S.searchHits = null;
  S.searchResults = null;
  document.body.classList.remove("searching");
  $("search-clear").classList.add("hidden");
  renderSearchResults();
  drawMap();
}
$("search-clear").onclick = clearSearch;
// the collapsed LATEST MEMORIES bar doubles as a "back" button while searching
document.querySelector("#memories-panel .panel-title").onclick = () => {
  if (document.body.classList.contains("searching")) clearSearch();
};

function renderSearchResults() {
  const box = $("search-results");
  if (!S.searchResults || !S.searchResults.text) {
    box.innerHTML = "";
    $("search-ms").textContent = "miniCOIL + dense · RRF";
    document.body.classList.remove("searching");
    $("search-clear").classList.add("hidden");
    return;
  }
  document.body.classList.add("searching");  // collapses LATEST MEMORIES
  $("search-clear").classList.remove("hidden");
  const { hits, text, ms } = S.searchResults;
  $("search-ms").innerHTML =
    `${hits.length} hit${hits.length === 1 ? "" : "s"} · <b>${fmtMs(ms || 0)}</b> on-device`;
  if (!hits.length) {
    box.innerHTML = `<div class="result-none">no memory matches «${esc(text)}»</div>`;
    return;
  }
  const max = Math.max(...hits.map((hd) => hd.score), 0.01);
  box.innerHTML = hits.map((hd) => {
    // the unit is the location: "last seen 4m ago · kitchen-unit" answers
    // "where did I leave it?" as well as a laptop honestly can
    const where = hd.local ? "this unit" : (hd.device || "fleet");
    const seen = hd.last_seen
      ? `<span class="seen">last seen ${relTime(hd.last_seen)} · ${esc(where)}</span>`
      : esc(where);
    return `
    <div class="result-row" data-id="${hd.object_id}">
      <img class="result-thumb" src="${hd.thumb ? "data:image/jpeg;base64," + hd.thumb : ""}" alt="">
      <div class="result-main">
        <div class="result-label">${esc(hd.label)} ${hd.similar ? '<span class="sim-tag">looks similar</span>' : ""}</div>
        <div class="result-meta">${hd.views} vector${hd.views === 1 ? "" : "s"}${hd.sightings ? ` · seen ${hd.sightings}×` : ""} · ${seen}</div>
      </div>
      <span class="result-score"><i style="width:${Math.round((hd.score / max) * 100)}%"></i></span>
    </div>`;
  }).join("");
  box.querySelectorAll(".result-row").forEach((el) => {
    el.onclick = () => {
      // open the memory drawer with every stored representation of this object
      pulseMapNode(el.dataset.id);
      S.expanded = el.dataset.id;
      openDrawer("inventory");
      requestAnimationFrame(() => {
        const row = document.querySelector(`#drawer-body .inv-item[data-id="${el.dataset.id}"]`);
        if (row) row.scrollIntoView({ block: "center" });
      });
    };
  });
}

// ---------- feed interactions ----------
view.addEventListener("click", (e) => {
  const m = mapping();
  if (!m) return;
  const px = e.offsetX * devicePixelRatio, py = e.offsetY * devicePixelRatio;
  // generous hit region: union of the drawn (eased) and latest detected box —
  // a moving object outruns its 8 Hz box — plus padding and the chip strip
  // above the box, so clicking "«item»? tap to answer" works too
  const pad = 6 * devicePixelRatio, chip = 30 * devicePixelRatio;
  let best = null, bestArea = Infinity;
  for (const b of S.boxes) {
    for (const box of [disp.get(b.tid), b.box]) {
      if (!box) continue;
      const [x1, y1, x2, y2] = box;
      const x = m.x + x1 * m.w - pad, y = m.y + y1 * m.h - chip;
      const w = (x2 - x1) * m.w + pad * 2, h = (y2 - y1) * m.h + chip + pad;
      if (px >= x && px <= x + w && py >= y && py <= y + h && w * h < bestArea) {
        best = b; bestArea = w * h;
      }
    }
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
    // picture pills: pick WHICH remembered object this is (red = an ignored look)
    const pills = (t.guesses || []).map((g) => `
      <button class="guess-pill ${g.ignored ? "ign" : ""}"
              ${g.ignored ? `data-ign="1" data-iglabel="${esc(g.label)}"` : `data-oid="${g.object_id}"`}>
        <img src="${g.thumb ? "data:image/jpeg;base64," + g.thumb : ""}" alt="">
        <span>${g.ignored ? "⊘ " : ""}${esc(g.label || "ignored look")}<i>${g.score.toFixed(2)}</i></span>
      </button>`).join("");
    html = `<h4>is it one of these?</h4>
      <div class="guess-grid">${pills}</div>
      <div class="pop-row">
        <button class="pop-btn ghost" data-act="reject">none of these</button>
        <button class="pop-btn ghost" data-act="ignore">ignore</button>
      </div>
      <input type="text" id="teach-name" placeholder="or teach a new name…">
      <div class="pop-row"><button class="pop-btn" data-act="teach">teach</button></div>`;
  } else {
    const title = t.state === "ignored" ? "ignored look · teach to rescue" : "unknown · teach me";
    html = `<h4>${title}</h4>
      ${guessChips(t.guesses, b.hints)}
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
  pop.onclick = (ev) => {
    const el = ev.target.closest("[data-oid],[data-hint],[data-act],[data-ign]");
    if (!el) return;
    const d = el.dataset;
    if (d.ign) {  // "it's that ignored thing" — the view joins the blocklist
      send({ cmd: "ignore_track", tid: S.pop.tid, epoch: S.pop.epoch, label: d.iglabel || "" });
      hidePop();
      return;
    }
    if (d.oid) {  // picture pill: confirm THIS instance
      send({ cmd: "confirm", tid: S.pop.tid, epoch: S.pop.epoch, object_id: d.oid });
      hidePop();
      return;
    }
    if (d.hint) {  // one-click teach with a suggested name
      const inp = $("teach-name");
      if (inp) { inp.value = d.hint; act("teach", t); }
      return;
    }
    if (d.act) act(d.act, t);
  };
}

function guessChips(guesses, hints) {
  // picture pills: nearest memories (teach that name); red pills: blocklisted
  // looks (click = it's that ignored thing — the view joins the blocklist,
  // never spawns an object); blue chips: detector guesses
  const gs = guesses || [];
  const mem = gs.filter((g) => !g.ignored).map((g) => `
    <button class="guess-pill sm" data-hint="${esc(g.label)}">
      <img src="${g.thumb ? "data:image/jpeg;base64," + g.thumb : ""}" alt="">
      <span>${esc(g.label)}<i>${g.score.toFixed(2)}</i></span>
    </button>`);
  const ign = gs.filter((g) => g.ignored).map((g) => `
    <button class="guess-pill sm ign" data-ign="1" data-iglabel="${esc(g.label)}">
      <img src="${g.thumb ? "data:image/jpeg;base64," + g.thumb : ""}" alt="">
      <span>⊘ ${esc(g.label || "ignored look")}<i>${g.score.toFixed(2)}</i></span>
    </button>`);
  const labels = new Set(gs.map((g) => (g.label || "").toLowerCase()).filter(Boolean));
  const det = (hints || [])
    .filter((hd) => !labels.has(hd.toLowerCase()))
    .map((hd) => `<button class="hint-chip" data-hint="${esc(hd)}">${esc(hd)}</button>`);
  const all = [...mem, ...ign, ...det];
  return all.length ? `<div class="hint-row">${all.join("")}</div>` : "";
}

function act(a, t) {
  const { tid, epoch } = S.pop || {};
  if (tid === undefined) return;
  if (a === "teach") {
    const label = ($("teach-name") || {}).value?.trim();
    if (!label) return;
    send({ cmd: "teach", tid, epoch, label });
    toast(`learning «${label}» · rotate it slowly`);
  } else if (a === "confirm") send({ cmd: "confirm", tid, epoch, object_id: t.object_id });
  else if (a === "reject") send({ cmd: "reject", tid, epoch, object_id: t.object_id });
  else if (a === "ignore") {
    // ignoring a RECOGNIZED box means "ignore that remembered object",
    // not "blocklist this one anonymous view"
    if (t.state === "recognized" && t.object_id)
      send({ cmd: "ignore_object", object_id: t.object_id });
    else send({ cmd: "ignore_track", tid, epoch });
  }
  hidePop();
}
function hidePop() { $("popover").classList.add("hidden"); S.pop = null; }

// ---------- drawers ----------
$("btn-unknowns").onclick = () => openDrawer("unknowns");
$("btn-inventory").onclick = () => { openDrawer("inventory"); send({ cmd: "inventory" }); };
$("drawer-close").onclick = () => closeDrawer();
// the input lives OUTSIDE #drawer-body, so re-rendering the list never rebuilds
// it — focus and caret survive every keystroke
$("inv-filter").oninput = () => { if (S.drawerMode === "inventory") renderInventory(); };
$("btn-tuning").onclick = () => $("tuning").classList.toggle("hidden");
$("tuning-close").onclick = () => $("tuning").classList.add("hidden");
$("btn-pull").onclick = () => { send({ cmd: "pull_now" }); toast("pulling fleet memory…"); };
$("btn-camera").onclick = () => send({ cmd: "camera", on: !S.cameraOn });

function openDrawer(mode) {
  S.drawerMode = mode;
  $("drawer-title").textContent = mode === "inventory" ? "memory · curation" : "unknowns";
  $("drawer").classList.remove("hidden");
  $("drawer-foot").classList.toggle("hidden", mode !== "inventory");
  $("drawer-search").classList.toggle("hidden", mode !== "inventory");
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
  // half-typed names must survive a re-render (tracks come and go constantly)
  const typed = {};
  let focusKey = null;
  body.querySelectorAll(".archived input").forEach((i) => {
    const k = i.dataset.tid + ":" + i.dataset.epoch;
    if (i.value) typed[k] = i.value;
    if (document.activeElement === i) focusKey = k;
  });
  let html = "";
  if (live.length) {
    html += `<div class="section-head">in view · click to teach</div>`;
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
    html += `<div class="section-head">recently seen · left the frame, still teachable</div>`;
    html += [...S.archived.values()].reverse().map((a) => `
      <div class="unknown-item archived" data-key="${a.tid}:${a.epoch}" style="cursor:default">
        <img class="unk-thumb zoomable" src="${a.thumb ? "data:image/jpeg;base64," + a.thumb : ""}" alt="" title="click to enlarge">
        <div class="inv-main">
          ${guessChips(a.guesses, [])}
          <div class="teach-inline">
            <input type="text" placeholder="what was this?" data-tid="${a.tid}" data-epoch="${a.epoch}">
            <button class="mini-btn" data-act="teach">teach</button>
            <button class="mini-btn" data-act="dismiss">✕</button>
          </div>
        </div>
      </div>`).join("");
  }
  body.innerHTML = html || `<p style="color:var(--faint);padding:8px">nothing unknown · show me something new</p>`;

  body.querySelectorAll(".unknown-item:not(.archived)").forEach((el) => {
    el.onclick = () => {
      const b = S.boxes.find((x) => x.tid === +el.dataset.tid);
      const m = mapping();
      if (b && m) {
        showPop(b, (m.x + b.box[0] * m.w) / devicePixelRatio + 20,
                   (m.y + b.box[1] * m.h) / devicePixelRatio + 20);
      }
    };
  });
  body.querySelectorAll(".archived").forEach((el) => {
    const input = el.querySelector("input");
    const key = input.dataset.tid + ":" + input.dataset.epoch;
    if (typed[key]) input.value = typed[key];
    if (focusKey === key) {
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    }
    const doTeach = () => {
      const label = input.value.trim();
      if (!label) return;
      send({ cmd: "teach", tid: +input.dataset.tid, epoch: +input.dataset.epoch, label });
    };
    input.onkeydown = (ev) => { if (ev.key === "Enter") doTeach(); };
    el.querySelector('[data-act="teach"]').onclick = doTeach;
    el.querySelector('[data-act="dismiss"]').onclick = () =>
      send({ cmd: "dismiss_unknown", tid: +input.dataset.tid, epoch: +input.dataset.epoch });
    el.querySelectorAll(".hint-chip, .guess-pill").forEach((chip) => {
      chip.onclick = () => {
        const d = chip.dataset;
        if (d.ign)  // "it was that ignored thing" — its view joins the blocklist
          send({ cmd: "ignore_track", tid: +input.dataset.tid, epoch: +input.dataset.epoch, label: d.iglabel || "" });
        else if (d.hint) { input.value = d.hint; doTeach(); }
      };
    });
    const img = el.querySelector(".zoomable");
    img.onclick = () => { if (img.src) openLightbox(img.src); };
  });
}

// ---------- inventory + curation ----------
function renderInventory() {
  const body = $("drawer-body");
  // filter client-side by label + device; empty label reads as "ignored look"
  // so typing "ignored" surfaces unnamed blocklist entries
  const q = ($("inv-filter") || {}).value ? $("inv-filter").value.trim().toLowerCase() : "";
  const match = (o) => !q
    || (o.label || "ignored look").toLowerCase().includes(q)
    || (o.device || "").toLowerCase().includes(q);
  const objects = S.inventory.filter((o) => !o.ignored && match(o));
  const ignored = S.inventory.filter((o) => o.ignored && match(o));
  // category boxes: same-name instances collapse to one row; a filter query
  // opens whatever it matched (typing "hat" should show the hats, not a box)
  const grouped = (list) => {
    const cats = new Map();
    for (const o of list) {
      const k = catKey(o);
      cats.has(k) ? cats.get(k).push(o) : cats.set(k, [o]);
    }
    let out = "";
    for (const [k, g] of cats) {
      if (g.length === 1) {
        out += invRow(g[0]);
        continue;
      }
      const open = !!q || S.openCats.has(k);
      const vecs = g.reduce((n, o) => n + o.views.length, 0);
      out += `
        <div class="inv-item cat" data-cat="${esc(k)}">
          <img class="inv-thumb" src="${g[0].thumb ? "data:image/jpeg;base64," + g[0].thumb : ""}" alt="">
          <div class="inv-main">
            <div class="inv-label">${esc(g[0].label || "ignored look")} <span class="cat-count">×${g.length}</span></div>
            <div class="inv-meta">${g.length} instances · ${vecs} vector${vecs === 1 ? "" : "s"}</div>
          </div>
          <span class="cat-caret">${open ? "▾" : "▸"}</span>
        </div>`;
      if (open) out += g.map((o) => invRow(o, "in-cat")).join("");
    }
    return out;
  };
  let html = "";
  if (!objects.length && !ignored.length) {
    html = q
      ? `<p style="color:var(--faint);padding:8px">no memories match «${esc(q)}»</p>`
      : `<p style="color:var(--faint);padding:8px">no memories yet — teach something</p>`;
  }
  if (objects.length) {
    html += `<div class="section-head">objects — select to push or merge</div>`;
    html += grouped(objects);
  }
  if (ignored.length) {
    html += `<div class="section-head">ignored — never tracked or asked about</div>`;
    html += grouped(ignored);
  }
  body.innerHTML = html;

  body.querySelectorAll(".inv-item.cat").forEach((el) => {
    el.onclick = () => {
      const k = el.dataset.cat;
      S.openCats.has(k) ? S.openCats.delete(k) : S.openCats.add(k);
      renderInventory();
    };
  });
  body.querySelectorAll(".inv-item[data-id]").forEach((el) => {
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
      if (actn === "ignore" && confirm("ignore this object? it will never be tracked or asked about again"))
        send({ cmd: "ignore_object", object_id: id });
      if (actn === "rename") {
        const label = prompt("new name:");
        if (label) send({ cmd: "rename", object_id: id, label });
      }
    };
  });
  updateCuration();
}

function catKey(o) {
  // objects and blocklist entries group separately even under the same name
  return (o.ignored ? "ign:" : "obj:") + (o.label || "").trim().toLowerCase();
}

function invRow(o, cls = "") {
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
    <div class="inv-item ${cls}" data-id="${o.object_id}">
      ${o.local ? `<input type="checkbox" class="inv-check" data-id="${o.object_id}"
        ${S.selected.has(o.object_id) ? "checked" : ""}>` : ""}
      <img class="inv-thumb" src="${o.thumb ? "data:image/jpeg;base64," + o.thumb : ""}" alt="">
      <div class="inv-main">
        <div class="inv-label">${esc(o.label || (o.ignored ? "ignored look" : "(unnamed)"))} ${badge}</div>
        <div class="inv-meta">${o.views.length} vector${o.views.length === 1 ? "" : "s"} · ${esc(o.device || "")}</div>
      </div>
      <div class="inv-actions">${actions}</div>
    </div>
    ${S.expanded === o.object_id ? viewsRow(o) : ""}`;
}

function viewsRow(o) {
  // fleet objects show their vectors read-only: the mirror can't be pruned,
  // and per-view thumbnails exist only on the unit that saw them (only the
  // object thumb rides a fleet payload) — missing files render dimmed
  const editable = o.local || o.ignored;
  return `<div class="views-row">
    ${o.views.map((v) => `
      <span class="view-cell ${v.human ? "human" : ""}" title="${v.human ? "taught/confirmed by a human" : "auto-captured while recognized"}">
        <img src="/thumbs/${v.view_id}.jpg" alt="" onerror="this.style.opacity=.12">
        ${editable ? `<button class="view-x" data-oid="${o.object_id}" data-vid="${v.view_id}" title="prune this vector">✕</button>` : ""}
      </span>`).join("")}
    <span class="views-legend">${editable
      ? "green = taught by you · grey = auto-captured"
      : "fleet memory · pictures live on the unit that saw them · confirm it live to edit"}</span>
  </div>`;
}

function updateCuration() {
  const n = S.selected.size;
  const pushable = [...S.selected].filter((id) => {
    const o = S.inventory.find((x) => x.object_id === id);
    return o && !o.ignored;
  }).length;
  $("btn-push").disabled = pushable === 0 || !S.fleetOn;
  $("btn-push").textContent = pushable ? `⛟ PUSH ${pushable} TO FLEET` : "⛟ PUSH TO FLEET";
  $("btn-merge").disabled = n !== 2;
}
$("btn-push").onclick = () => {
  if (!S.selected.size) return;
  send({ cmd: "push", object_ids: [...S.selected] });
  toast(`pushing ${S.selected.size} to the fleet…`);
  S.selected.clear();
  updateCuration();
  if (S.drawerMode === "inventory") renderInventory();
};
$("btn-merge").onclick = () => {
  const [a, b] = [...S.selected];
  if (b && confirm("merge the two selected objects? the first keeps its name"))
    send({ cmd: "merge", keep_id: a, fold_id: b });
  S.selected.clear();
  updateCuration();
  if (S.drawerMode === "inventory") renderInventory();
};

// ---------- tuning ----------
function initSliders(t, conf, maxArea, targetFps) {
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
  wire("s-fps", "v-fps", targetFps, () => send({ cmd: "target_fps", value: +$("s-fps").value }),
       (v) => `${Math.round(v)}/s`);
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
  if (!on) {
    S.frame = null;
    S.boxes = [];
    draw();
    renderUnknownCount();
    if (S.drawerMode === "unknowns") renderUnknowns();  // "in view" list must empty
  }
}
function setFleet(on) {
  S.fleetOn = on;
  const pill = $("fleet-pill");
  $("fleet-dot").className = "unit-dot " + (on ? "" : "off");
  if (!S.fleetConfigured) {
    pill.className = "pill off local";
    $("fleet-label").textContent = "LOCAL ONLY";
    pill.dataset.tip = "Running fully on-device: detection, embeddings, and vector search " +
      "all happen in this process, no server. Set QDRANT_URL in .env to link a shared " +
      "fleet memory in Qdrant Cloud, so every unit knows what any unit learned.";
  } else if (on) {
    pill.className = "pill on";
    $("fleet-label").textContent = "FLEET LINKED";
    pill.dataset.tip = "Connected to the shared fleet memory (Qdrant Cloud). Objects you " +
      "curate and push become recognizable to every unit in the fleet; new fleet " +
      "memories arrive here automatically (~30 s).";
  } else {
    pill.className = "pill off";
    $("fleet-label").textContent = "FLEET OFFLINE";
    pill.dataset.tip = "The shared fleet memory (Qdrant Cloud) isn't reachable right now. " +
      "Everything still works on-device; your teachings stay local and sync " +
      "automatically when the fleet comes back.";
  }
  $("btn-pull").classList.toggle("hidden", !on);
  if (S.drawerMode === "inventory") updateCuration();
}
addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "s" && !e.metaKey && !e.ctrlKey
      && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    send({ cmd: "scale", on: !S.scaleOn });
  }
});

// ---------- lightbox ----------
function openLightbox(src) {
  const lb = $("lightbox");
  lb.querySelector("img").src = src;
  lb.classList.remove("hidden");
}
$("lightbox").onclick = () => $("lightbox").classList.add("hidden");

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
