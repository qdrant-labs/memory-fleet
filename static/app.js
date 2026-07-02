/* Fleet Memory UI — vanilla JS, one WebSocket, canvas overlay. */
"use strict";

const $ = (id) => document.getElementById(id);
const ws = new WebSocket(`ws://${location.host}/ws`);
const send = (m) => ws.readyState === 1 && ws.send(JSON.stringify(m));

// ---------- state ----------
const S = {
  tracks: new Map(),   // tid -> {state, label, object_id, score, epoch}
  boxes: [],           // latest frame's boxes
  bursts: new Map(),   // tid -> {have, want}
  frame: null,         // Image
  frameW: 0, frameH: 0,
  latencies: [],       // rolling query ms
  memories: 0,
  inventory: [],
  drawerMode: null,    // "unknowns" | "inventory" | null
  pop: null,           // popover context {tid, epoch, kind}
  selected: new Set(), // curation: object ids checked for push/merge
  expanded: null,      // curation: object id with views row open
  fleetOn: false,
  searchHits: null,    // Set of object ids matching the search box (null = no search)
  mapPoints: [],
  scaleOn: false,
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
    $("device-name").textContent = m.device || "unit";
    setFleet(m.fleet);
    S.memories = m.memories;
    initSliders(m.thresholds, m.detector_conf);
  },
  frame(m) {
    const img = new Image();
    img.onload = () => { S.frame = img; draw(); };
    img.src = "data:image/jpeg;base64," + m.jpg;
    S.boxes = m.boxes;
    for (const b of m.boxes) {
      const t = S.tracks.get(b.tid);
      if (t && t.epoch !== b.epoch) S.tracks.delete(b.tid); // stale binding
    }
    if (S.drawerMode === "unknowns") renderUnknowns();
  },
  query(m) {
    S.latencies.push(m.ms);
    if (S.latencies.length > 120) S.latencies.shift();
    S.memories = m.searched;
    $("hud-ms").textContent = fmtMs(m.ms);
    $("hud-count").textContent = S.memories.toLocaleString();
    drawSpark();
  },
  track_update(m) {
    S.tracks.set(m.tid, m);
    if (m.state !== "capturing") S.bursts.delete(m.tid);
    if (S.pop && S.pop.tid === m.tid && m.state === "recognized") hidePop();
    if (S.drawerMode === "unknowns") renderUnknowns();
  },
  burst_progress(m) { S.bursts.set(m.tid, m); },
  stats(m) { S.memories = m.memories; setFleet(m.fleet); },
  object_created(m) { toast(`taught «${m.label}»`); refreshInv(); },
  object_updated() { refreshInv(); },
  object_deleted() { toast("forgotten"); refreshInv(); },
  objects_merged() { toast("merged"); refreshInv(); },
  rename_conflict(m) {
    if (confirm(`«${m.label}» already exists — merge into it?`))
      send({ cmd: "merge", keep_id: m.existing_id, fold_id: m.object_id });
  },
  push_done(m) { toast(m.count ? `${m.count} pushed — the fleet knows` : "nothing to push"); refreshInv(); },
  fleet_error(m) { toast(`fleet: ${m.message}`); },
  inventory(m) { S.inventory = m.items; if (S.drawerMode === "inventory") renderInventory(); },
  thresholds() {},
  pull_applied(m) { toast(`fleet pull applied (${m.deduped} deduped)`); refreshInv(); },
  search_results(m) {
    S.searchHits = new Set(m.hits.map((h) => h.object_id));
    if (m.text && !m.hits.length) toast(`no memory matches «${m.text}»`);
    if (S.drawerMode === "inventory") renderInventory();
    drawMap();
  },
  map(m) { S.mapPoints = m.points; drawMap(); },
  scale(m) {
    S.scaleOn = m.on;
    $("scale-banner").classList.toggle("hidden", !m.on);
    toast(m.on ? "a year of robot memories loaded" : "stunt shard detached");
  },
  error(m) { toast(m.message); },
};

function refreshInv() { if (S.drawerMode === "inventory") send({ cmd: "inventory" }); }

// ---------- canvas ----------
const view = $("view"), ctx = view.getContext("2d");
const COLORS = { recognized: "#35d49a", suggest: "#f5b83d", unknown: "#7aa2ff",
                 ignored: "rgba(120,130,150,.35)", capturing: "#dc244c", pending: "#7aa2ff" };

function fit() {
  view.width = view.clientWidth * devicePixelRatio;
  view.height = view.clientHeight * devicePixelRatio;
}
addEventListener("resize", () => { fit(); draw(); });
fit();

// video letterboxed into canvas; returns mapping
function mapping() {
  if (!S.frame) return null;
  const cw = view.width, ch = view.height;
  const s = Math.min(cw / S.frame.width, ch / S.frame.height);
  const w = S.frame.width * s, hgt = S.frame.height * s;
  return { x: (cw - w) / 2, y: (ch - hgt) / 2, w, h: hgt };
}

function draw() {
  ctx.clearRect(0, 0, view.width, view.height);
  const m = mapping();
  if (!m) return;
  $("no-feed").style.display = "none";
  ctx.drawImage(S.frame, m.x, m.y, m.w, m.h);

  for (const b of S.boxes) {
    const t = S.tracks.get(b.tid) || { state: "pending" };
    const st = t.state === "capturing" && S.bursts.has(b.tid) ? "capturing" : t.state;
    const [x1, y1, x2, y2] = b.box;
    const x = m.x + x1 * m.w, y = m.y + y1 * m.h;
    const w = (x2 - x1) * m.w, hgt = (y2 - y1) * m.h;
    drawBox(x, y, w, hgt, st, t, b);
  }
}

function drawBox(x, y, w, h, state, t, b) {
  ctx.save();
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.strokeStyle = COLORS[state] || COLORS.pending;
  ctx.setLineDash(state === "suggest" ? [10, 6] : state === "unknown" || state === "pending" ? [3, 5] : []);
  if (state === "ignored") ctx.globalAlpha = 0.35;
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);

  const px = 12 * devicePixelRatio;
  ctx.font = `600 ${px}px ui-monospace, Menlo, monospace`;

  if (state === "recognized") {
    chip(x, y - 8 * devicePixelRatio, `${t.label} · ${t.score.toFixed(2)}`, COLORS.recognized);
  } else if (state === "suggest") {
    chip(x, y - 8 * devicePixelRatio, `${t.label}? tap to answer`, COLORS.suggest);
  } else if (state === "capturing") {
    const bp = S.bursts.get(b.tid) || { have: 0, want: 6 };
    burstRing(x + w / 2, y + h / 2, Math.min(w, h) * 0.28, bp.have / bp.want);
    chip(x, y - 8 * devicePixelRatio, `learning ${t.label}…`, COLORS.capturing);
  } else if (state !== "ignored") {
    chip(x, y - 8 * devicePixelRatio, "?", COLORS.unknown);
  }
  ctx.restore();
}

function chip(x, y, text, color) {
  const pad = 6 * devicePixelRatio;
  const wt = ctx.measureText(text).width + pad * 2;
  const ht = 20 * devicePixelRatio;
  ctx.fillStyle = "rgba(8,12,18,.85)";
  ctx.strokeStyle = color;
  ctx.lineWidth = devicePixelRatio;
  ctx.beginPath();
  ctx.roundRect(x, y - ht, wt, ht, 5 * devicePixelRatio);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(text, x + pad, y - ht / 3.2);
}

function burstRing(cx, cy, r, frac) {
  ctx.beginPath();
  ctx.strokeStyle = "rgba(220,36,76,.35)";
  ctx.lineWidth = 4 * devicePixelRatio;
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = "#dc244c";
  ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
  ctx.stroke();
}

// sparkline
const spark = $("spark"), sctx = spark.getContext("2d");
function drawSpark() {
  const w = spark.width, h = spark.height;
  sctx.clearRect(0, 0, w, h);
  if (S.latencies.length < 2) return;
  const max = Math.max(...S.latencies, 1);
  sctx.beginPath();
  sctx.strokeStyle = "#35d49a";
  sctx.lineWidth = 1.5;
  S.latencies.forEach((v, i) => {
    const x = (i / (S.latencies.length - 1)) * w;
    const y = h - 4 - (v / max) * (h - 10);
    i ? sctx.lineTo(x, y) : sctx.moveTo(x, y);
  });
  sctx.stroke();
}

function fmtMs(ms) { return ms < 1 ? `${Math.round(ms * 1000)} µs` : `${ms.toFixed(1)} ms`; }

// ---------- interactions ----------
view.addEventListener("click", (e) => {
  const m = mapping();
  if (!m) return;
  const px = e.offsetX * devicePixelRatio, py = e.offsetY * devicePixelRatio;
  let best = null, bestArea = Infinity;
  for (const b of S.boxes) {
    const [x1, y1, x2, y2] = b.box;
    const x = m.x + x1 * m.w, y = m.y + y1 * m.h, w = (x2 - x1) * m.w, h = (y2 - y1) * m.h;
    if (px >= x && px <= x + w && py >= y && py <= y + h && w * h < bestArea) {
      best = b; bestArea = w * h;
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
  pop.style.top = Math.min(cy, view.clientHeight - 160) + "px";
  const inp = $("teach-name");
  if (inp) { inp.focus(); inp.onkeydown = (ev) => { if (ev.key === "Enter") act("teach", t); }; }
  pop.onclick = (ev) => {
    const a = ev.target.dataset && ev.target.dataset.act;
    if (a) act(a, t);
  };
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
$("btn-settings").onclick = () => $("settings").classList.toggle("hidden");
$("settings-close").onclick = () => $("settings").classList.add("hidden");

function openDrawer(mode) {
  S.drawerMode = mode;
  $("drawer-title").textContent = mode === "inventory" ? "inventory · curation" : mode;
  $("drawer").classList.remove("hidden");
  $("drawer-foot").classList.toggle("hidden", mode !== "inventory");
  mode === "unknowns" ? renderUnknowns() : renderInventory();
}
function closeDrawer() { S.drawerMode = null; $("drawer").classList.add("hidden"); }

function unknowns() {
  return S.boxes
    .filter((b) => ["unknown", "pending"].includes((S.tracks.get(b.tid) || { state: "pending" }).state))
    .sort((a, b) => b.stability - a.stability);
}

function renderUnknowns() {
  $("unknown-count").textContent = unknowns().length;
  if (S.drawerMode !== "unknowns") return;
  const body = $("drawer-body");
  body.innerHTML = unknowns().map((b) => `
    <div class="unknown-item" data-tid="${b.tid}">
      <div class="inv-main">
        <div class="inv-label">track ${b.tid}</div>
        <div class="stab" style="width:${Math.min(b.stability * 8, 100)}%"></div>
      </div>
      <span class="inv-meta">teach →</span>
    </div>`).join("") || `<p style="color:var(--dim);padding:8px">nothing unknown in view</p>`;
  body.querySelectorAll(".unknown-item").forEach((el) => {
    el.onclick = () => {
      const b = S.boxes.find((x) => x.tid === +el.dataset.tid);
      if (b) {
        const m = mapping();
        showPop(b, (m.x + b.box[0] * m.w) / devicePixelRatio + 20,
                   (m.y + b.box[1] * m.h) / devicePixelRatio + 20);
      }
    };
  });
}

function renderInventory() {
  const body = $("drawer-body");
  $("drawer-foot").classList.remove("hidden");
  if (!S.inventory.length) {
    body.innerHTML = `<p style="color:var(--dim);padding:8px">no memories yet — teach something</p>`;
    updateCuration();
    return;
  }
  body.innerHTML = S.inventory.map((o) => `
    <div class="inv-item" data-id="${o.object_id}"
         style="${S.searchHits && !S.searchHits.has(o.object_id) ? "opacity:.25" : ""}">
      ${o.local ? `<input type="checkbox" class="inv-check" data-id="${o.object_id}"
        ${S.selected.has(o.object_id) ? "checked" : ""}>` : ""}
      <img class="inv-thumb" src="${o.thumb ? "data:image/jpeg;base64," + o.thumb : ""}" alt="">
      <div class="inv-main">
        <div class="inv-label">${esc(o.label)}
          <span class="badge ${o.local ? "" : "fleet"}">${o.local ? o.pushed ? "pushed" : "local" : "fleet"}</span>
        </div>
        <div class="inv-meta">${o.views.length} views · ${esc(o.device || "")}</div>
      </div>
      <div class="inv-actions">
        ${o.local ? `<button class="mini-btn" data-act="rename" title="rename">✎</button>
        <button class="mini-btn" data-act="ignore" title="ignore (blocklist)">⊘</button>
        <button class="mini-btn" data-act="forget" title="forget">✕</button>` : ""}
      </div>
    </div>
    ${o.local && S.expanded === o.object_id ? viewsRow(o) : ""}`).join("");

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
      if (btn.dataset.act === "forget" && confirm("forget this object?"))
        send({ cmd: "forget", object_id: id });
      if (btn.dataset.act === "ignore" && confirm("blocklist this object? it will never be tracked again"))
        send({ cmd: "ignore_object", object_id: id });
      if (btn.dataset.act === "rename") {
        const label = prompt("new name:");
        if (label) send({ cmd: "rename", object_id: id, label });
      }
    };
  });
  updateCuration();
}

function viewsRow(o) {
  return `<div class="views-row">
    ${o.views.map((v) => `
      <span class="view-cell ${v.human ? "human" : ""}" title="${v.human ? "human-taught" : "auto view"}">
        <img src="/thumbs/${v.view_id}.jpg" alt="" onerror="this.style.opacity=.15">
        <button class="view-x" data-oid="${o.object_id}" data-vid="${v.view_id}" title="prune this view">✕</button>
      </span>`).join("")}
  </div>`;
}

function updateCuration() {
  const n = S.selected.size;
  $("btn-push").disabled = n === 0 || !S.fleetOn;
  $("btn-push").textContent = n ? `⛟ push ${n} to fleet` : "⛟ push to fleet";
  $("btn-merge").disabled = n !== 2;
}

$("btn-push").onclick = () => {
  if (!S.selected.size) return;
  send({ cmd: "push", object_ids: [...S.selected] });
  toast(`pushing ${S.selected.size} to the fleet…`);
  S.selected.clear();
  updateCuration();
};
$("btn-merge").onclick = () => {
  const [a, b] = [...S.selected];
  if (b && confirm("merge the two selected objects? the first keeps its name"))
    send({ cmd: "merge", keep_id: a, fold_id: b });
  S.selected.clear();
};
$("btn-pull").onclick = () => send({ cmd: "pull_now" });

// ---------- settings ----------
function initSliders(t, conf) {
  const wire = (id, vid, val, fn) => {
    const el = $(id);
    el.value = val;
    $(vid).textContent = (+val).toFixed(2);
    el.oninput = () => { $(vid).textContent = (+el.value).toFixed(2); fn(); };
  };
  const sendT = () => send({ cmd: "thresholds", s_same: +$("s-same").value,
                             s_suggest: +$("s-suggest").value, s_ignore: +$("s-ignore").value });
  wire("s-same", "v-same", t.s_same, sendT);
  wire("s-suggest", "v-suggest", t.s_suggest, sendT);
  wire("s-ignore", "v-ignore", t.s_ignore, sendT);
  wire("s-conf", "v-conf", conf, () => send({ cmd: "conf", value: +$("s-conf").value }));
}

// ---------- misc ----------
// ---------- search / map / scale stunt ----------
let searchTimer = null;
$("search").oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const text = $("search").value.trim();
    if (!text) { S.searchHits = null; renderInventory(); drawMap(); return; }
    send({ cmd: "search", text });
  }, 250);
};

$("btn-map").onclick = () => {
  $("map-panel").classList.toggle("hidden");
  if (!$("map-panel").classList.contains("hidden")) send({ cmd: "map" });
};
$("map-close").onclick = () => $("map-panel").classList.add("hidden");
setInterval(() => {
  if (!$("map-panel").classList.contains("hidden")) send({ cmd: "map" });
}, 5000);

const mapC = $("map-canvas"), mctx = mapC.getContext("2d");
function drawMap() {
  if ($("map-panel").classList.contains("hidden")) return;
  const w = mapC.width, h = mapC.height, pad = 26;
  mctx.clearRect(0, 0, w, h);
  mctx.font = "10px ui-monospace, Menlo, monospace";
  for (const p of S.mapPoints) {
    const x = pad + p.x * (w - pad * 2), y = pad + p.y * (h - pad * 2);
    const hit = !S.searchHits || S.searchHits.has(p.object_id);
    mctx.globalAlpha = hit ? 1 : 0.18;
    mctx.fillStyle = p.local ? "#dc244c" : "#35d49a";
    mctx.beginPath();
    mctx.arc(x, y, S.searchHits && hit ? 6 : 4, 0, Math.PI * 2);
    mctx.fill();
    mctx.fillStyle = "#7d8ca0";
    mctx.fillText(p.label.slice(0, 14), x + 7, y + 3);
  }
  mctx.globalAlpha = 1;
}

addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "s" && !e.metaKey && !e.ctrlKey
      && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    send({ cmd: "scale", on: !S.scaleOn });
  }
});

function setFleet(on) {
  S.fleetOn = on;
  $("fleet-dot").className = "fleet-dot " + (on ? "on" : "off");
  $("fleet-label").textContent = on ? "fleet linked" : "fleet offline";
  $("btn-pull").classList.toggle("hidden", !on);
}
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
setInterval(renderUnknowns, 1000);
