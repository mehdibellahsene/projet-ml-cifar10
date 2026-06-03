"use strict";
/* ============================================================
   ML Playground — design "handoff" (3 themes, mascotte, confettis,
   animations) cable sur NOTRE backend reel (modele EfficientNetB5 +
   images CINIC-10). Vanilla JS, sans build.
   ============================================================ */

const CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"];
const FR = {
  airplane: "avion", automobile: "automobile", bird: "oiseau", cat: "chat",
  deer: "cerf", dog: "chien", frog: "grenouille", horse: "cheval",
  ship: "bateau", truck: "camion",
};
const PALETTE = ["#2b2b2b", "#e23b3b", "#ef8a2b", "#f4c531", "#3fae5a", "#2f74d0", "#8a4fc4", "#8a5a35"];

const TXT = {
  subtitle: "Un reseau de neurones (EfficientNet-B5) a appris a reconnaitre 10 objets avec ~97,9 % de reussite. Trois mini-jeux pour le defier.",
  duelName: "Le Duel",
  duelTag: "Toi contre la machine. 3 secondes par image, 10 manches — qui reconnait le mieux ?",
  duelIntro: "Une image s'affiche 3 secondes. Clique la bonne categorie (ou tape 1, 2, 3) avant que l'IA ne reponde.",
  pictoName: "Dessine je devine",
  pictoTag: "Dessine le mot impose et regarde le modele deviner — ou paniquer — trait apres trait.",
  pictoIntro: "Dessine le mot demande. Entraine sur des photos 32x32, le modele retente sa prediction a chaque trait.",
  cinicName: "Test Ultime CINIC-10",
  cinicTag: "Des images qu'il n'a jamais vues. Sauras-tu reperer quand il se trompe ?",
  cinicIntro: "3 images issues d'un autre jeu de donnees, jamais vues a l'entrainement. Clique une carte pour reveler la prediction du modele.",
};

const THEMES = [{ v: "tableau", l: "Tableau" }, { v: "cahier", l: "Cahier" }, { v: "recre", l: "Recre" }];

window.__juice = 1.2;
window.__mascot = true;

// version de l'appli = cache-buster ?v=N du <script> (une seule source de verite)
const APP_VERSION = (() => {
  const sc = document.querySelector('script[src*="app.js"]');
  const m = sc && sc.src.match(/[?&]v=(\d+)/);
  return m ? "v" + m[1] : "dev";
})();

/* ---------------- icones ---------------- */
const ICONS = {
  logo: '<path d="M5 12h3l2-5 4 12 2-7h3" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  duel: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4l9 9M4 7V4h3M13 13l3 3 1 4-4-1-3-3"/><path d="M20 4l-9 9M20 7V4h-3M11 13l-3 3-1 4 4-1 3-3"/></g>',
  brush: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l4 4L9 19l-5 1 1-5z"/><path d="M13 7l4 4"/></g>',
  scope: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="6"/><path d="M11 5v2M11 15v2M5 11h2M15 11h2"/><circle cx="11" cy="11" r="1.6" fill="currentColor" stroke="none"/></g>',
  arrow: '<path d="M5 12h13M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  back: '<path d="M19 12H6M11 6l-5 6 5 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  refresh: '<path d="M4 11a8 8 0 0 1 14-5l2 2M20 13a8 8 0 0 1-14 5l-2-2M18 4v4h-4M6 20v-4h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
  trash: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></g>',
  dice: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="4"/><circle cx="9" cy="9" r="1.2" fill="currentColor"/><circle cx="15" cy="15" r="1.2" fill="currentColor"/><circle cx="15" cy="9" r="1.2" fill="currentColor"/><circle cx="9" cy="15" r="1.2" fill="currentColor"/></g>',
  spark: '<path d="M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z" fill="currentColor"/>',
  eraser: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 15l6-6 6 6-4 4H9z"/><path d="M8 21h12"/></g>',
  fill: '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 11l6-6 6 6-6 6z"/><path d="M11 5V3"/><path d="M19 14s2 2 2 3.5a2 2 0 0 1-4 0C17 16 19 14 19 14z" fill="currentColor"/></g>',
};
function icon(name, size = 24, style = "") {
  return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" style="${style}" aria-hidden="true">${ICONS[name]}</svg>`;
}

/* ---------------- mascotte robot ---------------- */
function mascot(mood = "idle", size = 64, bob = true) {
  if (window.__mascot === false) return "";
  const eye = ({ idle: 3.2, think: 2.4, happy: 3.4, sad: 3, win: 3.6 })[mood] || 3.2;
  const a = "var(--accent)";
  let eyes;
  if (mood === "happy" || mood === "win")
    eyes = `<g stroke="${a}" stroke-width="2.6" fill="none" stroke-linecap="round"><path d="M22 30q3 -4 6 0"/><path d="M36 30q3 -4 6 0"/></g>`;
  else if (mood === "sad")
    eyes = `<g stroke="${a}" stroke-width="2.6" fill="none" stroke-linecap="round"><path d="M22 31q3 3 6 0"/><path d="M36 31q3 3 6 0"/></g>`;
  else
    eyes = `<g fill="${a}"><circle cx="25" cy="30" r="${eye}"/><circle cx="39" cy="30" r="${eye}"/></g>`;
  return `<span class="mascot${bob ? " bob" : ""}" style="width:${size}px;height:${size}px"><svg viewBox="0 0 64 64" width="${size}" height="${size}"><g class="m-body">
    <line x1="32" y1="9" x2="32" y2="16" stroke="var(--muted)" stroke-width="2.4" stroke-linecap="round"/>
    <circle cx="32" cy="7" r="3.4" fill="${a}"/>
    <rect x="13" y="15" width="38" height="32" rx="12" fill="var(--surface-2)" stroke="var(--line)" stroke-width="2"/>
    <rect x="18" y="20" width="28" height="20" rx="8" fill="var(--bg-2)" opacity=".55"/>
    ${eyes}
    <circle cx="19.5" cy="35" r="2" fill="var(--p-pink)" opacity=".8"/>
    <circle cx="44.5" cy="35" r="2" fill="var(--p-pink)" opacity=".8"/>
    <rect x="9" y="26" width="4" height="10" rx="2" fill="var(--muted)"/>
    <rect x="51" y="26" width="4" height="10" rx="2" fill="var(--muted)"/>
    <rect x="22" y="47" width="20" height="9" rx="5" fill="var(--surface-2)" stroke="var(--line)" stroke-width="2"/>
  </g></svg></span>`;
}

/* ---------------- confettis ---------------- */
let _cc = null, _parts = [], _raf = null;
function ensureCanvas() {
  if (_cc) return _cc;
  _cc = document.createElement("canvas");
  _cc.className = "confetti-canvas";
  document.body.appendChild(_cc);
  const resize = () => { _cc.width = innerWidth; _cc.height = innerHeight; };
  resize(); addEventListener("resize", resize);
  return _cc;
}
function _tick() {
  const ctx = _cc.getContext("2d");
  ctx.clearRect(0, 0, _cc.width, _cc.height);
  _parts = _parts.filter((p) => p.life > 0);
  _parts.forEach((p) => {
    p.vy += 0.28; p.x += p.vx; p.y += p.vy; p.rot += p.vr; p.life--;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot);
    ctx.globalAlpha = Math.min(1, p.life / 24); ctx.fillStyle = p.c;
    if (p.shape === 0) ctx.fillRect(-p.s / 2, -p.s / 2, p.s, p.s);
    else { ctx.beginPath(); ctx.arc(0, 0, p.s / 2, 0, 7); ctx.fill(); }
    ctx.restore();
  });
  if (_parts.length) _raf = requestAnimationFrame(_tick);
  else { cancelAnimationFrame(_raf); _raf = null; }
}
function readVar(n, fb) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  return v || fb;
}
function fireConfetti(x, y, opts = {}) {
  if (window.__juice === 0) return;
  ensureCanvas();
  const cols = [readVar("--p-pink", "#f3a8c1"), readVar("--p-yellow", "#f0db95"), readVar("--p-mint", "#a6e0c1"),
                readVar("--p-blue", "#a7cdf1"), readVar("--p-lilac", "#cab7e9"), readVar("--p-orange", "#f2bb8c"), readVar("--accent", "#a6e0c1")];
  x = x ?? innerWidth / 2; y = y ?? innerHeight / 2;
  const n = Math.round((opts.count || 90) * (window.__juice ?? 1));
  for (let i = 0; i < n; i++) {
    const ang = Math.random() * Math.PI * 2, sp = 4 + Math.random() * 11;
    _parts.push({ x, y, vx: Math.cos(ang) * sp, vy: Math.sin(ang) * sp - 4, vr: (Math.random() - .5) * .4,
                  rot: Math.random() * 7, s: 5 + Math.random() * 8, c: cols[i % cols.length], shape: Math.random() < .5 ? 0 : 1, life: 55 + Math.random() * 35 });
  }
  if (!_raf) _raf = requestAnimationFrame(_tick);
}

/* ---------------- utilitaires ---------------- */
const $ = (id) => document.getElementById(id);
function shuffle(a) { const r = a.slice(); for (let i = r.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [r[i], r[j]] = [r[j], r[i]]; } return r; }
function pick(a) { return a[Math.floor(Math.random() * a.length)]; }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function centerOf(el) { const r = el ? el.getBoundingClientRect() : null; return r ? { x: r.left + r.width / 2, y: r.top + r.height / 2 } : { x: null, y: null }; }

function cheatFlash() {
  const o = document.createElement("div");
  o.className = "cheat-flash";
  o.innerHTML = `ARRETE DE<br>TRICHER ! &#128544;`;
  document.body.appendChild(o);
  setTimeout(() => o.remove(), 2600);
}

let _toastT = null;
function toast(msg) {
  let t = $("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show");
  clearTimeout(_toastT); _toastT = setTimeout(() => t.classList.remove("show"), 3200);
}

function dataUrlToBlob(dataUrl) {
  const [head, b64] = dataUrl.split(",");
  const mime = head.match(/:(.*?);/)[1];
  const bytes = atob(b64); const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}
async function predictBlob(blob) {
  const fd = new FormData(); fd.append("file", blob, "image.png");
  const res = await fetch("/api/predict", { method: "POST", body: fd });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || "Erreur de prediction"); }
  return res.json();
}
async function fetchRound(n) {
  const res = await fetch(`/api/game/round?n=${n}`);
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || "Images indisponibles"); }
  return (await res.json()).images;
}

/* ---------------- leaderboard ---------------- */
// notes decalees : le top 10 ne recoit que du A/B (affiche en vert),
// au-dela du top 10 : C (11-20) puis F (21+)
const GRADES = ["A+", "A", "A", "A-", "A-", "B+", "B+", "B", "B", "B-"];
const gradeFor = (i) => GRADES[Math.min(i, GRADES.length - 1)];
// egalite : meme valeur -> meme note (classement "competition")
function tieIdx(entries, valOf) {
  let idx = 0;
  return entries.map((e, i) => {
    if (i > 0 && valOf(e) !== valOf(entries[i - 1])) idx = i;
    return idx;
  });
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function fetchLeaderboard() {
  const empty = { duel: [], picto_fastest: [], picto_by_cat: {} };
  try { const r = await fetch("/api/leaderboard"); if (!r.ok) return empty; return r.json(); }
  catch { return empty; }
}
async function postScore(payload) {
  const r = await fetch("/api/score", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload) });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || "Erreur d'enregistrement"); }
  return r.json();
}
function floatPts(x, y, txt) {
  const d = document.createElement("div"); d.className = "float-pts"; d.style.left = x + "px"; d.style.top = y + "px";
  d.textContent = txt; document.body.appendChild(d); setTimeout(() => d.remove(), 1000);
}
function renderSubmit(container, game, score, image) {
  const last = esc(localStorage.getItem("mlp_name") || "");
  container.innerHTML = `<div style="display:flex;gap:8px"><input class="lb-input" maxlength="16" placeholder="Ton pseudo" value="${last}"><button class="btn primary lb-go">Enregistrer</button></div><div class="lb-res" style="margin-top:10px;font-weight:800;min-height:20px"></div>`;
  const inp = container.querySelector(".lb-input"), btn = container.querySelector(".lb-go"), res = container.querySelector(".lb-res");
  const send = async () => {
    const name = (inp.value || "").trim() || "Anonyme";
    localStorage.setItem("mlp_name", name); btn.disabled = true; inp.disabled = true;
    try {
      const r = await postScore({ game, name, score });
      res.innerHTML = r.rank != null
        ? `<span class="ok">Classe !</span> Note <b>${gradeFor(r.rank)}</b> &middot; #${r.rank + 1} du top`
        : `Pas dans le top 10 cette fois — retente !`;
    } catch (e) { res.textContent = e.message; btn.disabled = false; inp.disabled = false; }
  };
  btn.onclick = send; inp.onkeydown = (e) => { if (e.key === "Enter") send(); };
}
// top 10 : note verte (A/B) ; #11-20 : C ; #21+ : F
function rankBadge(ti) {
  if (ti < 10) return `<span class="lb-grade gA">${gradeFor(ti)}</span>`;
  return ti < 20 ? `<span class="lb-grade gC">C</span>` : `<span class="lb-grade gF">F</span>`;
}
function lbDuel(entries) {
  const tie = tieIdx(entries, (e) => e.score);
  const rows = entries.length
    ? entries.map((e, i) => `<tr><td>${rankBadge(tie[i])}</td>
        <td class="nm">${esc(e.name)}</td><td class="sc">${e.score} pts</td></tr>`).join("")
    : `<tr><td colspan="3" class="lb-empty">Aucun score — sois le premier !</td></tr>`;
  return `<div class="lb-card"><h3><span class="ti">${icon("duel", 15)}</span>${TXT.duelName}</h3>
    ${entries.length > 10 ? `<p class="lb-sub" style="margin:-6px 0 8px">${entries.length} joueurs &middot; scrolle pour voir au-dela du top 10</p>` : ""}
    <div class="lb-scroll"><table class="lb"><tbody>${rows}</tbody></table></div></div>`;
}
function lbFastest(entries) {
  const tie = tieIdx(entries, (e) => Number(e.time));
  const rows = entries.length
    ? entries.map((e, i) => `<tr>
        <td>${rankBadge(tie[i])}</td>
        <td>${e.image ? `<img class="lb-thumb" src="${e.image}" alt="">` : ""}</td>
        <td class="nm">${esc(e.name)}<div class="lb-sub">${FR[e.category] || ""}</div></td>
        <td class="sc">${Number(e.time).toFixed(2)} s</td></tr>`).join("")
    : `<tr><td colspan="4" class="lb-empty">Aucun dessin — sois le premier !</td></tr>`;
  return `<div class="lb-card"><h3><span class="ti">${icon("brush", 15)}</span>${TXT.pictoName} — les plus rapides</h3>
    ${entries.length > 10 ? `<p class="lb-sub" style="margin:-6px 0 8px">${entries.length} dessinateurs &middot; scrolle pour voir au-dela du top 10</p>` : ""}
    <div class="lb-scroll"><table class="lb"><tbody>${rows}</tbody></table></div></div>`;
}
function lbByCat(byCat) {
  const cells = CLASSES.map((c) => {
    const e = byCat[c];
    return `<div class="cat-cell">
      <div class="cat-pic">${e && e.image ? `<img src="${e.image}" alt="">` : `<span class="cat-empty">?</span>`}</div>
      <div class="cat-name">${FR[c]}</div>
      <div class="cat-meta">${e ? `${esc(e.name)} &middot; ${Number(e.time).toFixed(2)} s` : "libre"}</div>
    </div>`;
  }).join("");
  return `<div class="lb-card" style="margin-top:22px"><h3><span class="ti">${icon("brush", 15)}</span>Champions par categorie</h3><div class="cat-grid">${cells}</div></div>`;
}
function lbHonor(entries) {
  if (!entries || !entries.length)
    return `<div class="lb-card" style="margin-top:22px"><h3><span class="ti">${icon("brush", 15)}</span>Mentions honorables</h3><p class="lb-empty">Les dessins devines apparaitront ici.</p></div>`;
  const items = entries.map((e) => `<div class="honor-item">
      ${e.image ? `<img src="${e.image}" alt="">` : `<div class="honor-ph">?</div>`}
      <div class="honor-cap"><b>${esc(e.name)}</b><br>${FR[e.category] || ""} &middot; ${Number(e.time).toFixed(2)} s</div>
    </div>`).join("");
  return `<div class="lb-card" style="margin-top:22px">
    <h3><span class="ti">${icon("brush", 15)}</span>Mentions honorables</h3>
    <p class="lb-sub" style="margin:-6px 0 10px">Tous les dessins devines, meme les moins rapides &middot; defile &rarr;</p>
    <div class="honor-strip">${items}</div></div>`;
}

/* ---------------- theme (fixe : cahier) ---------------- */
function applyTheme() {
  document.documentElement.setAttribute("data-theme", "cahier");
  document.documentElement.style.setProperty("--juice", String(window.__juice));
}

/* ---------------- shell + routeur ---------------- */
let cleanup = null;
function screenHead(ic, title) {
  return `<div class="screen-head">
    <button class="crumb" id="crumbHome">${icon("back", 14, "vertical-align:-2px;margin-right:5px")}Accueil</button>
    <h2 class="screen-title"><span class="ti">${icon(ic, 22)}</span><span class="tt">${title}</span></h2>
  </div>`;
}
function wireHome(scope) { const c = scope.querySelector("#crumbHome"); if (c) c.onclick = go.bind(null, "home"); }

function renderShell() {
  const root = $("root");
  root.innerHTML =
    `<div class="wrap">
       <header class="topbar">
         <h1 class="wordmark" id="wordmark"><span class="logo" style="background:transparent;box-shadow:none;transform:none">${mascot("happy", 56, true)}</span> ML Playground</h1>
         <p class="subtitle">${TXT.subtitle}</p>
       </header>
       <main id="screen"></main>
       <div class="foot"><span>EfficientNet-B5 &middot; projet ML CIFAR-10 &middot; BELLAHSENE Mehdi Redha</span><span>10 categories &middot; ~97,9 % en test &middot; ${APP_VERSION}</span></div>
     </div>`;
  $("wordmark").onclick = () => go("home");
}

function go(name) {
  if (cleanup) { cleanup(); cleanup = null; }
  const s = $("screen");
  document.body.dataset.screen = name;   // sur mobile, l'en-tete se replie en jeu
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "home") return renderHome(s);
  if (name === "duel") return startDuel(s);
  if (name === "picto") return startPicto(s);
  if (name === "cinic") return startCinic(s);
}

/* ---------------- accueil ---------------- */
function renderHome(s) {
  const card = (accent, ic, title, desc, target) =>
    `<button class="game-card" data-go="${target}" style="--cardaccent:${accent}">
       <span class="badge">${icon(ic, 30)}</span>
       <h3>${title}</h3><p>${desc}</p>
       <span class="go">Jouer ${icon("arrow", 14)}</span>
     </button>`;
  s.innerHTML = `<div class="fadeup">
    <div class="pseudo-bar">
      <span class="pseudo-q">Qui joue ?</span>
      <input id="homeName" class="lb-input" maxlength="16" placeholder="Ton pseudo" autocomplete="off">
    </div>
    <div class="home-grid">
      ${card("var(--p-blue)", "duel", TXT.duelName, TXT.duelTag, "duel")}
      ${card("var(--p-pink)", "brush", TXT.pictoName, TXT.pictoTag, "picto")}
      ${card("var(--p-mint)", "scope", TXT.cinicName, TXT.cinicTag, "cinic")}
    </div>
    <div id="lbWrap" class="lb-wrap"></div>
    <div id="lbCat"></div>
    <div id="lbHonor"></div></div>`;
  s.querySelectorAll(".game-card").forEach((c) => (c.onclick = () => go(c.dataset.go)));
  const hn = $("homeName");
  hn.value = localStorage.getItem("mlp_name") || "";
  hn.oninput = () => { const v = hn.value.trim(); if (v) localStorage.setItem("mlp_name", v); };

  // fleche incitative : il y a les classements + dessins plus bas
  const cue = document.createElement("div");
  cue.className = "scrollcue";
  cue.innerHTML = `<span>Classements &amp; dessins en bas</span>${icon("arrow", 20, "transform:rotate(90deg)")}`;
  document.body.appendChild(cue);
  cue.onclick = () => { const w = $("lbWrap"); if (w) w.scrollIntoView({ behavior: "smooth", block: "start" }); };
  const onScroll = () => {
    // hysteresis : la barre d'adresse mobile genere des scrolls parasites,
    // un seuil unique ferait clignoter la fleche
    const notScrollable = document.documentElement.scrollHeight <= window.innerHeight + 40;
    if (notScrollable || window.scrollY > 140) cue.classList.add("hide");
    else if (window.scrollY < 80) cue.classList.remove("hide");
  };
  addEventListener("scroll", onScroll);
  cleanup = () => { removeEventListener("scroll", onScroll); cue.remove(); };

  fetchLeaderboard().then((lb) => {
    if ($("lbWrap")) $("lbWrap").innerHTML = lbDuel(lb.duel || []) + lbFastest(lb.picto_fastest || []);
    if ($("lbCat")) $("lbCat").innerHTML = lbByCat(lb.picto_by_cat || {});
    if ($("lbHonor")) $("lbHonor").innerHTML = lbHonor(lb.picto_history || []);
    onScroll();
  });
  setTimeout(onScroll, 60);
}

/* ============================================================
   JEU 1 — LE DUEL (vrai modele : l'IA = prediction du modele)
   ============================================================ */
const TIME_MS = 3000;
const JOKERS = [
  { type: "duo",   label: "50 / 50",        cls: "jk-blue", msg: "Joker Duo : 2 choix au prochain tour !" },
  { type: "mult",  mult: 10,  label: "x10",  cls: "jk-gold", msg: "Multiplicateur x10 arme !" },
  { type: "mult",  mult: 50,  label: "x50",  cls: "jk-gold", msg: "Multiplicateur x50 arme !" },
  { type: "mult",  mult: 100, label: "x100", cls: "jk-gold", msg: "Jackpot x100 arme !" },
  { type: "blind", label: "Aveugler l'IA",  cls: "jk-pink", msg: "IA debranchee au prochain tour !" },
  { type: "addq",  add: 3, label: "+3 questions", cls: "jk-mint", msg: "+3 questions ajoutees !" },
];
// pseudo "Nono" : mode triche — uniquement des boosters surpuissants
const NONO_JOKERS = [
  { type: "mult", mult: 100,    label: "x100",    cls: "jk-gold", msg: "Booster x100 arme !" },
  { type: "mult", mult: 1000,   label: "x1000",   cls: "jk-gold", msg: "Booster x1000 arme !" },
  { type: "mult", mult: 10000,  label: "x10000",  cls: "jk-gold", msg: "Booster x10000 arme !!!" },
  { type: "pts",  add: 100000,  label: "+100 000 pts", cls: "jk-mint", msg: "+100 000 points cadeaux !" },
];
const isNono = () => (localStorage.getItem("mlp_name") || "").trim().toLowerCase() === "nono";
function startDuel(s) {
  const S = { imgs: [], i: 0, me: 0, ai: 0, score: 0, total: 10, roundStart: 0, phase: "load",
              picked: null, timer: null, advance: null, aiP: null,
              duo: false, mult: 1, blind: false, blindActive: false, jokerTimer: null, jokerEls: [] };
  s.innerHTML = screenHead("duel", TXT.duelName) +
    `<div class="fadeup" id="duelBody"><p style="text-align:center;color:var(--muted)">Chargement du duel...</p></div>`;
  wireHome(s);

  const onKey = (e) => {
    if (S.phase !== "play") return;
    const i = parseInt(e.key, 10) - 1;
    const btns = s.querySelectorAll(".opt-btn");
    if (i >= 0 && i < btns.length) resolve(btns[i].dataset.c);
  };
  addEventListener("keydown", onKey);
  cleanup = () => {
    clearInterval(S.timer); clearInterval(S.advance); clearTimeout(S.jokerTimer);
    S.jokerEls.forEach((e) => e.remove()); S.jokerEls = [];
    removeEventListener("keydown", onKey);
  };

  fetchRound(12).then((imgs) => { S.imgs = imgs; renderPlay(); scheduleJoker(); }).catch((e) => toast(e.message));

  /* ----- jokers facon casino : popent sur les cotes, a cliquer ----- */
  function scheduleJoker() {
    clearTimeout(S.jokerTimer);
    const delay = isNono() ? 1000 + Math.random() * 1800 : 2600 + Math.random() * 3800;
    S.jokerTimer = setTimeout(() => { if (S.phase === "play") spawnJoker(); scheduleJoker(); }, delay);
  }
  function spawnJoker(force) {
    if (!force && S.jokerEls.length >= 2) return;
    const pool = isNono() ? NONO_JOKERS : JOKERS;
    const j = pool[Math.floor(Math.random() * pool.length)];
    const el = document.createElement("button");
    el.className = "joker " + j.cls;
    el.innerHTML = `<span class="jk-l">${j.label}</span>`;
    el.style[Math.random() < 0.5 ? "left" : "right"] = "12px";
    el.style.top = (18 + Math.random() * 58) + "vh";
    document.body.appendChild(el);
    S.jokerEls.push(el);
    const rm = () => { el.classList.add("jk-out"); setTimeout(() => { el.remove(); S.jokerEls = S.jokerEls.filter((x) => x !== el); }, 220); };
    el.onclick = () => { const r = el.getBoundingClientRect(); fireConfetti(r.left + r.width / 2, r.top + r.height / 2, { count: 45 }); applyJoker(j); rm(); };
    setTimeout(() => { if (S.jokerEls.includes(el)) rm(); }, 3600);
  }
  function applyJoker(j) {
    if (j.type === "duo") S.duo = true;
    else if (j.type === "mult") S.mult = j.mult;
    else if (j.type === "blind") S.blind = true;
    else if (j.type === "addq") { S.total += j.add; if ($("dRound")) $("dRound").textContent = `Manche ${S.i + 1} / ${S.total}`; }
    else if (j.type === "pts") {       // points cadeaux (mode Nono)
      S.score += j.add;
      const c = centerOf(s.querySelector(".imgframe"));
      floatPts(c.x, c.y - 30, "+" + j.add.toLocaleString("fr-FR"));
    }
    toast(j.msg);
    updateEffects();
  }
  function updateEffects() {
    const el = $("dEffects"); if (!el) return;
    const chips = [];
    if (S.blindActive) chips.push(`<span class="eff eff-pink">IA debranchee (ce tour)</span>`);
    if (S.duo) chips.push(`<span class="eff eff-blue">50/50 arme</span>`);
    if (S.mult > 1) chips.push(`<span class="eff eff-gold">x${S.mult} arme</span>`);
    if (S.blind) chips.push(`<span class="eff eff-pink">IA aveuglee (prochain)</span>`);
    el.innerHTML = chips.join(" ");
  }

  function renderPlay() {
    $("duelBody").innerHTML = `
      <p class="game-intro" style="text-align:center;color:var(--muted);max-width:620px;margin:0 auto 22px;font-size:16px">${TXT.duelIntro}</p>
      <div class="duel-bar">
        <div class="score-chip me"><span class="who">Toi</span><span class="num" id="dMe">0</span></div>
        <div class="round-label" id="dRound"></div>
        <div class="score-chip ai"><span class="num" id="dAi">0</span><span class="who">IA</span></div>
      </div>
      <div class="timerbar" id="dTimerWrap"><i id="dTimer"></i></div>
      <div id="dEffects" class="effects"></div>
      ${isNono() ? `<div style="text-align:center;margin-top:8px"><button class="btn" id="dCheat" style="font-size:14px;padding:8px 16px">&#127920; Booster</button></div>` : ""}
      <div class="imgframe"><img id="dImg" alt="image a reconnaitre"></div>
      <div class="opt-grid" id="dOpts"></div>
      <div class="verdict" id="dVerdict"></div>
      <div id="dNext" style="text-align:center;margin-top:18px"></div>`;
    const cheat = $("dCheat");
    if (cheat) cheat.onclick = () => spawnJoker(true);
    showRound();
  }

  async function showRound() {
    if (S.i >= S.imgs.length) { try { const more = await fetchRound(10); S.imgs.push(...more); } catch (e) {} }
    S.phase = "play"; S.picked = null;
    const useDuo = S.duo; S.duo = false;
    S.blindActive = S.blind; S.blind = false;
    const cur = S.imgs[S.i], truth = cur.true_label;
    const nDist = useDuo ? 1 : 2;   // Duo -> 2 choix au total
    const opts = shuffle([truth, ...shuffle(CLASSES.filter((c) => c !== truth)).slice(0, nDist)]);
    $("dRound").textContent = `Manche ${S.i + 1} / ${S.total}`;
    updateEffects();
    $("dImg").src = cur.image;
    $("dVerdict").innerHTML = ""; $("dNext").innerHTML = "";
    $("dOpts").innerHTML = opts.map((c, k) => `<button class="opt-btn" data-c="${c}"><span class="key">${k + 1}</span>${FR[c]}</button>`).join("");
    $("dOpts").querySelectorAll(".opt-btn").forEach((b) => (b.onclick = () => resolve(b.dataset.c)));

    // l'IA (notre modele) reflechit en parallele
    S.aiP = predictBlob(dataUrlToBlob(cur.image)).then((p) => p.label).catch(() => null);

    // chrono 3 s
    const wrap = $("dTimerWrap"), bar = $("dTimer");
    const start = Date.now(); S.roundStart = start;
    bar.style.width = "100%"; wrap.classList.remove("danger");
    clearInterval(S.timer);
    S.timer = setInterval(() => {
      const left = TIME_MS - (Date.now() - start);
      bar.style.width = Math.max(0, (left / TIME_MS) * 100) + "%";
      wrap.classList.toggle("danger", left < 1000);
      if (left <= 0) { clearInterval(S.timer); resolve(null); }
    }, 50);
  }

  async function resolve(choice) {
    if (S.phase !== "play") return;
    S.phase = "reveal"; S.picked = choice; clearInterval(S.timer);
    const truth = S.imgs[S.i].true_label;
    s.querySelectorAll(".opt-btn").forEach((b) => (b.disabled = true));

    const aiGuess = await S.aiP;
    const playerOk = choice === truth;
    let aiOk = aiGuess === truth;
    if (S.blindActive) aiOk = false;   // IA debranchee : forcement fausse
    if (playerOk) { S.me++; bump("dMe"); }
    if (aiOk) { S.ai++; bump("dAi"); }
    $("dMe").textContent = S.me; $("dAi").textContent = S.ai;

    // points : 50 + bonus vitesse (jusqu'a +50) + bonus si tu bats l'IA (+25), x multiplicateur
    const m = S.mult;
    if (playerOk) {
      const frac = Math.max(0, (TIME_MS - (Date.now() - S.roundStart)) / TIME_MS);
      let pts = 50 + Math.round(50 * frac);
      if (!aiOk) pts += 25;
      pts *= m;
      S.score += pts;
      const c = centerOf(s.querySelector(".imgframe")); floatPts(c.x, c.y - 30, "+" + pts + (m > 1 ? ` (x${m})` : ""));
    }
    S.mult = 1;   // multiplicateur consomme

    s.querySelectorAll(".opt-btn").forEach((b) => {
      if (b.dataset.c === truth) b.classList.add("right");
      else if (b.dataset.c === choice) b.classList.add("wrong");
    });

    if (playerOk) { const c = centerOf(s.querySelector(".imgframe")); fireConfetti(c.x, c.y, { count: m > 1 ? 120 : 60 }); }
    updateEffects();

    const iaTxt = S.blindActive ? "debranchee" : aiGuess ? (aiOk ? "a trouve" : "s'est trompee") : "n'a pas repondu";
    $("dVerdict").innerHTML = `<span class="fadeup">
      <span class="${playerOk ? "ok" : "ko"}">${playerOk ? "Bravo, trouve !" : choice === null ? "Trop tard !" : "Rate"}</span>
      <span class="sep">&middot;</span>
      <span class="${aiOk ? "ok" : "ko"}">IA : ${iaTxt}</span>
      <span class="sep">&middot;</span>
      <span style="color:var(--muted)">reponse : <b style="color:var(--ink)">${FR[truth]}</b></span>
    </span>`;

    const last = S.i + 1 >= S.total;
    const proceed = () => { clearInterval(S.advance); if (last) done(); else { S.i++; showRound(); } };
    $("dNext").innerHTML = `<button class="btn primary" id="dNextBtn">${last ? "Voir le resultat" : "Manche suivante"} ${icon("arrow", 18, "vertical-align:-3px;margin-left:6px")}</button>
      <div class="muted" style="font-size:14px;margin-top:8px">Suivant dans <b id="dCount">5</b></div>`;
    $("dNextBtn").onclick = proceed;
    let remain = 5;
    clearInterval(S.advance);
    S.advance = setInterval(() => {
      remain--;
      if (remain <= 0) { clearInterval(S.advance); proceed(); }
      else if ($("dCount")) $("dCount").textContent = remain;
    }, 1000);
  }

  function bump(id) { const el = $(id); el.classList.remove("pop"); void el.offsetWidth; el.classList.add("pop"); }

  function done() {
    S.phase = "done";
    const win = S.me > S.ai, tie = S.me === S.ai;
    $("duelBody").innerHTML = `<div class="fadeup" style="text-align:center;padding:20px 0 40px">
      ${mascot(win ? "win" : tie ? "idle" : "sad", 96, true)}
      <h2 style="font-family:var(--font-display);font-size:40px;margin:14px 0 6px;letter-spacing:-.02em">${win ? "Tu as battu la machine !" : tie ? "Egalite parfaite." : "La machine l'emporte."}</h2>
      <p style="color:var(--muted);font-weight:700;font-size:18px">Toi <b style="color:var(--ink)">${S.me}</b> &nbsp;—&nbsp; IA <b style="color:var(--ink)">${S.ai}</b> &nbsp;sur ${S.total} manches</p>
      <p style="margin:14px 0"><span class="score-pill">${icon("spark", 18)} ${S.score} points</span></p>
      <div class="panel lb-submit"><div style="font-weight:800;margin-bottom:10px">Entre ton pseudo pour le classement</div><div id="dSubmit"></div></div>
      <div style="display:flex;gap:12px;justify-content:center;margin-top:22px">
        <button class="btn primary" id="dAgain">${icon("refresh", 18, "vertical-align:-3px;margin-right:6px")}Rejouer</button>
        <button class="btn ghost" id="dHome">Accueil</button>
      </div></div>`;
    if (win) setTimeout(() => fireConfetti(null, innerHeight * 0.4, { count: 160 }), 120);
    renderSubmit($("dSubmit"), "duel", S.score);
    $("dAgain").onclick = () => {
      S.i = 0; S.me = 0; S.ai = 0; S.score = 0; S.total = 10;
      S.duo = false; S.mult = 1; S.blind = false; S.blindActive = false;
      renderPlay();
    };
    $("dHome").onclick = () => go("home");
  }
}

/* ============================================================
   JEU 2 — PICTIONARY (vrai modele : prediction sur le dessin)
   ============================================================ */
function startPicto(s) {
  s.innerHTML = screenHead("brush", TXT.pictoName) + `<div class="fadeup" id="pictoBody"></div>`;
  wireHome(s);

  const P = { ctx: null, drawing: false, last: null, color: PALETTE[0], erase: false, tool: "pen", size: 8,
              word: pick(CLASSES), guess: null, thinking: false, found: false, pending: false,
              wordStart: 0, swTimer: null, lastTry: 0, lastCheat: 0 };

  $("pictoBody").innerHTML = `
    <p class="picto-head" style="text-align:center;color:var(--muted);max-width:680px;margin:0 auto 22px;font-size:16px"><span class="game-intro-txt">${TXT.pictoIntro} </span>Ton mot : <span class="prompt-chip" id="pWord" style="font-size:16px;padding:4px 12px"></span></p>
    <div class="picto-layout">
      <div class="draw-card"><canvas id="pCanvas"></canvas></div>
      <div class="picto-side">
        <div class="panel p-guess">
          <div class="guess-stage">
            <div class="stopwatch" id="pTime">0.00 s</div>
            <div style="display:flex;justify-content:center;margin:6px 0 8px" id="pMascot">${mascot("idle", 56)}</div>
            <div class="guess-word" id="pGuess"><span class="muted">en attente d'un trait</span></div>
            <div class="guess-sub" id="pSub">le modele observe ton croquis</div>
            <div class="confbar"><i id="pConf" style="width:4%"></i></div>
          </div>
        </div>
        <div class="panel p-tools">
          <p class="field-label">Couleurs</p>
          <div class="palette" id="pPalette"></div>
          <p class="field-label" style="margin-top:18px">Outil</p>
          <div class="toolrow" id="pTools">
            <button class="btn tool-btn on" data-tool="pen" style="flex:1;font-size:15px;padding:11px 14px">${icon("brush", 16, "vertical-align:-3px;margin-right:6px")}Pinceau</button>
            <button class="btn tool-btn" data-tool="fill" style="flex:1;font-size:15px;padding:11px 14px">${icon("fill", 16, "vertical-align:-3px;margin-right:6px")}Remplir</button>
          </div>
          <p class="field-label" style="margin-top:18px">Taille &middot; <span id="pSizeL">8</span>px</p>
          <input type="range" min="2" max="30" value="8" id="pBrush" style="width:100%">
          <div class="toolrow" style="margin-top:18px">
            <button class="btn" id="pClear" style="flex:1;font-size:15px;padding:11px 14px">${icon("trash", 16, "vertical-align:-3px;margin-right:6px")}Effacer</button>
            <button class="btn" id="pNew" style="flex:1;font-size:15px;padding:11px 14px">${icon("dice", 16, "vertical-align:-3px;margin-right:6px")}Autre mot</button>
          </div>
        </div>
        <div class="panel p-name">
          <p class="field-label">Ton pseudo</p>
          <input id="pName" class="lb-input" maxlength="16" placeholder="Anonyme" style="width:100%">
          <p class="field-label p-hint" style="margin:10px 0 0">Plus tu fais deviner vite, mieux c'est. Ton temps et ton dessin sont enregistres au classement quand l'IA trouve.</p>
          <div id="pRecord" style="margin-top:10px;font-weight:800;min-height:20px"></div>
        </div>
      </div>
    </div>`;

  // palette (couleurs + gomme en icone)
  $("pPalette").innerHTML = PALETTE.map((c) => `<button class="swatch" data-c="${c}" style="background:${c}" aria-label="${c}"></button>`).join("") +
    `<button class="swatch eraser" data-eraser="1" title="Gomme">${icon("eraser", 18)}</button>`;
  const refreshTools = () => {
    $("pPalette").querySelectorAll(".swatch").forEach((sw) => {
      sw.classList.toggle("on", sw.dataset.eraser ? P.erase : (!P.erase && sw.dataset.c === P.color));
    });
    $("pTools").querySelectorAll(".tool-btn").forEach((b) => b.classList.toggle("on", b.dataset.tool === P.tool));
  };
  $("pPalette").querySelectorAll(".swatch").forEach((sw) => (sw.onclick = () => {
    if (sw.dataset.eraser) P.erase = true; else { P.color = sw.dataset.c; P.erase = false; }
    refreshTools();
  }));
  $("pTools").querySelectorAll(".tool-btn").forEach((b) => (b.onclick = () => { P.tool = b.dataset.tool; refreshTools(); }));
  refreshTools();

  $("pBrush").oninput = (e) => { P.size = +e.target.value; $("pSizeL").textContent = P.size; };
  function nextWord() { let w; do { w = pick(CLASSES); } while (w === P.word); P.word = w; $("pWord").textContent = FR[w]; clearCanvas(); }
  $("pClear").onclick = () => clearCanvas();
  $("pNew").onclick = nextWord;
  $("pWord").textContent = FR[P.word];
  $("pName").value = localStorage.getItem("mlp_name") || "";
  $("pName").oninput = () => { const v = $("pName").value.trim(); if (v) localStorage.setItem("mlp_name", v); };
  // mobile : si le pseudo est deja connu, on ne l'affiche pas (gagne de la place)
  if ($("pName").value.trim()) s.querySelector(".p-name").classList.add("has-name");

  // chrono stressant (centiemes de seconde) : demarre avec chaque mot, s'arrete quand trouve
  function startSW() {
    clearInterval(P.swTimer); P.wordStart = Date.now();
    P.swTimer = setInterval(() => {
      const t = (Date.now() - P.wordStart) / 1000, el = $("pTime");
      if (!el) return;
      el.textContent = t.toFixed(2) + " s";
      el.style.color = t > 10 ? "var(--danger)" : t > 5 ? "var(--p-orange)" : "var(--ink)";
    }, 70);
  }
  function stopSW() { clearInterval(P.swTimer); P.swTimer = null; }

  const canvas = $("pCanvas");
  function captureThumb() { const tc = document.createElement("canvas"); tc.width = 88; tc.height = 88; tc.getContext("2d").drawImage(canvas, 0, 0, 88, 88); return tc.toDataURL("image/png"); }
  // triche : un remplissage couleur unie n'est pas un dessin (>= 95 % d'une seule couleur)
  function isSolidDrawing() {
    const tc = document.createElement("canvas"); tc.width = 32; tc.height = 32;
    const tx = tc.getContext("2d"); tx.drawImage(canvas, 0, 0, 32, 32);
    const d = tx.getImageData(0, 0, 32, 32).data, counts = {};
    let best = 0, bestK = -1;
    for (let i = 0; i < d.length; i += 4) {
      const k = (d[i] >> 4) * 289 + (d[i + 1] >> 4) * 17 + (d[i + 2] >> 4);
      const c = (counts[k] = (counts[k] || 0) + 1);
      if (c > best) { best = c; bestK = k; }
    }
    if (best / 1024 < 0.95) return false;
    const r = Math.floor(bestK / 289), g = Math.floor(bestK / 17) % 17, b = bestK % 17;
    return !(r >= 14 && g >= 14 && b >= 14);   // blanc papier tolere (dessin au trait)
  }
  function setup() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.fillStyle = "#fffef9"; ctx.fillRect(0, 0, rect.width, rect.height);
    P.ctx = ctx; P.cw = rect.width;
  }
  setup();
  startSW();
  // mobile : la barre d'adresse qui se replie emet "resize" sans changer la
  // largeur -> ne pas effacer le dessin dans ce cas
  const onResize = () => { if (Math.abs(canvas.getBoundingClientRect().width - P.cw) > 2) setup(); };
  addEventListener("resize", onResize);
  cleanup = () => { removeEventListener("resize", onResize); stopSW(); };

  function pos(e) { const r = canvas.getBoundingClientRect(); const t = e.touches ? e.touches[0] : e; return { x: t.clientX - r.left, y: t.clientY - r.top }; }
  function hexRGB(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
  function floodFill(p, hex) {
    const ctx = P.ctx, rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / rect.width, w = canvas.width, h = canvas.height;
    const x0 = Math.floor(p.x * dpr), y0 = Math.floor(p.y * dpr);
    if (x0 < 0 || y0 < 0 || x0 >= w || y0 >= h) return;
    const img = ctx.getImageData(0, 0, w, h), d = img.data;
    const at = (x, y) => (y * w + x) * 4;
    const s = at(x0, y0), tr = d[s], tg = d[s + 1], tb = d[s + 2];
    const [fr, fg, fb] = hexRGB(hex);
    if (Math.abs(tr - fr) < 6 && Math.abs(tg - fg) < 6 && Math.abs(tb - fb) < 6) return;
    const stack = [x0, y0];
    while (stack.length) {
      const y = stack.pop(), x = stack.pop();
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      const i = at(x, y);
      if (Math.abs(d[i] - tr) > 28 || Math.abs(d[i + 1] - tg) > 28 || Math.abs(d[i + 2] - tb) > 28) continue;
      d[i] = fr; d[i + 1] = fg; d[i + 2] = fb; d[i + 3] = 255;
      stack.push(x + 1, y, x - 1, y, x, y + 1, x, y - 1);
    }
    ctx.putImageData(img, 0, 0);
  }

  // tente une prediction "au fur et a mesure" (throttle + garde anti-surcharge)
  function maybeGuess() {
    if (P.found || P.pending) return;
    const now = Date.now();
    if (now - P.lastTry < 450) return;
    P.lastTry = now; runGuess();
  }

  function down(e) {
    if (P.found) return; e.preventDefault();
    if (P.tool === "fill") { floodFill(pos(e), P.erase ? "#fffef9" : P.color); maybeGuess(); return; }
    P.drawing = true; P.last = pos(e);
  }
  function move(e) {
    if (!P.drawing || P.found) return; e.preventDefault();
    const ctx = P.ctx, p = pos(e), l = P.last;
    ctx.strokeStyle = P.erase ? "#fffef9" : P.color;
    ctx.lineWidth = P.erase ? P.size * 2.4 : P.size;
    ctx.beginPath(); ctx.moveTo(l.x, l.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    P.last = p;
    maybeGuess();           // devine pendant le trace, pas seulement a la fin
  }
  function up() { if (!P.drawing) return; P.drawing = false; runGuess(); }
  canvas.addEventListener("mousedown", down); canvas.addEventListener("mousemove", move);
  canvas.addEventListener("mouseup", up); canvas.addEventListener("mouseleave", up);
  canvas.addEventListener("touchstart", down, { passive: false });
  canvas.addEventListener("touchmove", move, { passive: false });
  canvas.addEventListener("touchend", up);

  function setMood(m) { $("pMascot").innerHTML = mascot(m, 56); }

  function clearCanvas() { setup(); P.found = false; P.guess = null; P.thinking = false;
    $("pGuess").innerHTML = `<span class="muted">en attente d'un trait</span>`;
    $("pSub").textContent = "le modele observe ton croquis"; $("pConf").style.width = "4%"; $("pConf").style.background = "";
    $("pRecord").innerHTML = ""; setMood("idle"); startSW(); }

  async function runGuess() {
    if (P.found || P.pending) return;
    P.lastTry = Date.now();
    P.pending = true; P.thinking = true; setMood("think");
    $("pGuess").innerHTML = `<span class="muted">hmm...</span>`;
    try {
      const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
      const pred = await predictBlob(blob);
      if (P.found) return;
      P.thinking = false;
      const conf = Math.round(pred.confidence * 100);
      const isWord = pred.label === P.word;
      $("pConf").style.width = conf + "%";
      if (isWord && isSolidDrawing()) {
        // couleur unie : pas de victoire, pas d'enregistrement, gros message
        P.thinking = false; setMood("sad");
        $("pGuess").innerHTML = `<span class="ko">Couleur unie — triche !</span>`;
        $("pSub").textContent = "dessine vraiment quelque chose";
        if (Date.now() - P.lastCheat > 3000) { P.lastCheat = Date.now(); cheatFlash(); }
        return;
      }
      if (isWord) {
        P.found = true; setMood("win"); stopSW();
        const time = (Date.now() - P.wordStart) / 1000;
        const thumb = captureThumb();
        const word = P.word;
        $("pGuess").innerHTML = `<span style="color:var(--accent)">Felicitations ! &laquo; ${FR[pred.label]} &raquo;</span>`;
        $("pSub").textContent = `trouve en ${time.toFixed(2)} s`;
        $("pConf").style.width = "100%";
        $("pConf").style.background = "linear-gradient(90deg,var(--p-mint),var(--accent))";
        const c = centerOf(canvas); floatPts(c.x, c.y - 20, time.toFixed(2) + " s"); fireConfetti(c.x, c.y, { count: 120 });
        // enregistrement auto (temps + dessin) au classement de la categorie
        const name = ($("pName").value || "").trim() || "Anonyme";
        if (name !== "Anonyme") localStorage.setItem("mlp_name", name);
        $("pRecord").innerHTML = `<span class="muted">enregistrement...</span>`;
        postScore({ game: "picto", name, time, category: word, image: thumb })
          .then((r) => {
            let msg = "Enregistre !";
            if (r.category_first) msg = `Champion de « ${FR[word]} » !`;
            if (r.rank != null) msg += ` Top 10 (${gradeFor(r.rank)})`;
            $("pRecord").innerHTML = `<span class="ok">${msg}</span>`;
            toast(msg);   // visible aussi quand le panneau pseudo est masque (mobile)
          })
          .catch((e) => { $("pRecord").innerHTML = `<span class="ko">${e.message}</span>`; toast(e.message); });
        setTimeout(nextWord, 2400);
      } else {
        setMood("idle");
        $("pGuess").innerHTML = `<span>&laquo; ${FR[pred.label]} &raquo; ?</span>`;
        $("pSub").textContent = `confiance ${conf}%`;
        $("pConf").style.background = "";
      }
    } catch (e) { P.thinking = false; setMood("idle"); toast(e.message); }
    finally { P.pending = false; }
  }
}

/* ============================================================
   JEU 3 — CINIC-10 (vrai modele : prediction sur images inedites)
   ============================================================ */
function startCinic(s) {
  const C = { cards: [], correct: 0, tested: 0 };
  s.innerHTML = screenHead("scope", TXT.cinicName) +
    `<div class="fadeup" id="cinicBody">
      <p class="game-intro" style="text-align:center;color:var(--muted);max-width:680px;margin:0 auto 18px;font-size:16px">${TXT.cinicIntro}</p>
      <div class="stat-row">
        <div class="stat"><div class="n" id="cCorrect">0</div><div class="l">Modele correct</div></div>
        <div class="stat"><div class="n" id="cTested">0</div><div class="l">Images testees</div></div>
        <div class="stat"><div class="n" id="cAcc">—</div><div class="l">Precision reelle</div></div>
      </div>
      <div class="cinic-grid" id="cGrid"></div>
      <div style="text-align:center;margin-top:28px;display:flex;gap:12px;justify-content:center">
        <button class="btn primary" id="cNew">${icon("refresh", 18, "vertical-align:-3px;margin-right:6px")}Nouvelle manche</button>
        <button class="btn ghost" id="cHome">Accueil</button>
      </div>
    </div>`;
  wireHome(s);
  cleanup = () => {};
  $("cHome").onclick = () => go("home");
  $("cNew").onclick = () => loadRound();

  function loadRound() {
    $("cGrid").innerHTML = `<p style="color:var(--muted)">Chargement...</p>`;
    fetchRound(3).then((imgs) => {
      C.cards = imgs.map((im) => ({ ...im, revealed: false }));
      $("cGrid").innerHTML = C.cards.map((c, i) =>
        `<div class="polaroid" data-i="${i}"><div class="pic"><img src="${c.image}" alt=""></div>
          <div class="cap" id="cap${i}"><span class="hint">${icon("scope", 15, "vertical-align:-3px;margin-right:6px")}Clique pour la prediction</span></div></div>`).join("");
      $("cGrid").querySelectorAll(".polaroid").forEach((p) => (p.onclick = () => reveal(+p.dataset.i)));
    }).catch((e) => toast(e.message));
  }

  async function reveal(i) {
    const card = C.cards[i];
    if (card.revealed) return;
    card.revealed = true;
    const cap = $("cap" + i);
    cap.innerHTML = `<span class="hint">analyse...</span>`;
    try {
      const pred = await predictBlob(dataUrlToBlob(card.image));
      const ok = pred.label === card.true_label;
      C.tested++; if (ok) C.correct++;
      $("cCorrect").textContent = C.correct; $("cTested").textContent = C.tested;
      $("cAcc").textContent = C.tested ? Math.round((C.correct / C.tested) * 100) + "%" : "—";
      cap.innerHTML = `<div class="fadeup">
        <div class="reveal-row"><span class="tag guess">IA : ${FR[pred.label]}</span><span class="tag truth">vrai : ${FR[card.true_label]}</span></div>
        <div class="judge ${ok ? "ok" : "ko"}">${ok ? "✓ Bien vu" : "✗ Le modele s'est trompe"}</div></div>`;
      const el = $("cGrid").querySelectorAll(".polaroid")[i];
      if (ok) { const c = centerOf(el); fireConfetti(c.x, c.y, { count: 50 }); }
      else if (el) { el.classList.add("shake"); setTimeout(() => el.classList.remove("shake"), 600); }
    } catch (e) {
      card.revealed = false;
      cap.innerHTML = `<span class="hint">${e.message}</span>`;
    }
  }

  loadRound();
}

/* ---------------- init ---------------- */
renderShell();
applyTheme();
go("home");
