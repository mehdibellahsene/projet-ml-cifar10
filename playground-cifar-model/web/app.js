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
  pictoName: "Crash Test Pictionary",
  pictoTag: "Dessine le mot impose et regarde le modele deviner — ou paniquer — trait apres trait.",
  pictoIntro: "Dessine le mot demande. Entraine sur des photos 32x32, le modele retente sa prediction a chaque trait.",
  cinicName: "Test Ultime CINIC-10",
  cinicTag: "Des images qu'il n'a jamais vues. Sauras-tu reperer quand il se trompe ?",
  cinicIntro: "3 images issues d'un autre jeu de donnees, jamais vues a l'entrainement. Clique une carte pour reveler la prediction du modele.",
};

const THEMES = [{ v: "tableau", l: "Tableau" }, { v: "cahier", l: "Cahier" }, { v: "recre", l: "Recre" }];

window.__juice = 1.2;
window.__mascot = true;

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

/* ---------------- theme ---------------- */
function setTheme(v) {
  document.documentElement.setAttribute("data-theme", v);
  document.documentElement.style.setProperty("--juice", String(window.__juice));
  localStorage.setItem("mlp_theme_v2", v);
  document.querySelectorAll(".theme-switch button").forEach((b) => b.classList.toggle("on", b.dataset.t === v));
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
    `<div class="theme-switch">${THEMES.map((t) => `<button data-t="${t.v}">${t.l}</button>`).join("")}</div>
     <div class="wrap">
       <header class="topbar">
         <h1 class="wordmark" id="wordmark"><span class="logo" style="background:transparent;box-shadow:none;transform:none">${mascot("happy", 56, true)}</span> ML Playground</h1>
         <p class="subtitle">${TXT.subtitle}</p>
       </header>
       <main id="screen"></main>
       <div class="foot"><span>EfficientNet-B5 &middot; projet ML CIFAR-10 &middot; BELLAHSENE Mehdi Redha</span><span>10 categories &middot; ~97,9 % en test</span></div>
     </div>`;
  $("wordmark").onclick = () => go("home");
  document.querySelectorAll(".theme-switch button").forEach((b) => (b.onclick = () => setTheme(b.dataset.t)));
}

function go(name) {
  if (cleanup) { cleanup(); cleanup = null; }
  const s = $("screen");
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
  s.innerHTML = `<div class="fadeup"><div class="home-grid">
      ${card("var(--p-blue)", "duel", TXT.duelName, TXT.duelTag, "duel")}
      ${card("var(--p-pink)", "brush", TXT.pictoName, TXT.pictoTag, "picto")}
      ${card("var(--p-mint)", "scope", TXT.cinicName, TXT.cinicTag, "cinic")}
    </div></div>`;
  s.querySelectorAll(".game-card").forEach((c) => (c.onclick = () => go(c.dataset.go)));
}

/* ============================================================
   JEU 1 — LE DUEL (vrai modele : l'IA = prediction du modele)
   ============================================================ */
const ROUNDS = 10, TIME_MS = 3000;
function startDuel(s) {
  const S = { imgs: [], i: 0, me: 0, ai: 0, phase: "load", picked: null, timer: null, aiP: null };
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
  cleanup = () => { clearInterval(S.timer); removeEventListener("keydown", onKey); };

  fetchRound(ROUNDS).then((imgs) => { S.imgs = imgs; renderPlay(); }).catch((e) => toast(e.message));

  function renderPlay() {
    $("duelBody").innerHTML = `
      <p style="text-align:center;color:var(--muted);max-width:620px;margin:0 auto 22px;font-size:16px">${TXT.duelIntro}</p>
      <div class="duel-bar">
        <div class="score-chip me"><span class="who">Toi</span><span class="num" id="dMe">0</span></div>
        <div class="round-label" id="dRound"></div>
        <div class="score-chip ai"><span class="num" id="dAi">0</span><span class="who">IA</span></div>
      </div>
      <div class="timerbar" id="dTimerWrap"><i id="dTimer"></i></div>
      <div class="imgframe"><img id="dImg" alt="image a reconnaitre"></div>
      <div class="opt-grid" id="dOpts"></div>
      <div class="verdict" id="dVerdict"></div>
      <div id="dNext" style="text-align:center;margin-top:18px"></div>`;
    showRound();
  }

  function showRound() {
    S.phase = "play"; S.picked = null;
    const cur = S.imgs[S.i], truth = cur.true_label;
    const opts = shuffle([truth, ...shuffle(CLASSES.filter((c) => c !== truth)).slice(0, 2)]);
    $("dRound").textContent = `Manche ${S.i + 1} / ${ROUNDS}`;
    $("dImg").src = cur.image;
    $("dVerdict").innerHTML = ""; $("dNext").innerHTML = "";
    $("dOpts").innerHTML = opts.map((c, k) => `<button class="opt-btn" data-c="${c}"><span class="key">${k + 1}</span>${FR[c]}</button>`).join("");
    $("dOpts").querySelectorAll(".opt-btn").forEach((b) => (b.onclick = () => resolve(b.dataset.c)));

    // l'IA (notre modele) reflechit en parallele
    S.aiP = predictBlob(dataUrlToBlob(cur.image)).then((p) => p.label).catch(() => null);

    // chrono 3 s
    const wrap = $("dTimerWrap"), bar = $("dTimer");
    const start = Date.now();
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
    const playerOk = choice === truth, aiOk = aiGuess === truth;
    if (playerOk) { S.me++; bump("dMe"); }
    if (aiOk) { S.ai++; bump("dAi"); }
    $("dMe").textContent = S.me; $("dAi").textContent = S.ai;

    s.querySelectorAll(".opt-btn").forEach((b) => {
      if (b.dataset.c === truth) b.classList.add("right");
      else if (b.dataset.c === choice) b.classList.add("wrong");
    });

    if (playerOk) { const c = centerOf(s.querySelector(".imgframe")); fireConfetti(c.x, c.y, { count: 60 }); }

    $("dVerdict").innerHTML = `<span class="fadeup">
      <span class="${playerOk ? "ok" : "ko"}">${playerOk ? "Bravo, trouve !" : choice === null ? "Trop tard !" : "Rate"}</span>
      <span class="sep">&middot;</span>
      <span class="${aiOk ? "ok" : "ko"}">IA : ${aiGuess ? (aiOk ? "a trouve" : "s'est trompee") : "n'a pas repondu"}</span>
      <span class="sep">&middot;</span>
      <span style="color:var(--muted)">reponse : <b style="color:var(--ink)">${FR[truth]}</b></span>
    </span>`;

    const last = S.i + 1 >= ROUNDS;
    $("dNext").innerHTML = `<button class="btn primary" id="dNextBtn">${last ? "Voir le resultat" : "Manche suivante"} ${icon("arrow", 18, "vertical-align:-3px;margin-left:6px")}</button>`;
    $("dNextBtn").onclick = () => { if (last) done(); else { S.i++; showRound(); } };
  }

  function bump(id) { const el = $(id); el.classList.remove("pop"); void el.offsetWidth; el.classList.add("pop"); }

  function done() {
    const win = S.me > S.ai, tie = S.me === S.ai;
    $("duelBody").innerHTML = `<div class="fadeup" style="text-align:center;padding:20px 0 40px">
      ${mascot(win ? "win" : tie ? "idle" : "sad", 96, true)}
      <h2 style="font-family:var(--font-display);font-size:40px;margin:14px 0 6px;letter-spacing:-.02em">${win ? "Tu as battu la machine !" : tie ? "Egalite parfaite." : "La machine l'emporte."}</h2>
      <p style="color:var(--muted);font-weight:700;font-size:18px">Toi <b style="color:var(--ink)">${S.me}</b> &nbsp;—&nbsp; IA <b style="color:var(--ink)">${S.ai}</b> &nbsp;sur ${ROUNDS} manches</p>
      <div style="display:flex;gap:12px;justify-content:center;margin-top:22px">
        <button class="btn primary" id="dAgain">${icon("refresh", 18, "vertical-align:-3px;margin-right:6px")}Rejouer</button>
        <button class="btn ghost" id="dHome">Accueil</button>
      </div></div>`;
    if (win) setTimeout(() => fireConfetti(null, innerHeight * 0.4, { count: 160 }), 120);
    $("dAgain").onclick = () => { S.i = 0; S.me = 0; S.ai = 0; renderPlay(); };
    $("dHome").onclick = () => go("home");
  }
}

/* ============================================================
   JEU 2 — PICTIONARY (vrai modele : prediction sur le dessin)
   ============================================================ */
function startPicto(s) {
  s.innerHTML = screenHead("brush", TXT.pictoName) + `<div class="fadeup" id="pictoBody"></div>`;
  wireHome(s);

  const P = { ctx: null, drawing: false, last: null, color: PALETTE[0], erase: false, size: 8,
              word: pick(CLASSES), guess: null, thinking: false, found: false, pending: false };

  $("pictoBody").innerHTML = `
    <p style="text-align:center;color:var(--muted);max-width:680px;margin:0 auto 22px;font-size:16px">${TXT.pictoIntro} Ton mot : <span class="prompt-chip" id="pWord" style="font-size:16px;padding:4px 12px"></span></p>
    <div class="picto-layout">
      <div class="draw-card"><canvas id="pCanvas"></canvas></div>
      <div style="display:flex;flex-direction:column;gap:22px">
        <div class="panel" style="padding:20px 18px">
          <div class="guess-stage">
            <div style="display:flex;justify-content:center;margin-bottom:8px" id="pMascot">${mascot("idle", 56)}</div>
            <div class="guess-word" id="pGuess"><span class="muted">en attente d'un trait</span></div>
            <div class="guess-sub" id="pSub">le modele observe ton croquis</div>
            <div class="confbar"><i id="pConf" style="width:4%"></i></div>
          </div>
        </div>
        <div class="panel" style="padding:18px">
          <p class="field-label">Couleurs</p>
          <div class="palette" id="pPalette"></div>
          <p class="field-label" style="margin-top:18px">Pinceau &middot; <span id="pSizeL">8</span>px</p>
          <input type="range" min="2" max="30" value="8" id="pBrush" style="width:100%">
          <div class="toolrow" style="margin-top:18px">
            <button class="btn" id="pClear" style="flex:1;font-size:15px;padding:11px 14px">${icon("trash", 16, "vertical-align:-3px;margin-right:6px")}Effacer</button>
            <button class="btn" id="pNew" style="flex:1;font-size:15px;padding:11px 14px">${icon("dice", 16, "vertical-align:-3px;margin-right:6px")}Autre mot</button>
          </div>
        </div>
      </div>
    </div>`;

  // palette
  $("pPalette").innerHTML = PALETTE.map((c) => `<button class="swatch" data-c="${c}" style="background:${c}" aria-label="${c}"></button>`).join("") +
    `<button class="swatch eraser" data-eraser="1">gomme</button>`;
  const refreshPalette = () => $("pPalette").querySelectorAll(".swatch").forEach((sw) => {
    sw.classList.toggle("on", sw.dataset.eraser ? P.erase : (!P.erase && sw.dataset.c === P.color));
  });
  $("pPalette").querySelectorAll(".swatch").forEach((sw) => (sw.onclick = () => {
    if (sw.dataset.eraser) P.erase = true; else { P.color = sw.dataset.c; P.erase = false; }
    refreshPalette();
  }));
  refreshPalette();

  $("pBrush").oninput = (e) => { P.size = +e.target.value; $("pSizeL").textContent = P.size; };
  $("pClear").onclick = () => clearCanvas();
  $("pNew").onclick = () => { let w; do { w = pick(CLASSES); } while (w === P.word); P.word = w; $("pWord").textContent = FR[w]; clearCanvas(); };
  $("pWord").textContent = FR[P.word];

  const canvas = $("pCanvas");
  function setup() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.fillStyle = "#fffef9"; ctx.fillRect(0, 0, rect.width, rect.height);
    P.ctx = ctx;
  }
  setup();
  const onResize = () => setup();
  addEventListener("resize", onResize);
  cleanup = () => removeEventListener("resize", onResize);

  function pos(e) { const r = canvas.getBoundingClientRect(); const t = e.touches ? e.touches[0] : e; return { x: t.clientX - r.left, y: t.clientY - r.top }; }
  function down(e) { if (P.found) return; e.preventDefault(); P.drawing = true; P.last = pos(e); }
  function move(e) {
    if (!P.drawing || P.found) return; e.preventDefault();
    const ctx = P.ctx, p = pos(e), l = P.last;
    ctx.strokeStyle = P.erase ? "#fffef9" : P.color;
    ctx.lineWidth = P.erase ? P.size * 2.4 : P.size;
    ctx.beginPath(); ctx.moveTo(l.x, l.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    P.last = p;
  }
  function up() { if (!P.drawing) return; P.drawing = false; if (!P.erase) runGuess(); }
  canvas.addEventListener("mousedown", down); canvas.addEventListener("mousemove", move);
  canvas.addEventListener("mouseup", up); canvas.addEventListener("mouseleave", up);
  canvas.addEventListener("touchstart", down, { passive: false });
  canvas.addEventListener("touchmove", move, { passive: false });
  canvas.addEventListener("touchend", up);

  function setMood(m) { $("pMascot").innerHTML = mascot(m, 56); }

  function clearCanvas() { setup(); P.found = false; P.guess = null; P.thinking = false;
    $("pGuess").innerHTML = `<span class="muted">en attente d'un trait</span>`;
    $("pSub").textContent = "le modele observe ton croquis"; $("pConf").style.width = "4%"; setMood("idle"); }

  async function runGuess() {
    if (P.found || P.pending) return;
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
      if (isWord) {
        P.found = true; setMood("win");
        $("pGuess").innerHTML = `<span style="color:var(--accent)">&laquo; ${FR[pred.label]} &raquo; !</span>`;
        $("pSub").textContent = "Cette fois il a trouve !";
        $("pConf").style.background = "linear-gradient(90deg,var(--p-mint),var(--accent))";
        const c = centerOf(canvas); fireConfetti(c.x, c.y, { count: 120 });
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
      <p style="text-align:center;color:var(--muted);max-width:680px;margin:0 auto 18px;font-size:16px">${TXT.cinicIntro}</p>
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
setTheme(localStorage.getItem("mlp_theme_v2") || "cahier");
go("home");
