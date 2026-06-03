"use strict";

const CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"];
const FR = {
  airplane: "avion", automobile: "voiture", bird: "oiseau", cat: "chat",
  deer: "cerf", dog: "chien", frog: "grenouille", horse: "cheval",
  ship: "bateau", truck: "camion",
};

/* ============================ utilitaires ============================ */
function $(id) { return document.getElementById(id); }
function setStatus(msg, isErr = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.classList.toggle("err", isErr);
}
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function dataUrlToBlob(dataUrl) {
  const [head, b64] = dataUrl.split(",");
  const mime = head.match(/:(.*?);/)[1];
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

async function predictBlob(blob) {
  const fd = new FormData();
  fd.append("file", blob, "image.png");
  const res = await fetch("/api/predict", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Erreur de prediction");
  }
  return res.json();
}

async function fetchRound(n) {
  const res = await fetch(`/api/game/round?n=${n}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Impossible de charger les images");
  }
  return (await res.json()).images;
}

/* ============================ routeur ============================ */
function showView(id) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $(id).classList.add("active");
  if (id !== "duel") duelStop();
  if (id === "cinic" && !$("cinic-cards").children.length) cinicNewRound();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-go]");
  if (go) showView(go.dataset.go);
});
$("logo").addEventListener("click", () => showView("home"));

/* ============================ JEU 1 : DUEL ============================ */
const duel = { images: [], idx: 0, you: 0, ia: 0, answered: false, timer: null, predP: null };

$("duel-start").addEventListener("click", duelStart);
$("duel-again").addEventListener("click", duelStart);

async function duelStart() {
  $("duel-intro").classList.add("hidden");
  $("duel-end").classList.add("hidden");
  $("duel-play").classList.remove("hidden");
  setStatus("Chargement du duel...");
  try {
    duel.images = await fetchRound(10);
  } catch (err) { setStatus(err.message, true); return; }
  setStatus("");
  duel.idx = 0; duel.you = 0; duel.ia = 0;
  $("duel-you").textContent = "0";
  $("duel-ia").textContent = "0";
  duelRenderChoices();
  duelShow();
}

function duelRenderChoices() {
  const box = $("duel-choices");
  box.innerHTML = "";
  CLASSES.forEach((c) => {
    const b = document.createElement("button");
    b.className = "choice";
    b.textContent = FR[c];
    b.dataset.cls = c;
    b.addEventListener("click", () => duelAnswer(c));
    box.appendChild(b);
  });
}

function duelShow() {
  const item = duel.images[duel.idx];
  duel.answered = false;
  $("duel-idx").textContent = String(duel.idx + 1);
  $("duel-feedback").innerHTML = "";
  $("duel-img").src = item.image;

  // reactive les boutons
  document.querySelectorAll("#duel-choices .choice").forEach((b) => {
    b.disabled = false;
    b.className = "choice";
  });

  // l'IA reflechit en parallele (en arriere-plan, pendant les 3 s)
  duel.predP = predictBlob(dataUrlToBlob(item.image)).catch(() => null);

  // chrono 3 s
  const bar = document.querySelector(".timer-bar");
  bar.classList.remove("run");
  void bar.offsetWidth; // reflow pour relancer l'animation
  bar.classList.add("run");
  clearTimeout(duel.timer);
  duel.timer = setTimeout(() => duelAnswer(null), 3000);
}

async function duelAnswer(choice) {
  if (duel.answered) return;
  duel.answered = true;
  clearTimeout(duel.timer);
  document.querySelector(".timer-bar").classList.remove("run");

  const item = duel.images[duel.idx];
  const truth = item.true_label;
  document.querySelectorAll("#duel-choices .choice").forEach((b) => (b.disabled = true));

  const pred = await duel.predP;
  const iaGuess = pred ? pred.label : null;

  const youOk = choice === truth;
  const iaOk = iaGuess === truth;
  if (youOk) duel.you += 1;
  if (iaOk) duel.ia += 1;
  $("duel-you").textContent = String(duel.you);
  $("duel-ia").textContent = String(duel.ia);

  // surlignage
  document.querySelectorAll("#duel-choices .choice").forEach((b) => {
    const c = b.dataset.cls;
    if (c === truth) b.classList.add("correct");
    if (c === choice && !youOk) b.classList.add("wrong");
    if (c === iaGuess) b.classList.add("ia");
  });

  const youTxt = choice === null
    ? "Trop tard !"
    : youOk ? "Bravo !" : `Rate (${FR[choice]})`;
  const iaTxt = iaGuess
    ? `${iaOk ? "a trouve" : "s'est plante"} (${FR[iaGuess]})`
    : "n'a pas repondu";
  $("duel-feedback").innerHTML =
    `<span class="${youOk ? "ok" : "ko"}">Toi : ${youTxt}</span> &nbsp;|&nbsp; ` +
    `<span class="${iaOk ? "ok" : "ko"}">IA : ${iaTxt}</span> ` +
    `&middot; vraie reponse : <strong>${FR[truth]}</strong>`;

  await sleep(1700);
  duel.idx += 1;
  if (duel.idx >= duel.images.length) duelEnd();
  else duelShow();
}

function duelEnd() {
  $("duel-play").classList.add("hidden");
  $("duel-end").classList.remove("hidden");
  $("duel-final-you").textContent = String(duel.you);
  $("duel-final-ia").textContent = String(duel.ia);
  let verdict;
  if (duel.you > duel.ia) verdict = "Tu bats la machine !";
  else if (duel.you < duel.ia) verdict = "La machine gagne...";
  else verdict = "Egalite parfaite !";
  $("duel-verdict").textContent = verdict;
}

function duelStop() {
  clearTimeout(duel.timer);
  document.querySelector(".timer-bar") && document.querySelector(".timer-bar").classList.remove("run");
}

/* ============================ JEU 2 : PICTIONARY ============================ */
const picto = { ctx: null, drawing: false, target: null, last: null, pending: false, dirty: false };

function pictoInit() {
  const cv = $("picto-canvas");
  picto.ctx = cv.getContext("2d");
  pictoClear();
  pictoNewWord();

  const pos = (e) => {
    const r = cv.getBoundingClientRect();
    const p = e.touches ? e.touches[0] : e;
    return { x: (p.clientX - r.left) * (cv.width / r.width),
             y: (p.clientY - r.top) * (cv.height / r.height) };
  };
  const start = (e) => { e.preventDefault(); picto.drawing = true; picto.last = pos(e); };
  const move = (e) => {
    if (!picto.drawing) return;
    e.preventDefault();
    const p = pos(e);
    const ctx = picto.ctx;
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 14;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(picto.last.x, picto.last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    picto.last = p;
    picto.dirty = true;
    pictoGuessThrottled();
  };
  const end = () => { if (picto.drawing) { picto.drawing = false; pictoGuess(); } };

  cv.addEventListener("mousedown", start);
  cv.addEventListener("mousemove", move);
  window.addEventListener("mouseup", end);
  cv.addEventListener("touchstart", start, { passive: false });
  cv.addEventListener("touchmove", move, { passive: false });
  cv.addEventListener("touchend", end);

  $("picto-clear").addEventListener("click", pictoClear);
  $("picto-new").addEventListener("click", () => { pictoClear(); pictoNewWord(); });
}

function pictoClear() {
  const cv = $("picto-canvas");
  picto.ctx.fillStyle = "#fff";
  picto.ctx.fillRect(0, 0, cv.width, cv.height);
  picto.dirty = false;
  $("picto-guess").textContent = "...";
  $("picto-bar").style.width = "0%";
  $("picto-conf").textContent = "en attente d'un trait";
}

function pictoNewWord() {
  picto.target = CLASSES[Math.floor(Math.random() * CLASSES.length)];
  $("picto-target").textContent = FR[picto.target];
}

let pictoTimer = null;
function pictoGuessThrottled() {
  if (pictoTimer) return;
  pictoTimer = setTimeout(() => { pictoTimer = null; pictoGuess(); }, 450);
}

async function pictoGuess() {
  if (!picto.dirty || picto.pending) return;
  picto.pending = true;
  try {
    const blob = await new Promise((res) => $("picto-canvas").toBlob(res, "image/png"));
    const pred = await predictBlob(blob);
    const pct = Math.round(pred.confidence * 100);
    const found = pred.label === picto.target;
    $("picto-guess").innerHTML = `${found ? "Oui, " : ""}${FR[pred.label]}${found ? " !" : "..."}`;
    $("picto-guess").style.color = found ? "var(--accent-2)" : "var(--text)";
    $("picto-bar").style.width = pct + "%";
    $("picto-conf").textContent = `sur a ${pct}% (cible : ${FR[picto.target]})`;
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    picto.pending = false;
  }
}

/* ============================ JEU 3 : CINIC ============================ */
const cinic = { score: 0, total: 0 };

$("cinic-new").addEventListener("click", cinicNewRound);

async function cinicNewRound() {
  const box = $("cinic-cards");
  box.innerHTML = "";
  setStatus("Chargement...");
  try {
    const images = await fetchRound(3);
    images.forEach((img) => box.appendChild(cinicCard(img)));
    setStatus("");
  } catch (e) { setStatus(e.message, true); }
}

function cinicCard(item) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<img src="${item.image}" alt="image" />
    <div class="card-body"><span class="muted">Clique pour la prediction</span></div>`;
  let done = false;
  card.addEventListener("click", async () => {
    if (done) return;
    done = true;
    const body = card.querySelector(".card-body");
    body.innerHTML = `<span class="muted">Analyse...</span>`;
    try {
      const pred = await predictBlob(dataUrlToBlob(item.image));
      const ok = pred.label === item.true_label;
      cinic.total += 1;
      if (ok) cinic.score += 1;
      $("cinic-score").textContent = String(cinic.score);
      $("cinic-total").textContent = String(cinic.total);
      const pct = Math.round(pred.confidence * 100);
      body.innerHTML =
        `<div class="verdict ${ok ? "ok" : "ko"}">${ok ? "Bien classe" : "Rate"} : ${FR[pred.label]}</div>` +
        `<div class="muted">vraie classe : ${FR[item.true_label]} &middot; ${pct}%</div>`;
      setTimeout(() => {
        card.classList.add("fade");
        setTimeout(() => { const im = card.querySelector("img"); if (im) im.remove(); }, 350);
      }, 1700);
    } catch (e) {
      body.innerHTML = `<span class="ko">${e.message}</span>`;
      done = false;
    }
  });
  return card;
}

/* ============================ init ============================ */
pictoInit();
showView("home");
