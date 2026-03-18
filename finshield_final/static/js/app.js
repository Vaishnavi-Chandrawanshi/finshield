// FinShield v2 — Shared JS

async function loadUser() {
  try {
    const u = await fetch("/api/user").then(r => r.json());
    if (!u.id) return;
    const ni = document.getElementById("sb-initial");
    const nn = document.getElementById("sb-name");
    if (ni) ni.textContent = u.name[0];
    if (nn) nn.textContent = u.name;
    const acc = document.getElementById("chip-acc");
    const bal = document.getElementById("chip-bal");
    if (acc) acc.innerHTML = `<strong>${u.account_number}</strong>`;
    if (bal) bal.innerHTML = `<strong>$${parseFloat(u.balance).toLocaleString("en-US",{minimumFractionDigits:2})}</strong>`;
    const fb = document.getElementById("freeze-btn");
    if (fb) {
      if (u.account_frozen) { fb.textContent = "🔓 Unfreeze"; fb.classList.add("frozen"); }
      else { fb.textContent = "🔒 Freeze"; fb.classList.remove("frozen"); }
    }
    const nb = document.getElementById("nav-alert-badge");
    if (nb && u.unread_alerts > 0) { nb.textContent = u.unread_alerts; nb.style.display = "inline-flex"; }
  } catch(e) {}
}

async function toggleFreeze() {
  const fb = document.getElementById("freeze-btn");
  const frozen = fb.classList.contains("frozen");
  const r = await fetch("/api/freeze", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({freeze: !frozen}) }).then(r=>r.json());
  if (r.ok) {
    if (!frozen) { fb.textContent="🔓 Unfreeze"; fb.classList.add("frozen"); showToast("Account frozen. No transactions will process.", "warn"); }
    else { fb.textContent="🔒 Freeze"; fb.classList.remove("frozen"); showToast("Account unfrozen. Transactions enabled.", "success"); }
  }
}

async function markRead(id, el) {
  await fetch(`/api/alerts/read/${id}`, {method:"POST"});
  el.classList.remove("unread");
  const btn = el.querySelector(".mark-btn");
  if (btn) btn.replaceWith(Object.assign(document.createElement("span"), {className:"read-tag", textContent:"✓ read"}));
  loadUser();
}

function showToast(msg, type="info") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = `toast toast-${type}`;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 4000);
}

function timeAgo(str) {
  try {
    const d = new Date(str.includes("T") ? str : str.replace(" ","T")+"Z");
    const s = Math.floor((Date.now()-d)/1000);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s/60)}m ago`;
    if (s < 86400) return `${Math.floor(s/3600)}h ago`;
    return `${Math.floor(s/86400)}d ago`;
  } catch { return "recently"; }
}

function merchantIcon(m) {
  m = (m||"").toLowerCase();
  if (m.includes("amazon"))    return "📦";
  if (m.includes("starbucks") || m.includes("coffee")) return "☕";
  if (m.includes("netflix") || m.includes("spotify") || m.includes("streaming")) return "🎬";
  if (m.includes("uber") || m.includes("lyft"))  return "🚗";
  if (m.includes("apple"))     return "🍎";
  if (m.includes("walmart") || m.includes("whole foods") || m.includes("target")) return "🛒";
  if (m.includes("delta") || m.includes("airline")) return "✈️";
  if (m.includes("shell") || m.includes("fuel") || m.includes("gas")) return "⛽";
  if (m.includes("atm"))       return "🏧";
  if (m.includes("wire") || m.includes("offshore")) return "💸";
  if (m.includes("casino") || m.includes("gambling")) return "🎰";
  if (m.includes("crypto"))    return "🪙";
  if (m.includes("unknown"))   return "❓";
  return "💳";
}

function riskColor(score) {
  if (score >= 80) return "#dc2626";
  if (score >= 60) return "#ea580c";
  if (score >= 35) return "#ca8a04";
  return "#16a34a";
}

window.addEventListener("load", loadUser);
