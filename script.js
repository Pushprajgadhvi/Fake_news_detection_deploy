/* ─── Constants ─── */
const API = "/predict";

/* ─── Character counter ─── */
const newsInput = document.getElementById("newsInput");
const charCount = document.getElementById("charCount");

newsInput.addEventListener("input", () => {
    const n = newsInput.value.length;
    charCount.textContent = `${n.toLocaleString()} character${n !== 1 ? "s" : ""}`;
});

/* ─── Clear ─── */
function clearInput() {
    newsInput.value = "";
    charCount.textContent = "0 characters";
    hide("resultsSection");
    hide("loadingSection");
}

/* ─────────────────────────────────────
   MAIN: analyzeNews
───────────────────────────────────── */
async function analyzeNews() {
    const text = newsInput.value.trim();
    if (!text || text.length < 30) {
        showShake(newsInput);
        return;
    }

    // Show loading
    hide("resultsSection");
    show("loadingSection");
    document.getElementById("analyzeBtn").disabled = true;

    try {
        const res = await fetch(API, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();

        hide("loadingSection");
        renderResults(data);

    } catch (err) {
        hide("loadingSection");
        showError(err.message);
    } finally {
        document.getElementById("analyzeBtn").disabled = false;
    }
}

/* ─────────────────────────────────────
   RENDER RESULTS
───────────────────────────────────── */
function renderResults(data) {
    // ── Verdict banner ──
    const banner = document.getElementById("verdictBanner");
    if (data.uncertain) {
        banner.className = "verdict-banner uncertain animate-pop";
        banner.innerHTML = `
      <span class="verdict-emoji">🤔</span>
      <div>
        <div>Result is <strong>Uncertain</strong> (avg. confidence: ${data.avg_conf}%)</div>
        <div class="verdict-sub">The AI models disagree or have low confidence. This may be outside the model's training domain (primarily US political news). Please verify with a trusted source.</div>
      </div>`;
    } else if (data.is_real) {
        banner.className = "verdict-banner real animate-pop";
        banner.innerHTML = `
      <span class="verdict-emoji">✅</span>
      <div>
        <div>This appears to be <strong>Real News</strong></div>
        <div class="verdict-sub">Majority of AI models classified this article as genuine.</div>
      </div>`;
    } else {
        banner.className = "verdict-banner fake animate-pop";
        banner.innerHTML = `
      <span class="verdict-emoji">🚨</span>
      <div>
        <div>This appears to be <strong>Fake News</strong></div>
        <div class="verdict-sub">Majority of AI models flagged this article as potentially misleading.</div>
      </div>`;
    }

    // ── Model cards ──
    const modelMap = {
        "Logistic Regression": "lr",
        "Passive-Aggressive": "pac",
        "Random Forest": "rfc",
    };

    for (const [name, key] of Object.entries(modelMap)) {
        const m = data.models[name];
        if (!m) continue;
        const isReal = m.is_real;
        const cls = isReal ? "real" : "fake";

        // Card border
        document.getElementById(`card-${key}`).className =
            `model-card card-glass animate-slide ${cls}`;

        // Badge
        const badge = document.getElementById(`badge-${key}`);
        badge.className = `model-badge badge-${cls}`;
        badge.textContent = m.label;

        // Bar
        const bar = document.getElementById(`bar-${key}`);
        bar.className = `confidence-bar bar-${cls}`;
        // Animate width with a brief delay
        setTimeout(() => { bar.style.width = m.confidence + "%"; }, 100);

        // Value
        const val = document.getElementById(`val-${key}`);
        val.className = `confidence-value val-${cls}`;
        val.textContent = m.confidence.toFixed(1) + "%";
    }

    // ── Summary (real news only) ──
    const panel = document.getElementById("summaryPanel");
    if (data.is_real && data.summary) {
        const { category, bullets } = data.summary;

        document.getElementById("categoryBadge").textContent = "🏷 " + category;

        const ul = document.getElementById("bulletList");
        ul.innerHTML = "";
        bullets.forEach((b, i) => {
            const li = document.createElement("li");
            li.style.animationDelay = `${i * 0.08}s`;
            li.innerHTML = `<span class="bullet-num">${i + 1}</span><span>${escHtml(b)}</span>`;
            ul.appendChild(li);
        });

        show("summaryPanel");
        panel.classList.add("animate-pop");
    } else {
        hide("summaryPanel");
    }

    // ── Fake Reasons (fake news only) ──
    const fakePanel = document.getElementById("fakeReasonsPanel");
    if (data.is_real === false && data.fake_reasons && data.fake_reasons.length) {
        const ul = document.getElementById("fakeReasonsList");
        ul.innerHTML = "";
        data.fake_reasons.forEach((reason, i) => {
            const li = document.createElement("li");
            li.style.animationDelay = `${i * 0.1}s`;
            li.innerHTML = `<span class="reason-icon">⚠️</span><span>${escHtml(reason)}</span>`;
            ul.appendChild(li);
        });
        show("fakeReasonsPanel");
        fakePanel.classList.add("animate-pop");
    } else {
        hide("fakeReasonsPanel");
    }

    show("resultsSection");
    document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth" });
}

/* ─────────────────────────────────────
   HELPERS
───────────────────────────────────── */
function show(id) { document.getElementById(id).classList.remove("hidden"); }
function hide(id) { document.getElementById(id).classList.add("hidden"); }

function escHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function showShake(el) {
    el.classList.add("shake");
    setTimeout(() => el.classList.remove("shake"), 600);
}

function showError(msg) {
    const banner = document.getElementById("verdictBanner");
    banner.className = "verdict-banner fake animate-pop";
    banner.innerHTML = `
    <span class="verdict-emoji">⚠️</span>
    <div>
      <div><strong>Error</strong></div>
      <div class="verdict-sub">${escHtml(msg)}</div>
    </div>`;
    hide("summaryPanel");
    show("resultsSection");
}

/* ─── Keyboard shortcut: Ctrl+Enter ─── */
newsInput.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyzeNews();
});

/* ─── Shake animation (injected dynamically) ─── */
const shakeStyle = document.createElement("style");
shakeStyle.textContent = `
  @keyframes shake {
    0%,100%{transform:translateX(0)}
    20%{transform:translateX(-6px)}
    40%{transform:translateX(6px)}
    60%{transform:translateX(-4px)}
    80%{transform:translateX(4px)}
  }
  .shake { animation: shake .5s ease; border-color: #f87171 !important; }
`;
document.head.appendChild(shakeStyle);
