/* delphi.js - vanilla javascript, no build step. four actions, one results area.
   the browser never computes a verdict; it renders what the server returns. */

(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const results = $("#results");
  const forms = { price: $("#form-price"), ladder: $("#form-price"), market: $("#form-market"), method: $("#form-method") };
  let action = "price";

  // ------------------------------------------------------------ formatting
  const usd = (n) => n == null ? "—" : "$" + Math.round(n).toLocaleString("en-US");
  const pct = (x, d = 1) => x == null ? "—" : (x > 0 ? "+" : "") + (100 * x).toFixed(d) + "%";
  const num = (x, d = 0) => x == null ? "—" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d });
  const word = (v) => (v || "").replace(/_/g, " ");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ------------------------------------------------------------ health chips
  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("#health-chips").innerHTML =
      `<span class="chip">engine <b>${esc(h.narrator)}</b></span>` +
      `<span class="chip">sales <b>${num(h.sales_rows)}</b></span>` +
      `<span class="chip">market rows <b>${num(h.market_rows)}</b></span>`;
  }).catch(() => { $("#health-chips").innerHTML = `<span class="chip">engine <b>offline</b></span>`; });

  // ------------------------------------------------------------ action tabs
  $$(".action").forEach((btn) => btn.addEventListener("click", () => {
    action = btn.dataset.action;
    $$(".action").forEach((b) => b.classList.toggle("is-on", b === btn));
    Object.values(forms).forEach((f) => (f.hidden = true));
    forms[action].hidden = false;
    $("#form-price .go").textContent = action === "ladder" ? "Build the price ladder" : "Run price check";
  }));

  $$(".ex").forEach((b) => b.addEventListener("click", () => {
    const f = forms.price;
    f.address.value = b.dataset.address || "";
    f.price.value = b.dataset.price || "";
    f.dom.value = b.dataset.dom || "";
    f.requestSubmit();
  }));
  $$(".exq").forEach((b) => b.addEventListener("click", () => {
    forms.method.question.value = b.textContent;
    forms.method.requestSubmit();
  }));

  // ------------------------------------------------------------ status line
  // the stages mirror the planner's turns. they advance on a timer while the request is
  // in flight, then all flip to done when the answer lands - honest about being a progress
  // indicator, not a live feed.
  const STAGES = {
    price: ["Retrieving record", "Pulling comps", "Reading market", "Scoring model", "Verifying", "Composing"],
    market: ["Reading market", "Grounding", "Composing"],
    method: ["Retrieving", "Ranking", "Quoting"],
  };
  function startStatus(el, kind) {
    const stages = STAGES[kind];
    el.innerHTML = stages.map((s) => `<span class="stage">${s}</span>`).join("<span>→</span>");
    const spans = $$(".stage", el);
    let i = 0;
    spans[0].classList.add("on");
    const timer = setInterval(() => {
      if (i < spans.length - 1) { spans[i].classList.replace("on", "done"); i += 1; spans[i].classList.add("on"); }
    }, 350);
    return {
      done() { clearInterval(timer); spans.forEach((s) => { s.classList.remove("on"); s.classList.add("done"); }); },
      fail(msg) { clearInterval(timer); el.innerHTML = `<span class="err">${esc(msg)}</span>`; },
    };
  }

  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({ ok: false, error: `http ${r.status}` }));
    if (!r.ok || !data.ok) throw new Error(data.error || `http ${r.status}`);
    return data;
  }

  // ------------------------------------------------------------ price check / ladder
  forms.price.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = forms.price;
    const body = { address: f.address.value.trim(), price: f.price.value || null, dom: f.dom.value || null,
                   narrator: f.narrator.value };
    const st = startStatus($("#status"), "price");
    $(".go", f).disabled = true;
    try {
      const data = await post("/api/price-check", body);
      st.done();
      renderPrice(data, action === "ladder");
    } catch (err) {
      st.fail(err.message);
      results.innerHTML = `<div class="error">${esc(err.message)}</div>`;
    } finally {
      $(".go", f).disabled = false;
    }
  });

  function renderPrice(d, ladderFirst) {
    const v = d.verdict, ev = d.evidence || {};
    const listing = ev.lookup_listing || {}, comps = ev.comp_stats || {}, market = ev.market_context || {}, model = ev.predict_price || {};
    const conf = v.confidence || 0;
    const low = conf < 0.55;

    const verdictCard = `
      <div class="card">
        <div class="verdict-card">
          <div>
            <div class="label">verdict</div>
            <div class="verdict-word ${v.verdict}">${word(v.verdict)}</div>
            ${v.provisional_verdict ? `<div class="label">would have been: ${word(v.provisional_verdict)}</div>` : ""}
          </div>
          <div>
            <div class="label">confidence ${conf.toFixed(2)}${low ? " · below the 0.55 floor" : ""}</div>
            <div class="meter ${low ? "low" : ""}"><i style="width:${Math.round(100 * conf)}%"></i></div>
            <div class="label" style="margin-top:6px">${esc(listing.address || d.question)}${listing.list_price ? " · asking " + usd(listing.list_price) : ""}${listing.days_on_market != null ? " · " + listing.days_on_market + " dom" : ""}</div>
          </div>
          <div><span class="route ${d.route}" data-tip="${esc(d.route_reason)}">${d.route === "auto" ? "auto" : "route to analyst"}</span></div>
        </div>
        ${d.model_text ? `<p class="summary">${esc(d.model_text)}</p>` : ""}
        <ul class="reasons">${(v.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
        <div class="toolbar">
          <button class="btn" id="copy-summary">Copy seller summary</button>
          <button class="btn" id="dl-trace">Download trace (json)</button>
          <span class="label" style="align-self:center">${d.turns} turns · ${d.elapsed_ms} ms · narrator: ${d.narrator}</span>
        </div>
      </div>`;

    const ladderCard = d.ladder && d.ladder.length ? renderLadder(d.ladder, listing.list_price) : "";

    const compsRows = (comps.comps || []).map((c) =>
      `<tr><td>${esc(c.address)}</td><td>${esc(c.sale_date)}</td><td>${usd(c.sale_price)}</td><td>${num(c.sqft)}</td><td>${c.ppsf ? "$" + Math.round(c.ppsf) : "—"}</td></tr>`).join("");

    const evidenceCards = `
      <div class="cards">
        <div class="card">
          <h3>Record</h3>
          <table class="kv">
            <tr><th>address</th><td>${esc(listing.address || "—")}</td></tr>
            <tr><th>beds / baths</th><td>${listing.beds ?? "—"} / ${listing.baths ?? "—"}</td></tr>
            <tr><th>sqft · built</th><td>${num(listing.sqft)} · ${listing.year_built ?? "—"}</td></tr>
            <tr><th>building</th><td>${esc(listing.building || listing.source || "—")}</td></tr>
            <tr><th>last sale</th><td>${listing.last_sale_price ? usd(listing.last_sale_price) + " · " + esc(listing.last_sale_date) : "—"}</td></tr>
            <tr><th>assessed</th><td>${usd(listing.assessed_value)}</td></tr>
          </table>
        </div>
        <div class="card">
          <h3>Comps <span class="hint" data-tip="Same zip, same bed count, last ${comps.window_months || 12} months. Median is the anchor; $/sqft is checked against size-matched comps only.">?</span></h3>
          ${comps.error ? `<div class="error">${esc(comps.error)}</div>` : `
          <table class="kv">
            <tr><th>comps</th><td>${comps.comp_count ?? 0} in ${esc(comps.zip_code || "")}</td></tr>
            <tr><th>median sale</th><td>${usd(comps.median_sale_price)}</td></tr>
            <tr><th>median $/sqft</th><td>${comps.median_ppsf ? "$" + Math.round(comps.median_ppsf) : "—"}</td></tr>
            <tr><th>size-matched $/sqft</th><td>${comps.median_ppsf_similar ? "$" + Math.round(comps.median_ppsf_similar) + " (k=" + comps.similar_size_count + ")" : "—"}</td></tr>
            <tr><th>as of</th><td>${esc(comps.as_of || "—")}</td></tr>
          </table>
          ${compsRows ? `<details><summary>show comps</summary><table class="tbl"><tr><th>address</th><th>sold</th><th>price</th><th>sqft</th><th>$/sqft</th></tr>${compsRows}</table></details>` : ""}`}
        </div>
        <div class="card">
          <h3>Market <span class="hint" data-tip="Latest complete month from Redfin's county tracker. Regime from months of supply; trend from median dom vs the prior three months.">?</span></h3>
          ${market.error ? `<div class="error">${esc(market.error)}</div>` : `
          <table class="kv">
            <tr><th>region</th><td>${esc(market.region || "—")}</td></tr>
            <tr><th>period</th><td>${esc(market.period || "—")}</td></tr>
            <tr><th>median dom</th><td>${num(market.median_dom)} days · ${esc(market.dom_trend || "")}</td></tr>
            <tr><th>months of supply</th><td>${market.months_of_supply ?? "—"}</td></tr>
            <tr><th>regime</th><td>${esc(market.regime || "—")}</td></tr>
            <tr><th>median sale</th><td>${usd(market.median_sale_price)}</td></tr>
          </table>`}
        </div>
        <div class="card">
          <h3>Model <span class="hint" data-tip="Hedonic ridge + one-hidden-layer MLP on log price, blended by geometric mean. Spread between the two lowers the model's weight.">?</span></h3>
          ${model.error ? `<div class="error">${esc(model.error)}</div>` : `
          <table class="kv">
            <tr><th>estimate</th><td>${usd(model.predicted_price)}</td></tr>
            <tr><th>ridge · mlp</th><td>${usd(model.ridge_price)} · ${usd(model.mlp_price)}</td></tr>
            <tr><th>spread</th><td>${model.model_spread_pct != null ? model.model_spread_pct + "%" : "—"}</td></tr>
            <tr><th>zip in training</th><td>${model.zip_in_training_vocab == null ? "—" : model.zip_in_training_vocab ? "yes" : "no"}</td></tr>
            <tr><th>hold-out mape</th><td>${model.holdout_mape ? "ridge " + pct(model.holdout_mape.ridge, 0).replace("+", "") + " · mlp " + pct(model.holdout_mape.mlp, 0).replace("+", "") : "—"}</td></tr>
          </table>`}
        </div>
      </div>`;

    const critic = d.critic || { pass: true, flags: [] };
    const criticCard = `
      <div class="card">
        <h3>Critic <span class="hint" data-tip="A deterministic second reader. Any flag routes the file to an analyst.">?</span></h3>
        <div class="critic ${critic.pass ? "ok" : "flag"}">${critic.pass ? "✓ the verdict's story is consistent with the evidence" : "⚑ " + esc(critic.flags.join("; "))}</div>
        <details><summary>rubric</summary><ul class="reasons">${(critic.rubric || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul></details>
      </div>`;

    const notes = ev.search_notes && ev.search_notes.hits ? ev.search_notes.hits : [];
    const citeCard = notes.length ? `
      <div class="card">
        <h3>Method, quoted</h3>
        ${notes.map((h) => `<div class="cite">${esc(h.text)}<div class="src">${esc(h.id)} · score ${h.score}</div></div>`).join("")}
      </div>` : "";

    const traceCard = `
      <div class="card">
        <details><summary>trace — ${d.trace.length} events</summary>
          <ul class="trace">${d.trace.map((e) => `<li>${esc(traceLine(e))}</li>`).join("")}</ul>
        </details>
      </div>`;

    results.innerHTML = ladderFirst
      ? verdictCard + ladderCard + evidenceCards + criticCard + citeCard + traceCard
      : verdictCard + evidenceCards + ladderCard + criticCard + citeCard + traceCard;

    $("#copy-summary").addEventListener("click", () => copyText(sellerSummary(d)));
    $("#dl-trace").addEventListener("click", () => download(`delphi-trace-${Date.now()}.json`, JSON.stringify(d, null, 2)));
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderLadder(rows, current) {
    const flipUp = rows.find((r) => r.step_pct > 0 && r.verdict === "overpriced");
    const flipDown = [...rows].reverse().find((r) => r.step_pct < 0 && r.verdict === "underpriced");
    const bars = rows.map((r) => `<i class="${r.verdict} ${r.price === Math.round(current / 1000) * 1000 ? "now" : ""}" data-tip="${usd(r.price)} (${r.step_pct > 0 ? "+" : ""}${r.step_pct}%): ${word(r.verdict)} ${r.confidence.toFixed(2)}"></i>`).join("");
    const trs = rows.map((r) => `<tr class="${r.step_pct === 0 ? "now" : ""}"><td>${r.step_pct > 0 ? "+" : ""}${r.step_pct}%</td><td>${usd(r.price)}</td><td><span class="pill ${r.verdict}">${word(r.verdict)}</span>${r.provisional ? ` <span class="label">(${word(r.provisional)})</span>` : ""}</td><td>${r.confidence.toFixed(2)}</td><td>${pct(r.combined_delta)}</td></tr>`).join("");
    return `
      <div class="card">
        <h3>Price ladder <span class="hint" data-tip="The same evidence, re-scored at eleven asking prices. No new tool calls — this is the verdict engine alone.">?</span></h3>
        <div class="ladder-bar">${bars}</div>
        <div class="label">−15% … +15% around ${usd(current)}${flipUp ? ` · flips to overpriced at ${usd(flipUp.price)}` : ""}${flipDown ? ` · flips to underpriced at ${usd(flipDown.price)}` : ""}</div>
        <details><summary>show the ladder</summary>
          <table class="tbl"><tr><th>step</th><th>asking</th><th>verdict</th><th>confidence</th><th>combined delta</th></tr>${trs}</table>
        </details>
      </div>`;
  }

  function traceLine(e) {
    const t = e.kind;
    if (t === "model_call") return `model   turn ${e.turn} → ${e.stop_reason} (${e.latency_ms} ms)`;
    if (t === "tool_call") return `tool    ${e.name}(${JSON.stringify(e.input || {}).slice(0, 60)}) → ${e.error ? "error: " + e.error : "ok"} (${e.latency_ms} ms)`;
    if (t === "guardrail") return `guard   ${e.stage}: ${typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail)}`;
    if (t === "critic") return `critic  ${e.passed ? "pass" : "flags: " + (e.flags || []).join("; ")}`;
    if (t === "gate") return `gate    → ${e.route} (${e.why})`;
    if (t === "run_end") return `end     ok=${e.ok} total ${e.total_ms} ms`;
    return t;
  }

  function sellerSummary(d) {
    // plain text a listing agent can paste into an email: the numbers, then the caveat
    const v = d.verdict, ev = d.evidence || {}, l = ev.lookup_listing || {}, c = ev.comp_stats || {}, m = ev.market_context || {}, p = ev.predict_price || {};
    const lines = [
      `${l.address || d.question} — pricing check`,
      l.list_price ? `Asking: ${usd(l.list_price)}` : "",
      c.comp_count ? `Comparable closed sales: ${c.comp_count} in ${c.zip_code}, median ${usd(c.median_sale_price)}` : "",
      p.predicted_price ? `Model estimate: ${usd(p.predicted_price)}` : "",
      m.median_dom ? `Market: ${m.region}, median ${Math.round(m.median_dom)} days on market, ${m.months_of_supply} months of supply (${m.regime})` : "",
      `Verdict: ${word(v.verdict)} (confidence ${(v.confidence || 0).toFixed(2)}) — ${d.route === "auto" ? "numbers stand on their own" : "flagged for analyst review"}`,
      "",
      "Prepared with Delphi from public records; advisory only — a licensed professional makes the final call.",
    ];
    return lines.filter((s) => s !== "").join("\n");
  }

  // ------------------------------------------------------------ market pulse
  forms.market.addEventListener("submit", async (e) => {
    e.preventDefault();
    const st = startStatus($("#status-market"), "market");
    try {
      const d = await post("/api/market", { region: forms.market.region.value });
      st.done();
      renderMarket(d);
    } catch (err) {
      st.fail(err.message);
      results.innerHTML = `<div class="error">${esc(err.message)}</div>`;
    }
  });

  function renderMarket(d) {
    const c = d.context, h = d.history || [];
    const rows = [...h].reverse().map((r) => `<tr><td>${esc(r.period_begin.slice(0, 7))}</td><td>${usd(r.median_sale_price)}</td><td>${num(r.median_dom)}</td><td>${r.months_of_supply ?? "—"}</td><td>${num(r.inventory)}</td><td>${num(r.homes_sold)}</td><td>${r.avg_sale_to_list ? (100 * r.avg_sale_to_list).toFixed(1) + "%" : "—"}</td></tr>`).join("");
    results.innerHTML = `
      <div class="card">
        <div class="verdict-card">
          <div><div class="label">market pulse</div><div class="verdict-word ${c.regime === "sellers" ? "overpriced" : c.regime === "buyers" ? "underpriced" : "fairly_priced"}">${esc(c.regime)} market</div></div>
          <div><div class="label">${esc(c.region)} · ${esc(c.period)}</div>
               <p class="summary">Median ${Math.round(c.median_dom)} days on market (${esc(c.dom_trend)}), ${c.months_of_supply} months of supply, ${num(c.homes_sold)} homes sold, median sale ${usd(c.median_sale_price)}. A fair-price band here is ±${c.regime === "sellers" ? 7 : c.regime === "buyers" ? 3 : 5}%.</p></div>
          <div><span class="route auto" data-tip="Regime from months of supply: under 4 sellers, over 6 buyers. Trend from median dom vs the prior three months.">rule of thumb</span></div>
        </div>
      </div>
      <div class="cards">
        <div class="card"><h3>Median days on market · months of supply (12 mo)</h3>${sparkline(h)}<div class="spark-legend"><span><i></i>median dom</span><span><i class="two"></i>months of supply</span></div></div>
        <div class="card"><h3>Latest month</h3>
          <table class="kv">
            <tr><th>median sale price</th><td>${usd(c.median_sale_price)}</td></tr>
            <tr><th>median dom</th><td>${num(c.median_dom)} days</td></tr>
            <tr><th>months of supply</th><td>${c.months_of_supply}</td></tr>
            <tr><th>inventory</th><td>${num(c.inventory)}</td></tr>
            <tr><th>homes sold</th><td>${num(c.homes_sold)}</td></tr>
            <tr><th>sale-to-list</th><td>${c.avg_sale_to_list ? (100 * c.avg_sale_to_list).toFixed(1) + "%" : "—"}</td></tr>
          </table></div>
      </div>
      <div class="card"><details open><summary>last 12 months</summary>
        <table class="tbl"><tr><th>month</th><th>median sale</th><th>dom</th><th>supply</th><th>inventory</th><th>sold</th><th>sale/list</th></tr>${rows}</table></details></div>
      ${d.notes && d.notes.length ? `<div class="card"><h3>Method, quoted</h3>${d.notes.map((n) => `<div class="cite">${esc(n.text)}<div class="src">${esc(n.id)}</div></div>`).join("")}</div>` : ""}
      <div class="card"><div class="label">source</div><div style="font-size:13px;color:var(--ink-2)">${esc(c.source)}. Data from Redfin. Monthly, county level; the newest month is dropped when it is clearly partial.</div></div>`;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function sparkline(h) {
    // two series on one small svg; each scaled to its own range so both are readable
    if (!h.length) return "";
    const W = 600, H = 64, pad = 4;
    const line = (key, cls) => {
      const vals = h.map((r) => Number(r[key] || 0));
      const lo = Math.min(...vals), hi = Math.max(...vals) || 1;
      const pts = vals.map((v, i) => {
        const x = pad + (i / Math.max(1, vals.length - 1)) * (W - 2 * pad);
        const y = H - pad - ((v - lo) / (hi - lo || 1)) * (H - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      return `<path class="${cls}" d="M${pts.join(" L")}"></path>`;
    };
    return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="dom and supply over time">${line("median_dom", "")}${line("months_of_supply", "two")}</svg>`;
  }

  // ------------------------------------------------------------ ask the method
  forms.method.addEventListener("submit", async (e) => {
    e.preventDefault();
    const st = startStatus($("#status-method"), "method");
    try {
      const d = await post("/api/ask-method", { question: forms.method.question.value.trim(), k: 4 });
      st.done();
      results.innerHTML = `
        <div class="card">
          <div class="label">you asked</div>
          <p class="summary">${esc(d.question)}</p>
          ${d.hits.length ? d.hits.map((h) => `<div class="cite">${esc(h.text)}<div class="src">${esc(h.source)} · paragraph ${esc(h.id.split("#")[1])} · score ${h.score}</div></div>`).join("")
                          : `<div class="error">nothing relevant in the method notes — try different words</div>`}
          <div class="label" style="margin-top:8px">retrieval: tf-idf over the method notes, cosine similarity, top ${d.hits.length}. every answer is a quote with a source; nothing is generated.</div>
        </div>`;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      st.fail(err.message);
      results.innerHTML = `<div class="error">${esc(err.message)}</div>`;
    }
  });

  // ------------------------------------------------------------ small utilities
  function copyText(text) {
    navigator.clipboard?.writeText(text).then(() => flash("copied"), () => flash("copy failed"));
  }
  function download(name, text) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  function flash(msg) {
    const el = document.createElement("div");
    el.className = "chip"; el.textContent = msg;
    Object.assign(el.style, { position: "fixed", bottom: "18px", right: "18px", background: "#16212b", color: "#fff", zIndex: 50 });
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 1400);
  }
})();
