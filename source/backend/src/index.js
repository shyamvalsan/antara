// Antara guest book and hearts. Cloudflare Worker + D1.
const ALLOWED_ORIGINS = new Set([
  "https://shyamvalsan.github.io",
  "http://localhost:8765",
  "http://127.0.0.1:8765",
]);
const SLUG_RE = /^[a-z0-9_-]{1,120}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_BODY = 4096;
const HOUR = 3600, DAY = 86400;

class HttpError extends Error {
  constructor(status, msg) { super(msg); this.status = status; }
}

function corsHeaders(req) {
  const o = req.headers.get("Origin");
  const h = { "Vary": "Origin" };
  if (o && ALLOWED_ORIGINS.has(o)) {
    h["Access-Control-Allow-Origin"] = o;
    h["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS";
    h["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
    h["Access-Control-Max-Age"] = "86400";
  }
  return h;
}

function json(req, data, status = 200, extra = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      ...corsHeaders(req), ...extra,
    },
  });
}

const now = () => Math.floor(Date.now() / 1000);

async function sha256hex(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function ipHash(req, env) {
  const ip = req.headers.get("CF-Connecting-IP") || "unknown";
  return sha256hex((env.IP_SALT || "") + "|" + ip);
}

function timingSafeEqual(a, b) {
  const enc = new TextEncoder();
  const x = enc.encode(a), y = enc.encode(b);
  // compare fixed-length digests-free: pad to same length and fold length difference in
  const n = Math.max(x.length, y.length);
  let diff = x.length ^ y.length;
  for (let i = 0; i < n; i++) diff |= (x[i] || 0) ^ (y[i] || 0);
  return diff === 0;
}

function requireAdmin(req, env) {
  const m = /^Bearer (.+)$/.exec(req.headers.get("Authorization") || "");
  if (!env.ADMIN_TOKEN || !m || !timingSafeEqual(m[1], env.ADMIN_TOKEN)) {
    throw new HttpError(401, "unauthorized");
  }
}

async function readJson(req) {
  const len = parseInt(req.headers.get("Content-Length") || "0", 10);
  if (len > MAX_BODY) throw new HttpError(413, "body too large");
  const text = await req.text();
  if (new TextEncoder().encode(text).length > MAX_BODY) throw new HttpError(413, "body too large");
  let v;
  try { v = JSON.parse(text); } catch { throw new HttpError(400, "invalid json"); }
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new HttpError(400, "expected a json object");
  return v;
}

function clean(s) {
  return String(s).replace(/[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u2028-\u202e\u2066-\u2069\ufeff]/g, " ")
    .replace(/\s+/g, " ").trim();
}

// ---------- hearts ----------
async function getHearts(req, env, url) {
  const { results } = await env.DB.prepare(
    "SELECT w.slug AS slug, COUNT(h.visitor) AS n FROM works w LEFT JOIN hearts h ON h.slug = w.slug GROUP BY w.slug"
  ).all();
  const counts = {};
  for (const r of results) counts[r.slug] = r.n;
  let mine = [];
  const v = url.searchParams.get("visitor");
  if (v) {
    if (!UUID_RE.test(v)) throw new HttpError(400, "invalid visitor");
    const r = await env.DB.prepare("SELECT slug FROM hearts WHERE visitor = ?").bind(v.toLowerCase()).all();
    mine = r.results.map(x => x.slug);
  }
  return json(req, { counts, mine }, 200, { "Cache-Control": "no-store" });
}

async function postHeart(req, env, slug) {
  if (!SLUG_RE.test(slug)) throw new HttpError(400, "invalid slug");
  const body = await readJson(req);
  if (typeof body.visitor !== "string" || !UUID_RE.test(body.visitor)) throw new HttpError(400, "invalid visitor");
  if (typeof body.on !== "boolean") throw new HttpError(400, "on must be true or false");
  const visitor = body.visitor.toLowerCase();
  const known = await env.DB.prepare("SELECT 1 FROM works WHERE slug = ?").bind(slug).first();
  if (!known) throw new HttpError(404, "unknown work");

  const ih = await ipHash(req, env);
  const t = now();
  const hourN = await env.DB.prepare("SELECT COUNT(*) AS n FROM heart_events WHERE iphash = ? AND ts > ?").bind(ih, t - HOUR).first();
  if (hourN.n >= 60) throw new HttpError(429, "too many requests, try later");
  const seen = await env.DB.prepare(
    "SELECT DISTINCT visitor FROM heart_events WHERE iphash = ? AND slug = ? AND ts > ?"
  ).bind(ih, slug, t - DAY).all();
  const vs = new Set(seen.results.map(r => r.visitor));
  if (!vs.has(visitor) && vs.size >= 3) throw new HttpError(429, "too many requests, try later");

  const stmts = [
    env.DB.prepare("INSERT INTO heart_events (iphash, slug, visitor, ts) VALUES (?, ?, ?, ?)").bind(ih, slug, visitor, t),
    body.on
      ? env.DB.prepare("INSERT OR IGNORE INTO hearts (slug, visitor, created) VALUES (?, ?, ?)").bind(slug, visitor, t)
      : env.DB.prepare("DELETE FROM hearts WHERE slug = ? AND visitor = ?").bind(slug, visitor),
    env.DB.prepare("SELECT COUNT(*) AS n FROM hearts WHERE slug = ?").bind(slug),
  ];
  const res = await env.DB.batch(stmts);
  return json(req, { count: res[2].results[0].n, on: body.on });
}

// ---------- guest book ----------
async function postGuestbook(req, env) {
  const b = await readJson(req);
  if (b.website !== undefined && b.website !== null && b.website !== "") throw new HttpError(400, "rejected");
  if (typeof b.visitor !== "string" || !UUID_RE.test(b.visitor)) throw new HttpError(400, "invalid visitor");
  if (typeof b.name !== "string" || typeof b.message !== "string") throw new HttpError(400, "name and message are required");
  const name = clean(b.name), message = clean(b.message);
  let place = null, work = null;
  if (b.from !== undefined && b.from !== null && b.from !== "") {
    if (typeof b.from !== "string") throw new HttpError(400, "invalid from");
    place = clean(b.from) || null;
  }
  if (b.work !== undefined && b.work !== null && b.work !== "") {
    if (typeof b.work !== "string" || !SLUG_RE.test(b.work)) throw new HttpError(400, "invalid work");
    work = b.work;
  }
  if (name.length < 1 || name.length > 60) throw new HttpError(400, "name must be 1 to 60 characters");
  if (place && place.length > 60) throw new HttpError(400, "from must be at most 60 characters");
  if (message.length < 2 || message.length > 1000) throw new HttpError(400, "message must be 2 to 1000 characters");
  const urls = message.match(/https?:\/\/|www\.|\b[a-z0-9-]+\.(com|net|org|io|ru|xyz|info|biz|co|me|ly)\b/gi) || [];
  if (urls.length > 2) throw new HttpError(400, "too many links");

  const ih = await ipHash(req, env);
  const t = now();
  const r = await env.DB.prepare(
    "SELECT SUM(created > ?) AS h, COUNT(*) AS d FROM entries WHERE iphash = ? AND created > ?"
  ).bind(t - HOUR, ih, t - DAY).first();
  if ((r.h || 0) >= 3 || r.d >= 10) throw new HttpError(429, "too many submissions, try later");

  await env.DB.prepare(
    "INSERT INTO entries (name, place, message, work, visitor, iphash, status, created) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)"
  ).bind(name, place, message, work, b.visitor.toLowerCase(), ih, t).run();
  return json(req, { ok: true, pending: true }, 201);
}

async function getGuestbook(req, env, url) {
  let limit = parseInt(url.searchParams.get("limit") || "50", 10);
  if (!Number.isFinite(limit) || limit < 1) limit = 50;
  limit = Math.min(limit, 100);
  let before = url.searchParams.get("before");
  if (before !== null && !/^\d{1,15}$/.test(before)) throw new HttpError(400, "invalid before");
  const q = before !== null
    ? env.DB.prepare("SELECT id, name, place, message, work, created FROM entries WHERE status='approved' AND id < ? ORDER BY id DESC LIMIT ?").bind(parseInt(before, 10), limit + 1)
    : env.DB.prepare("SELECT id, name, place, message, work, created FROM entries WHERE status='approved' ORDER BY id DESC LIMIT ?").bind(limit + 1);
  const { results } = await q.all();
  const more = results.length > limit;
  const rows = results.slice(0, limit);
  const entries = rows.map(r => ({ id: r.id, name: r.name, from: r.place, message: r.message, work: r.work, created: r.created }));
  return json(req, { entries, next: more ? rows[rows.length - 1].id : null }, 200, { "Cache-Control": "public, max-age=30" });
}

async function getStats(req, env) {
  const h = await env.DB.prepare("SELECT COUNT(*) AS n FROM hearts").first();
  const e = await env.DB.prepare("SELECT COUNT(*) AS n FROM entries WHERE status='approved'").first();
  const v = await env.DB.prepare("SELECT COUNT(*) AS n FROM visits").first();
  return json(req, { hearts: h.n, entries: e.n, visitors: v.n }, 200, { "Cache-Control": "public, max-age=30" });
}

// ---------- visitors ----------
async function postVisit(req, env) {
  const body = await readJson(req);
  if (typeof body.visitor !== "string" || !UUID_RE.test(body.visitor)) throw new HttpError(400, "invalid visitor");
  const visitor = body.visitor.toLowerCase();
  const t = now();
  const known = await env.DB.prepare("SELECT 1 FROM visits WHERE visitor = ?").bind(visitor).first();
  if (!known) {
    // a new id: at most 30 new ids per IP hash per day
    const ih = await ipHash(req, env);
    const n = await env.DB.prepare("SELECT COUNT(*) AS n FROM visit_events WHERE iphash = ? AND ts > ?").bind(ih, t - DAY).first();
    if (n.n >= 30) throw new HttpError(429, "too many requests, try later");
    await env.DB.batch([
      env.DB.prepare("INSERT INTO visit_events (iphash, ts) VALUES (?, ?)").bind(ih, t),
      env.DB.prepare("INSERT OR IGNORE INTO visits (visitor, first, last) VALUES (?, ?, ?)").bind(visitor, t, t),
    ]);
  } else {
    await env.DB.prepare("UPDATE visits SET last = ? WHERE visitor = ?").bind(t, visitor).run();
  }
  const c = await env.DB.prepare("SELECT COUNT(*) AS n FROM visits").first();
  return json(req, { visitors: c.n, new: !known });
}

// ---------- admin ----------
async function adminEntries(req, env, url) {
  const status = url.searchParams.get("status") || "pending";
  if (!["pending", "approved", "rejected"].includes(status)) throw new HttpError(400, "invalid status");
  const { results } = await env.DB.prepare(
    "SELECT id, name, place AS \"from\", message, work, status, created FROM entries WHERE status = ? ORDER BY id DESC LIMIT 500"
  ).bind(status).all();
  return json(req, { entries: results });
}

async function adminSetEntry(req, env, id) {
  const b = await readJson(req);
  if (!["approved", "rejected"].includes(b.status)) throw new HttpError(400, "status must be approved or rejected");
  const r = await env.DB.prepare("UPDATE entries SET status = ? WHERE id = ?").bind(b.status, id).run();
  if (!r.meta.changes) throw new HttpError(404, "no such entry");
  return json(req, { ok: true, id, status: b.status });
}

async function adminDeleteEntry(req, env, id) {
  const r = await env.DB.prepare("DELETE FROM entries WHERE id = ?").bind(id).run();
  if (!r.meta.changes) throw new HttpError(404, "no such entry");
  return json(req, { ok: true, id });
}

async function adminWorks(req, env) {
  const b = await readJson(req).catch(e => { throw e; });
  if (!Array.isArray(b.slugs) || b.slugs.length > 1000 || !b.slugs.every(s => typeof s === "string" && SLUG_RE.test(s))) {
    throw new HttpError(400, "slugs must be an array of valid slugs");
  }
  const slugs = [...new Set(b.slugs)];
  const stmts = [env.DB.prepare("DELETE FROM works")];
  for (const s of slugs) stmts.push(env.DB.prepare("INSERT INTO works (slug) VALUES (?)").bind(s));
  await env.DB.batch(stmts);
  return json(req, { ok: true, count: slugs.length });
}

const ADMIN_HTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Antara guest book</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0e0e0d;color:#cfcac0;font:16px/1.5 Georgia,serif}
main{max-width:46rem;margin:0 auto;padding:2rem 1.2rem 4rem}
h1{font-weight:normal;letter-spacing:.08em;font-size:1.3rem;margin:0 0 1.2rem}
nav{display:flex;gap:.4rem;margin:1rem 0}
button,input{font:inherit;background:#1a1a18;color:#cfcac0;border:1px solid #3a3935;padding:.35rem .8rem;border-radius:3px;cursor:pointer}
input{cursor:text;min-width:16rem}
button:hover{border-color:#8a857a}
button.on{border-color:#cfcac0}
.e{border-top:1px solid #2a2927;padding:1rem 0}
.m{white-space:pre-wrap;margin:.4rem 0 .7rem}
.s{color:#8a857a;font-size:.85rem}
.err{color:#d98a7a}
.act{display:flex;gap:.5rem}
</style></head><body><main>
<h1>ANTARA GUEST BOOK</h1>
<div id="login"><input id="tok" type="password" placeholder="Admin token" autocomplete="off"> <button id="go">Enter</button></div>
<div id="app" hidden>
<nav id="tabs"></nav><div id="msg" class="s"></div><div id="list"></div></div>
<script>
const $=id=>document.getElementById(id);
let token=sessionStorage.getItem("antara-admin")||"",tab="pending";
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e}
async function api(path,opt={}){
  const r=await fetch(path,{...opt,headers:{"Authorization":"Bearer "+token,"Content-Type":"application/json"}});
  if(r.status===401){sessionStorage.removeItem("antara-admin");token="";show();throw new Error("Wrong token")}
  const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);return j}
function show(){$("login").hidden=!!token;$("app").hidden=!token;if(token)load()}
async function load(){
  const tabs=$("tabs");tabs.replaceChildren();
  for(const t of ["pending","approved","rejected"]){const b=el("button",t===tab?"on":"",t[0].toUpperCase()+t.slice(1));b.onclick=()=>{tab=t;load()};tabs.append(b)}
  const list=$("list");list.replaceChildren();$("msg").textContent="";
  try{
    const {entries}=await api("/admin/entries?status="+tab);
    $("msg").textContent=entries.length+" "+tab;
    for(const e of entries){
      const d=el("div","e");
      d.append(el("div","",e.name+(e.from?", "+e.from:"")));
      d.append(el("div","m",e.message));
      d.append(el("div","s","#"+e.id+" · "+new Date(e.created*1000).toLocaleString()+(e.work?" · on "+e.work:"")));
      const a=el("div","act");
      const mk=(label,fn)=>{const b=el("button","",label);b.onclick=async()=>{try{await fn();load()}catch(x){$("msg").textContent=x.message;$("msg").className="s err"}};a.append(b)};
      const set=s=>()=>api("/admin/entries/"+e.id,{method:"POST",body:JSON.stringify({status:s})});
      if(tab==="pending"){mk("Approve",set("approved"));mk("Reject",set("rejected"))}
      else{mk(tab==="approved"?"Reject":"Approve",set(tab==="approved"?"rejected":"approved"));mk("Delete",()=>confirm("Delete permanently?")?api("/admin/entries/"+e.id,{method:"DELETE"}):Promise.reject(new Error("cancelled")))}
      d.append(a);list.append(d)}
  }catch(x){$("msg").textContent=x.message;$("msg").className="s err"}
}
$("go").onclick=()=>{token=$("tok").value.trim();sessionStorage.setItem("antara-admin",token);$("tok").value="";show()};
$("tok").onkeydown=e=>{if(e.key==="Enter")$("go").click()};
show();
</script></main></body></html>`;

// ---------- router ----------
export default {
  async fetch(req, env) {
    try {
      const url = new URL(req.url);
      const path = url.pathname.replace(/\/+$/, "") || "/";
      const m = req.method;
      if (m === "OPTIONS") return new Response(null, { status: 204, headers: corsHeaders(req) });

      if (path === "/admin" && m === "GET") {
        return new Response(ADMIN_HTML, { headers: {
          "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store",
          "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
          "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
        } });
      }
      if (path === "/hearts" && m === "GET") return await getHearts(req, env, url);
      let x;
      if ((x = /^\/hearts\/([^/]+)$/.exec(path)) && m === "POST") return await postHeart(req, env, x[1]);
      if (path === "/guestbook" && m === "POST") return await postGuestbook(req, env);
      if (path === "/guestbook" && m === "GET") return await getGuestbook(req, env, url);
      if (path === "/visit" && m === "POST") return await postVisit(req, env);
      if (path === "/stats" && m === "GET") return await getStats(req, env);

      if (path.startsWith("/admin/")) {
        requireAdmin(req, env);
        if (path === "/admin/entries" && m === "GET") return await adminEntries(req, env, url);
        if (path === "/admin/works" && m === "PUT") return await adminWorks(req, env);
        if ((x = /^\/admin\/entries\/(\d{1,15})$/.exec(path))) {
          const id = parseInt(x[1], 10);
          if (m === "POST") return await adminSetEntry(req, env, id);
          if (m === "DELETE") return await adminDeleteEntry(req, env, id);
        }
      }
      throw new HttpError(404, "not found");
    } catch (e) {
      if (e instanceof HttpError) return json(req, { error: e.message }, e.status);
      return json(req, { error: "server error" }, 500);
    }
  },
};
