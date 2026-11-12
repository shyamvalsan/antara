// Antara: the communal part. Hearts and the guest book, served by a small Worker.
// Every network call here fails silently: if the backend is down or slow, nothing in the gallery changes.
// Plain ES module, shared by the gallery and the plain pages. Visitor text is only ever set with textContent.

const TIMEOUT = 6000;
let BASE = '';
export const configure = base => { BASE = String(base || '').replace(/\/+$/, ''); };

const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const NS = 'http://www.w3.org/2000/svg';

// ---------------------------------------------------------------- visitor id
let memId = null;
function makeId() {
  try { if (crypto.randomUUID) return crypto.randomUUID(); } catch {}
  const b = new Uint8Array(16);
  try { crypto.getRandomValues(b); } catch { for (let i = 0; i < 16; i++) b[i] = Math.floor(Math.random() * 256); }
  b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80;
  const h = [...b].map(x => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}
export function visitorId() {
  if (memId) return memId;
  try {
    let v = localStorage.getItem('antara.visitor');
    if (!v || v.length < 16) { v = makeId(); localStorage.setItem('antara.visitor', v); }
    return (memId = v);
  } catch { return (memId = makeId()); }
}

// ---------------------------------------------------------------- one fetch, with a timeout, that never throws
async function call(path, body) {
  const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), TIMEOUT);
  try {
    const r = await fetch(BASE + path, body === undefined ? { signal: ctl.signal }
      : { method: 'POST', signal: ctl.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    let data = null; try { data = await r.json(); } catch {}
    return { ok: r.ok, status: r.status, data };
  } catch { return null; } finally { clearTimeout(t); }
}

// ---------------------------------------------------------------- hearts
const counts = new Map(), mine = new Set(), pending = new Set(), live = new Map();   // live: slug -> elements to keep in step
let heartsLoad = null;
const HEART_D = 'M12 20.6s-7.4-4.6-9.2-9.1C1.6 8.4 3.6 5 7 5c2 0 3.6 1 5 3 1.4-2 3-3 5-3 3.4 0 5.4 3.4 4.2 6.5-1.8 4.5-9.2 9.1-9.2 9.1z';
function heartSvg() {
  const s = document.createElementNS(NS, 'svg'); s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS(NS, 'path'); p.setAttribute('d', HEART_D); s.append(p); return s;
}
function paint(slug) {
  const set = live.get(slug); if (!set) return;
  const n = counts.get(slug) || 0, on = mine.has(slug);
  for (const e of [...set]) {
    if (e.isConnected) e._seen = true; else if (e._seen) { set.delete(e); continue; }   // forget elements the page has dropped
    if (e.tagName === 'BUTTON') e.setAttribute('aria-pressed', on ? 'true' : 'false');
    else e.classList.toggle('mine', on);
    e.querySelector('.n').textContent = n > 0 ? String(n) : '';
    if (e.tagName !== 'BUTTON') e.title = n > 0 ? `${n} ${n === 1 ? 'heart' : 'hearts'}` : '';
  }
}
const track = (slug, e) => { if (!live.has(slug)) live.set(slug, new Set()); live.get(slug).add(e); paint(slug); };

export function loadHearts() {
  if (!heartsLoad) heartsLoad = call('/hearts?visitor=' + encodeURIComponent(visitorId())).then(r => {
    const d = r && r.ok && r.data; if (!d) return;
    for (const [k, v] of Object.entries(d.counts || {})) if (!pending.has(k)) counts.set(k, +v || 0);
    for (const k of d.mine || []) if (!pending.has(k)) mine.add(k);
    for (const k of live.keys()) paint(k);
  });
  return heartsLoad;
}

async function toggle(slug, btn) {
  if (pending.has(slug)) return;
  pending.add(slug);
  const was = mine.has(slug), prev = counts.get(slug) || 0, on = !was;
  if (on) mine.add(slug); else mine.delete(slug);
  counts.set(slug, Math.max(0, prev + (on ? 1 : -1)));
  paint(slug);
  if (on && btn) { btn.classList.remove('pulse'); void btn.offsetWidth; btn.classList.add('pulse'); }
  const r = await call('/hearts/' + encodeURIComponent(slug), { visitor: visitorId(), on });
  pending.delete(slug);
  if (r && r.ok && r.data && typeof r.data.count === 'number') {
    counts.set(slug, r.data.count);
    if (r.data.on) mine.add(slug); else mine.delete(slug);
  } else {   // quietly put it back
    counts.set(slug, prev);
    if (was) mine.add(slug); else mine.delete(slug);
  }
  paint(slug);
}

/** A working heart button for a work. */
export function heartButton(slug) {
  const b = el('button', 'heart'); b.type = 'button'; b.setAttribute('aria-label', 'Heart this work'); b.setAttribute('aria-pressed', 'false');
  b.append(heartSvg(), el('span', 'n'));
  b.addEventListener('click', () => toggle(slug, b));
  b.addEventListener('animationend', () => b.classList.remove('pulse'));
  track(slug, b); return b;
}
/** A small read-only heart and count, for lists of works. */
export function heartBadge(slug) {
  const s = el('span', 'hb'); s.append(heartSvg(), el('span', 'n')); track(slug, s); return s;
}
/** Plain page: fill every <span data-heart-slot="slug"> with a working heart button. */
export function bindHearts(root = document) {
  for (const s of root.querySelectorAll('[data-heart-slot]')) { const b = heartButton(s.dataset.heartSlot); s.replaceChildren(b); }
  loadHearts();
}

// ---------------------------------------------------------------- totals
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
/** Fills `node` with "N hearts, M notes", or leaves it hidden if the call fails or both are zero. */
export async function showStats(node) {
  if (!node) return;
  const r = await call('/stats'), d = r && r.ok && r.data;
  if (!d || !(+d.hearts || +d.entries)) return;
  node.textContent = `${plural(+d.hearts || 0, 'heart')}, ${plural(+d.entries || 0, 'note')}`;
  node.hidden = false;
}

// ---------------------------------------------------------------- guest book
const fmtDate = s => new Date(s * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
let uid = 0;

/**
 * Mounts the entries and the signing form into `root`.
 * o.titles: {slug: title}; o.workLink(slug, anchor): make the anchor go to that work.
 * Returns { open(workSlug|null), setWork, load }.
 */
export function mountGuestbook(root, o = {}) {
  const titles = o.titles || {}, id = 'gb' + (++uid);
  let work = null, nextId = null, loadedAt = 0, busy = false;

  const list = el('div', 'gb-list'), entries = el('div', 'gb-entries'), note = el('p', 'gb-note'), more = el('button', 'gb-more', 'Load more');
  more.type = 'button'; more.hidden = true; note.hidden = true;
  list.append(entries, note, more);

  const form = el('form', 'gb-form'); form.noValidate = true;
  const head = el('h3', null, 'Sign the guest book');
  const ctx = el('p', 'gb-ctx'); ctx.hidden = true;
  const field = (name, label, opt, tag = 'input') => {
    const w = el('div', 'gb-field'), l = el('label', null, label); l.htmlFor = `${id}-${name}`;
    if (opt) l.append(el('span', 'opt', ` ${opt}`));
    const i = document.createElement(tag); i.id = `${id}-${name}`; i.name = name;
    if (tag === 'input') { i.type = 'text'; i.maxLength = 60; i.autocomplete = name === 'name' ? 'name' : 'off'; } else { i.rows = 5; i.maxLength = 1000; }
    w.append(l, i); return [w, i];
  };
  const [fName, iName] = field('name', 'Name');
  const [fFrom, iFrom] = field('from', 'Where you are writing from', '(optional)');
  const [fMsg, iMsg] = field('message', 'Message', '', 'textarea');
  const count = el('span', 'gb-count', '0 / 1000'); fMsg.append(count);
  const trap = el('div', 'gb-trap'); trap.setAttribute('aria-hidden', 'true');
  const iTrap = el('input'); iTrap.type = 'text'; iTrap.name = 'website'; iTrap.tabIndex = -1; iTrap.autocomplete = 'off'; trap.append(iTrap);
  const status = el('p', 'gb-status'); status.setAttribute('role', 'status');
  const send = el('button', 'gb-send', 'Sign the guest book'); send.type = 'submit';
  form.append(head, ctx, fName, fFrom, fMsg, trap, status, send);
  const thanks = el('div', 'gb-thanks'); thanks.hidden = true; thanks.setAttribute('role', 'status');
  const wrap = el('div', 'gb-sign'); wrap.append(form, thanks);
  root.append(list, wrap);

  const say = (t, bad) => { status.textContent = t; status.classList.toggle('bad', !!bad); };
  iMsg.addEventListener('input', () => { count.textContent = `${iMsg.value.length} / 1000`; });

  function setWork(slug) {
    work = slug && titles[slug] ? slug : null;
    ctx.hidden = !work; ctx.replaceChildren();
    if (work) {
      const off = el('button', 'link', 'Not about this work'); off.type = 'button'; off.addEventListener('click', () => setWork(null));
      ctx.append(`Your note will be linked to “${titles[work]}”. `, off);
    }
  }

  function entry(e) {
    const a = el('article', 'gb-entry');
    const msg = el('p', 'gb-msg', e.message);
    const by = el('p', 'gb-by'), who = el('b', null, e.name); by.append(who);
    if (e.from) by.append(' from ', el('span', null, e.from));
    by.append(el('span', 'dot', ' · '), el('time', null, fmtDate(e.created)));
    a.append(msg, by);
    if (e.work && titles[e.work]) {
      const l = el('a', 'gb-work', titles[e.work]); l.href = '#';
      if (o.workLink) o.workLink(e.work, l);
      const p = el('p', 'gb-about'); p.append('About ', l); a.append(p);
    }
    return a;
  }

  async function load(more_) {
    if (busy) return; busy = true; note.hidden = true;
    const r = await call('/guestbook?limit=20' + (more_ && nextId != null ? '&before=' + encodeURIComponent(nextId) : ''));
    busy = false;
    const d = r && r.ok && r.data;
    if (!d || !Array.isArray(d.entries)) {
      if (!entries.children.length) { note.textContent = 'The entries could not be loaded just now.'; note.hidden = false; }
      return;
    }
    if (!more_) entries.replaceChildren();
    for (const e of d.entries) entries.append(entry(e));
    nextId = d.next == null ? null : d.next; more.hidden = nextId == null;
    loadedAt = Date.now();
    if (!entries.children.length) { note.textContent = 'No one has signed yet. Be the first.'; note.hidden = false; }
  }
  more.addEventListener('click', () => load(true));

  form.addEventListener('submit', async ev => {
    ev.preventDefault();
    const name = iName.value.trim(), from = iFrom.value.trim(), message = iMsg.value.trim();
    if (!name) { say('Please add your name.', true); iName.focus(); return; }
    if (message.length < 2) { say('Please write at least a few words.', true); iMsg.focus(); return; }
    send.disabled = true; say('Sending…');
    const body = { name, message, visitor: visitorId(), website: iTrap.value };
    if (from) body.from = from; if (work) body.work = work;
    const r = await call('/guestbook', body);
    send.disabled = false;
    if (r && (r.status === 201 || (r.ok && r.data && r.data.ok))) {
      form.hidden = true; thanks.hidden = false; thanks.textContent = 'Thank you. Your note will appear once it has been read.';
      iMsg.value = ''; count.textContent = '0 / 1000'; say('');
    } else if (r && r.status === 429) say('You have signed a few times already. Try again later.', true);
    else if (r && r.status === 400) say('That note could not be accepted. Please check the fields, and keep to two links or fewer.', true);
    else if (r && r.status === 413) say('That note is too long to send.', true);
    else say('The guest book could not be reached just now. Please try again in a little while.', true);
  });

  return {
    load: () => load(false),
    setWork,
    open(slug) {
      setWork(slug); form.hidden = false; thanks.hidden = true; say('');
      if (!loadedAt || Date.now() - loadedAt > 60000) load(false);
    },
    focusForm() { wrap.scrollIntoView({ block: 'start' }); iName.focus({ preventScroll: true }); },
  };
}
