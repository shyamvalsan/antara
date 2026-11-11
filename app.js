// A Year and a Day: the ticket, the days left on the wall, and stepping closer.
(() => {
  // ---- ticket -------------------------------------------------------------
  const KEY = 'ayd-ticket';
  const read = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const write = v => { try { localStorage.setItem(KEY, v); } catch {} };
  const badge = document.getElementById('tk');
  const showBadge = no => { if (badge) { badge.textContent = 'Ticket ' + no; badge.hidden = false; } };

  let no = read();
  if (no) showBadge(no);

  const take = document.getElementById('take');
  const ticket = document.getElementById('ticket');
  if (take && ticket) {
    const issue = () => {
      if (!no) {
        no = String(Math.floor(Math.random() * 90000) + 10000);
        write(no);
      }
      document.getElementById('tno').textContent = no;
      ticket.hidden = false;
      take.textContent = 'Your ticket';
      take.disabled = true;
      showBadge(no);
    };
    if (no) issue();
    take.addEventListener('click', () => {
      issue();
      ticket.querySelector('a').focus({ preventScroll: true });
    });
  }

  // ---- days left on the wall ---------------------------------------------
  document.querySelectorAll('[data-down]').forEach(el => {
    const d = Math.ceil((new Date(el.dataset.down + 'T00:00:00') - Date.now()) / 864e5);
    el.textContent = d > 366 ? 'a year and a day' : d > 1 ? d + ' days' : d === 1 ? 'one day' : 'coming down';
  });

  // ---- stepping closer ----------------------------------------------------
  const dlg = document.getElementById('closer');
  if (!dlg || !dlg.showModal) return;
  const stage = dlg.querySelector('.stage');
  const img = dlg.querySelector('img');
  const cap = dlg.querySelector('.cap');
  let s = 1, x = 0, y = 0, fitS = 1, nw = 1, nh = 1;

  const maxS = () => Math.max(fitS, 1.5);
  const apply = () => {
    const W = stage.clientWidth, H = stage.clientHeight, w = nw * s, h = nh * s;
    x = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, x));
    y = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, y));
    img.style.transform = `translate(${x}px,${y}px) scale(${s})`;
  };
  const fit = () => {
    fitS = Math.min(stage.clientWidth / nw, stage.clientHeight / nh);
    s = fitS; x = y = 0; apply();
  };
  const zoomAt = (cx, cy, f) => {
    const ns = Math.min(maxS(), Math.max(fitS, s * f));
    x = cx - (cx - x) * ns / s;
    y = cy - (cy - y) * ns / s;
    s = ns; apply();
  };

  document.querySelectorAll('.pic[data-z]').forEach(b => b.addEventListener('click', () => {
    nw = +b.dataset.w; nh = +b.dataset.h;
    img.style.width = nw + 'px'; img.style.height = nh + 'px';
    img.src = b.querySelector('img').currentSrc || b.querySelector('img').src;
    img.alt = b.querySelector('img').alt;
    cap.textContent = b.dataset.cap + '. Scroll or pinch to go closer, drag to look around.';
    dlg.showModal(); fit();
    const big = new Image();
    big.onload = () => { if (dlg.open) img.src = big.src; };
    big.src = b.dataset.z;
  }));

  dlg.querySelector('[data-close]').addEventListener('click', () => dlg.close());
  dlg.querySelectorAll('[data-zoom]').forEach(b => b.addEventListener('click', () =>
    zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, +b.dataset.zoom)));
  stage.addEventListener('wheel', e => {
    e.preventDefault();
    zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0016));
  }, { passive: false });
  stage.addEventListener('dblclick', e => zoomAt(e.clientX, e.clientY, s > fitS * 1.05 ? 0 : maxS() / s));

  const pts = new Map();
  let lastD = 0;
  stage.addEventListener('pointerdown', e => { stage.setPointerCapture(e.pointerId); pts.set(e.pointerId, e); lastD = 0; });
  stage.addEventListener('pointermove', e => {
    const p = pts.get(e.pointerId);
    if (!p) return;
    if (pts.size === 1) { x += e.clientX - p.clientX; y += e.clientY - p.clientY; apply(); }
    pts.set(e.pointerId, e);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (lastD) zoomAt((a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2, d / lastD);
      lastD = d;
    }
  });
  const up = e => { pts.delete(e.pointerId); lastD = 0; };
  stage.addEventListener('pointerup', up);
  stage.addEventListener('pointercancel', up);
  dlg.addEventListener('keydown', e => {
    const c = [stage.clientWidth / 2, stage.clientHeight / 2];
    if (e.key === '+' || e.key === '=') zoomAt(...c, 1.4);
    else if (e.key === '-') zoomAt(...c, 1 / 1.4);
    else if (e.key.startsWith('Arrow')) {
      const k = e.key.slice(5), st = 120;
      x += k === 'Left' ? st : k === 'Right' ? -st : 0;
      y += k === 'Up' ? st : k === 'Down' ? -st : 0;
      apply(); e.preventDefault();
    }
  });
  addEventListener('resize', () => { if (dlg.open) fit(); });
})();
