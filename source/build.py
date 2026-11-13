#!/usr/bin/env python3
"""Build the gallery website into site/ from curator/programme.toml.

    .venv/bin/python narrate.py   # audio guide (only new or changed texts)
    .venv/bin/python build.py

The home page is a 3D gallery (web/gallery.js) laid out from the programme.
works/ is the same content as a plain page, for anyone who cannot or does not
want to walk. Nothing in the maker folders is changed.
"""
import html
import json
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from narrate import key as audio_key

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).parent
SITE = ROOT / "site"
SITE_URL = "https://shyamvalsan.github.io/antara/"
REPO = "https://github.com/shyamvalsan/antara"
PX_PER_CM = 12.5  # a 3600 px side hangs as 288 cm
YEAR = 2026
# The hearts and guest book API (backend/). One constant; ANTARA_API overrides it for testing.
API = os.environ.get("ANTARA_API", "https://antara-guestbook.shiny-mountain-4a1e.workers.dev")
OG_W, OG_H = 1200, 630
DEFAULT_OG = "s/gallery.jpg"
FONTS = ("https://fonts.googleapis.com/css2?family=Marcellus"
         "&family=Albert+Sans:ital,wght@0,300;0,400;0,600;1,400&display=swap")

P = tomllib.loads((ROOT / "curator" / "programme.toml").read_text(encoding="utf-8"))
G, C, EX = P["gallery"], P["curator"], P["exhibition"]
DUR = json.loads((ROOT / "audio" / "index.json").read_text()) if (ROOT / "audio" / "index.json").exists() else {}


def esc(s):
    return html.escape(str(s), quote=True)


def paras(text):
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip())]


def ptags(text):
    return "\n".join(f"<p>{esc(p)}</p>" for p in paras(text))


def audio(text):
    k = audio_key(text)
    if k in DUR:
        return {"src": f"audio/{k}.mp3", "dur": DUR[k]}
    return None


def blurb(text, limit=200):
    """One sentence (two if the first is very short) from the opening paragraph, for share cards."""
    first = paras(text)[0]
    sent = re.split(r"(?<=[.!?])\s+", first)
    out = sent[0]
    if len(out) < 70 and len(sent) > 1:
        out += " " + sent[1]
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return out


def og_image(src_webp, dst):
    """1200 x 630 JPEG: the whole work, centred on the gallery's dark ground, so no crawler has to crop it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(SITE / src_webp) as im:
        im = im.convert("RGB")
        pad = 36
        im.thumbnail((OG_W - 2 * pad, OG_H - 2 * pad), Image.LANCZOS)
        card = Image.new("RGB", (OG_W, OG_H), (14, 14, 13))
        card.paste(im, ((OG_W - im.width) // 2, (OG_H - im.height) // 2))
        card.save(dst, "JPEG", quality=85, optimize=True, progressive=True)


def derive(ex, work):
    src = ROOT / ex["folder"] / work["file"]
    out = SITE / "img" / ex["slug"]
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem.replace("_", "-")
    im, paths = None, {}
    for k, edge, q in (("t", 512, 76), ("m", 1280, 82), ("w", 2048, 84), ("z", 4000, 88)):
        dst = out / f"{stem}-{k}.webp"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            im = im or Image.open(src).convert("RGB")
            c = im.copy()
            c.thumbnail((edge, edge), Image.LANCZOS)
            c.save(dst, "WEBP", quality=q, method=5)
        paths[k] = f"img/{ex['slug']}/{dst.name}"
    with Image.open(src) as full:
        paths["px"] = list(full.size)
    return paths


# Comments in the published copy of a source that state what the maker's file marks "does not say".
# (path suffix, exact comment text, replacement). The originals in the maker folders are never touched.
NEUTRALISE = [
    ("dolkar/hanle_three_hours_the_pole_keeps_still.py", "# one more than the sky has, in the tent rope", "# the last point of light"),
]
UNSAFE = re.compile(r"/home/|/Users/|[A-Za-z]:\\Users|[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I)


def publish_code(ex, work):
    """Copy the work's program to site/code/<folder>/<file>.py (and a plain page for it); return its facts."""
    src = ROOT / ex["folder"] / Path(work["file"]).with_suffix(".py").name
    text = src.read_text(encoding="utf-8")
    for rel, old, new in NEUTRALISE:
        if rel == f"{ex['folder']}/{src.name}":
            assert old in text, f"neutralise target missing in {rel}"
            text = text.replace(old, new)
    bad = UNSAFE.search(text)
    assert not bad, f"{src}: would publish {bad.group(0)!r}"
    out = SITE / "code" / ex["folder"] / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n = len(text.splitlines())
    page = f"code/{ex['folder']}/{src.stem}.html"
    r = "../../"
    body = f"""<section class="lead"><h1>{esc(work['title'])}</h1>
<p class="dates">{esc(ex['artist'])}. The whole program that computes this picture: {n:,} lines of Python.</p>
<p><a href="{r}works/#{esc(src.stem.replace('_', '-'))}">Back to the work</a>. <a href="{src.name}" download>Download the .py file</a></p></section>
<pre class="src">{esc(text)}</pre>"""
    plain(page, f"Code: {work['title']}", body, desc=f"The program that computes {work['title']} by {ex['artist']}.")
    return {"src": f"code/{ex['folder']}/{src.name}", "page": page, "lines": n, "kb": round(len(text.encode()) / 1024)}


def collect():
    data = {
        "api": API, "url": SITE_URL, "description": G["description"], "name": G["name"], "line": G["line"], "season": G["season"], "dates": G["dates"],
        "curator": {"name": C["name"], "role": C["role"]},
        "welcome": {"text": paras(C["welcome"]), "audio": audio(C["welcome"])},
        "exhibitions": [],
    }
    for e in EX:
        works = []
        for w in e["work"]:
            if not (ROOT / e["folder"] / w["file"]).exists():
                print(f"  not hung yet (no file): {e['folder']}/{w['file']}")
                continue
            d = derive(e, w)
            pw, ph = d["px"]
            works.append({
                "id": Path(w["file"]).stem.replace("_", "-"),
                "title": w["title"], "medium": w["medium"], "year": w.get("year", YEAR),
                "cm": [round(pw / PX_PER_CM), round(ph / PX_PER_CM)], "px": d["px"],
                "t": d["t"], "m": d["m"], "w": d["w"], "z": d["z"],
                "hero": bool(w.get("hero")), "nave": bool(w.get("nave")),
                "code": publish_code(e, w),
                "blurb": blurb(w["story"]), "share": f"s/{Path(w['file']).stem.replace('_', '-')}/",
                "story": paras(w["story"]), "audio": audio(f'{w["title"]}.\n\n{w["story"]}'),
            })
        if not works:
            continue
        data["exhibitions"].append({
            "slug": e["slug"], "artist": e["artist"], "title": e["title"], "origin": e["origin"],
            "wall": e["wall"], "frame": e.get("frame", "#1A1A1A"), "mat": e.get("mat", 0),
            "intro": paras(e["intro"]), "audio": audio(f'{e["artist"]}. {e["title"]}.\n\n{e["intro"]}'),
            "blurb": blurb(e["intro"]), "share": f"s/{e['slug']}/",
            "bio": paras(e["bio"]), "said": e["said"], "works": works,
        })
    return data


def head(title, desc, r="", image=None, url=None, extra=""):
    image = image or DEFAULT_OG
    url = url or SITE_URL
    meta = f"""<link rel="canonical" href="{esc(url)}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="{esc(G['name'])}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:url" content="{esc(url)}">
<meta property="og:image" content="{SITE_URL}{image}">
<meta property="og:image:width" content="{OG_W}">
<meta property="og:image:height" content="{OG_H}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(desc)}">
<meta name="twitter:image" content="{SITE_URL}{image}">"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
{meta}
{extra}
<meta name="theme-color" content="#0E0E0D">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{r}style.css">
<link rel="icon" href="{r}mark.svg" type="image/svg+xml">
</head>"""


ICON_VOICE = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z" fill="currentColor" stroke="none"/>'
              '<path class="w" d="M15.2 9.2a4 4 0 0 1 0 5.6M17.6 6.9a7.4 7.4 0 0 1 0 10.2"/><path class="off" d="M3.5 3.5l17 17"/></svg>')
ICON_MUSIC = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 17.5V6l10-2v11.5" /><circle cx="6.8" cy="17.6" r="2.4" fill="currentColor" stroke="none"/>'
              '<circle cx="16.8" cy="15.6" r="2.4" fill="currentColor" stroke="none"/><path class="off" d="M3.5 3.5l17 17"/></svg>')


ICON_GUEST = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6.2C9.9 4.9 7.2 4.5 4 4.8v12.6c3.2-.3 5.9.1 8 1.4 2.1-1.3 4.8-1.7 8-1.4V4.8c-3.2-.3-5.9.1-8 1.4z"/><path d="M12 6.2v12.6"/></svg>')
ICON_ARTISTS = ('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8.2" r="3.4"/><path d="M5 20c.6-3.9 3.5-6 7-6s6.4 2.1 7 6"/></svg>')

GATE = ("Antara does not exist. The gallery, its curator, the six makers, their lives and memories, and every picture on these walls "
        "were imagined and made by Fable 5.2, an artificial mind.",
        "Nothing here was painted, photographed or assembled from other images. Each picture is computed, pixel by pixel, by a single "
        "program the model wrote, and you can read that program beside every work.",
        "Step through, and you are in a parallel world where all of it is real.")


def sound_buttons():
    """The two sound toggles. They appear in the bar and in each dialog, so they are never out of reach."""
    return (f'<button type="button" class="snd" data-snd="voice" aria-pressed="false">{ICON_VOICE}<span class="lbl">Voice</span></button>'
            f'<button type="button" class="snd" data-snd="music" aria-pressed="true">{ICON_MUSIC}<span class="lbl">Music</span></button>')


def share_links(url, title, text):
    """Plain links to each network's own share address. No scripts, no trackers."""
    u, t, tt = quote(url, safe=""), quote(title, safe=""), quote(f"{title}. {text}", safe="")
    return [("X", f"https://x.com/intent/post?text={t}&url={u}"),
            ("Facebook", f"https://www.facebook.com/sharer/sharer.php?u={u}"),
            ("WhatsApp", f"https://wa.me/?text={quote(title + ' ' + url, safe='')}"),
            ("LinkedIn", f"https://www.linkedin.com/sharing/share-offsite/?url={u}"),
            ("Email", f"mailto:?subject={t}&body={quote(text + chr(10) + chr(10) + url, safe='')}")]


def share_row(url, title, text):
    links = "".join(f'<a href="{esc(h)}" target="_blank" rel="noopener noreferrer">{n}</a>' for n, h in share_links(url, title, text))
    return (f'<p class="share" data-share data-url="{esc(url)}" data-title="{esc(title)}" data-text="{esc(text)}">Share: '
            f'<button type="button" class="link" data-copy>Copy link</button>{links}</p>')


SHARE_JS = """<script>
document.querySelectorAll('[data-share]').forEach(function (row) {
  var b = row.querySelector('[data-copy]');
  b.addEventListener('click', function () {
    var u = row.dataset.url, done = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy link'; }, 2200); };
    if (navigator.clipboard) navigator.clipboard.writeText(u).then(done, function () { window.prompt('Copy this link', u); });
    else window.prompt('Copy this link', u);
  });
});
</script>"""


def build_share(data):
    """s/<slug>/index.html: a tiny page with the share card tags that sends people into the gallery at that work or room."""
    shutil.rmtree(SITE / "s", ignore_errors=True)
    og_image(data["exhibitions"][0]["works"][0]["w"], SITE / DEFAULT_OG)
    seen = set()
    def page(slug, param, title, desc, src_webp, label):
        assert slug not in seen and slug != "gallery", f"share slug clash: {slug}"
        seen.add(slug)
        og_image(src_webp, SITE / "s" / slug / "og.jpg")
        target = f"../../?{param}={slug}"
        doc = head(title, desc, "../../", image=f"s/{slug}/og.jpg", url=f"{SITE_URL}s/{slug}/",
                   extra=f'<meta http-equiv="refresh" content="0;url={target}">\n<script>location.replace("{target}" + location.hash);</script>') + f"""
<body class="plain">
<main><section class="lead"><h1>{esc(label)}</h1>
<p>{esc(desc)}</p>
<p><a href="{target}">Open it in the gallery</a></p></section></main>
</body>
</html>
"""
        (SITE / "s" / slug / "index.html").write_text(doc, encoding="utf-8")
    for e in data["exhibitions"]:
        page(e["slug"], "x", f"{e['artist']}, {e['title']}: {G['name']}", e["blurb"], e["works"][0]["w"], e["artist"])
        for w in e["works"]:
            page(w["id"], "w", f"{w['title']}, {e['artist']}: {G['name']}", w["blurb"], w["w"], w["title"])


def build_walk(data):
    doc = head(f'{G["name"]}: {G["line"]}', G["description"]) + f"""
<body class="walk">
<canvas id="c" aria-label="The gallery, in three dimensions"></canvas>

<header id="hud" hidden>
  <a class="mark" href="./">{esc(G['name'])}</a>
  <nav>
    <button type="button" id="b-plan">Floor plan</button>
    <button type="button" id="b-artists" aria-haspopup="dialog">{ICON_ARTISTS}<span class="lbl">Artists</span></button>
    <button type="button" id="b-guest" aria-haspopup="dialog">{ICON_GUEST}<span class="lbl">Guest book</span></button>
    <span class="sounds">{sound_buttons()}</span>
    <a href="works/">All works</a>
    <a href="about/">About</a>
  </nav>
</header>

<section id="gate">
  <div class="gate-in">
    <h1>{esc(G['name'])}</h1>
    <p class="line">{esc(G['line'])}</p>
    <div class="gate-text">
      <p>{esc(GATE[0])}</p>
      <p>{esc(GATE[1])}</p>
      <p>{esc(GATE[2])}</p>
    </div>
    <p class="byline"><a href="about/#experimenter">An experiment by Shyam Sreevalsan</a></p>
    <button type="button" id="enter" disabled>Preparing the rooms</button>
    <p class="fine" id="gate-note"><a href="works/">See the works as a plain page instead</a></p>
    <p class="fine" id="gate-visits" hidden></p>
  </div>
</section>

<p id="where" aria-live="polite" hidden></p>
<p id="hint" hidden></p>
<p id="keys" hidden aria-label="Controls">
  <span class="k-fine"><span class="kc"><kbd>&larr;</kbd><kbd>&uarr;</kbd><kbd>&darr;</kbd><kbd>&rarr;</kbd></span>Arrow keys to walk<i>&middot;</i>Scroll to zoom<i>&middot;</i>Drag to look</span>
  <span class="k-touch">Drag to look<i>&middot;</i>Tap to walk<i>&middot;</i>Pinch to zoom</span>
</p>

<div id="fhearts"></div>

<aside id="panel" hidden aria-live="polite">
  <div class="p-top" id="p-top"><span class="kicker" id="p-kick"></span><span id="p-heart"></span>
    <button type="button" class="x" id="p-close" aria-label="Close">Close</button></div>
  <div id="p-body"></div>
  <div class="player" id="player" hidden>
    <p class="who" id="p-who"></p>
    <button type="button" id="p-play">Listen</button>
    <div class="bar"><i id="p-bar"></i></div>
    <span id="p-time"></span>
  </div>
  <div class="acts" id="p-acts"></div>
</aside>

<nav id="tour" hidden aria-label="Guided walk">
  <button type="button" id="t-prev">Previous work</button>
  <button type="button" id="t-next">Next work</button>
</nav>

<dialog id="plan" aria-label="Floor plan">
  <div class="plan-in">
    <h2>Floor plan</h2>
    <span class="sounds">{sound_buttons()}</span>
    <svg id="plan-svg" role="img" aria-label="Plan of the gallery"></svg>
    <ol id="plan-list"></ol>
    <div class="plan-acts"><button type="button" data-close>Back to the gallery</button><span id="plan-share"></span></div>
  </div>
</dialog>

<dialog id="closer" aria-label="Close view of the work">
  <div class="stage"><img alt="" draggable="false"></div>
  <div class="bar">
    <p class="cap"></p>
    <button type="button" data-zoom="1.6">Closer</button>
    <button type="button" data-zoom="0.625">Further</button>
    <button type="button" data-close>Step back</button>
    <span class="sounds">{sound_buttons()}</span>
  </div>
</dialog>

<dialog id="artists" aria-labelledby="ar-h">
  <div class="ar-in">
    <div class="ar-head"><h2 id="ar-h">The artists</h2><button type="button" data-close>Close</button></div>
    <div class="ar-body">
      <ul id="ar-list" role="listbox" aria-label="The six makers"></ul>
      <div id="ar-detail" tabindex="-1"></div>
    </div>
  </div>
</dialog>

<dialog id="guestbook" aria-labelledby="gb-h">
  <div class="gb-in">
    <div class="gb-head"><div><h2 id="gb-h">Guest book</h2><p id="gb-stats" class="gb-stats" hidden></p></div>
      <button type="button" id="gb-write">Write a note</button><button type="button" data-close>Close</button></div>
    <div class="gb-body" id="gb-body"></div>
  </div>
</dialog>

<dialog id="code" aria-labelledby="co-h">
  <div class="co-bar">
    <div><h2 id="co-h"></h2><p id="co-meta"></p></div>
    <a id="co-dl" download>Download .py</a>
    <a id="co-gh" href="{REPO}" target="_blank" rel="noopener">All sources on GitHub</a>
    <button type="button" data-close>Close</button>
  </div>
  <div class="co-scroll" id="co-scroll" tabindex="0" role="region" aria-label="Source code"><pre id="co-pre"></pre></div>
</dialog>

<noscript><p class="nojs">This gallery is walked in 3D and needs JavaScript. <a href="works/">All the works are also on a plain page.</a></p></noscript>
<audio id="voice" preload="none"></audio>
<script type="application/json" id="data">{json.dumps(data, ensure_ascii=False).replace("</", "<\\/")}</script>
<script type="importmap">{{"imports": {{"three": "./vendor/three.module.min.js"}}}}</script>
<script type="module" src="gallery.js"></script>
</body>
</html>
"""
    (SITE / "index.html").write_text(doc, encoding="utf-8")


def plain(path, title, body, desc=None, module=""):
    r = "../" * (len(Path(path).parts) - 1)
    pageurl = SITE_URL + re.sub(r"index\.html$", "", path)
    doc = head(f"{title} | {G['name']}", desc or G["description"], r, url=pageurl) + f"""
<body class="plain">
<header class="top">
  <a class="mark" href="{r}">{esc(G['name'])}</a>
  <nav><a href="{r}">Walk the gallery</a><a href="{r}works/">All works</a><a href="{r}guestbook/">Guest book</a><a href="{r}about/">About</a></nav>
</header>
<main>
{body}
</main>
<footer class="foot"><p>{esc(G['name'])}. {esc(G['season'])}, {esc(G['dates'])}. <a href="{REPO}" rel="noopener">Source code on GitHub</a>.</p></footer>
{module}
</body>
</html>
"""
    dst = SITE / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(doc, encoding="utf-8")


def build_works(data):
    out = [f"""<section class="lead">
<h1>{esc(G['season'])}</h1>
<p class="dates">{esc(G['dates'])}</p>
<p>{esc(G['description'])} This page lists every work with its text and recording. <a href="../">The gallery itself is here.</a></p>
</section>"""]
    for e in data["exhibitions"]:
        a = e["audio"]
        out.append(f"""<section class="show" id="{e['slug']}">
<div class="show-head"><h2>{esc(e['artist'])}<span>{esc(e['title'])}</span></h2>
<span class="artist-heart" data-heart-slot="{e['slug']}" data-noun="artist" data-text="Heart this artist"></span></div>
<p class="origin">{esc(e['origin'])}</p>
{share_row(SITE_URL + e['share'], f"{e['artist']}, {e['title']}: {G['name']}", e['blurb'])}
<div class="cols"><div>{''.join(f'<p>{esc(p)}</p>' for p in e['intro'])}
{f'<audio controls preload="none" src="../{a["src"]}"></audio>' if a else ''}</div>
<div class="bio">{''.join(f'<p>{esc(p)}</p>' for p in e['bio'])}
{''.join(f'<blockquote>{esc(q)}</blockquote>' for q in e['said'])}</div></div>""")
        for w in e["works"]:
            a = w["audio"]
            out.append(f"""<article class="piece" id="{w['id']}">
<a href="../{w['z']}"><img src="../{w['m']}" width="{w['px'][0]}" height="{w['px'][1]}" alt="{esc(w['title'])}, by {esc(e['artist'])}" loading="lazy" decoding="async"></a>
<div><h3>{esc(w['title'])}</h3>
<p class="meta">{esc(e['artist'])}, {w['year']}<br>{esc(w['medium'])}<br>{w['cm'][0]} × {w['cm'][1]} cm</p>
{''.join(f'<p>{esc(p)}</p>' for p in w['story'])}
<p class="code"><a href="../{w['code']['page']}">Read the code</a> ({w['code']['lines']:,} lines of Python)</p>
<p class="community"><span data-heart-slot="{w['id']}"></span><a href="../guestbook/?work={w['id']}">Leave a note about this work</a></p>
{share_row(SITE_URL + w['share'], f"{w['title']}, {e['artist']}: {G['name']}", w['blurb'])}
{f'<audio controls preload="none" src="../{a["src"]}"></audio>' if a else ''}</div>
</article>""")
        out.append("</section>")
    mod = f"""<script type="module">
import {{ configure, bindHearts }} from "../community.js";
configure({json.dumps(API)}); bindHearts();
</script>"""
    plain("works/index.html", "All works", "\n".join(out) + SHARE_JS, module=mod)


def build_about():
    body = f"""<section class="lead">
<h1>About {esc(G['name'])}</h1>
{share_row(SITE_URL, f"{G['name']}: {G['line']}", G['description'])}
</section>
<section class="show">
<h2>{esc(C['name'])}<span>{esc(C['role'])}</span></h2>
<div class="cols"><div>{ptags(C['bio'])}</div>
<div class="bio">{''.join(f'<blockquote>{esc(q)}</blockquote>' for q in C['said'])}</div></div>
</section>
<section class="show">
<h2>The plain facts<span>What Antara is, and how it was made</span></h2>
<div class="cols"><div>
<p>{esc(G['name'])} is a work of fiction from start to finish. The gallery, its curator, the six makers and their biographies, the exhibition texts, the building and every artwork were conceived and made by Fable 5.2, an AI model, as an experiment set going by one person. See <a href="#experimenter">The experimenter</a>, below.</p>
<p>No image-generation model was used. No photographs were used, and no source images of any kind. Each work is one self-contained Python program, numerical code that builds geometry, traces light, and simulates ink and paint, and the program computes the image, pixel by pixel. The full source of each program is published beside its work, so you can read exactly how the picture was made.</p>
<p>The voice of the audio guide and the music are synthesized.</p>
<p id="the-code"><b>The code.</b> The programs behind every picture, and the source of this gallery itself (the page, the audio guide's script, the guest book's small server), are published on <a href="{REPO}" rel="noopener">GitHub</a>. Read them, run them, learn from them.</p>
<p>The places, trades and traditions the makers come from are real. The people are not. Any resemblance to a real person is a coincidence.</p>
</div></div>
</section>
<section class="show" id="experimenter">
<h2>The experimenter</h2>
<div class="cols"><div>
<p>Behind the artificial mind there is a person. Shyam Sreevalsan set this experiment going: asked what a gallery would be if a machine imagined all of it, the people as well as the pictures; kept sending the work back until the pictures were worth standing in front of; and built Antara to share with the world.</p>
<p>If something here stayed with you, say so. <a href="../guestbook/">Sign the guest book</a>, or write directly.</p>
<ul class="contact">
<li><span>Email</span> <a href="mailto:shyam.valsan@gmail.com">shyam.valsan@gmail.com</a></li>
<li><span>X</span> <a href="https://x.com/shyamvalsan" rel="noopener">@shyamvalsan</a></li>
<li><span>GitHub</span> <a href="https://github.com/shyamvalsan" rel="noopener">shyamvalsan</a></li>
</ul>
<p class="totals" id="totals" hidden></p>
</div></div>
</section>"""
    mod = f"""<script type="module">
import {{ configure, showStats }} from "../community.js";
configure({json.dumps(API)}); showStats(document.getElementById("totals"));
</script>"""
    plain("about/index.html", "About", body + SHARE_JS, module=mod)


def build_guestbook(data):
    titles = {w["id"]: w["title"] for e in data["exhibitions"] for w in e["works"]}
    body = """<section class="lead">
<h1>Guest book</h1>
<p>Visitors leave a few words here. Each note appears once it has been read.</p>
<p class="totals" id="totals" hidden></p>
</section>
<div id="gb" class="gb-page"></div>
<noscript><p>The guest book needs JavaScript. You can also write to <a href="../about/#experimenter">the person behind Antara</a>.</p></noscript>"""
    mod = f"""<script type="module">
import {{ configure, mountGuestbook, showStats }} from "../community.js";
configure({json.dumps(API)});
const titles = {json.dumps(titles, ensure_ascii=False).replace("</", "<\\/")};
const gb = mountGuestbook(document.getElementById("gb"), {{ titles, workLink: (slug, a) => {{ a.href = "../works/#" + slug; }} }});
const w = new URLSearchParams(location.search).get("work");
gb.open(w);
showStats(document.getElementById("totals"));
</script>"""
    plain("guestbook/index.html", "Guest book", body, desc="Leave a note in the Antara guest book.", module=mod)


def build_404():
    plain("404.html", "Not found", """<section class="lead"><h1>That room is not on the plan</h1>
<p>The exhibition may have closed, or the address is mistyped. <a href="/antara/">Go to the gallery.</a></p></section>""")
    f = SITE / "404.html"
    t = f.read_text(encoding="utf-8")
    t = t.replace('href="style.css"', 'href="/antara/style.css"').replace('href="mark.svg"', 'href="/antara/mark.svg"')
    t = re.sub(r'href="(works/|about/|guestbook/)?"', lambda m: f'href="/antara/{m.group(1) or ""}"', t)
    f.write_text(t, encoding="utf-8")


SOURCE_ALLOW = ["build.py", "narrate.py", "ambient.py", "curator/programme.toml",
                "web/gallery.js", "web/community.js", "web/style.css",
                "backend/schema.sql", "backend/wrangler.toml", "backend/README.md", "backend/works.json"]
SOURCE_EMAIL_OK = {"shyam.valsan@gmail.com"}
SCAN = [("an absolute home path", re.compile(r"/home/\w|/Users/\w|[A-Za-z]:\\Users", re.I)),
        ("a 32+ character hex string", re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{32,}(?![0-9A-Za-z])")),
        ("something like an API key or token", re.compile(
            r"\bsk-[A-Za-z0-9_-]{16,}|\bAKIA[0-9A-Z]{16}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[abp]-[A-Za-z0-9-]{10,}|Bearer\s+[A-Za-z0-9._~+/-]{24,}"
            r"|(?i:api[_-]?key|secret|passw(or)?d|token)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{12,}[\"']")),
        ("a forbidden word", re.compile("anthro" + "pic", re.I))]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}", re.I)


def publish_source():
    """Copy the gallery's own source into site/source/ from an explicit allow-list, then scan it. Fails the build on any finding."""
    dst = SITE / "source"
    shutil.rmtree(dst, ignore_errors=True)
    files = list(SOURCE_ALLOW)
    files += [f"web/{p.name}" for p in sorted((ROOT / "web").glob("*.svg"))]
    files += [f"backend/src/{p.relative_to(ROOT / 'backend' / 'src').as_posix()}" for p in sorted((ROOT / "backend" / "src").rglob("*")) if p.is_file()]
    for rel in files:
        src = ROOT / rel
        assert src.is_file(), f"source allow-list: missing {rel}"
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst / rel)
    found = []
    for f in sorted(dst.rglob("*")):
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        rel = f.relative_to(dst).as_posix()
        for what, rx in SCAN:
            for m in rx.finditer(text):
                found.append(f"{rel}: {what}: {m.group(0)[:40]!r}")
        for m in EMAIL.finditer(text):
            if m.group(0).lower() not in SOURCE_EMAIL_OK:
                found.append(f"{rel}: an email address: {m.group(0)!r}")
    if found:
        shutil.rmtree(dst, ignore_errors=True)
        raise SystemExit("source/ scan failed, nothing published:\n  " + "\n  ".join(found))
    print(f"  source/: {len(files)} files, scan clean (home paths, other emails, 32+ hex, keys and tokens, forbidden words)")
    return files


PIP_NAME = {"PIL": "pillow", "skimage": "scikit-image"}


def build_readme(data):
    """site/README.md: what GitHub shows on the repository's front page."""
    import ast
    mods = set()
    for f in (SITE / "code").rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else []
            mods |= {m.split(".")[0] for m in names if m.split(".")[0] not in sys.stdlib_module_names}
    pip = " ".join(sorted(PIP_NAME.get(m, m) for m in mods))
    rows, nworks = [], 0
    for e in data["exhibitions"]:
        for w in e["works"]:
            nworks += 1
            c = w["code"]
            rows.append(f"| {w['title']} | {e['artist']} | [{Path(c['src']).name}]({c['src']}) | {c['lines']:,} |")
    first = data["exhibitions"][0]["works"][0]["code"]["src"]
    text = f"""# {G['name']}

{G['line']}

{G['name']} is a walkable gallery: you move through a long hall and {len(data['exhibitions'])} rooms, one for each artist, and stand in front of {nworks} pictures, each with a short text, a recording and the program that made it.

It is entirely imagined. The gallery, its curator, the six artists and their lives, the exhibition texts, the building and every picture were conceived and made by Fable 5.2, an AI model, as an experiment by Shyam Sreevalsan. The artists are fictional. The places and trades they come from are real.

Live: **{SITE_URL}**

## How the pictures are made

Every artwork is one self-contained Python program that computes the picture. There is no image-generation model and there are no source images, photographs or textures: each program builds its own geometry, traces its own light and simulates its own paint and ink, and writes the finished image pixel by pixel. The program beside each work in the gallery is the whole of it.

## The works

| Work | Artist | Source | Lines |
| --- | --- | --- | ---: |
{chr(10).join(rows)}

## Run a picture yourself

You need Python 3.10 or later.

    pip install {pip}
    python {Path(first).name}

Each program writes its picture (a `.png`) next to itself. They are compute-heavy, and numba compiles on the first run, so be patient. Most also take a scale such as `0.5` for a quicker, smaller study; the top of each file says how.

## Where things are

- `index.html`, `gallery.js`, `community.js`, `style.css`: the 3D gallery (three.js, in `vendor/`).
- `code/<artist>/<work>.py`: the program for each picture, also as a readable page beside it.
- `works/`, `about/`, `guestbook/`: the same content as plain pages.
- `img/`: each picture at several sizes. `audio/`, `music/`: the synthesized audio guide and music.
- `s/`: small pages that give each work and room a share card.
- `source/`: the source of the gallery itself: `build.py` (makes this site), `narrate.py` and `ambient.py` (audio guide and music), `curator/programme.toml` (every text), `web/` (the page), and `backend/` (the small Cloudflare Worker behind hearts, the visitor count and the guest book).

## Licence

Artworks, texts and code © 2026 Shyam Sreevalsan. You may read, run and learn from the code; please ask before reusing the artworks or texts.

Contact: shyam.valsan@gmail.com
"""
    (SITE / "README.md").write_text(text, encoding="utf-8")


def main():
    SITE.mkdir(exist_ok=True)
    for old in ("rooms", "attendant", "colophon", "app.js", "tag.svg"):
        p = SITE / old
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    shutil.rmtree(SITE / "code", ignore_errors=True)
    data = collect()
    data["music"] = (ROOT / "web" / "ambient.mp3").exists()
    # per-room loops: web/music/nave.mp3 and web/music/<exhibition-slug>.mp3, whichever exist; the rest fall back to ambient.mp3
    wanted = ["nave"] + [e["slug"] for e in data["exhibitions"]]
    data["musicTracks"] = [k for k in wanted if (ROOT / "web" / "music" / f"{k}.mp3").exists()]
    print(f"  music: {len(data['musicTracks'])} of {len(wanted)} space tracks present: {', '.join(data['musicTracks']) or 'none'}")
    live = {e["slug"] for e in data["exhibitions"]}
    for d in (SITE / "img").iterdir():
        if d.name not in live:
            shutil.rmtree(d)
    for name in ("style.css", "gallery.js", "community.js", "mark.svg"):
        shutil.copy(ROOT / "web" / name, SITE / name)
    (SITE / "vendor").mkdir(exist_ok=True)
    shutil.copy(ROOT / "web" / "vendor" / "three.module.min.js", SITE / "vendor" / "three.module.min.js")
    if data["music"]:
        shutil.copy(ROOT / "web" / "ambient.mp3", SITE / "ambient.mp3")
    else:
        (SITE / "ambient.mp3").unlink(missing_ok=True)
    shutil.rmtree(SITE / "music", ignore_errors=True)
    if data["musicTracks"]:
        (SITE / "music").mkdir()
        for k in data["musicTracks"]:
            shutil.copy(ROOT / "web" / "music" / f"{k}.mp3", SITE / "music" / f"{k}.mp3")
    shutil.rmtree(SITE / "audio", ignore_errors=True)
    (SITE / "audio").mkdir()
    for k in DUR:
        shutil.copy(ROOT / "audio" / f"{k}.mp3", SITE / "audio" / f"{k}.mp3")
    (SITE / ".nojekyll").write_text("")
    build_walk(data)
    build_share(data)
    build_works(data)
    build_about()
    build_guestbook(data)
    build_404()
    publish_source()
    build_readme(data)
    n = sum(len(e["works"]) for e in data["exhibitions"])
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file() and ".git" not in f.parts)
    print(f"built {len(data['exhibitions'])} exhibitions, {n} works, {size / 1e6:.1f} MB -> {SITE}")


if __name__ == "__main__":
    main()
