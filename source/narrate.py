#!/usr/bin/env python3
"""Speak the audio guide: every narrated text in curator/programme.toml becomes
an MP3 in audio/, named by a hash of voice and text, so only new or changed
texts are spoken. Uses ElevenLabs (key in .env); one voice per speaker.

    .venv/bin/python narrate.py
"""
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).parent
OUT = ROOT / "audio"
P = tomllib.loads((ROOT / "curator" / "programme.toml").read_text(encoding="utf-8"))
MODEL = "eleven_turbo_v2_5"
# One voice per speaker (ElevenLabs voice id, speed). Leela speaks the welcome
# and each exhibition's introduction; each maker speaks about their own works.
VOICES = {
    "Leela": ("pFZP5JQG7iQjIQuC4Bku", 0.95),
    "Stanzin Dolkar": ("EXAVITQu4vr4xnSDxMaL", 0.95),
    "Go Sun-deok": ("XrExE9yKIg1WjnnlVkGX", 0.88),
    "Ruhi Vaghela": ("hpp4J3VqNfWAUOO0d1Us", 0.95),
    "Halina Mróz Quispe": ("Xb7hH8MSUJpSbSDYk0k2", 0.92),
    "Tobi Adeyemi": ("bIHbv24MWmeRgasZH58o", 1.0),
    "Ilanji": ("pqHfZKP75CvOlQylNhV4", 0.9),
}

# Respellings for the voice only; the printed text is never changed.
SAY = {"25:00": "twenty-five o'clock"}


def spoken(text):
    for k in sorted(SAY, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", SAY[k], text)
    return text


def key(text):
    v, s = SPEAKER.get(text, VOICES["Leela"])
    return hashlib.sha1(f"{v}|{MODEL}|{s}|{spoken(text)}".encode()).hexdigest()[:14]


def items():
    yield "welcome", P["curator"]["welcome"]
    for e in P["exhibition"]:
        yield e["slug"], f'{e["artist"]}. {e["title"]}.\n\n{e["intro"]}'
        for w in e["work"]:
            yield w["file"], f'{w["title"]}.\n\n{w["story"]}'


SPEAKER = {}
for _e in P["exhibition"]:
    for _w in _e["work"]:
        SPEAKER[f'{_w["title"]}.\n\n{_w["story"]}'] = VOICES[_e["artist"]]


def main():
    OUT.mkdir(exist_ok=True)
    todo = [(n, t) for n, t in items() if not (OUT / f"{key(t)}.mp3").exists()]
    index = {}
    if todo:
        import os, urllib.request
        env = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines() if "=" in l)
        api = env["ELEVENLABS_API_KEY"].strip().strip('"\'')
        print(f"{sum(len(spoken(t)) for _, t in todo)} characters to speak")
        if os.environ.get("DRY"):
            return
        for name, text in todo:
            v, s = SPEAKER.get(text, VOICES["Leela"])
            body = json.dumps({"text": spoken(text).strip(), "model_id": MODEL,
                               "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "speed": s}}).encode()
            rq = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{v}?output_format=mp3_44100_96",
                                        data=body, headers={"xi-api-key": api, "Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=180) as r:
                (OUT / f"{key(text)}.mp3").write_bytes(r.read())
            print(f"spoke {name}")
    used = set()
    for name, text in items():
        f = OUT / f"{key(text)}.mp3"
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                                   capture_output=True, text=True).stdout or 0)
        index[key(text)] = round(dur, 1)
        used.add(f.name)
    for f in OUT.glob("*.mp3"):
        if f.name not in used:
            f.unlink()
    (OUT / "index.json").write_text(json.dumps(index, indent=0))
    print(f"{len(index)} recordings, {sum(index.values()) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
