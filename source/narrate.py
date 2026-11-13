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
MODEL = "eleven_v4"
# One voice per speaker (ElevenLabs voice ids; the makers' are Voice Library
# voices added to the account). Leela speaks the welcome and each exhibition's
# introduction; each maker speaks about their own works.
VOICES = {
    "Leela": "pFZP5JQG7iQjIQuC4Bku",
    "Stanzin Dolkar": "T6T157Ans2FrOfpaKzCO",
    "Go Sun-deok": "Zjb2Dbq5IbWDKpVOllIo",
    "Ruhi Vaghela": "OIIrFPBzLAigdcttMGWZ",
    "Halina Mróz Quispe": "pTX8uGyVgHCWLj6IkcbC",
    "Tobi Adeyemi": "LEvd0YiWkwZ6hTZOmdVE",
    "Ilanji": "ybsn8GUgoNB8oDLyFqwG",
}
# Direction for the voice: how each speaker talks, and a few lines that get
# their own tone. Bracketed tags are read by the model as direction, not text.
BASE = {
    "Leela": "[Warm, low, unhurried]",
    "Stanzin Dolkar": "[Quiet, dry, matter of fact]",
    "Go Sun-deok": "[Old woman, blunt, slow]",
    "Ruhi Vaghela": "[Warm, precise, a little teasing]",
    "Halina Mróz Quispe": "[Dry, deadpan, official]",
    "Tobi Adeyemi": "[Quick, warm, confident]",
    "Ilanji": "[Old man, gentle, slow, tender]",
}
LINES = {
    "I am not going to tell you which.": "[faintly amused]",
    "Then step back and let the whole wheel turn.": "[softly]",
    "This does not happen here.": "[with quiet wonder]",
    "Whoever was awake in there saw it too.": "[softly]",
    "Then it was light again.": "[softly]",
    "That was the best thing about it.": "[with a small laugh]",
    "The light at the top always looks nearer than it is.": "[tired, tender]",
    "That is the picture.": "[quietly]",
    "My father used to say": "[fondly]",
    "Now find her.": "[quietly]",
    "Then look at the sky again": "[softly]",
    "Then check the window.": "[a touch amused]",
    "It does one thing, everywhere, always.": "[slowly]",
    "It is not. It is behind you": "[firmly]",
    "It is where I went down in 2022.": "[quieter]",
    "That one I do not answer.": "[quietly]",
    "because I looked at it twice": "[amused]",
    "My grandmother lit one like it every dusk.": "[softly]",
    "Please do not.": "[very gently]",
    "I drew a lamp.": "[softly]",
    "Will they leave a grove alone?": "[wondering]",
    "The lamp is still lit.": "[softly]",
}

# Respellings for the voice only; the printed text is never changed.
SAY = {"25:00": "twenty-five o'clock", "Tso Moriri": "So Moriri", "Thrikkarthika": "Thrik-kaar-thika",
       "Go Sun-deok": "Go Soon-duk", "Kavu": "Kaavu", "kavu": "kaavu", "Kuvo": "Koovo", "kuvo": "koovo",
       "Vav": "Vaav", "Kund": "Koond"}


def spoken(text):
    who = SPEAKER.get(text, "Leela")
    for k in sorted(SAY, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", SAY[k], text)
    for k, tag in LINES.items():
        text = text.replace(k, f"{tag} {k}")
    return f"{BASE[who]} {text.strip()}"


def key(text):
    return hashlib.sha1(f"{VOICES[SPEAKER.get(text, 'Leela')]}|{MODEL}|{spoken(text)}".encode()).hexdigest()[:14]


def items():
    yield "welcome", P["curator"]["welcome"]
    for e in P["exhibition"]:
        yield e["slug"], f'{e["artist"]}. {e["title"]}.\n\n{e["intro"]}'
        for w in e["work"]:
            yield w["file"], f'{w["title"]}.\n\n{w["story"]}'


SPEAKER = {}
for _e in P["exhibition"]:
    for _w in _e["work"]:
        SPEAKER[f'{_w["title"]}.\n\n{_w["story"]}'] = _e["artist"]


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
            v = VOICES[SPEAKER.get(text, "Leela")]
            body = json.dumps({"text": spoken(text), "model_id": MODEL}).encode()
            rq = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{v}?output_format=mp3_44100_128",
                                        data=body, headers={"xi-api-key": api, "Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=300) as r:
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
