#!/usr/bin/env python3
"""Clean the mental-health statements CSV for BERT fine-tuning.

Usage: python clean_data.py [in.csv] [out.csv]
Needs only pandas + the standard library (no emoji/ftfy install required).
"""
import csv
import html
import re
import sys
import unicodedata

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "data_cleaned.csv"

# ---------------------------------------------------------------- mojibake
# "â€™" / "ðŸ˜­" are UTF-8 bytes that were wrongly read as cp1252. Re-decode
# what we can; truncated leftovers (the stray "™", "‚", "Ÿ" ...) are dropped.
def _byte(ch):
    o = ord(ch)
    if o < 0x80:
        return None
    if o < 0x100 and not 0x80 <= o <= 0x9F:
        return o
    try:
        return ch.encode("cp1252")[0]
    except UnicodeEncodeError:
        return o if o < 0x100 else None


_LEAD, _CONT = [], []
for _b in range(0x80, 0x100):
    _ch = bytes([_b]).decode("cp1252", errors="ignore") or chr(_b)
    if 0xC2 <= _b <= 0xF4:
        _LEAD.append(re.escape(_ch))
    if _b <= 0xBF:
        _CONT.append(re.escape(_ch))
        _CONT.append(re.escape(chr(_b)))
MOJI = re.compile("[%s][%s]{0,3}" % ("".join(_LEAD), "".join(_CONT)))
LONE_LEAD = re.compile(r"[âðÃÂ](?![A-Za-z])")


def _fix_moji(m):
    run = m.group(0)
    try:
        return bytes(_byte(c) for c in run).decode("utf-8")
    except Exception:
        return " " if len(run) > 1 else run


# ---------------------------------------------------------------- patterns
PUNCT = str.maketrans({
    "’": "'", "‘": "'", "‚": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "«": " ", "»": " ",
    "–": "-", "—": "-", "−": "-", "…": "...",
    "•": " ", "●": " ", "·": " ",
    "\u00a0": " ", "\u200b": "", "\u200c": "", "\u2060": "", "\ufeff": "",
    "\t": " ", "\r": " ", "\n": " ",
})
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"(?:https?://|www\.)\S+|\b\S+@\S+\.\S+\b", re.I)
BARE_URL = re.compile(
    r"\bhttps?\b\s+(?:(?:ww+|[a-z0-9-]{1,30})\s+){0,3}?"
    r"(?:com|org|net|edu|gov|io|co|uk|eu|me|ly|tv|info|html?|php)\b", re.I)
REDDIT = re.compile(r"(?<![\w/])/?[ru]/\w+")
REMOVED = re.compile(r"\[(?:removed|deleted)\]", re.I)

# text emoticons -> :name:
EMOTICON = re.compile(
    r"(?<![\d/])([:;=])(['^-]?)(\)+|\]+|\(+|\[+|D+|[Pp]+|[Oo]+|\|+|\*+|/+|\\+)(?![\w/\\])")
MOUTH = {")": "smile", "]": "smile", "(": "sad", "[": "sad", "D": "big_smile",
         "P": "tongue_out", "p": "tongue_out", "O": "surprised", "o": "surprised",
         "|": "neutral", "*": "kiss", "/": "skeptical", "\\": "skeptical"}
OTHER_EMOTICONS = [
    (re.compile(r"(?<![\w<])<3+(?!\w)"), "heart"),
    (re.compile(r"(?<!\w)(?:XD|xD)(?!\w)"), "laughing"),
    (re.compile(r"(?<!\w)(?:XP|xP)(?!\w)"), "tongue_out"),
    (re.compile(r"(?<!\w)\^[_.\-]?\^(?!\w)"), "happy"),
    (re.compile(r"(?<!\w)[-~]_[-~](?!\w)"), "annoyed"),
    (re.compile(r"(?<!\w)(?:T[_.]T|;_+;)(?!\w)"), "crying"),
    (re.compile(r"(?<!\w)[oO]_[oO](?!\w)"), "surprised"),
]


def _emoticon(m):
    eyes, nose, mouth = m.group(1), m.group(2), m.group(3)[0]
    name = MOUTH[mouth]
    if name == "sad" and (eyes == ";" or nose == "'"):
        name = "crying"
    elif name == "smile" and eyes == ";":
        name = "wink"
    elif name == "smile" and nose == "'":
        name = "happy_tears"
    return f" :{name}: "


# unicode emoji -> :name:
_E = "\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2300-\u23FF\u203C\u2049"
EMOJI_RUN = re.compile(f"[{_E}][{_E}\u200d\ufe0f\u20e3]*")
OVERRIDES = {"\u2764": "red_heart", "\u2665": "heart_suit",
             "\u263a": "smiling_face", "\u2639": "frowning_face"}


def _emoji_run(m):
    names, after_zwj, flags = [], False, 0
    for c in m.group(0):
        if c == "\u200d":
            after_zwj = True
            continue
        if c in "\ufe0f\u20e3" or "\U0001F3FB" <= c <= "\U0001F3FF":
            continue
        if "\U0001F1E6" <= c <= "\U0001F1FF":
            flags += 1
            if flags % 2 == 0:
                names.append("flag")
            continue
        if after_zwj and c in "\u2640\u2642\u2695\u2696\u2708":
            after_zwj = False
            continue
        after_zwj = False
        n = OVERRIDES.get(c)
        if n is None:
            try:
                n = re.sub(r"[^a-z0-9]+", "_", unicodedata.name(c).lower()).strip("_")
            except ValueError:
                continue
        names.append(n)
    return " " + " ".join(f":{n}:" for n in names) + " "


JUNK = re.compile(r"[^\x20-\x7e\w]")          # keep ASCII + any-script letters/digits
DEBRIS = re.compile(r"[²³¹¼½¾ªºƒŒœŠšŸŽžˆ]")   # leftover mojibake letters/symbols
REPEAT_EMOJI = re.compile(r"(:[a-z0-9_]+:)(?:\s+\1){2,}")


def clean(t):
    t = MOJI.sub(_fix_moji, t)
    t = LONE_LEAD.sub(" ", t)
    t = html.unescape(html.unescape(t))             # "&amp;#x200B;" needs two passes
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"\\([\\'\"~*_()\[\]<>#!.\-])", r"\1", t)   # reddit "\'" escapes
    t = re.sub(r"\\+", "/", t)
    t = LINK.sub(r"\1", t).replace("](", " ")
    t = URL.sub(" ", t)
    t = re.sub(r"\bhttps?\b", " ", BARE_URL.sub(" ", t), flags=re.I)
    t = REDDIT.sub(" ", t)                          # r/depression etc. leak the label
    t = REMOVED.sub(" ", t)
    t = re.sub(r"\*+|~~", " ", t)                   # markdown bold / censor stars
    t = t.translate(PUNCT).replace("''", '"')
    t = EMOTICON.sub(_emoticon, t)
    for pat, name in OTHER_EMOTICONS:
        t = pat.sub(f" :{name}: ", t)
    t = EMOJI_RUN.sub(_emoji_run, t)
    t = DEBRIS.sub(" ", JUNK.sub(" ", t))
    t = re.sub(r"\s+([,.!?;])", r"\1", t)           # no space before punctuation
    t = re.sub(r",(?=[A-Za-z])", ", ", t)           # space after comma
    t = re.sub(r"([!?;])(?=[A-Za-z])", r"\1 ", t)
    t = re.sub(r"(?<=[a-z]{2})\.(?=[A-Z][a-z])", ". ", t)
    t = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", t)     # sooooo -> soo
    t = re.sub(r"([!?])\1{3,}", r"\1\1\1", t)
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r",{2,}", ",", t)
    t = re.sub(r"([^\w\s:])\1{4,}", r"\1\1\1", t)
    t = REPEAT_EMOJI.sub(r"\1 \1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 1 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2:
        t = t[1:-1].strip()
    return t


def main():
    df = pd.read_csv(SRC)
    n0 = len(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["statement"])
    n_null = n0 - len(df)

    df["statement"] = df["statement"].astype(str).map(clean)
    df = df[df["statement"].str.contains(r"[A-Za-z]{2}") & (df["statement"].str.len() >= 3)]
    n_short = n0 - n_null - len(df)

    key = df["statement"].str.lower()
    # same text with different labels -> label is unreliable, drop every copy
    conflict = df.groupby(key)["status"].transform("nunique") > 1
    n_conflict = int(conflict.sum())
    df = df[~conflict]
    n_before = len(df)
    df = df[~df["statement"].str.lower().duplicated()]
    n_dup = n_before - len(df)

    df = df.reset_index(drop=True)
    df.to_csv(DST, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n", encoding="utf-8")

    print(f"rows in: {n0} | null: {n_null} | too short/no letters: {n_short} | "
          f"conflicting labels: {n_conflict} | exact dups: {n_dup} | rows out: {len(df)}")
    print(df["status"].value_counts().to_string())
    print(df.groupby("status")["statement"].apply(lambda s: s.str.len().median()).round().to_string())


if __name__ == "__main__":
    main()
