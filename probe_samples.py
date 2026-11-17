"""Score every questionnaire in samples.json with the trained model.

    python probe_samples.py [model_dir]

Shows, per sample group, the average probability per label and how many answers got each
top label. This is the quickest check for the length shortcut: compare "Normal (short)"
with "Normal (long)", and watch whether Depression answers leak into Suicidal.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clean_data import clean
from prepare_data import _despace_tokens

model_dir = sys.argv[1] if len(sys.argv) > 1 else "model_output/final"
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

samples = json.loads((Path(__file__).parent / "samples.json").read_text(encoding="utf-8"))
for name, group in samples.items():
    answers = [it["answer"] for it in group["items"] if it.get("answer")]
    texts = [_despace_tokens(clean(a)) for a in answers]   # same preprocessing as training
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    top = Counter(labels[int(p.argmax())] for p in probs)
    avg = probs.mean(0)
    print(f"\n== {name}  (n={len(texts)})")
    print("   top labels:", dict(top))
    print("   avg probs :", {l: round(float(avg[i]), 2) for i, l in enumerate(labels) if avg[i] > 0.03})
