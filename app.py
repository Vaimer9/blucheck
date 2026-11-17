import json
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "model_output/final"
SAMPLES_PATH = Path(__file__).resolve().parent / "samples.json"
FALLBACK_LABELS = [
    "Anxiety", "Bipolar", "Depression", "Normal",
    "Personality Disorder", "Stress", "Suicidal",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # inference mode: disables dropout etc.

# Use the label names stored in the model config if you saved them there,
# otherwise fall back to the hardcoded list (LabelEncoder sorts alphabetically).
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
if labels[0].startswith("LABEL_"):
    labels = FALLBACK_LABELS

# --- short-answer penalty configuration ---
NORMAL_LABEL = "Normal"
SHORT_ANSWER_WORDS = 10   # average words per answer below this triggers the rule
NORMAL_KEEP = 0.3         # keep 30% of Normal -> deduct 70%


def score_answers(texts):
    """Run the classifier on a list of texts -> tensor of shape (n_texts, n_labels)."""
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)


def penalise_normal(probs, avg_words):
    """If answers are very short, cut Normal by 70% and renormalise the rest.

    probs: dict of {label: probability} (may be floats or tensors)
    Returns (adjusted_probs, was_adjusted).
    """
    if avg_words >= SHORT_ANSWER_WORDS or NORMAL_LABEL not in probs:
        return probs, False

    adjusted = {k: float(v) for k, v in probs.items()}
    adjusted[NORMAL_LABEL] *= NORMAL_KEEP
    total = sum(adjusted.values())
    if total > 0:                      # renormalise so scores still sum to 1
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted, True


# ------------------------------------------------------- questionnaire JSON --
#
# A questionnaire is {"items": [{"question": "...", "answer": "..."}, ...]}.
# "answer" is optional (it prefills the form). A bare list of items, or of plain
# question strings, is accepted too. samples.json maps a button name to a
# questionnaire plus an optional "note".

def parse_items(data):
    """Normalise a questionnaire from JSON into (questions, answers)."""
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        raise ValueError('expected a list of items or {"items": [...]}')

    qs, answers = [], []
    for item in data:
        if isinstance(item, str):
            question, answer = item, ""
        elif isinstance(item, dict):
            question, answer = item.get("question", ""), item.get("answer", "")
        else:
            continue
        question = str(question).strip()
        if question:
            qs.append(question)
            answers.append(str(answer).strip())
    if not qs:
        raise ValueError("no questions found")
    return qs, answers


def load_samples():
    try:
        return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"Could not load {SAMPLES_PATH.name}: {e}")
        return {}


SAMPLES = load_samples()


# ---------------------------------------------------------------- builder ----

def fmt_questions(qs):
    if not qs:
        return "*No questions yet. Add one above.*"
    return "\n".join(f"{i + 1}. {q}" for i, q in enumerate(qs))


def add_question(qs, text):
    text = (text or "").strip()
    if text:
        qs = qs + [text]
    return qs, fmt_questions(qs), ""  # last value clears the textbox


def remove_last(qs):
    qs = qs[:-1]
    return qs, fmt_questions(qs)


def clear_all():
    return [], [], fmt_questions([]), ""


def make_sample_loader(name):
    sample = SAMPLES[name]
    qs, answers = parse_items(sample)
    note = sample.get("note", "") if isinstance(sample, dict) else ""

    def load():
        msg = f"Loaded **{name}**: {len(qs)} questions with answers prefilled. {note}"
        return qs, answers, fmt_questions(qs), msg

    return load


def load_uploaded(file):
    if file is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    path = getattr(file, "name", file)  # str in newer Gradio, tempfile in older
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        qs, answers = parse_items(data)
    except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
        raise gr.Error(f"Could not read that questionnaire: {e}")
    n_filled = sum(1 for a in answers if a)
    msg = f"Loaded {len(qs)} questions from file ({n_filled} with prefilled answers)."
    return qs, answers, fmt_questions(qs), msg


# ------------------------------------------------------------------- UI ------

JSON_EXAMPLE = """```json
{
  "items": [
    {"question": "How have you been feeling lately?", "answer": "optional prefilled answer"},
    {"question": "How have you been sleeping?"}
  ]
}
