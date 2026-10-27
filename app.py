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


def score_answers(texts):
    """Run the classifier on a list of texts -> tensor of shape (n_texts, n_labels)."""
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)


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
```
A plain list of question strings also works."""

with gr.Blocks(title="Questionnaire Designer") as demo:
    gr.Markdown(
        "# Questionnaire Designer (prototype)\n"
        "Build a questionnaire, let someone fill it in, then see how the model "
        "scores the answers.  \n"
        "*Prototype only. The model is trained on a public dataset and is not a "
        "diagnostic tool.*"
    )

    questions = gr.State([])
    prefill = gr.State([])  # prefilled answers, matched to questions by position

    with gr.Tabs():
        # ---- Tab 1: author ----
        with gr.Tab("1. Build"):
            gr.Markdown("### Start from a sample")
            with gr.Row():
                sample_btns = {name: gr.Button(name) for name in SAMPLES}
            if not SAMPLES:
                gr.Markdown(f"*No samples found ({SAMPLES_PATH.name} is missing or invalid).*")
            status = gr.Markdown()

            gr.Markdown("### Or load your own file")
            upload = gr.File(label="Questionnaire (.json)", file_types=[".json"])
            with gr.Accordion("JSON format", open=False):
                gr.Markdown(JSON_EXAMPLE)

            gr.Markdown("### Or build one by hand")
            new_q = gr.Textbox(
                label="New question",
                placeholder="e.g. How have you been feeling this week?",
            )
            with gr.Row():
                add_btn = gr.Button("Add question", variant="primary")
                remove_btn = gr.Button("Remove last")
                clear_btn = gr.Button("Clear all")
            q_list = gr.Markdown(fmt_questions([]))

            add_btn.click(add_question, [questions, new_q], [questions, q_list, new_q])
            new_q.submit(add_question, [questions, new_q], [questions, q_list, new_q])
            remove_btn.click(remove_last, questions, [questions, q_list])
            clear_btn.click(clear_all, None, [questions, prefill, q_list, status])
            upload.upload(load_uploaded, upload, [questions, prefill, q_list, status])
            for name, btn in sample_btns.items():
                btn.click(make_sample_loader(name), None, [questions, prefill, q_list, status])

        # ---- Tab 2: respondent + results ----
        with gr.Tab("2. Answer"):

            # Re-runs whenever the questions or prefilled answers change, so the
            # form always matches what was built or loaded.
            @gr.render(inputs=[questions, prefill])
            def render_form(qs, answers):
                if not qs:
                    gr.Markdown("*Build or load a questionnaire in the first tab.*")
                    return

                boxes = [
                    gr.Textbox(
                        label=f"{i + 1}. {q}",
                        lines=3,
                        value=answers[i] if i < len(answers) else "",
                    )
                    for i, q in enumerate(qs)
                ]
                submit = gr.Button("Submit answers", variant="primary")

                def on_submit(*texts):
                    # skip unanswered questions so they don't pull the average around
                    pairs = [
                        (q, a.strip()) for q, a in zip(qs, texts) if a and a.strip()
                    ]
                    if not pairs:
                        raise gr.Error("Answer at least one question first.")

                    probs = score_answers([a for _, a in pairs])  # (n, n_labels)
                    avg = probs.mean(dim=0)
                    avg_dict = {l: float(avg[i]) for i, l in enumerate(labels)}

                    rows = []
                    for (q, a), p in zip(pairs, probs):
                        row = {"Question": q, "Answer": a, "Top label": labels[int(p.argmax())]}
                        row.update({l: round(float(p[i]) * 100, 1) for i, l in enumerate(labels)})
                        rows.append(row)

                    return avg_dict, pd.DataFrame(rows)

                submit.click(on_submit, inputs=boxes, outputs=[avg_label, per_question])

            gr.Markdown("## Results")
            avg_label = gr.Label(
                label="Average score across all answers",
                num_top_classes=len(labels),
            )
            per_question = gr.Dataframe(
                label="Per-question scores (% per label)", interactive=False, wrap=True
            )

demo.launch()
