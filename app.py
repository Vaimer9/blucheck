import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "model_output/final"
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


def clear_all(_qs):
    return [], fmt_questions([])


# ------------------------------------------------------------------- UI ------

with gr.Blocks(title="Questionnaire Designer") as demo:
    gr.Markdown(
        "# Questionnaire Designer (prototype)\n"
        "Build a questionnaire, let someone fill it in, then see how the model "
        "scores the answers.  \n"
        "*Prototype only. The model is trained on a public dataset and is not a "
        "diagnostic tool.*"
    )

    questions = gr.State([])

    with gr.Tabs():
        # ---- Tab 1: author ----
        with gr.Tab("1. Build"):
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
            clear_btn.click(clear_all, questions, [questions, q_list])

        # ---- Tab 2: respondent + results ----
        with gr.Tab("2. Answer"):

            # Re-runs whenever `questions` changes, so the form always matches the builder.
            @gr.render(inputs=questions)
            def render_form(qs):
                if not qs:
                    gr.Markdown("*Build a questionnaire in the first tab.*")
                    return

                boxes = [
                    gr.Textbox(label=f"{i + 1}. {q}", lines=3)
                    for i, q in enumerate(qs)
                ]
                submit = gr.Button("Submit answers", variant="primary")

                def on_submit(*answers):
                    # skip unanswered questions so they don't pull the average around
                    pairs = [
                        (q, a.strip()) for q, a in zip(qs, answers) if a and a.strip()
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
