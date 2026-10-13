import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model_output/final")
model = AutoModelForSequenceClassification.from_pretrained("model_output/final")
model.eval()  # inference mode — disables dropout etc.

labels = ["Anxiety", "Bipolar", "Depression", "Normal", "Personality Disorder", "Stress", "Suicidal"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():          # no gradients needed, we're not training
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(fn=predict, inputs="text", outputs="label", title="Mental Health Text Classifier").launch()
