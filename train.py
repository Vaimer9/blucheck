"""Fine-tune BERT on the cleaned mental-health statements.

Quick start:
    python train.py

Fast smoke test (checks everything runs):
    python train.py --limit 3000 --epochs 1
"""
import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from prepare_data import load_clean, make_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/statements.csv", help="raw CSV (statement,status)")
    p.add_argument("--cleaned", default="data/statements_cleaned.csv", help="cache for cleaned CSV")
    p.add_argument("--reclean", action="store_true")
    p.add_argument("--out", default="model_output/final", help="where app.py loads the model from (old one is backed up)")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--class-weights", choices=["none", "sqrt", "balanced"], default="balanced")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="cap rows per class (e.g. 6000) to cut training time / rebalance")
    p.add_argument("--limit", type=int, default=None, help="use only N rows (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class EncDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in ("input_ids", "attention_mask") if k in self.enc}
        item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(outputs.logits.float(), labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted")}


def length_baseline(train, test):
    """How well does text LENGTH alone predict the label? (the shortcut check)"""
    def feats(d):
        s = d["statement"]
        return np.c_[s.str.len(), s.str.split().str.len()]
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(feats(train), train["label"])
    pred = tree.predict(feats(test))
    return accuracy_score(test["label"], pred), f1_score(test["label"], pred, average="macro")


def save_confusion(cm, names, out_dir):
    pd.DataFrame(cm, index=names, columns=names).to_csv(out_dir / "confusion_matrix.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Row-normalised confusion matrix")
        fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)
    except ImportError:
        print("(matplotlib not installed, skipped PNG; CSV saved)")


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):          # don't clobber the model app.py currently uses
        backup = out.with_name(out.name + "_prev_" + time.strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(out), str(backup))
        print(f"[backup] previous model moved to {backup}")
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    df = load_clean(args.data, args.cleaned, args.reclean)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed)
    train, val, test, label2id = make_splits(df, seed=args.seed, max_per_class=args.max_per_class)
    id2label = {v: k for k, v in label2id.items()}
    names = [id2label[i] for i in range(len(id2label))]
    print(f"\nsplit sizes  train={len(train)}  val={len(val)}  test={len(test)}")
    print(train["status"].value_counts().to_string())

    acc, mf1 = length_baseline(train, test)
    print(f"\n[shortcut check] a depth-4 tree on text length ALONE gets acc={acc:.3f}, macro-F1={mf1:.3f}")
    print("Your BERT needs to clearly beat this to be learning more than length.\n")

    # ---------------- tokenise ----------------
    tok = AutoTokenizer.from_pretrained(args.model)
    def enc(d):
        return tok(d["statement"].tolist(), truncation=True, max_length=args.max_len)
    ds_train = EncDataset(enc(train), train["label"].values)
    ds_val = EncDataset(enc(val), val["label"].values)
    ds_test = EncDataset(enc(test), test["label"].values)

    # ---------------- class weights ----------------
    cw = None
    if args.class_weights != "none":
        w = compute_class_weight("balanced", classes=np.arange(len(names)), y=train["label"].values)
        if args.class_weights == "sqrt":
            w = np.sqrt(w)
        cw = torch.tensor(w, dtype=torch.float)
        print("class weights:", {n: round(float(x), 2) for n, x in zip(names, w)})

    # ---------------- model ----------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(names), id2label=id2label, label2id=label2id)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    total_steps = math.ceil(len(ds_train) / args.batch_size) * args.epochs
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=int(0.06 * total_steps),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",   # accuracy hides minority-class failures
        greater_is_better=True,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        dataloader_num_workers=2,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, class_weights=cw)
    trainer.train()

    # ---------------- evaluate ----------------
    pred_out = trainer.predict(ds_test)
    preds = pred_out.predictions.argmax(-1)
    y = test["label"].values
    print("\n=== TEST SET ===")
    print(classification_report(y, preds, target_names=names, digits=3))
    cm = confusion_matrix(y, preds, labels=range(len(names)))
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=names, columns=names).to_string())
    save_confusion(cm, names, out)

    # does it still work when length can't give the answer away?
    words = test["statement"].str.split().str.len()
    buckets = pd.cut(words, [0, 15, 50, 150, np.inf], labels=["<=15w", "16-50w", "51-150w", ">150w"])
    print("\nperformance by length bucket (a big gap = length shortcut):")
    for b in buckets.cat.categories:
        m = (buckets == b).values
        if m.sum() > 0:
            print(f"  {b:8s} n={m.sum():5d}  acc={accuracy_score(y[m], preds[m]):.3f}  "
                  f"macro-F1={f1_score(y[m], preds[m], average='macro', zero_division=0):.3f}")

    test.assign(pred=[id2label[p] for p in preds]).to_csv(out / "test_predictions.csv", index=False)

    # ---------------- save ----------------
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"\nSaved model + tokenizer to {out}/  (label names are now real, not LABEL_0..6)")


if __name__ == "__main__":
    main()
