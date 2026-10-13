from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load and split data
df = pd.read_csv("data/statements.csv")
le = LabelEncoder()
df["label"] = le.fit_transform(df["status"])  # e.g. "Depression" -> 2
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# 2. Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["statement"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# 3. Load pretrained model, swap in your classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(le.classes_)
)

# 4. Set training config
args = TrainingArguments(
    output_dir="model_output",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    warmup_steps=500,          # this is your "ramp up"
    lr_scheduler_type="linear", # this is your "ramp down"
    eval_strategy="epoch",
    save_strategy="epoch",
)

# 5. Train
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()

# 6. Save the finished model + tokenizer
model.save_pretrained("model_output/final")
tokenizer.save_pretrained("model_output/final")
