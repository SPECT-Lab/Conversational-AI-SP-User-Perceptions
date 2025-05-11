import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, RobertaForSequenceClassification
import numpy as np
import evaluate
from transformers import default_data_collator, TrainingArguments, Trainer
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
import sys

if len(sys.argv) < 2:
    raise ValueError("No CSV file provided. Please run the script with a CSV file as an argument. Example: python train_model.py <path_to_your_csv_file>")

csv_file = sys.argv[1]
data = pd.read_csv(csv_file)

if not {'text', 'label'}.issubset(data.columns):
    raise ValueError("The CSV file must contain 'text' and 'label' columns.")

data = data[['text', 'label']]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
raw_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

print(raw_dataset)

MODEL = "FacebookAI/roberta-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = RobertaForSequenceClassification.from_pretrained(MODEL, num_labels=2)

model.config.max_position_embeddings

MAX_LENGTH = 256

def tokenize_function(examples):
    encoding = tokenizer(examples["text"], padding="max_length", max_length = MAX_LENGTH, truncation=True)
    return encoding

train_dataset = raw_dataset["train"].map(tokenize_function, batched=True)
eval_dataset = raw_dataset["test"].map(tokenize_function, batched=True)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1_score = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels)["precision"],
        "recall": recall.compute(predictions=predictions, references=labels)["recall"],
        "f1_score": f1_score.compute(predictions=predictions, references=labels)["f1"]
    }

args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    learning_rate=1e-5,
    eval_strategy = "epoch",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    fp16=False,
    load_best_model_at_end=True,         
    metric_for_best_model="f1_score",
    greater_is_better=True
)

data_collator = default_data_collator

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        class_weights = torch.tensor([1.000, 7.707])
        loss_fn = CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits.squeeze(), labels.squeeze())

        return (loss, outputs) if return_outputs else loss

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
