import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight

# 1. Cargar dataset completo
df = pd.read_csv("BullyingFinal.csv")
df = df[["Text", "oh_label"]].dropna()
df["oh_label"] = df["oh_label"].astype(int)

# 2. Tomar una muestra del 10% estratificada por 'oh_label'
df_reducido, _ = train_test_split(
    df,
    train_size=0.10,  # Puedes ajustar esto según tu RAM
    stratify=df["oh_label"],
    random_state=42
)

# 3. Renombrar columna y crear dataset HF
df_reducido["labels"] = df_reducido["oh_label"]
df_reducido = df_reducido[["Text", "labels"]]
dataset = Dataset.from_pandas(df_reducido)

# 4. Tokenizador
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["Text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Dividir dataset
split_dataset = tokenized_dataset.train_test_split(test_size=0.3)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 6. Calcular class weights
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df_reducido["labels"]),
    y=df_reducido["labels"]
)
class_weights = torch.tensor(weights, dtype=torch.float)

# 7. Modelo con pérdida personalizada
class WeightedModel(nn.Module):
    def __init__(self, base_model_name, num_labels, class_weights):
        super(WeightedModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}

model = WeightedModel(model_name, num_labels=2, class_weights=class_weights)

# 8. Métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 9. Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./xlmr_bullying_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    do_eval=True,
    save_steps=500,
    eval_strategy="steps",           # ← ¡CORREGIDO! es eval_strategy, no es evaluation_strategy
    save_strategy="steps",                 # ← Para dejarlo claro
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

# 10. Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# 11. Entrenar
trainer.train()

# Guardar el modelo y tokenizer entrenados
model.save_pretrained("./xlmr_bullying_final_model")
tokenizer.save_pretrained("./xlmr_bullying_final_model")
print("✅ Modelo y tokenizer guardados en './xlmr_bullying_final_model'")

# 12. Evaluación detallada
predictions_output = trainer.predict(eval_dataset)
preds = np.argmax(predictions_output.predictions, axis=1)
labels = predictions_output.label_ids

report = classification_report(
    labels,
    preds,
    target_names=["No-Cyberbullying", "Cyberbullying"]
)

print("\n📋 Reporte de clasificación detallado:")
print(report)

