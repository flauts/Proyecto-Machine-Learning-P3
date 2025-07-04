# 1. Importar librerías
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamWeightDecay, \
    get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import dataframe_image as dfi
import os  # Import os to create directory

# 2. Verificar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Cargar los datos
try:
    df = pd.read_csv("BullyingMultiClase.csv")
except FileNotFoundError:
    print("❌ ERROR: Main data file 'BullyingMultiClase.csv' not found. Please ensure it is in the correct directory.")
    exit()

# 4. Separar entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42)

# 5. Tokenizador
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


# 6. Dataset personalizado
class BullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_dataset = BullyingDataset(train_texts, train_labels, tokenizer)
val_dataset = BullyingDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 7. Cargar modelo
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4)
model = model.to(device)

# 8. Optimizador y Scheduler
optimizer = AdamWeightDecay(model.parameters(), lr=2e-5)
epochs = 4
num_training_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 9. Entrenamiento
training_log_output = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss_log = f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}"
    print(avg_loss_log)
    training_log_output.append(avg_loss_log)

# 10. Evaluación
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 11. Guardar Reporte de Clasificación y Logs
print("\n" + "=" * 50)
print("Saving training and evaluation results...")
print("=" * 50)
report_dict = classification_report(true_labels, predictions, digits=4, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
dfi.export(report_df, "classification_report.png", table_conversion='matplotlib')
print(f"✅ Classification report saved to classification_report.png")
with open("training_log.txt", 'w') as f:
    f.write("--- Training Log ---\n")
    f.write("\n".join(training_log_output))
print(f"✅ Training log saved to training_log.txt")

# 11.5 SAVE THE FINE-TUNED MODEL FOR LATER USE
# ==================================================
output_dir = "./saved_cyberbullying_model"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model and tokenizer saved to {output_dir}")

# 12. PREDICT ON NEW DATA AND SAVE OUTPUT
# (This section remains the same, running predictions right after training)
# ... (The prediction code from section 12 in the previous answer goes here) ...