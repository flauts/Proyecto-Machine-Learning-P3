# 1. Importar librerías
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup
from torch.optim import AdamW as AdamWeightDecay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import dataframe_image as dfi
import os  # Import os to create directory
from tqdm import tqdm  # Para barras de progreso
import json  # Para guardar configuraciones
import pickle  # Para guardar objetos Python

# 2. Verificar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Cargar los datos
try:
    df = pd.read_csv("BullyingMultiClase.csv")
except FileNotFoundError:
    print("❌ ERROR: Main data file 'BullyingMultiClase.csv' not found. Please ensure it is in the correct directory.")
    exit()

# 🔧 FIX CRÍTICO: Mapeo de etiquetas string a números
label_mapping = {
    'not_cyberbullying': 0,
    'ethnicity/race': 1, 
    'religion': 2,
    'gender/sexual': 3
}

# Verificar etiquetas y aplicar mapeo
print(f"📊 Etiquetas encontradas: {df['label'].unique()}")
df['label_numeric'] = df['label'].map(label_mapping)
df = df.dropna(subset=['label_numeric'])  # Eliminar etiquetas no reconocidas
df['label_numeric'] = df['label_numeric'].astype(int)

print(f"📈 Distribución de clases:")
print(df['label'].value_counts())
print(f"📊 Total de muestras válidas: {len(df)}")

# 4. Separar entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label_numeric"].tolist(), test_size=0.2, random_state=42, stratify=df["label_numeric"])

# 5. Tokenizador
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


# 6. Dataset personalizado - CORREGIDO
class BullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Asegurar que es string
        label = int(self.labels[idx])  # Asegurar que es entero
        
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)  # 🔧 FIX: 'labels' no 'label'
        }


train_dataset = BullyingDataset(train_texts, train_labels, tokenizer)
val_dataset = BullyingDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 🔧 Batch más grande
val_loader = DataLoader(val_dataset, batch_size=32)

# 7. Cargar modelo
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4)
model = model.to(device)

# 8. Optimizador y Scheduler - MEJORADO
optimizer = AdamWeightDecay(model.parameters(), lr=2e-5, weight_decay=0.01)
epochs = 6  # 🔧 Más épocas
num_training_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,  # 🔧 Warmup steps
    num_training_steps=num_training_steps
)

# Variables para early stopping
best_val_loss = float('inf')
best_model_state = None
patience = 3
patience_counter = 0

# 📁 CREAR ESTRUCTURA DE CARPETAS PARA GUARDAR TODO
import os
from datetime import datetime

# Crear carpeta con timestamp para este experimento
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"experiment_{timestamp}"
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(f"{experiment_dir}/models", exist_ok=True)
os.makedirs(f"{experiment_dir}/results", exist_ok=True)
os.makedirs(f"{experiment_dir}/embeddings", exist_ok=True)
os.makedirs(f"{experiment_dir}/plots", exist_ok=True)
os.makedirs(f"{experiment_dir}/logs", exist_ok=True)

print(f"📁 Experimento guardado en: {experiment_dir}")

# Guardar configuración del experimento
config = {
    'model_name': 'xlm-roberta-base',
    'num_labels': 4,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'max_epochs': 6,
    'patience': 3,
    'max_length': 128,
    'train_samples': len(train_texts),
    'val_samples': len(val_texts),
    'label_mapping': label_mapping,
    'timestamp': timestamp
}

import json
with open(f"{experiment_dir}/config.json", 'w') as f:
    json.dump(config, f, indent=2)
print(f"⚙️ Configuración guardada: {experiment_dir}/config.json")

# 9. Entrenamiento - CON VALIDACIÓN Y EARLY STOPPING
training_log_output = []
epoch_metrics = []  # 📊 Guardar métricas por época

for epoch in range(epochs):
    # FASE DE ENTRENAMIENTO
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    
    print(f"\n🔄 Epoch {epoch + 1}/{epochs} - Entrenando...")
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # 🔧 FIX: 'labels' no 'label'
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
        
        # Calcular accuracy
        predictions = torch.argmax(outputs.logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        
        # Progress cada 50 batches
        if batch_idx % 50 == 0:
            current_acc = train_correct / train_total
            print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Acc: {current_acc:.4f}")
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    
    # FASE DE VALIDACIÓN
    model.eval()
    total_val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    
    # 📊 Guardar métricas de esta época
    epoch_metric = {
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy
    }
    epoch_metrics.append(epoch_metric)
    
    # Logging mejorado
    epoch_log = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_accuracy:.4f}"
    print(f"📊 {epoch_log}")
    training_log_output.append(epoch_log)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"✅ Nuevo mejor modelo! Val Loss: {best_val_loss:.4f}")
        
        # 💾 Guardar checkpoint del mejor modelo
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        torch.save(checkpoint, f"{experiment_dir}/models/best_checkpoint.pth")
        
    else:
        patience_counter += 1
        print(f"⏳ Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping activado después de {epoch + 1} épocas")
            break

# Cargar el mejor modelo
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("🔄 Cargado el mejor modelo del entrenamiento")

# 📊 Guardar métricas por época en CSV
import pandas as pd
metrics_df = pd.DataFrame(epoch_metrics)
metrics_df.to_csv(f"{experiment_dir}/results/epoch_metrics.csv", index=False)
print(f"📊 Métricas por época guardadas: {experiment_dir}/results/epoch_metrics.csv")

# 10. Evaluación Final
model.eval()
predictions, true_labels = [], []
all_probabilities = []

print("\n🔍 Evaluación final...")
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # 🔧 FIX: 'labels' no 'label'
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# Mapeo inverso para mostrar nombres de clases
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
target_names = [reverse_label_mapping[i] for i in range(len(label_mapping))]

# 📏 EXTRACCIÓN Y GUARDADO DE EMBEDDINGS BERT (como backup/análisis)
print("\n🧮 Extrayendo embeddings BERT para análisis...")
from transformers import AutoModel as BertModel

# Cargar modelo base para extracción
bert_base = BertModel.from_pretrained("xlm-roberta-base").to(device)
bert_base.eval()

def extract_bert_embeddings(texts, model, tokenizer, batch_size=32):
    embeddings = []
    dataloader = DataLoader(texts, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extrayendo BERT embeddings"):
            encoded = tokenizer(batch, padding=True, truncation=True, 
                              max_length=128, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            outputs = model(**encoded)
            # Mean pooling
            mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            pooled = summed / counts
            
            embeddings.append(pooled.cpu())
    
    return torch.cat(embeddings, dim=0)

# Extraer embeddings de train y validation
train_embeddings = extract_bert_embeddings(train_texts, bert_base, tokenizer)
val_embeddings = extract_bert_embeddings(val_texts, bert_base, tokenizer)

# Guardar embeddings
torch.save({
    'train_embeddings': train_embeddings,
    'train_labels': torch.tensor(train_labels),
    'val_embeddings': val_embeddings,
    'val_labels': torch.tensor(val_labels),
    'label_mapping': label_mapping
}, f"{experiment_dir}/embeddings/bert_embeddings.pth")

print(f"💾 Embeddings BERT guardados: {experiment_dir}/embeddings/bert_embeddings.pth")
print(f"   - Train embeddings: {train_embeddings.shape}")
print(f"   - Val embeddings: {val_embeddings.shape}")

# 11. Guardar Reporte de Clasificación y Logs
print("\n" + "=" * 50)
print("💾 Guardando resultados de entrenamiento y evaluación...")
print("=" * 50)

# Reporte de clasificación con nombres de clases
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

report_dict = classification_report(true_labels, predictions, target_names=target_names, digits=4, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Guardar reporte
try:
    import dataframe_image as dfi
    dfi.export(report_df, f"{experiment_dir}/results/classification_report.png", table_conversion='matplotlib')
    print(f"✅ Reporte de clasificación guardado: {experiment_dir}/results/classification_report.png")
except ImportError:
    print("📝 Reporte de clasificación:")
    print(classification_report(true_labels, predictions, target_names=target_names, digits=4))
    
report_df.to_csv(f"{experiment_dir}/results/classification_report.csv")
print(f"✅ Reporte CSV guardado: {experiment_dir}/results/classification_report.csv")

# Matriz de confusión
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión - Modelo Fine-tuned')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{experiment_dir}/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Matriz de confusión guardada: {experiment_dir}/plots/confusion_matrix.png")

# Gráfico de métricas por época
plt.figure(figsize=(15, 5))

# Loss por época
plt.subplot(1, 3, 1)
epochs_list = [m['epoch'] for m in epoch_metrics]
train_losses = [m['train_loss'] for m in epoch_metrics]
val_losses = [m['val_loss'] for m in epoch_metrics]
plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', marker='o')
plt.plot(epochs_list, val_losses, 'r-', label='Val Loss', marker='s')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Pérdida por Época')
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy por época
plt.subplot(1, 3, 2)
train_accs = [m['train_accuracy'] for m in epoch_metrics]
val_accs = [m['val_accuracy'] for m in epoch_metrics]
plt.plot(epochs_list, train_accs, 'b-', label='Train Acc', marker='o')
plt.plot(epochs_list, val_accs, 'r-', label='Val Acc', marker='s')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.title('Precision por Época')
plt.legend()
plt.grid(True, alpha=0.3)

# Distribución de predicciones
plt.subplot(1, 3, 3)
pred_labels = [reverse_label_mapping[p] for p in predictions]
pred_counts = pd.Series(pred_labels).value_counts()
plt.bar(range(len(pred_counts)), pred_counts.values)
plt.xticks(range(len(pred_counts)), pred_counts.index, rotation=45)
plt.ylabel('Cantidad')
plt.title('Distribución de Predicciones')

plt.tight_layout()
plt.savefig(f'{experiment_dir}/plots/training_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Gráficos de métricas guardados: {experiment_dir}/plots/training_metrics.png")

# Guardar logs de entrenamiento detallados
with open(f"{experiment_dir}/logs/training_log.txt", 'w', encoding='utf-8') as f:
    f.write("=== REGISTRO COMPLETO DE ENTRENAMIENTO ===\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Dispositivo usado: {device}\n\n")
    
    f.write(f"Configuración del modelo:\n")
    for key, value in config.items():
        f.write(f"  {key}: {value}\n")
    
    f.write(f"\n=== DATOS ===\n")
    f.write(f"Total muestras: {len(df)}\n")
    f.write(f"Muestras entrenamiento: {len(train_texts)}\n")
    f.write(f"Muestras validación: {len(val_texts)}\n")
    f.write(f"\nDistribución de clases:\n")
    class_dist = df['label'].value_counts()
    for label, count in class_dist.items():
        percentage = (count / len(df)) * 100
        f.write(f"  {label}: {count} ({percentage:.1f}%)\n")
    
    f.write(f"\n=== PROGRESO DEL ENTRENAMIENTO ===\n")
    f.write("\n".join(training_log_output))
    
    f.write(f"\n\n=== RESULTADOS FINALES ===\n")
    f.write(f"Mejor validation loss: {best_val_loss:.4f}\n")
    final_accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
    f.write(f"Accuracy final en validación: {final_accuracy:.4f}\n")
    f.write(f"\nReporte de clasificación por clase:\n")
    f.write(classification_report(true_labels, predictions, target_names=target_names, digits=4))
    
print(f"✅ Log detallado guardado: {experiment_dir}/logs/training_log.txt")

# 11.5 GUARDAR EL MODELO FINE-TUNED COMPLETO
# ==================================================
print(f"\n💾 Guardando modelo fine-tuned completo...")

# Guardar modelo y tokenizer en la estructura de carpetas
model_dir = f"{experiment_dir}/models/final_model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"✅ Modelo y tokenizer guardados: {model_dir}")

# Guardar mapeo de etiquetas y configuración
import pickle
with open(f"{model_dir}/label_mapping.pkl", 'wb') as f:
    pickle.dump(label_mapping, f)
with open(f"{model_dir}/config_training.json", 'w') as f:
    json.dump(config, f, indent=2)
print(f"✅ Mapeo de etiquetas y configuración guardados en {model_dir}")

# Crear script de carga rápida
quick_load_script = f"""# SCRIPT DE CARGA RÁPIDA DEL MODELO
# Generado automáticamente el {timestamp}

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json

# Cargar modelo
model_path = "{model_dir}"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Cargar mapeo de etiquetas
with open("{model_dir}/label_mapping.pkl", 'rb') as f:
    label_mapping = pickle.load(f)

# Cargar configuración
with open("{model_dir}/config_training.json", 'r') as f:
    training_config = json.load(f)

print("Modelo cargado exitosamente!")
print(f"Etiquetas: {{list(label_mapping.keys())}}")
print(f"Accuracy final: {{training_config.get('final_accuracy', 'N/A')}}")

# Función de predicción
def predict_text(text, model, tokenizer, label_mapping, device='cpu'):
    model.eval()
    encoded = tokenizer(text, padding=True, truncation=True, 
                       max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    reverse_mapping = {{v: k for k, v in label_mapping.items()}}
    predicted_label = reverse_mapping[prediction]
    confidence = probabilities[0][prediction].item()
    
    return {{
        'label': predicted_label,
        'confidence': confidence,
        'all_probabilities': {{reverse_mapping[i]: prob.item() 
                             for i, prob in enumerate(probabilities[0])}}
    }}

# Ejemplo de uso:
# resultado = predict_text("Este es un texto de ejemplo", model, tokenizer, label_mapping)
# print(resultado)
"""

with open(f"{experiment_dir}/quick_load_model.py", 'w') as f:
    f.write(quick_load_script)
print(f"✅ Script de carga rápida creado: {experiment_dir}/quick_load_model.py")

# 12. PREDICCIÓN EN NUEVOS DATOS
try:
    print("\n🔮 Cargando datos para predicción...")
    df_predict = pd.read_csv("BullyingPredict.csv")
    
    # Dataset para predicción
    predict_dataset = BullyingDataset(
        df_predict["text"].tolist(), 
        [0] * len(df_predict),  # Labels dummy
        tokenizer
    )
    predict_loader = DataLoader(predict_dataset, batch_size=32)
    
    # Hacer predicciones
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    print(f"🔄 Procesando {len(df_predict)} textos para predicción...")
    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="Haciendo predicciones"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convertir predicciones a etiquetas
    predicted_labels = [reverse_label_mapping[pred] for pred in all_predictions]
    
    # Crear DataFrame con resultados completos
    results_df = df_predict.copy()
    results_df['predicted_label'] = predicted_labels
    results_df['confidence'] = [max(probs) for probs in all_probabilities]
    results_df['prediction_id'] = range(len(results_df))
    
    # Agregar probabilidades por clase
    for i, label in enumerate(target_names):
        results_df[f'prob_{label}'] = [probs[i] for probs in all_probabilities]
    
    # Guardar resultados principales
    results_df.to_csv(f"{experiment_dir}/results/predictions_complete.csv", index=False)
    print(f"✅ Predicciones completas guardadas: {experiment_dir}/results/predictions_complete.csv")
    
    # Guardar solo las predicciones principales (más limpio)
    simple_results = results_df[['text', 'predicted_label', 'confidence']].copy()
    simple_results.to_csv(f"{experiment_dir}/results/predictions_simple.csv", index=False)
    print(f"✅ Predicciones simples guardadas: {experiment_dir}/results/predictions_simple.csv")
    
    # Análisis de resultados
    pred_analysis = {
        'total_predictions': len(predicted_labels),
        'label_distribution': dict(pd.Series(predicted_labels).value_counts()),
        'confidence_stats': {
            'mean': float(np.mean([max(probs) for probs in all_probabilities])),
            'std': float(np.std([max(probs) for probs in all_probabilities])),
            'min': float(np.min([max(probs) for probs in all_probabilities])),
            'max': float(np.max([max(probs) for probs in all_probabilities]))
        },
        'high_confidence_predictions': sum(1 for probs in all_probabilities if max(probs) > 0.9),
        'low_confidence_predictions': sum(1 for probs in all_probabilities if max(probs) < 0.5)
    }
    
    with open(f"{experiment_dir}/results/prediction_analysis.json", 'w') as f:
        json.dump(pred_analysis, f, indent=2)
    print(f"✅ Análisis de predicciones guardado: {experiment_dir}/results/prediction_analysis.json")
    
    # Mostrar muestra de resultados
    print("\n📝 Muestra de predicciones:")
    print("-" * 100)
    sample_size = min(10, len(results_df))
    for idx in range(sample_size):
        text = results_df.iloc[idx]['text'][:80]
        pred = results_df.iloc[idx]['predicted_label']
        conf = results_df.iloc[idx]['confidence']
        print(f"\n{idx+1:2d}. Texto: {text}{'...' if len(results_df.iloc[idx]['text']) > 80 else ''}")
        print(f"    Predicción: {pred} (Confianza: {conf:.3f})")
    
    # Estadísticas finales
    print(f"\n📊 ESTADÍSTICAS DE PREDICCIONES:")
    print(f"Total de textos procesados: {len(predicted_labels)}")
    print(f"\nDistribución de predicciones:")
    pred_counts = pd.Series(predicted_labels).value_counts()
    for label, count in pred_counts.items():
        percentage = (count / len(predicted_labels)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nEstadísticas de confianza:")
    confidences = [max(probs) for probs in all_probabilities]
    print(f"  Confianza promedio: {np.mean(confidences):.3f}")
    print(f"  Predicciones alta confianza (>0.9): {sum(1 for c in confidences if c > 0.9):,} ({sum(1 for c in confidences if c > 0.9)/len(confidences)*100:.1f}%)")
    print(f"  Predicciones baja confianza (<0.5): {sum(1 for c in confidences if c < 0.5):,} ({sum(1 for c in confidences if c < 0.5)/len(confidences)*100:.1f}%)")
    
    # Extraer embeddings de datos de predicción (para análisis posterior)
    print(f"\n🧮 Extrayendo embeddings de datos de predicción...")
    predict_embeddings = extract_bert_embeddings(df_predict["text"].tolist(), bert_base, tokenizer)
    
    # Guardar embeddings de predicción
    torch.save({
        'predict_embeddings': predict_embeddings,
        'predictions': torch.tensor(all_predictions),
        'probabilities': torch.tensor(all_probabilities),
        'texts': df_predict["text"].tolist(),
        'label_mapping': label_mapping
    }, f"{experiment_dir}/embeddings/predict_embeddings.pth")
    
    print(f"💾 Embeddings de predicción guardados: {experiment_dir}/embeddings/predict_embeddings.pth")
    print(f"   - Predict embeddings: {predict_embeddings.shape}")
        
except FileNotFoundError:
    print("⚠️ Archivo BullyingPredict.csv no encontrado - saltando predicciones")
except Exception as e:
    print(f"❌ Error en predicciones: {e}")
    import traceback
    traceback.print_exc()

# RESUMEN FINAL
print("\n" + "=" * 80)
print("🎉 ENTRENAMIENTO Y PREDICCIONES COMPLETADOS")
print("=" * 80)
print(f"📁 Todos los resultados guardados en: {experiment_dir}/")
print(f"\n💾 Archivos generados:")
print(f"  ✅ Modelo final: {experiment_dir}/models/final_model/")
print(f"  ✅ Mejor checkpoint: {experiment_dir}/models/best_checkpoint.pth")
print(f"  ✅ Embeddings BERT: {experiment_dir}/embeddings/")
print(f"  ✅ Métricas y reportes: {experiment_dir}/results/")
print(f"  ✅ Gráficos: {experiment_dir}/plots/")
print(f"  ✅ Logs detallados: {experiment_dir}/logs/")
print(f"  ✅ Script de carga: {experiment_dir}/quick_load_model.py")
print(f"\n🎯 Para usar el modelo, ejecuta: python {experiment_dir}/quick_load_model.py")
print("=" * 80)