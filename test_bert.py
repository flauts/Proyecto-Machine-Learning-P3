# predict.py
# This script loads the pre-trained model and makes predictions on new data.

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import dataframe_image as dfi
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
MODEL_DIR = "./saved_cyberbullying_model"  # Path to your saved model
UNLABELED_CSV_PATH = "BullyingPredict.csv"  # The new data you want to classify
OUTPUT_IMAGE_PATH = "new_predictions.png"  # Name for the output image
BATCH_SIZE = 32  # You can adjust this for performance

# Check for CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


label_encoder = LabelEncoder()
# 6. Dataset personalizado
class BullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = label_encoder.fit_transform(labels)
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


# --- SCRIPT EXECUTION ---
try:
    # 1. Load Model and Tokenizer from the saved directory
    print(f"Loading model from {MODEL_DIR}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # 2. Load and prepare new data
    print(f"Loading data from {UNLABELED_CSV_PATH}...")
    unlabeled_df = pd.read_csv(UNLABELED_CSV_PATH)
    unlabeled_texts = unlabeled_df["text"].tolist()

    unlabeled_labels = [0] * len(unlabeled_texts)  # Dummy labels
    unlabeled_dataset = BullyingDataset(unlabeled_texts, unlabeled_labels, tokenizer)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE)

    # 3. Perform Prediction
    print("Running predictions...")
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for batch in unlabeled_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            top_probability, top_class = torch.max(probabilities, dim=1)

            all_predictions.extend(top_class.cpu().numpy())
            all_probabilities.extend(top_probability.cpu().numpy())

    # 4. Format and Save Results
    unlabeled_df['predicted_class'] = all_predictions
    unlabeled_df['probability'] = [f"{p:.2%}" for p in all_probabilities]

    df_styled = unlabeled_df[['text', 'predicted_class', 'probability']].style.set_properties(**{'text-align': 'left'})
    df_styled.set_table_styles([dict(selector="th", props=[("text-align", "left")])])

    dfi.export(df_styled, OUTPUT_IMAGE_PATH, table_conversion='matplotlib')
    print(f"✅ Predictions complete. Output saved to {OUTPUT_IMAGE_PATH}")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find '{UNLABELED_CSV_PATH}' or model at '{MODEL_DIR}'. Please check paths.")
except Exception as e:
    print(f"❌ An error occurred: {e}")