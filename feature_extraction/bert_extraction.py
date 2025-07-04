import os
if not os.path.exists('features/bert'):
    os.makedirs('features/bert')

bert_folder = "features/bert"
#%%
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np

# Load and prepare data
df_train = pd.read_csv('datasets/train.csv')
label_mapping = {
    'not_cyberbullying': 0,
    'gender/sexual': 1,
    'ethnicity/race': 2,
    'religion': 3
}

# Apply the mapping to your dataframe
df_train['label'] = df_train['label'].map(label_mapping)
unmapped = df_train['label'].isna().sum()
if unmapped > 0:
    print(f"Warning: {unmapped} labels couldn't be mapped!")
    print("Unique labels in data:", df_train['label'].unique())
train_texts = df_train['text']
train_labels = df_train['label']
#%%
# Initialize model and tokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
model = AutoModel.from_pretrained(model_ckpt).to(device)

def extract_features(texts, max_length=200):
    """Extract features from texts using BERT"""
    all_hidden_states = []

    # Process in batches to avoid memory issues
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,  # You can tune this parameter
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token representation (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_hidden_states.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_hidden_states)

#%%

# Extract features for all splits
print("Extracting features...")
X_train = extract_features(train_texts)

# Convert labels to numpy arrays
y_train = np.array(train_labels)
# Save extracted features
print("Saving extracted features...")

dump(X_train, os.path.join(bert_folder,'X_train_features.joblib'))

dump(y_train,  os.path.join(bert_folder,'y_train_features.joblib'))

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")