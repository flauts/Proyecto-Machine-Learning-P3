#%%
import pandas as pd
import os
#%%
train_df = pd.read_csv('BullyingMultiClase.csv')
predict_df = pd.read_csv('BullyingPredict.csv')
#%% md
# # Feature extraction
#%%
if not os.path.exists('features'):
    os.makedirs('features')
# # BERT_EMBEDDING
#%%
if not os.path.exists('features/tfidf'):
    os.makedirs('features/tfidf')

bert_folder = "features/bert"
#%%
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm  # optional progress bar

# 1. Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load tokenizer and base model (no classification head)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base").to(device)
model.eval()  # Turn off dropout, etc.


# 3. Define a mean pooling function
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return summed / counts  # Shape: [batch_size, 768]


# 4. Function to extract features from a list of texts
from tqdm import tqdm


def extract_features(texts, batch_size=64):
    all_embeddings = []
    dataloader = DataLoader(texts, batch_size=batch_size)
    for batch in tqdm(dataloader, desc="Extracting features"):
        # Tokenize a batch of texts
        encoded = tokenizer(batch, padding=True, truncation=True,
                            return_tensors="pt", max_length=128)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            embeddings = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# 5. Extract features for train data
x_train = extract_features(train_df["text"].tolist(), batch_size=256)
y_train = train_df["label"]

# 6. Extract features for predict data
x_predict = extract_features(predict_df["text"].tolist(), batch_size=256)

# 7. Save the features and labels
torch.save(x_train, os.path.join(bert_folder, "x_train.pt"))
torch.save(y_train, os.path.join(bert_folder, "y_train.pt"))
torch.save(x_predict, os.path.join(bert_folder, "x_predict.pt"))

#%%
