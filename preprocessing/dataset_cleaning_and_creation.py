#%%
import pandas as pd
#%%
df1 = pd.read_csv("../datasets/original/cb_multi_labeled_balanced.csv")
df2 = pd.read_csv("../datasets/original/cyberbullying_tweets.csv")
print(df1.shape)
print(df2.shape)
print(df1.label.unique())
#%%
print(df1.label.unique())
print(df2.cyberbullying_type.unique())
df3 = pd.DataFrame(columns=['text'])
for i in df2.index:
    if df2.loc[i].cyberbullying_type == "gender":
        new_row = pd.DataFrame({'text': [df2.loc[i].tweet_text], 'label': ['gender/sexual']})
        df1 = pd.concat([df1, new_row], ignore_index=True)
    elif df2.loc[i].cyberbullying_type == "ethnicity":
        new_row = pd.DataFrame({'text': [df2.loc[i].tweet_text], 'label': ['ethnicity/race']})
        df1 = pd.concat([df1, new_row], ignore_index=True)
    elif df2.loc[i].cyberbullying_type == "religion":
        new_row = pd.DataFrame({'text': [df2.loc[i].tweet_text], 'label': ['religion']})
        df1 = pd.concat([df1, new_row], ignore_index=True)
    elif df2.loc[i].cyberbullying_type == "not_cyberbullying":
        new_row = pd.DataFrame({'text': [df2.loc[i].tweet_text], 'label': ['not_cyberbullying']})
        df1 = pd.concat([df1, new_row], ignore_index=True)
    else:
        new_row = pd.DataFrame({'text': [df2.loc[i].tweet_text]})
        df3 = pd.concat([df3, new_row], ignore_index=True)
#%%
print(df1.shape)
print(df1['label'].value_counts())
print(df3.shape)
#%%
import re
import re
import pandas as pd

# Regex pattern to match emojis and symbols
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002500-\U00002BEF"  # Chinese characters and lines
    "\U00002702-\U000027B0"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # more supplemental
    "\u200d"                 # zero-width joiner
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"                 # dingbats
    "\u3030"
    "]+",
    flags=re.UNICODE
)

def clean_text(text):
    if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)       # Remove @mentions
    text = re.sub(r"http\S+", "", text)              # Remove URLs
    text = EMOJI_PATTERN.sub(r"", text)              # Remove emojis
    # Convert hashtags to regular words (remove # but keep content)
    text = re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)  # #MachineLearning -> MachineLearning

    text = re.sub(r"[^a-z0-9]", " ", text)           # Keep alphanumerics
    text = re.sub(r"\s+", " ", text).strip()         # Normalize whitespace
    return text if text else "empty"

df1["text"] = df1["text"].apply(clean_text)
df3["text"] = df3["text"].apply(clean_text)

#%%
df1.head()
#%%
df1.to_csv('BullyingMultiClase.csv', index=False)
df3.to_csv('BullyingPredict.csv', index=False)
#%%
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full dataset
df = pd.read_csv('../datasets/original/BullyingMultiClase.csv')

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']  # Maintains class distribution
)

# Save to separate files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f"Original dataset: {len(df)} samples")
print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Check class distribution
print("\nClass distribution:")
print("Original:")
print(df['label'].value_counts(normalize=True))
print("\nTrain:")
print(train_df['label'].value_counts(normalize=True))
print("\nTest:")
print(test_df['label'].value_counts(normalize=True))