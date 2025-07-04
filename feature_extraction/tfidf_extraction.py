#%%
import pandas as pd
import os
#%%
train_df = pd.read_csv('datasets/train.csv')
#%% md
# # Feature extraction
#%%
if not os.path.exists('features'):
    os.makedirs('features')
#%% md
# # TF-IDF
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

if not os.path.exists('features/tfidf'):
    os.makedirs('features/tfidf')

tfidf_folder = "features/tfidf"
#%%
tfidf= TfidfVectorizer(sublinear_tf=True, min_df=5,
                       ngram_range=(1, 2), stop_words='english', max_features=5000)
features_train = tfidf.fit_transform(train_df.text).toarray()
labels_train = train_df.label
#%%
import os
from joblib import dump

# Make sure the folder exists
os.makedirs(tfidf_folder, exist_ok=True)

# Save the TF-IDF vectorizer and features
dump(tfidf, os.path.join(tfidf_folder, "tfidf_vectorizer.joblib"))
dump(features_train, os.path.join(tfidf_folder, "features_train.joblib"))
dump(labels_train, os.path.join(tfidf_folder, "labels_train.joblib"))
