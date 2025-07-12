# demo_pipeline.py - Example usage of CSAIEvaluator with 20 Newsgroups dataset

import torch
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import umap
import warnings

from csai import CSAIEvaluator

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state*", category=UserWarning)

# Step 1: SBERT embedding
def get_sbert_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_tensor=False)

# Step 2: UMAP dimensionality reduction
def reduce_with_umap(df, emb_col="Embedding", output_col="key_umap", n_components=10):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    emb_array = np.array(df[emb_col].tolist())
    reduced = reducer.fit_transform(emb_array)
    df[output_col] = reduced.tolist()
    return df

# Step 3: Load and preprocess data
def run_pipeline():
    newsgroups = fetch_20newsgroups(subset='all')
    data = newsgroups.data
    df = pd.DataFrame(data, columns=["text"])
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)

    texts = df["text"].fillna("").apply(lambda x: re.sub(r"\d+|[^\w\s]|\s+@", " ", x.lower()).strip()).tolist()
    embeddings = get_sbert_embeddings(texts)
    df["SBERT_Embedding"] = embeddings.tolist()

    df = reduce_with_umap(df, emb_col="SBERT_Embedding", output_col="key_umap", n_components=10)
    return df

df_result = run_pipeline()

# Step 4: Train/test split
X_train, X_test = train_test_split(df_result, test_size=0.30, random_state=42)

# Step 5: Define clustering function
def kmeans_label_func(embeddings, n_clusters=7):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels, model

# Step 6: CSAI Evaluation
csai = CSAIEvaluator()
score = csai.run_csai_evaluation(X_train, X_test, key_col="key_umap", label_func=kmeans_label_func, n_splits=5)

print("CSAIEvaluator Score:", score)
