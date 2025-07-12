# CSAIEvaluator: Cluster Structure Alignment Index

 **CSAIEvaluator** is a Python package for evaluating the stability of clustering algorithms using cross-validation and distribution alignment. It is especially useful in text and high-dimensional embeddings, using UMAP projections to compare cluster distributions across data splits.

---

## Features

-  Uses UMAP for embedding visualization and dimensionality reduction  
- Supports multiple clustering algorithms (e.g., KMeans, Agglomerative)  
- Compares cluster distribution consistency between train/test splits  
- Easy integration with scikit-learn and SentenceTransformers  
- Designed for unsupervised and semi-supervised clustering analysis  

---

## Installation

```bash
pip install git+https://github.com/yourusername/cluster-validity-csai.git

#### From local source

Clone the repo and run:

```bash
pip install .

---

## Example Usage

This example shows how to use the CSAIEvaluator with SBERT embeddings and UMAP-reduced vectors on the 20 Newsgroups dataset.

```python
from csai import CSAIEvaluator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import umap
import re
import warnings

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state*", category=UserWarning)

# Step 1: Load and clean 20 Newsgroups text data
newsgroups = fetch_20newsgroups(subset='all')
df = pd.DataFrame(newsgroups.data, columns=["text"])
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Step 2: Text preprocessing and SBERT embedding
texts = df["text"].fillna("").apply(lambda x: re.sub(r"\d+|[^\w\s]|\s+", " ", x.lower()).strip()).tolist()
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sbert_model.encode(texts, convert_to_tensor=False)
df["SBERT_Embedding"] = embeddings.tolist()

# Step 3: UMAP dimensionality reduction
reducer = umap.UMAP(n_components=10, random_state=42)
df["key_umap"] = reducer.fit_transform(np.array(df["SBERT_Embedding"].tolist())).tolist()

# Step 4: Train-test split
X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)

# Step 5: Define a clustering function
def kmeans_label_func(embeddings, n_clusters=6):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels, model

# Step 6: Run CSAI evaluation
csai = CSAIEvaluator()
score = csai.run_csai_evaluation(
    X_train,
    X_test,
    key_col="key_umap",
    label_func=kmeans_label_func,
    n_splits=5
)

print("CSAIEvaluator Score:", score)
