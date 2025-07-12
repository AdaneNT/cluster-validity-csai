from csai import CSAIEvaluator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings

# Suppress UMAP warning
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state*", category=UserWarning)

# Simulate minimal data
np.random.seed(42)
dummy_embeddings = np.random.rand(100, 10)
df = pd.DataFrame({
    "key_umap": dummy_embeddings.tolist()
})

X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)

def kmeans_label_func(embeddings, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels, model

csai = CSAIEvaluator()
score = csai.run_csai_evaluation(X_train, X_test, key_col="key_umap", label_func=kmeans_label_func, n_splits=5)

print("CSAIEvaluator Score:", score)
