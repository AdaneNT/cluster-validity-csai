# CSAIEvaluator: Cluster Structure Alignment Index

CSAIEvaluator is a Python library for evaluating the robustness of clustering across train/test splits using UMAP projections and distribution alignment.

## Installation

```bash
pip install git+https://github.com/yourusername/cluster-validity-csai.git
```

## Example

```python
from csai import CSAIEvaluator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

csai = CSAIEvaluator()
X_train, X_test = train_test_split(df, test_size=0.3)

def kmeans_label_func(embeddings):
    model = KMeans(n_clusters=7)
    labels = model.fit_predict(embeddings)
    return labels, model

csai_score = csai.run_csai_evaluation(
    X_train, X_test,
    key_col="key_umap",
    label_func=kmeans_label_func,
    n_splits=5
)
```

## License

MIT
