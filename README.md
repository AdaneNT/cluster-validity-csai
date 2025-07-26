## CSAIEvaluator: Clustering Stability Assessment Index (CSAI)

**CSAIEvaluator** is a Python package for evaluating the quality of clustering algorithms using the CSAI Index — a novel method for assessing both the validity and stability of clustering solutions. Unlike traditional methods that rely on cluster centroids, CSAI measures the distributional alignment of aggregated feature structures across data partitions. It is a simple, effective, and model-agnostic approach to quantify the performance and reproducibility of unsupervised models.

---

## Features
- Model-agnostic approach: Supports multiple clustering algorithms (e.g., KMeans, Agglomerative, Density based clusters,etc)
- Compares cluster distribution consistency accross multiple partitions of data
- Uses UMAP for embedding visualization and dimensionality reduction    
- Easy integration with scikit-learn and SentenceTransformers  
---
## Requirements

```bash
pip install -r requirements.txt
```

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/AdaneNT/cluster-validity-csai.git
```

#### From  PyPI
```bash
pip install cluster-validity-csai
```
#### From local source  
Clone the repo and run:

```bash
git clone https://github.com/AdaneNT/cluster-validity-csai.git
cd cluster-validity-csai
pip install .
```

---

## Example Usage

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

# Step 1: Load and clean data 
newsgroups = fetch_20newsgroups(subset='all')
df = pd.DataFrame(newsgroups.data, columns=["text"])
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Step 2: Text preprocessing 
texts = df["text"].fillna("").apply(lambda x: re.sub(r"\d+|[^\w\s]|\s+", " ", x.lower()).strip()).tolist()
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sbert_model.encode(texts, convert_to_tensor=False)
df["SBERT_Embedding"] = embeddings.tolist()

# Step 3:  Dimensionality reduction 
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
```

---
## Complementary Statistical Analysis

The [Chi-Square module](https://github.com/AdaneNT/cluster-validity-csai/tree/main/chi_square_test.py) can be considered complementary to `CSAIEvaluator`. The CSAI measures the stability of clustering across partitions based on embedding structure, while `Chi-Square module` provides **statistical significance testing** on categorical variables across clusters using the **Chi-Square test**.

This helps to:
- Validate if cluster assignments are associated with meaningful feature differences.
- Quantify confidence in cluster-specific feature distributions.

📘 Example output:
```
| Feature   | Chi2  | DF | CriticalVal | PValue |
|-----------|-------|----|-------------|--------|
| feature1  | 12.34 | 2  | 5.99        | 0.002  |
| feature2  | 8.56  | 2  | 5.99        | 0.013  |
| feature3  | 15.67 | 2  | 5.99        | 0.000  |
```
## Citation & Origin

The CSAI method was first introduced in the following publication:
> Tarekegn, A. N., Tessem, B., Rabbi, F. (2025).  
> **A New Cluster Validation Index Based on Stability Analysis.**  
> In *Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods - ICPRAM*,  
> ISBN 978-989-758-730-6; ISSN 2184-4313, SciTePress, pages 377–384.  
> DOI: [10.5220/0013309100003905](https://doi.org/10.5220/0013309100003905)

If you find this code useful in your work, please cite this publication.

## License

This software is licensed for academic, non-commercial use only. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

This research work is funded by **SFI MediaFutures Partners** and the **Research Council of Norway**  
(Grant number: 309339).
