import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import umap
import matplotlib.pyplot as plt

class CSAIEvaluator:
    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self._init_umap_model()

    def _init_umap_model(self):
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            transform_seed=self.random_state,
            force_approximation_algorithm=False,
            init="spectral",
            metric="euclidean"
        )

    def run_umap(self, X):
        self._init_umap_model()
        return self.umap_model.fit_transform(X)

    def preprocess_partition(self, X_partition, key_col):
        assert key_col in X_partition.columns, f"Missing column: {key_col}"
        embedding_values_array = np.array(X_partition[key_col].tolist())
        return self.run_umap(embedding_values_array)

    def compute_distribution(self, df, cluster_col, value_col):
        grouped = df.groupby(cluster_col)[value_col].sum()
        total = grouped.sum()
        return (grouped / total).sort_index()

    def run_csai_evaluation(self, X_train, X_test, key_col, label_func, n_splits=4):
        assert callable(label_func), "You must provide a callable label_func."
        overall_CSAI_all_samples = []
        cv = KFold(n_splits=n_splits, random_state=self.random_state, shuffle=True)

        for counter, (train_idx, _) in enumerate(cv.split(X_train)):
            X_train_prt = X_train.iloc[train_idx].copy()
            X_test_copy = X_test.copy()

            umap_train = self.preprocess_partition(X_train_prt, key_col)
            labels_train, cluster_model = label_func(umap_train)
            X_train_prt["Cluster"] = labels_train.astype(str)

            umap_test = self.preprocess_partition(X_test_copy, key_col)
            if hasattr(cluster_model, "predict"):
                labels_test = cluster_model.predict(umap_test)
            else:
                labels_test = label_func(umap_test)[0]
            X_test_copy["Cluster"] = labels_test.astype(str)

            X_train_prt['sum_value_train'] = X_train_prt[key_col].apply(np.sum)
            X_test_copy['sum_value_test'] = X_test_copy[key_col].apply(np.sum)

            train_dist = self.compute_distribution(X_train_prt, "Cluster", "sum_value_train")
            test_dist = self.compute_distribution(X_test_copy, "Cluster", "sum_value_test")

            range_y = max(train_dist.max(), test_dist.max()) - min(train_dist.min(), test_dist.min())

            cluster_rmse = []
            for cluster in sorted(train_dist.index):
                rmse = np.sqrt((train_dist[cluster] - test_dist.get(cluster, 0)) ** 2) / (range_y + 1e-8)
                cluster_rmse.append(rmse)

            overall_rmse = np.mean(cluster_rmse)
            overall_CSAI_all_samples.append(overall_rmse)

            print(f"Sample {counter + 1} | CSAI per cluster: {[round(float(r), 4) for r in cluster_rmse]} |  CSAI across all clusters: {round(float(overall_rmse), 4)}")

        overall_CSAI = np.mean(overall_CSAI_all_samples)
        print(f"\nOverall CSAI across all samples: {round(float(overall_CSAI), 4)}")
        return overall_CSAI

    def visualize_clusters(self, embeddings, labels, title="Cluster Visualization"):
        plt.figure(figsize=(8, 6))
        labels = np.array(labels)
        scatter = plt.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=labels.astype(int), cmap='tab10', alpha=0.7
        )
        plt.title(title)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
