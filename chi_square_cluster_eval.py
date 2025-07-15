# ChiSquareClusterEval Module

This module provides statistical evaluation between clusters using the Chi-Square test.

```python
import pandas as pd
import numpy as np
import scipy.stats as stats

class ChiSquareClusterEval:
    def __init__(self, cluster_files, features, total_size=None):
        self.cluster_files = cluster_files
        self.features = features
        self.total_size = total_size

    def load_data(self):
        self.data = [pd.read_csv(f, index_col=0)[self.features] for f in self.cluster_files]
        if self.total_size is None:
            self.total_size = sum([len(df) for df in self.data])

    def compute_statistics(self):
        self.report = pd.DataFrame({})
        for i, feature in enumerate(self.features):
            # Build frequency tables per cluster
            freq_tables = [pd.crosstab(index=df.iloc[:, i], columns="") for df in self.data]
            tb = pd.concat(freq_tables, axis=1, ignore_index=True)
            tb.columns = [f"c{j}" for j in range(len(self.data))]
            observed = tb.iloc[:, :len(self.data)]

            # Compute expected values
            col_totals = observed.sum(axis=0)
            row_totals = observed.sum(axis=1)
            expected = np.outer(row_totals, col_totals) / self.total_size
            expected = pd.DataFrame(expected, index=observed.index, columns=observed.columns)

            # Chi-Square test
            chi_stat = (((observed - expected)**2) / expected).sum().sum()
            dof = (len(row_totals)-1)*(len(col_totals)-1)
            crit_val = stats.chi2.ppf(q=0.95, df=dof)
            p_value = 1 - stats.chi2.cdf(x=chi_stat, df=dof)

            # Summary table
            row = {
                'Feature': feature,
                'Chi2': round(chi_stat, 4),
                'DF': dof,
                'CriticalVal': round(crit_val, 4),
                'PValue': round(p_value, 4)
            }
            self.report = pd.concat([self.report, pd.DataFrame([row])], ignore_index=True)

        return self.report

    def save_report(self, filename="ChiSquare_Report.xlsx"):
        self.report.to_excel(filename, index=False)
```

### Example Usage
```python
features = ['hyper', 'diabet', 'fatty']
files = ["Cluster_1.csv", "Cluster_2.csv", "Cluster_3.csv"]
eval_module = ChiSquareClusterEval(files, features)
eval_module.load_data()
report_df = eval_module.compute_statistics()
eval_module.save_report("Pvalue_detail_Km_chronic_dataset.xlsx")
print(report_df)
```

---

## Complementary Statistical Evaluation

The `ChiSquareClusterEval` module can **absolutely be considered complementary** to `CSAIEvaluator`. While CSAI measures the stability of clustering across partitions based on embedding structure, `ChiSquareClusterEval` provides **statistical significance testing** on categorical variables across clusters using the **Chi-Square test**.

This helps to:
- Validate if cluster assignments are associated with meaningful feature differences.
- Quantify confidence in cluster-specific feature distributions.

📂 Source code: [ChiSquareClusterEval Module](https://github.com/AdaneNT/cluster-validity-csai/tree/main/chisquare_eval)

📘 Example output:
```
Feature    Chi2    DF    CriticalVal    PValue
hyper      12.34   2     5.99           0.002
diabet      8.56   2     5.99           0.013
fatty      15.67   2     5.99           0.000
```

This module is particularly useful for **healthcare datasets** or any scenario where **categorical health indicators or labels** are involved.
