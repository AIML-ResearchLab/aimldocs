<h2 style="color:red;">✅ Chi-Square Test</h2>

**📊 Real-Time Example 3: Chi-Square Test for Independence**

```
import pandas as pd
from scipy.stats import chi2_contingency

# Survey results: Gender vs Preferred Learning Style
data = pd.DataFrame({
    "Visual": [25, 30],
    "Auditory": [20, 15],
    "Kinesthetic": [15, 20]
}, index=["Male", "Female"])

chi2, p, dof, expected = chi2_contingency(data)
print(f"Chi-Square: {chi2:.3f}, p-value: {p:.3f}")
```


Chi-Square: 1.686, p-value: 0.430

**📌 Summary Table**

| Test           | Best For              | Python Function     |
| -------------- | --------------------- | ------------------- |
| Mann–Whitney U | 2 independent groups  | `mannwhitneyu`      |
| Wilcoxon       | 2 related groups      | `wilcoxon`          |
| Kruskal–Wallis | 3+ independent groups | `kruskal`           |
| Friedman       | 3+ related groups     | `friedmanchisquare` |
| Chi-Square     | Categorical variables | `chi2_contingency`  |


