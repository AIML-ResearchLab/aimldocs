<h2 style="color:red;">✅ Mann–Whitney U Test</h2>

**📊 Real-Time Example 1: Mann–Whitney U Test**

Compare test scores between **2 teaching methods** (non-normally distributed).

```
import numpy as np
from scipy.stats import mannwhitneyu

group1 = [88, 90, 85, 95, 92]
group2 = [75, 78, 72, 70, 80]

stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
print(f"U-Statistic: {stat}, p-value: {p:.3f}")
```

U-Statistic: 25.0, p-value: 0.008

