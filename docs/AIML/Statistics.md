For any Machine Learning project, understanding the **underlying statistics** and using the **right plots** is crucial during **EDA (Exploratory Data Analysis), feature selection, model evaluation**, and **interpretability**.


# 🔍 1. Descriptive Statistics
Used to summarize and understand the data.

| Task              | Statistics                              |
|-------------------|------------------------------------------|
| Central Tendency  | Mean, Median, Mode                      |
| Spread            | Standard Deviation, Variance, IQR       |
| Shape             | Skewness, Kurtosis                      |
| Outliers          | Z-score, IQR method                     |
| Data Distribution | Count, Frequency tables                 |


# 📊 2. EDA & Visualization Plots

| Purpose            | Plot Type                          | Use Case                             |
|--------------------|------------------------------------|---------------------------------------|
| Univariate analysis| Histogram, KDE Plot, Boxplot       | Distribution of a single feature      |
| Bivariate analysis | Scatter plot, Line plot, Heatmap   | Feature relationships                 |
| Outlier detection  | Boxplot, Violin plot               | Detect extreme values                 |
| Skewness check     | Histogram, QQ plot                 | Check Normality assumption            |
| Class imbalance    | Bar plot, Pie chart                | Classification target imbalance       |
| Correlation        | Heatmap, Pairplot                  | Check multicollinearity               |
| Missing values     | Heatmap, Bar plot                  | Identify null patterns                |
| Time-series        | Line plot, Rolling Mean            | Temporal patterns                     |



# 📈 3. Distribution-Specific Plots (for Statistical Analysis)

| Distribution | When Used                          | Plot                          |
|--------------|------------------------------------|-------------------------------|
| Normal       | Continuous features, model assumption | Histogram + KDE, QQ Plot     |
| Binomial     | Binary outcomes                    | Bar plot                      |
| Poisson      | Event count in time/window         | PMF, Histogram                |
| Exponential  | Time until event                   | Histogram, PDF                |
| Uniform      | Simulations                        | Flat histogram                |
| Logistic     | Classification (0-1 outcome)       | Sigmoid curve                 |
| Multinomial  | Categorical features               | Bar plot (for categories)     |


# 📊 4. Model Evaluation Plots

| Task              | Plot                                      | Use Case                             |
|-------------------|-------------------------------------------|---------------------------------------|
| Classification    | Confusion Matrix, ROC Curve, PR Curve     | Evaluate classification performance   |
| Regression        | Residual plot, QQ plot, Predicted vs Actual | Evaluate regression fit               |
| Model Selection   | Learning Curve, Validation Curve          | Diagnose overfitting/underfitting     |
| Feature Importance| Bar plot, SHAP, Permutation               | Interpret model                       |
| Clustering        | Elbow Plot, Silhouette Plot               | Choose number of clusters             |


# 🧠 5. Advanced & ML-specific Visualization

| Technique                | Visual                          | Usage                             |
|--------------------------|----------------------------------|------------------------------------|
| PCA / t-SNE / UMAP       | 2D/3D scatter plots              | Visualize high-dimensional data    |
| SHAP / LIME              | Force plots, Beeswarm plots     | Explain predictions                |
| Decision Trees           | Tree plot                       | Interpret model rules              |
| Time-Series Forecasting  | Trend/Seasonality Decomposition | Understand components              |


# 🛠️ Bonus Tools
- **Seaborn:** For statistical visualizations

- **Matplotlib:** Core plotting

- **Plotly:** Interactive plots

- **Pandas Profiling / Sweetviz / D-Tale:** Automated EDA tools


