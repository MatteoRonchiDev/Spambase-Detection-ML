# Spambase Detection ML
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tentorifrancescaDev/spambase-detection-ml/blob/main/Progetto_SpamBase.ipynb)
 
**Spambase Detection ML** is a supervised Machine Learning project that automatically classifies e-mails as *spam* or *not spam*, based on the frequency of specific keywords and special characters in the text. The project compares a **Decision Tree** and a **Support Vector Machine (SVM)** and was developed as a university assignment for the Machine Learning course.
 
---
 
## Technologies
 
* **Python & Google Colab:** development and execution of the whole project in a Jupyter notebook.
* **ucimlrepo:** dataset loaded directly from the UCI Machine Learning Repository, with no local files.
* **pandas & NumPy:** data manipulation and construction of the DataFrame used for the analysis.
* **Matplotlib & Seaborn:** visualizations for the exploratory analysis (box plots, pair plots, correlation heatmaps) and for the results (confusion matrices, ROC curves).
* **SciPy:** computation of the Spearman correlation coefficient and its p-value.
* **scikit-learn:** preprocessing (`StandardScaler`), stratified data splitting, hyperparameter tuning (`GridSearchCV`), models (`DecisionTreeClassifier`, `SVC`) and evaluation metrics.
---
 
## Project Pipeline
 
* **Exploratory Data Analysis (EDA):** study of the target class distribution (61% not spam, 39% spam) and of the discriminative power of each feature through box plots and the normalized difference between the medians of the two classes.
* **Feature Selection with Spearman:** use of the Spearman correlation, better suited than Pearson to data with many zeros, outliers and non-linear relationships, to detect multicollinearity; p-value analysis (α = 0.05) to discard non-significant features. **7 redundant or uninformative features** were removed.
* **Stratified Hold-Out:** dataset split into 80% training and 20% test, keeping the same class proportions in both sets.
* **Standardization without Data Leakage:** `StandardScaler` fitted on the training set only and then applied to the test set.
* **Hyperparameter Tuning:** `GridSearchCV` with 5-fold Cross-Validation on the training set only, and `class_weight='balanced'` to compensate for the class imbalance.
* **Comparative Evaluation:** classification reports, confusion matrices, ROC curve with AUC, and training times.
---
 
## Models
 
* **Baseline:** "dummy" model that always predicts the majority class (not spam) and sets the minimum accuracy threshold to beat.
* **Decision Tree:** Gini impurity criterion, `max_depth` tuned in the range [1, 20] (best value: **18**). Trained on the full dataset without standardization, thanks to its ability to implicitly select the most informative features.
* **Linear-kernel SVM:** regularization constant `C` searched in {0.01, 0.1, 1, 10, 100} (best value: **C = 10**). Trained on the dataset reduced to 50 features and standardized, with `probability=True` to compute the ROC curve.
---
 
## Results
 
Metrics on the test set (921 e-mails, weighted average across the two classes):
 
| Model          | Accuracy | Precision | Recall | F1-Score | AUC  | Training time |
| -------------- | :------: | :-------: | :----: | :------: | :--: | :-----------: |
| Baseline       | 0.61     | 0.37      | 0.61   | 0.46     | –    | –             |
| Decision Tree  | 0.91     | 0.91      | 0.91   | 0.91     | 0.90 | ~0.1 s        |
| **SVM**        | **0.93** | **0.93**  | **0.93** | **0.93** | **0.97** | ~13 s     |
 
* The **SVM** is the most reliable model: it correctly detects 328 out of 363 spam e-mails (Recall 0.90 on the spam class) and reduces false negatives from 45 to 35 compared to the Decision Tree.
* The **Decision Tree** achieves slightly lower results without standardization or feature reduction, and is **over 100 times faster** to train: a concrete advantage on larger datasets or when frequent retraining is needed.
---
 
## Development Team
 
University project developed by:
* [Francesca Tentori](https://github.com/tentorifrancescaDev)
* [Matteo Ronchi](https://github.com/MatteoRonchiDev)
* Gabriel Pesce

---
 
## Project Structure
 
```
├── Progetto_SpamBase.ipynb             # Colab notebook: EDA, preprocessing, model training and evaluation
├── Progetto_Machine_Learning.pdf       # Full project report (in Italian)
├── Presentazione Machine Learning.pdf  # Project presentation slides (in Italian)
└── README.md
```
 
> **Documentation Note:** the full analysis, including the description of all 57 features, the correlation matrices, the experimental setup and the discussion of the results, is available in the **`Progetto_Machine_Learning.pdf`** file.
 
---
 
## How to Run
 
* **On Google Colab (recommended):** open the notebook with the *Open in Colab* badge at the top of this page and run all cells with *Runtime > Run all*. The dataset is downloaded automatically from the UCI Machine Learning Repository.
 
---
 
## Dataset
 
The project uses the **Spambase** dataset (4,601 e-mails, 57 continuous numerical features), released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license:
 
> Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53G6X
