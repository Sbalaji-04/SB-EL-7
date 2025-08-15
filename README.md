## Support Vector Machines (SVM)

### Steps

1. **Data Loading and Preprocessing**
   - Loaded the Breast Cancer dataset using `pandas`.
   - Dropped the `id` column (not useful for modeling).
   - Encoded the `diagnosis` column: `M` as 1 (malignant), `B` as 0 (benign).
   - Split the dataset into:
     - Features (`X`)
     - Labels (`y`)
   - Used `train_test_split` with an 80-20 split.
   - Scaled the features using `StandardScaler`.

2. **Training SVM Models**
   - Trained a **Linear SVM** using `SVC(kernel='linear')`.
   - Trained a **Non-linear SVM (RBF Kernel)** using `SVC(kernel='rbf')`.

3. **Model Evaluation**
   - **Linear SVM Accuracy**: `95.61%`
   - **RBF SVM Accuracy**: `98.25%`

4. **Hyperparameter Tuning**
   - Used `GridSearchCV` to tune:
     - `C`: [0.1, 1, 10, 100]
     - `gamma`: ['scale', 0.01, 0.1, 1]
     - `kernel`: ['rbf']
   - **Best Parameters**: `C=1`, `gamma='scale'`, `kernel='rbf'`
   - **Best Cross-validation Score**: `97.58%`

5. **Cross-Validation**
   - Performed 5-fold cross-validation using `cross_val_score` with the best estimator.
   - **Cross-validation scores**: `[0.8509, 0.8947, 0.9298, 0.9474, 0.9381]`
   - **Mean CV Accuracy**: `91.21%`

6. **Dimensionality Reduction and Visualization**
   - Reduced features to 2D using **PCA** (`n_components=2`).
   - Trained an **RBF SVM** on the 2D data.
   - Plotted the decision boundary using `matplotlib`.

