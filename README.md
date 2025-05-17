\'\'\'python
# CODCREDIT
Repository for CODSOFT Data Science Internship tasks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Load the dataset
# Ensure the path is correct, the dataset should be in the same directory, or specify the correct path
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("Error: creditcard.csv not found. Please make sure the file is in the correct directory.")
    exit()

# Explore the dataset
print("Dataset shape:", df.shape)
print("Dataset head:")
print(df.head())
print("Dataset info:")
df.info()

# --- Data Cleaning ---
# Check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum().sum())

# Handle missing values (if any) - using 0 for simplicity.  You might want to use mean, median, or imputation.
df.fillna(0, inplace=True)  # Replace NaN values with 0
# df = df.dropna() #another way to handle missing values

print("Missing values after cleaning:")
print(df.isnull().sum().sum())

# Check for duplicate values
print("Duplicate values before cleaning:", df.duplicated().sum())

# Remove duplicate values
df = df.drop_duplicates()

print("Duplicate values after cleaning:", df.duplicated().sum())


# Class distribution
print("Class distribution:")
print(df['Class'].value_counts())

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Class Imbalance Handling ---
# 1. SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 2. Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)

# 3. Combined: SMOTE + Under-sampling (Optional - can sometimes give best results)
# Apply SMOTE first to oversample the minority class, then apply RandomUnderSampler to the majority class
smote_combined = SMOTE(sampling_strategy=0.5, random_state=42) # You can adjust sampling_strategy
rus_combined = RandomUnderSampler(sampling_strategy=0.8, random_state=42) # You can adjust sampling_strategy
X_train_combined, y_train_combined = smote_combined.fit_resample(X_train_scaled, y_train) # Apply SMOTE
X_train_combined, y_train_combined = rus_combined.fit_resample(X_train_combined, y_train_combined) # Apply RUS


# --- Model Training and Evaluation ---
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a classification model and prints performance metrics.  Also generates a ROC curve.

    Args:
        model: Trained classification model.
        X_test: Scaled test features.
        y_test: True labels for the test set.
        model_name: Name of the model for display purposes.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    print(f"\n--- {model_name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    # Generate ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    
    return y_pred, y_pred_proba # Return both predictions and probabilities


# 1. Logistic Regression
logistic_model = LogisticRegression(random_state=42, solver='liblinear') #solver specified
logistic_model.fit(X_train_scaled, y_train) #default
y_pred_logistic, y_pred_proba_logistic = evaluate_model(logistic_model, X_test_scaled, y_test, "Logistic Regression (Original)")

logistic_model_smote = LogisticRegression(random_state=42, solver='liblinear')
logistic_model_smote.fit(X_train_smote, y_train_smote)
y_pred_logistic_smote, y_pred_proba_logistic_smote = evaluate_model(logistic_model_smote, X_test_scaled, y_test, "Logistic Regression (SMOTE)")

logistic_model_rus = LogisticRegression(random_state=42, solver='liblinear')
logistic_model_rus.fit(X_train_rus, y_train_rus)
y_pred_logistic_rus, y_pred_proba_logistic_rus = evaluate_model(logistic_model_rus, X_test_scaled, y_test, "Logistic Regression (RUS)")

logistic_model_combined = LogisticRegression(random_state=42, solver='liblinear')
logistic_model_combined.fit(X_train_combined, y_train_combined)
y_pred_logistic_combined, y_pred_proba_logistic_combined = evaluate_model(logistic_model_combined, X_test_scaled, y_test, "Logistic Regression (Combined)")


# 2. Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train) #default
y_pred_rf, y_pred_proba_rf = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest (Original)")

rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_train_smote, y_train_smote)
y_pred_rf_smote, y_pred_proba_rf_smote = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest (SMOTE)")

rf_model_rus = RandomForestClassifier(random_state=42)
rf_model_rus.fit(X_train_rus, y_train_rus)
y_pred_rf_rus, y_pred_proba_rf_rus = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest (RUS)")

rf_model_combined = RandomForestClassifier(random_state=42)
rf_model_combined.fit(X_train_combined, y_train_combined)
y_pred_rf_combined, y_pred_proba_rf_combined  = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest (Combined)")


# --- Comparison of Models (Optional) ---
# You can compare the performance of different models and choose the best one.
# For example, you can compare the ROC AUC scores of the models.
print("\n--- Model Comparison (ROC AUC Scores) ---")
print(f"Logistic Regression (Original): {roc_auc_score(y_test, y_pred_proba_logistic):.4f}")
print(f"Logistic Regression (SMOTE): {roc_auc_score(y_test, y_pred_proba_logistic_smote):.4f}")
print(f"Logistic Regression (RUS): {roc_auc_score(y_test, y_pred_proba_logistic_rus):.4f}")
print(f"Logistic Regression (Combined): {roc_auc_score(y_test, y_pred_proba_logistic_combined):.4f}")
print(f"Random Forest (Original): {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
print(f"Random Forest (SMOTE): {roc_auc_score(y_test, y_pred_proba_rf_smote):.4f}")
print(f"Random Forest (RUS): {roc_auc_score(y_test, y_pred_proba_rf_rus):.4f}")
print(f"Random Forest (Combined): {roc_auc_score(y_test, y_pred_proba_rf_combined):.4f}")

# Optional:  Further analysis
# 1. Feature Importance (Random Forest)
if 'rf_model' in locals(): # Check if the variable exists
    feature_importances = rf_model.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(feature_importances)[::-1]  # Sort in descending order

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()

# 2. Precision-Recall Curve (Example with Logistic Regression - SMOTE)
from sklearn.metrics import precision_recall_curve

if 'y_pred_proba_logistic_smote' in locals():
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_logistic_smote)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Logistic Regression - SMOTE)')
    plt.legend(loc="lower left")
    plt.show()

\'\'\'


Result:

Dataset shape: (284807, 31)
Dataset head:
   Time        V1        V2        V3        V4        V5  ...       V25       V26       V27       V28  Amount  Class
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  ...  0.128539 -0.189115  0.133558 -0.021053  149.62      0
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018  ...  0.167170  0.125895 -0.008983  0.014724    2.69      0
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  ... -0.327642 -0.139097 -0.055353 -0.059752  378.66      0
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  ...  0.647376 -0.221929  0.062723  0.061458  123.50      0
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  ... -0.206010  0.502292  0.219422  0.215153   69.99      0

[5 rows x 31 columns]
Dataset info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype
---  ------  --------------   -----
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
Missing values before cleaning:
0
Missing values after cleaning:
0
Duplicate values before cleaning: 1081
Duplicate values after cleaning: 0
Class distribution:
Class
0    283253
1       473
Name: count, dtype: int64

--- Logistic Regression (Original) ---
Confusion Matrix:
[[84973    11]
 [   60    74]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     84984
           1       0.87      0.55      0.68       134

    accuracy                           1.00     85118
   macro avg       0.93      0.78      0.84     85118
weighted avg       1.00      1.00      1.00     85118

ROC AUC Score: 0.9695674058400459

--- Logistic Regression (SMOTE) ---
Confusion Matrix:
[[82923  2061]
 [   14   120]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     84984
           1       0.06      0.90      0.10       134

    accuracy                           0.98     85118
   macro avg       0.53      0.94      0.55     85118
weighted avg       1.00      0.98      0.99     85118

ROC AUC Score: 0.9771626897986767

--- Logistic Regression (RUS) ---
Confusion Matrix:
[[81639  3345]
 [   15   119]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.96      0.98     84984
           1       0.03      0.89      0.07       134

    accuracy                           0.96     85118
   macro avg       0.52      0.92      0.52     85118
weighted avg       1.00      0.96      0.98     85118

ROC AUC Score: 0.9736323940169246

--- Logistic Regression (Combined) ---
Confusion Matrix:
[[83375  1609]
 [   15   119]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     84984
           1       0.07      0.89      0.13       134

    accuracy                           0.98     85118
   macro avg       0.53      0.93      0.56     85118
weighted avg       1.00      0.98      0.99     85118

ROC AUC Score: 0.977112460853035

![c](https://github.com/user-attachments/assets/fb9e7115-d297-4450-9be3-1f5cc326e300)
![r](https://github.com/user-attachments/assets/51c60f8b-dac3-435d-bdca-1c289349a5b0)
![e](https://github.com/user-attachments/assets/ad851e8a-e725-4cdf-a628-5cd8b16ad02e)
![d](https://github.com/user-attachments/assets/5fa621ac-c334-4768-aa0c-152cd7920c49)

