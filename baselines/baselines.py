import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, balanced_accuracy_score,    # <-- NEW
)

RANDOM_STATE = 42

# LOAD CSV
df = pd.read_csv("flights_prepared.csv")
print("Loaded CSV with shape:", df.shape)
TARGET = "delay_15"
DATE_COL = "FL_DATE"

#splitting
df = df.sort_values(by=DATE_COL)

train_df, temp_df = train_test_split(df, test_size=0.30, shuffle=False)
val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=False)

print("Split sizes -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# preprocessing
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

#training and evaluation


all_metrics = [] 
def train_and_eval(model, model_name):
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    try:
        proba = pipe.predict_proba(X_val)[:, 1]
    except Exception:
        proba = None

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(y_val, preds)

    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    if proba is not None:
        roc_auc = roc_auc_score(y_val, proba)
    else:
        roc_auc = np.nan

    print(f"\n===== {model_name} =====")
    print("Accuracy           :", acc)
    print("Balanced Accuracy  :", bal_acc)
    print("Precision (positive class=1):", prec)
    print("Recall (Sensitivity):", rec)
    print("Specificity        :", specificity)
    print("F1 Score           :", f1)
    print("ROC-AUC            :", roc_auc)
    print("Confusion Matrix:\n", cm)
    print("\nClassification report:\n",
          classification_report(y_val, preds, digits=3))

    # store metrics for summary
    all_metrics.append({
        "model": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

    return pipe, cm


def plot_cm(cm, title):
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()



# Logistic Regression
lr_model, lr_cm = train_and_eval(
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Logistic Regression"
)
plot_cm(lr_cm, "Logistic Regression CM")

# Decision Tree
dt_model, dt_cm = train_and_eval(
    DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree"
)
plot_cm(dt_cm, "Decision Tree CM")

# Random Forest
rf_model, rf_cm = train_and_eval(
    RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Random Forest"
)
plot_cm(rf_cm, "Random Forest CM")

# Multi-Layer Perceptron
mlp_model, mlp_cm = train_and_eval(
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True
    ),
    "MLPClassifier"
)
plot_cm(mlp_cm, "MLP Classifier CM")

metrics_df = pd.DataFrame(all_metrics)
print("\n\n===== Validation Metrics Summary =====")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

#Gnereate CSV file of metrics
metrics_df.to_csv("baselines_metrics_summary.csv", index=False)
print("\nSaved summary metrics to 'baselines_metrics_summary.csv'")
