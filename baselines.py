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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42


#loading csv
df = pd.read_csv("flights_prepared.csv")
print("Loaded CSV with shape:", df.shape)
TARGET = "delay_15"
DATE_COL = "FL_DATE"


#splitting
df = df.sort_values(by=DATE_COL)

train_df, temp_df = train_test_split(df, test_size=0.30, shuffle=False)
val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=False)

print(len(train_df), len(val_df), len(test_df))

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


#preprocessing 
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ]
)


#training and evaluation function
def train_and_eval(model, model_name):
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    cm = confusion_matrix(y_val, preds)

    print(f"\n===== {model_name} =====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("Confusion Matrix:\n", cm)

    return pipe, cm


def plot_cm(cm, title):
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center")

    plt.show()


# Logistic Regression
lr_model, lr_cm = train_and_eval(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Logistic Regression"
)
plot_cm(lr_cm, "Logistic Regression CM")

# Decision Tree
dt_model, dt_cm = train_and_eval(
    DecisionTreeClassifier(max_depth=12, class_weight="balanced"),
    "Decision Tree"
)
plot_cm(dt_cm, "Decision Tree CM")

# Random Forest
rf_model, rf_cm = train_and_eval(
    RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Random Forest"
)
plot_cm(rf_cm, "Random Forest CM")