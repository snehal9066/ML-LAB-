# iris_feature_selection.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print("\n===== STEP 1: Load Dataset =====")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print("Feature Names:", feature_names)
print("Dataset Shape:", X.shape)


print("\n===== STEP 2: Exploratory Data Analysis =====")

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())


print("\n===== STEP 3: Split Dataset =====")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


print("\n===== STEP 4: Model Before Feature Selection =====")

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy BEFORE Feature Selection:",
      accuracy_score(y_test, y_pred))


print("\n===== STEP 5A: Univariate Feature Selection =====")

selector = SelectKBest(score_func=f_classif, k=2)
X_train_uni = selector.fit_transform(X_train, y_train)
X_test_uni = selector.transform(X_test)

selected_features_uni = np.array(feature_names)[selector.get_support()]
print("Selected Features (Univariate):", selected_features_uni)

model.fit(X_train_uni, y_train)
y_pred_uni = model.predict(X_test_uni)

print("Accuracy AFTER Univariate Selection:",
      accuracy_score(y_test, y_pred_uni))


print("\n===== STEP 5B: Feature Importance (Random Forest) =====")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_

for name, score in zip(feature_names, importances):
    print(f"{name}: {score:.4f}")

important_features = np.argsort(importances)[-2:]

X_train_rf = X_train[:, important_features]
X_test_rf = X_test[:, important_features]

print("Top 2 Important Features:",
      np.array(feature_names)[important_features])

model.fit(X_train_rf, y_train)
y_pred_rf = model.predict(X_test_rf)

print("Accuracy AFTER Random Forest Selection:",
      accuracy_score(y_test, y_pred_rf))


print("\n===== STEP 5C: RFE using SVM =====")

svm = SVC(kernel="linear")
rfe = RFE(svm, n_features_to_select=2)

X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

selected_features_rfe = np.array(feature_names)[rfe.support_]
print("Selected Features (RFE):", selected_features_rfe)

model.fit(X_train_rfe, y_train)
y_pred_rfe = model.predict(X_test_rfe)

print("Accuracy AFTER RFE:",
      accuracy_score(y_test, y_pred_rfe))


print("\n===== STEP 6: Comparison Summary =====")

print("Before Feature Selection Accuracy:",
      accuracy_score(y_test, y_pred))
print("Univariate Selection Accuracy:",
      accuracy_score(y_test, y_pred_uni))
print("Random Forest Selection Accuracy:",
      accuracy_score(y_test, y_pred_rf))
print("RFE Selection Accuracy:",
      accuracy_score(y_test, y_pred_rfe))
