import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

# -------------------------------
# Load MNIST dataset
# -------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images (28x28 → 784)
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# -------------------------------
# Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=1000, solver='lbfgs')

model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Hyperparameter Tuning using GridSearchCV
# -------------------------------
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [500, 1000]
}

grid = GridSearchCV(
    LogisticRegression(solver='lbfgs'),
    param_grid,
    cv=3,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print("\nBest Parameters from Grid Search:")
print(grid.best_params_)

best_model = grid.best_estimator_

# -------------------------------
# Evaluate Tuned Model
# -------------------------------
y_pred_best = best_model.predict(X_test)

print("\nTuned Model Accuracy:", accuracy_score(y_test, y_pred_best))

# -------------------------------
# Decision Boundary Visualization (PCA)
# -------------------------------
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

best_model.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=5)
plt.title("Decision Boundary (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


