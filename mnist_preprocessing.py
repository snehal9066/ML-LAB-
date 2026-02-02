import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Original Training Shape:", X_train.shape)
print("Original Test Shape:", X_test.shape)

# Normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot encode labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

# Split training data into training & validation
X_train, X_val, y_train_encoded, y_val = train_test_split(
    X_train,
    y_train_encoded,
    test_size=0.2,
    random_state=42
)

print("\nAfter Preprocessing:")
print("Training Set:", X_train.shape)
print("Validation Set:", X_val.shape)
print("Test Set:", X_test.shape)
