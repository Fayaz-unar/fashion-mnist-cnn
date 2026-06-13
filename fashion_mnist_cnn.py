# ═══════════════════════════════════════════════════════════════
#           Fashion-MNIST CNN Classification
#           TensorFlow / Keras
# ═══════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ═══════════════════════════════════════════════════════════════
# Task 1: Data Loading and Preprocessing
# ═══════════════════════════════════════════════════════════════
print("=" * 55)
print("    Task 1: Data Loading and Preprocessing")
print("=" * 55)

# Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"Training images shape : {X_train.shape}")
print(f"Test images shape     : {X_test.shape}")
print(f"Training labels shape : {y_train.shape}")
print(f"Total Classes         : {len(class_names)}")

# Normalize pixel values to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0
print(f"\nPixel range after normalization: Min={X_train.min()}  Max={X_train.max()}")

# Reshape for CNN input (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1,  28, 28, 1)
print(f"\nAfter Reshaping:")
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)
print(f"\nAfter One-Hot Encoding:")
print(f"  y_train : {y_train_cat.shape}")
print(f"  y_test  : {y_test_cat.shape}")

# Show sample images
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title(class_names[y_train[i]], fontsize=9)
    plt.axis("off")
plt.suptitle("Sample Images from Fashion-MNIST Dataset",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
print("\nTask 1 Complete! ✅")


# ═══════════════════════════════════════════════════════════════
# Task 2: Design / Build the CNN Model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("         Task 2: Build the CNN Model")
print("=" * 55)

model = models.Sequential([

    # Convolutional Layer 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                  input_shape=(28, 28, 1), name="Conv_Layer_1"),
    layers.MaxPooling2D((2, 2), name="MaxPool_1"),

    # Convolutional Layer 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                  name="Conv_Layer_2"),
    layers.MaxPooling2D((2, 2), name="MaxPool_2"),

    # Convolutional Layer 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                  name="Conv_Layer_3"),

    # Flatten
    layers.Flatten(name="Flatten"),

    # Dense Layers
    layers.Dense(256, activation="relu", name="Dense_1"),
    layers.Dropout(0.5, name="Dropout"),
    layers.Dense(10, activation="softmax", name="Output_Layer")

], name="FashionMNIST_CNN")

model.summary()

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Compiled Successfully!")
print("  Optimizer : Adam")
print("  Loss      : Categorical Crossentropy")
print("  Metric    : Accuracy")
print("\nTask 2 Complete! ✅")


# ═══════════════════════════════════════════════════════════════
# Task 3: Compile and Train the Model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("         Task 3: Train the Model")
print("=" * 55)

history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print("\nTask 3 Complete! ✅")


# ═══════════════════════════════════════════════════════════════
# Task 4: Model Evaluation
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("         Task 4: Model Evaluation")
print("=" * 55)

# Test accuracy and loss
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# Predictions
y_pred     = model.predict(X_test)
y_pred_cls = np.argmax(y_pred, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cls, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_cls)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
print("\nTask 4 Complete! ✅")


# ═══════════════════════════════════════════════════════════════
# Task 5: Visualization and Analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("         Task 5: Visualization and Analysis")
print("=" * 55)

# Plot Training & Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"],     label="Training Loss",   linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
plt.title("Training & Validation Loss Across Epochs",
          fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Training & Validation Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"],     label="Training Accuracy",   linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
plt.title("Training & Validation Accuracy Across Epochs",
          fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot test images with predictions
plt.figure(figsize=(13, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    pred  = class_names[y_pred_cls[i]]
    true  = class_names[y_test[i]]
    color = "green" if pred == true else "red"
    plt.title(f"P: {pred}\nT: {true}", fontsize=8, color=color)
    plt.axis("off")
plt.suptitle("Test Image Predictions (Green=Correct, Red=Wrong)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print("\nTask 5 Complete! ✅")
print("\n" + "=" * 55)
print("        ALL TASKS COMPLETE! 🎉")
print(f"   Final Test Accuracy: {test_acc * 100:.2f}%")
print("=" * 55)