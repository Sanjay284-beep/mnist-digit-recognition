"""
MNIST Handwritten Digit Recognition
Author: Sanjay
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

os.makedirs('results', exist_ok=True

print("=" * 60)
print("MNIST DIGIT RECOGNITION PROJECT")
print("=" * 60)

print("\n[1/9] Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"✓ Training samples: {X_train.shape[0]}")
print(f"✓ Test samples: {X_test.shape[0]}")
print(f"✓ Image shape: {X_train.shape[1]}x{X_train.shape[2]}")

print("\n[2/9] Preprocessing data...")
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print("✓ Data normalized to [0, 1] range")
print("✓ Data reshaped for CNN input")

print("\n[3/9] Creating sample visualizations...")

plt.figure(figsize=(12, 5))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f'Label: {y_train[i]}', fontsize=10)
    plt.axis('off')
plt.suptitle('Sample Handwritten Digits from Training Set', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/01_sample_digits.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: results/01_sample_digits.png")

print("\n[4/9] Building Simple Neural Network (Baseline)...")

model_simple = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
], name='Simple_NN')

model_simple.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Simple NN built successfully")
print(f"  Total parameters: {model_simple.count_params():,}")

print("\n[5/9] Training Simple Neural Network...")
print("This will take 3-5 minutes...\n")

history_simple = model_simple.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

print("\n✓ Simple NN training completed")

print("\n[6/9] Building Convolutional Neural Network (CNN)...")

model_cnn = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
], name='CNN')

model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ CNN built successfully")
print(f"  Total parameters: {model_cnn.count_params():,}")

print("\n[7/9] Training Convolutional Neural Network...")
print("This will take 5-7 minutes...\n")

history_cnn = model_cnn.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

print("\n✓ CNN training completed")

print("\n[8/9] Evaluating models on test set...")

test_loss_simple, test_acc_simple = model_simple.evaluate(X_test, y_test, verbose=0)
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(X_test, y_test, verbose=0)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"\nSimple Neural Network:")
print(f"  Test Accuracy:  {test_acc_simple * 100:.2f}%")
print(f"  Test Loss:      {test_loss_simple:.4f}")
print(f"\nConvolutional Neural Network:")
print(f"  Test Accuracy:  {test_acc_cnn * 100:.2f}%")
print(f"  Test Loss:      {test_loss_cnn:.4f}")
print(f"\nImprovement: {(test_acc_cnn - test_acc_simple) * 100:.2f}% higher accuracy with CNN")
print("=" * 60)

print("\n[9/9] Creating result visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_simple.history['accuracy'], 'b-', label='Simple NN (Train)', linewidth=2)
axes[0].plot(history_simple.history['val_accuracy'], 'b--', label='Simple NN (Val)', linewidth=2)
axes[0].plot(history_cnn.history['accuracy'], 'r-', label='CNN (Train)', linewidth=2)
axes[0].plot(history_cnn.history['val_accuracy'], 'r--', label='CNN (Val)', linewidth=2)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_simple.history['loss'], 'b-', label='Simple NN (Train)', linewidth=2)
axes[1].plot(history_simple.history['val_loss'], 'b--', label='Simple NN (Val)', linewidth=2)
axes[1].plot(history_cnn.history['loss'], 'r-', label='CNN (Train)', linewidth=2)
axes[1].plot(history_cnn.history['val_loss'], 'r--', label='CNN (Val)', linewidth=2)
axes[1].set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/02_training_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: results/02_training_comparison.png")

y_pred = model_cnn.predict(X_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - CNN Model\n(Test Set Predictions)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/03_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: results/03_confusion_matrix.png")

misclassified_idx = np.where(y_pred != y_test)[0][:20]

if len(misclassified_idx) > 0:
    plt.figure(figsize=(16, 8))
    for i, idx in enumerate(misclassified_idx[:20]):
        plt.subplot(4, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        
        pred_probs = model_cnn.predict(X_test[idx:idx+1], verbose=0)[0]
        confidence = pred_probs[y_pred[idx]] * 100
        
        plt.title(f'True: {y_test[idx]} | Pred: {y_pred[idx]}\nConf: {confidence:.1f}%', 
                  fontsize=9, color='red')
        plt.axis('off')
    
    plt.suptitle('Misclassified Examples (First 20)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/04_misclassified_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results/04_misclassified_examples.png")

correct_idx = np.where(y_pred == y_test)[0][:20]

plt.figure(figsize=(16, 8))
for i, idx in enumerate(correct_idx):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    
    pred_probs = model_cnn.predict(X_test[idx:idx+1], verbose=0)[0]
    confidence = pred_probs[y_pred[idx]] * 100
    
    plt.title(f'Digit: {y_test[idx]}\nConf: {confidence:.1f}%', 
              fontsize=9, color='green')
    plt.axis('off')

plt.suptitle('Correctly Classified Examples', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/05_correct_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: results/05_correct_predictions.png")

print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, digits=4))

with open('results/summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("MNIST DIGIT RECOGNITION - PROJECT SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training Samples: {X_train.shape[0]:,}\n")
    f.write(f"Test Samples: {X_test.shape[0]:,}\n\n")
    f.write("MODEL ARCHITECTURES\n")
    f.write("-" * 60 + "\n")
    f.write(f"Simple NN Parameters: {model_simple.count_params():,}\n")
    f.write(f"CNN Parameters: {model_cnn.count_params():,}\n\n")
    f.write("RESULTS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Simple NN Test Accuracy: {test_acc_simple * 100:.2f}%\n")
    f.write(f"CNN Test Accuracy: {test_acc_cnn * 100:.2f}%\n")
    f.write(f"Improvement: {(test_acc_cnn - test_acc_simple) * 100:.2f}%\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 60 + "\n")
    f.write(classification_report(y_test, y_pred, digits=4))

print("✓ Saved: results/summary.txt")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nAll results saved in 'results/' folder:")
print("  • 01_sample_digits.png")
print("  • 02_training_comparison.png")
print("  • 03_confusion_matrix.png")
print("  • 04_misclassified_examples.png")
print("  • 05_correct_predictions.png")
print("  • summary.txt")
print("\nNext steps:")
print("  1. Review all generated images")
print("  2. Read summary.txt for detailed metrics")
print("  3. Update README.md with your results")
print("  4. Push to GitHub")
print("=" * 60 + "\n")
