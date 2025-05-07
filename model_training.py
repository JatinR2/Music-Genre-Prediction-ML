# Import required libraries
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Reshape the data from 3D to 2D
# Assuming X_train and X_val are in shape (n_samples, time_steps, n_features)
n_samples_train = X_train.shape[0]
n_samples_val = X_val.shape[0]

# Reshape to (n_samples, time_steps * n_features)
X_train_reshaped = X_train.reshape(n_samples_train, -1)
X_val_reshaped = X_val.reshape(n_samples_val, -1)

# Initialize models
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Dictionary to store models and their names
models = {
    'SVM': svm,
    'KNN': knn,
    'Random Forest': rf
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_reshaped, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_reshaped)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    
    # Print results
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Compare model performances
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.show()

# Save the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest performing model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
import joblib
joblib.dump(best_model, 'best_model.joblib')
print("Best model saved as 'best_model.joblib'") 