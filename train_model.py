import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading extracted features...")
# 1. Load the dataset you just created
df = pd.read_csv('final_training_data.csv')

# 2. Separate Features (X) and Labels (y)
# X contains all the math (mean, std, rms). y contains the answers (0 or 1).
X = df.drop('Label', axis=1) 
y = df['Label']

# 3. Split the data (80% for training, 20% for testing)
# The AI learns on the 80%, and takes a "blind test" on the remaining 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training AI on {len(X_train)} samples...")
print(f"Testing AI on {len(X_test)} blind samples...\n")

# 4. Initialize and Train the Random Forest Model
ai_model = RandomForestClassifier(n_estimators=100, random_state=42)
ai_model.fit(X_train, y_train)

# 5. Make Predictions on the blind test set
predictions = ai_model.predict(X_test)

# 6. Evaluate the Results
accuracy = accuracy_score(y_test, predictions)
print("================ AI MODEL RESULTS ================")
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['Normal (0)', 'Parkinson (1)']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("==================================================")

# (Optional) Save the trained model to a file for later use
import joblib
joblib.dump(ai_model, 'parkinsons_rf_model.pkl')
print("\nModel saved successfully as 'parkinsons_rf_model.pkl'!")
