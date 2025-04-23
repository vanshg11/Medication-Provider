import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Load training data
df = pd.read_csv('Training.csv')

# Separate features and label
disease_column = 'prognosis'
X = df.drop(columns=[disease_column])
y = df[disease_column]

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train RandomForest model (high accuracy from baseline)
model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("RandomForest accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Persist model artifacts
output_dir = 'mediactionproject/working'
os.makedirs(output_dir, exist_ok=True)

# Save the trained model
with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save the label encoder for mapping back
with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

# Save symptom column order for consistent feature construction
symptom_columns = X.columns.tolist()
with open(os.path.join(output_dir, 'symptom_columns.pkl'), 'wb') as f:
    pickle.dump(symptom_columns, f)