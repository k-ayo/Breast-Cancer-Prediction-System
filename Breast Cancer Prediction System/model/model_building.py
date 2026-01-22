
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

os.makedirs('model', exist_ok=True)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

df.rename(columns={
    'mean radius':'radius_mean',
    'mean perimeter':'perimeter_mean',
    'mean area':'area_mean',
    'mean smoothness':'smoothness_mean',
    'mean concavity':'concavity_mean'
}, inplace=True)

FEATURES = ['radius_mean','perimeter_mean','area_mean','smoothness_mean','concavity_mean']
X = df[FEATURES]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'model/breast_cancer_model.pkl')
print("Model saved as model/breast_cancer_model.pkl")
