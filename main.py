import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('D:/VIT Bhopal/GitHub/ML Projects/Stock Price Prediction/Dataset/Tesla.csv')

# Data Information
print("Dataset Shape:", df.shape)
print(df.describe())
print(df.info())

# Visualizing the Closing Price Over Time
plt.figure(figsize=(15, 5))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title('Tesla Closing Price Over Time', fontsize=15)
plt.xlabel('Time (Days)')
plt.ylabel('Price in USD')
plt.legend()
plt.show()

# Drop Redundant 'Adj Close' Column
df.drop(['Adj Close'], axis=1, inplace=True)

# Check for Null Values
print("\nMissing Values in Each Column:\n", df.isnull().sum())

# Distribution Plots for Continuous Features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True, color='purple')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Box Plots for Continuous Features
plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(df[col], color='orange')
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# Feature Engineering
df[['month', 'day', 'year']] = df['Date'].str.split('/', expand=True).astype(int)
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
df.drop('Date', axis=1, inplace=True)

# Visualize Yearly Average Stock Data
data_grouped = df.groupby('year').mean()
plt.figure(figsize=(20, 10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot(kind='bar', color='green')
    plt.title(f'Yearly Average {col}')
    plt.ylabel('Price in USD')
plt.tight_layout()
plt.show()

# Additional Features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Target Distribution
plt.figure(figsize=(6, 6))
plt.pie(df['target'].value_counts().values, labels=['Decrease', 'Increase'], autopct='%1.1f%%', colors=['red', 'green'])
plt.title('Target Distribution (Stock Movement)')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Data Splitting and Normalization
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)
print("\nTraining Set Shape:", X_train.shape, "Validation Set Shape:", X_valid.shape)

# Model Development and Evaluation
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss')
]

for model in models:
    model.fit(X_train, Y_train)
    print(f'\nModel: {model.__class__.__name__}')

    # Predictions
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    # Probabilities for ROC-AUC
    train_prob = model.predict_proba(X_train)[:, 1]
    valid_prob = model.predict_proba(X_valid)[:, 1]

    # Evaluation Metrics
    print("Training Metrics:")
    print(classification_report(Y_train, train_pred))
    print("ROC-AUC Score:", metrics.roc_auc_score(Y_train, train_prob))

    print("Validation Metrics:")
    print(classification_report(Y_valid, valid_pred))
    print("ROC-AUC Score:", metrics.roc_auc_score(Y_valid, valid_prob))

# Confusion Matrix for Logistic Regression
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid, cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()