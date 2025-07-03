# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

# Set pandas display options
pd.set_option('display.max_columns', None)

# Set random seed untuk reproducibility
np.random.seed(42)

# ===============================================================
# 1. LOAD DATA DAN EXPLORATORY DATA ANALYSIS (EDA)
# ===============================================================
print("1. Loading data and performing EDA...")

# Import library
import pandas as pd

# Load dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Dataset Overview
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(df.info())

# Descriptive Statistics untuk data numerik
print("\nDescriptive Statistics (Numerik):")
print(df.describe())

# Descriptive Statistics untuk data kategorik
print("\nDescriptive Statistics (Kategorikal):")
print(df.describe(include=['object']))

# Check Missing Values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

print("\nMissing Values:")
print(missing_values)

print("\nMissing Percentage:")
print(missing_percentage)

# Target Distribution (NObeyesdad adalah label/target klasifikasi)
print("\nTarget Distribution:")
print(df['NObeyesdad'].value_counts(normalize=True))


# ===============================================================
# 2. VISUALISASI DATA
# ===============================================================
print("\n2. Data Visualization...")


# Pastikan sebelumnya data sudah dimuat:
# df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Histogram untuk variabel numerik
numeric_features = df.select_dtypes(include=[np.number]).columns

print(f"Numerical Features: {list(numeric_features)}")

df[numeric_features].hist(figsize=(15, 10), bins=20, edgecolor='black', grid=False)
plt.tight_layout()
plt.savefig('numeric_histograms.png')
plt.close()
print("Saved histogram: numeric_histograms.png")

# Bar plot untuk variabel kategorikal
categorical_features = df.select_dtypes(include=['object']).columns

print(f"Categorical Features: {list(categorical_features)}")

for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    df[feature].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.ylabel('Count')
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f'distribution_{feature}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved bar plot: {filename}")

# Korelasi antara variabel numerik
correlation_matrix = df[numeric_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()
print("Saved correlation heatmap: correlation_matrix.png")

# ===============================================================
# 3. DATA CLEANING
# ===============================================================
print("\n3. Data Cleaning...")

# Handle missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Remove duplicates
df.drop_duplicates(inplace=True)
print(f"Number of rows after removing duplicates: {len(df)}")

# ===============================================================
# 4. FEATURE ENGINEERING & ENCODING
# ===============================================================
print("\n4. Feature Engineering & Encoding...")

from sklearn.preprocessing import LabelEncoder

# Simpan encoder untuk setiap kolom kategorikal
encoding_maps = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoding_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Tampilkan contoh hasil encoding
print("\nContoh Encoding Map:")
for key, value in encoding_maps.items():
    print(f"{key}: {value}")

# ===============================================================
# 5. Checking correlations after encoding...
# ===============================================================
print("\n5. Checking correlations after encoding...")

# Korelasi setelah encoding
correlation = df.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(correlation.round(2),
           annot=True,
           vmax=1,
           square=True,
           cmap='RdYlGn_r')
plt.title('Correlation Matrix After Encoding')
plt.tight_layout()
plt.savefig('correlation_after_encoding.png')
plt.close()
print("Saved: correlation_after_encoding.png")

# ===============================================================
# 6. Feature Selection...
# ===============================================================
print("\n6. Feature Selection...")

# Remove constant features
df = df.loc[:, df.apply(pd.Series.nunique) != 1]

# Fungsi untuk mencari fitur yang sangat berkorelasi
def correlation(dataset, threshold):
    col_corr = set()  # Menyimpan nama kolom yang berkorelasi tinggi
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

# Pisahkan target
data_tanpa_target = df.drop('NObeyesdad', axis=1)

# Cari fitur yang berkorelasi tinggi
corr_features = correlation(data_tanpa_target, 0.8)
print(f"Correlated features: {len(corr_features)}")
print(corr_features)

# Hapus fitur-fitur yang terlalu berkorelasi
df.drop(labels=corr_features, axis=1, inplace=True)

# Final correlation heatmap
correlation = df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation.round(2),
           annot=True,
           vmax=1,
           square=True,
           cmap='RdYlGn_r')
plt.title('Final Correlation Matrix After Feature Selection')
plt.tight_layout()
plt.savefig('final_correlation.png')
plt.close()
print("Saved: final_correlation.png")

# ===============================================================
# 7. MODEL TRAINING
# ===============================================================
print("\n7. Model Training...")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import joblib

# Split data
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Hyperparameter tuning
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_classifier = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(dt_classifier, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# ===============================================================
# 8. MODEL EVALUATION
# ===============================================================
print("\n8. Model Evaluation...")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Visualisasi pohon keputusan
plt.figure(figsize=(40, 30))
plot_tree(best_model,
          feature_names=X.columns,
          class_names=[str(i) for i in sorted(y.unique())],
          filled=True,
          rounded=True,
          max_depth=3)
plt.title('Decision Tree Visualization (Limited to Depth 3)')
plt.tight_layout()
plt.savefig('decision_tree.png')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Train/Test Metrics
train_pred = best_model.predict(X_train)
print('\nTrain Accuracy:', accuracy_score(y_train, train_pred))
print('Train Precision:', precision_score(y_train, train_pred, average='micro'))
print('Train Recall:', recall_score(y_train, train_pred, average='micro'))

print('\nTest Accuracy:', accuracy_score(y_test, y_pred))
print('Test Precision:', precision_score(y_test, y_pred, average='micro'))
print('Test Recall:', recall_score(y_test, y_pred, average='micro'))

# ===============================================================
# 9. SAVE MODEL COMPONENTS
# ===============================================================
print("\n9. Saving model components...")

model_components = {
    'model': best_model,
    'feature_names': X.columns.tolist(),
    'encoding_maps': encoding_maps,
    'model_params': best_params,
    'removed_features': list(corr_features) if len(corr_features) > 0 else [],
    'target_classes': list(encoding_maps['NObeyesdad'].keys())
}

joblib.dump(model_components, 'obesity_prediction_components.joblib')
print("Saved as 'obesity_prediction_components.joblib'")

# ===============================================================
# PREDICTION FUNCTION
# ===============================================================
def predict_obesity(data, model_components):
    """
    Predict obesity level.
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    model = model_components['model']
    encoding_maps = model_components['encoding_maps']
    feature_names = model_components['feature_names']
    
    for col in data.columns:
        if col in encoding_maps:
            data[col] = data[col].map(encoding_maps[col])
    
    data_for_pred = data[feature_names]
    prediction = model.predict(data_for_pred)[0]
    probabilities = model.predict_proba(data_for_pred)[0]

    target_map_inverse = {v: k for k, v in encoding_maps['NObeyesdad'].items()}
    
    return {
        'prediction': prediction,
        'prediction_label': target_map_inverse[prediction],
        'probability': probabilities[prediction]
    }

# Sample prediction test
test_sample = X_test.iloc[0].to_dict()
loaded_model = joblib.load('obesity_prediction_components.joblib')
result = predict_obesity(test_sample, loaded_model)

print("\nTest prediction result:")
print(result)
print("Actual:", y_test.iloc[0])
