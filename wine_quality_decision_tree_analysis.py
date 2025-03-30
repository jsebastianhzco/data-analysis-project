import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# 1. Load dataset
data = pd.read_csv('winequality-red.csv')

# Preview first rows
print("First rows of the dataset:")
print(data.head())

# Descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# Histograms for all features
print("\nHistograms for all features:")
data.hist(bins=15, figsize=(15, 10))
plt.show()

# Boxplots to detect outliers
print("\nBoxplots to detect outliers:")
data.plot(kind='box', subplots=True, layout=(4, 3), figsize=(15, 10), sharex=False, sharey=False)
plt.show()

# Correlation matrix
print("\nCorrelation matrix:")
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# 2. Data Preprocessing
print("\nData Preprocessing:")
if data.isnull().any().any():
    print("Missing values detected. Replacing with column means.")
    data.fillna(data.mean(), inplace=True)
else:
    print("No missing values detected in the dataset.")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('quality', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
scaled_data['quality'] = data['quality']
print("Numeric features scaled using StandardScaler.")

# 3. Feature Selection
print("\nFeature Selection:")
k = 8
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(scaled_data.drop('quality', axis=1), scaled_data['quality'])
selected_features = scaled_data.drop('quality', axis=1).columns[selector.get_support()]

print(f"Selected features ({k}):")
for feature, score in zip(selected_features, selector.scores_[selector.get_support()]):
    print(f"{feature}: {score:.2f}")

feature_scores = pd.DataFrame({
    'Feature': scaled_data.columns[:-1],
    'Score': selector.scores_
})
print("\nScores of all features:")
print(feature_scores.sort_values(by='Score', ascending=False))

# 4. Train-Test Split
print("\nSplitting data into training and test sets:")
X = scaled_data[selected_features]
y = scaled_data['quality']
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

print(f"Total samples: {len(data)}")
print(f"Training samples: {len(X_train)} ({(len(X_train) / len(data)) * 100:.2f}%)")
print(f"Test samples: {len(X_test)} ({(len(X_test) / len(data)) * 100:.2f}%)")

# 5. Train the Model
print("\nTraining Decision Tree model:")
max_depth = 10
min_samples_leaf = 5
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
model.fit(X_train, y_train)

print(f"Model trained with max_depth={max_depth} and min_samples_leaf={min_samples_leaf}")
if model:
    print("Decision tree model successfully created and trained.")
else:
    print("Error in model creation.")

# 6. Evaluate the Model
print("\nEvaluating model performance:")
y_pred = model.predict(X_test)

print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

# 7. Visualizations
print("\nVisualizing results:")

# Decision Tree plot
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=selected_features, class_names=True)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()
print("Confusion Matrix:\n", cm)

# Feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), selected_features[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
