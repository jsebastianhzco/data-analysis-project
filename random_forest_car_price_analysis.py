import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset from specified path
data = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# Initial info
print(data.info())

# Handle missing values
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Remove duplicates
data.drop_duplicates(inplace=True)

# Encode categorical variables using One-Hot Encoding
data = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Boxplots to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numeric_columns])
plt.title("Box Plot of Numeric Variables")
plt.show()

# Histograms to examine distributions
plt.figure(figsize=(12, 6))
data[numeric_columns].hist(bins=15, layout=(2, 3), alpha=0.7)
plt.tight_layout()
plt.show()

# Scatter plots to visualize relationships
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=data, x='km_driven', y='selling_price', ax=ax[0])
ax[0].set_title('Kilometers vs Selling Price')
sns.scatterplot(data=data, x='year', y='selling_price', ax=ax[1])
ax[1].set_title('Year vs Selling Price')
plt.show()

# Final check of prepared data
print(data.head())
print(data.describe())
print(data.info())

# Correlation analysis
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Feature importance using Random Forest
X = numeric_data.drop('selling_price', axis=1)
y = numeric_data['selling_price']
model = RandomForestRegressor()
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model configuration
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2 Score): {r2}")

# Visualization: Real vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Selling Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # perfect prediction line
plt.show()

# Prediction error distribution
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, color='blue')
plt.xlabel("Prediction Errors")
plt.title("Distribution of Prediction Errors")
plt.show()
