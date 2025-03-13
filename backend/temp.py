import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("mobile.csv")

# Encode categorical features (Brand, OS)
le_brand = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])

le_os = LabelEncoder()
df['OS'] = le_os.fit_transform(df['OS'])

# Apply log transformation to the target variable (Price)
df['Price'] = np.log(df['Price'])

# Define features (X) and target variable (y)
X = df[['Brand', 'ReleaseYear', 'OS', 'Ram', 'Storage', 'Battery']]
y = df['Price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the SVM model with an RBF kernel
svm_model = SVR(kernel='rbf')

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best parameters from grid search
print(f"Best Parameters: {grid_search.best_params_}")

# Use the best model to predict the prices
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Inverse log transformation to get the real price values
y_pred_real = np.exp(y_pred)
y_test_real = np.exp(y_test)

# Evaluate the model's performance using MSE and R^2
mse = mean_squared_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")