import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS
from flask import Flask, request, jsonify

# =========================
# Step 1: Data Reading
# =========================
df = pd.read_csv("data.csv")

# =========================
# Step 2: Data Preprocessing
# =========================
# One-Hot Encoding for 'Company'
df_encoded = pd.get_dummies(df, columns=["Company"], dtype=int)

# Identify boolean columns and numeric columns (excluding target "Price")
boolean_columns = ["Is5GSupport", "IsDualSimSupport"]
numeric_columns = [col for col in df_encoded.columns if col not in boolean_columns + ["Price"]]

# Normalize numerical features (Min-Max Scaling) on numeric columns only
df_encoded[numeric_columns] = (df_encoded[numeric_columns] - df_encoded[numeric_columns].min()) / \
                              (df_encoded[numeric_columns].max() - df_encoded[numeric_columns].min())

# =========================
# Step 3: Train-Test Split
# =========================
def train_test_split(df, test_size=0.2):
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_size))
    return shuffled_df[:split_index], shuffled_df[split_index:]

train_df, test_df = train_test_split(df_encoded)
# Convert feature arrays to float type to avoid dtype issues.
X_train = train_df.drop(columns=["Price"]).values.astype(np.float64)
y_train = train_df["Price"].values.astype(np.float64)
X_test = test_df.drop(columns=["Price"]).values.astype(np.float64)
y_test = test_df["Price"].values.astype(np.float64)

# =========================
# Step 4: SVR Implementation
# =========================
class SVR:
    def __init__(self, learning_rate=0.001, n_iters=1000, C=1.0, epsilon=0.1):
        """
        Parameters:
          learning_rate: Step size for gradient descent.
          n_iters: Number of iterations to run gradient descent.
          C: Regularization parameter (weight for the epsilon-insensitive loss).
          epsilon: Epsilon-insensitive threshold.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.C = C
        self.epsilon = epsilon
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Gradient (subgradient) descent loop
        for i in range(self.n_iters):
            # Compute predictions and errors
            y_pred = X.dot(self.w) + self.b
            error = y_pred - y
            
            # Start with the gradient from the regularization term (0.5 * ||w||^2)
            grad_w = np.copy(self.w)
            grad_b = 0.0
            
            # Determine indices where the epsilon-insensitive loss is active
            pos = error > self.epsilon      # Over-prediction beyond epsilon
            neg = error < -self.epsilon     # Under-prediction beyond epsilon
            
            # For samples where error > epsilon, add subgradient: C * x_i
            if np.any(pos):
                grad_w += self.C * np.sum(X[pos], axis=0)
            # For samples where error < -epsilon, subtract subgradient: C * x_i
            if np.any(neg):
                grad_w -= self.C * np.sum(X[neg], axis=0)
            
            # For the bias term, add C * (number of pos) and subtract C * (number of neg)
            grad_b = self.C * (np.sum(pos) - np.sum(neg))
            
            # Average the gradients over all samples
            grad_w /= n_samples
            grad_b /= n_samples
            
            # Update weights and bias
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        return X.dot(self.w) + self.b

# =========================
# Step 5: Train the SVR Model
# =========================
svr_model = SVR(learning_rate=0.001, n_iters=1000, C=10, epsilon=0.1)
svr_model.fit(X_train, y_train)

# =========================
# Step 6: Evaluate the Model
# =========================
y_pred = svr_model.predict(X_test)
mae = np.mean(np.abs(y_test - y_pred))
print("Mean Absolute Error:", mae)

# =========================
# Step 7: Save the Trained SVR Model as a .pkl File
# =========================
with open("svr_model.pkl", "wb") as file:
    pickle.dump(svr_model, file)

print("SVR model saved successfully as 'svr_model.pkl'")

# =========================
# Step 8: Load the Saved Model (for Flask Backend)
# =========================
with open("svr_model.pkl", "rb") as file:
    model = pickle.load(file)

# =========================
# Step 9: Build Flask Backend
# =========================
# The valid companies list must match the order used during training.
valid_companies = ["Samsung", "IPhone", "Oppo", "Vivo", "Realme", "OnePlus"]

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.json

        # Extract features with defaults if not provided
        battery = int(data.get("Battery", 4000))
        weight = int(data.get("Weight", 180))
        no_of_cameras = int(data.get("NoOfCameras", 2))
        is_5g = int(data.get("Is5GSupport", False))  # Convert boolean to int: True -> 1, False -> 0
        is_dual_sim = int(data.get("IsDualSimSupport", True))
        model_year = int(data.get("ModelYear", 2022))
        company = data.get("Company", "Samsung")
        frame_rate = int(data.get("FrameRate", 60))

        # One-hot encode the company (order must match training)
        company_encoding = [1 if company == comp else 0 for comp in valid_companies]

        # Construct the input vector (order: all features + one-hot encoding of Company)
        input_data = np.array(
            [battery, weight, no_of_cameras, is_5g, is_dual_sim, model_year, frame_rate] + company_encoding
        ).reshape(1, -1).astype(np.float64)

        # Get prediction from the loaded SVR model
        predicted_price = model.predict(input_data)[0]

        return jsonify({"Predicted Price": round(predicted_price, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =========================
# Step 10: Run the Flask Application
# =========================
if __name__ == '__main__':
    app.run(debug=True)