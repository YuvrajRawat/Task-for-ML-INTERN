import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("TASK-ML-INTERN.csv")

# Select numeric columns for processing
numeric_df = df.select_dtypes(include=['number'])

# Extract features and target
X = numeric_df.drop(columns=['vomitoxin_ppb']).values  # Features
y = numeric_df['vomitoxin_ppb'].values  # Target Variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction using PCA
pca = PCA(n_components=10)  # Reduce to 10 principal components
X_pca = pca.fit_transform(X_scaled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save Model Performance
with open("model_performance.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR² Score: {r2:.4f}\n")

# Save Processed Data
pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]).to_csv("processed_data.csv", index=False)

print("Processing Complete. Results saved in 'model_performance.txt' and 'processed_data.csv'")
