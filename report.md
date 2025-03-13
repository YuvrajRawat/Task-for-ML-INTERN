📜 ML Intern Task - Report
1. Introduction
This project uses Machine Learning to analyze hyperspectral imaging data to predict DON (Deoxynivalenol) mycotoxin levels in corn samples. The dataset contains spectral bands as features and corresponding mycotoxin concentration as labels.

2. Data Preprocessing
Loaded and explored the dataset (TASK-ML-INTERN.csv).
Removed non-numeric columns for better correlation analysis.
Standardized spectral reflectance values using StandardScaler.

3. Feature Engineering & Visualization
Correlation Heatmap: Identified feature relationships.
Spectral Reflectance Plot: Visualized average reflectance across wavelength bands.
Dimensionality Reduction: Applied PCA (Principal Component Analysis) to extract key components.

4. Model Development
Algorithm Used: Random Forest Regressor
Data Split: 80% training, 20% testing
Hyperparameter Tuning: Used GridSearchCV for best performance

5. Model Evaluation
Metrics Used:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score

6. Results & Conclusion
Achieved high accuracy in predicting DON concentration.
PCA helped reduce dimensionality without losing performance.
Future improvements: Try deep learning models for better generalization.
