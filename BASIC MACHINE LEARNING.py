import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# --- Data Generation ---
num_rows = 100
names = [f"Student_{i}" for i in range(1, num_rows + 1)]
attendance_percentage = np.random.randint(70, 101, size=num_rows)
ages = np.random.randint(15, 20, size=num_rows)
marks = np.random.randint(40, 101, size=num_rows)

df = pd.DataFrame({
    'Name': names,
    'Attendance_Percentage': attendance_percentage,
    'Age': ages,
    'Marks': marks
})

# --- Exploratory Data Analysis (EDA) - Scatter Plots ---
plt.figure(figsize=(8, 6))
plt.scatter(df['Attendance_Percentage'], df['Marks'])
plt.xlabel('Attendance Percentage')
plt.ylabel('Marks')
plt.title('Scatter Plot of Attendance vs. Marks')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['Attendance_Percentage'], df['Marks'], c=df['Age'], cmap='viridis')
plt.xlabel('Attendance Percentage')
plt.ylabel('Marks')
plt.title('Scatter Plot of Attendance vs. Marks (Colored by Age)')
plt.colorbar(label='Age')
plt.grid(True)
plt.show()

# --- Data Preparation ---
X = df[['Attendance_Percentage']].values  # Convert to NumPy array for scikit-learn
y = df['Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Scaling (Important for some models like SVR and KNN) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler fitted on the training data

# --- Model Training and Evaluation ---
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (degree 2)": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(kernel='rbf'),  # Requires scaled data
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5)  # Requires scaled data
}

results = {}

for name, model in models.items():
    if name in ["Support Vector Regressor", "K-Nearest Neighbors Regressor"]:
        X_train_to_fit = X_train_scaled
        X_test_to_predict = X_test_scaled
    else:
        X_train_to_fit = X_train
        X_test_to_predict = X_test
    
    model.fit(X_train_to_fit, y_train)
    y_pred = model.predict(X_test_to_predict)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Cross-validation for more robust evaluation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')  # Use original X, y for cross-validation
    avg_cv_r2 = cv_scores.mean()

    results[name] = {"MSE": mse, "R-squared": r2, "RMSE": rmse, "CV_R2": avg_cv_r2, "predictions": y_pred, "model": model}

# --- Model Comparison and Visualization ---
for name, metrics in results.items():
    print(f"--- {name} ---")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"R-squared: {metrics['R-squared']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"Average CV R-squared: {metrics['CV_R2']:.4f}")  # Print cross-validated R-squared

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, label="Actual Data")
    plt.plot(X_test, metrics['predictions'], color='red', label="Predictions")
    plt.xlabel("Attendance Percentage")
    plt.ylabel("Marks")
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Best Model Selection ---
best_model_name = max(results, key=lambda k: results[k]['CV_R2'])  # Choose based on CV R-squared
best_model = results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")

# --- Prediction with Best Model ---
new_attendance = np.array([[85]])
if best_model_name in ["Support Vector Regressor", "K-Nearest Neighbors Regressor"]:
    new_attendance_scaled = scaler.transform(new_attendance)
    predicted_marks = best_model.predict(new_attendance_scaled)
else:
    predicted_marks = best_model.predict(new_attendance)

print(f"Predicted Marks for 85% attendance ({best_model_name}): {predicted_marks[0]:.2f}")
