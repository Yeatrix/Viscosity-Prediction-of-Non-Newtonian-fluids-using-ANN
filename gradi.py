import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from joblib import Parallel, delayed
from sklearn.metrics import r2_score 

# Function to preprocess and split data
def load_and_preprocess_data(file_path, random_state=42):
    df = pd.read_excel(file_path)
    X = df[['Particle Size', 'Temperature', 'Shear Rate']]
    y = df['Viscosity']
    scaler = StandardScaler()
    X['Shear Rate'] = scaler.fit_transform(X[['Shear Rate']])
    return X, y, scaler

# Function to train and evaluate a model for a specific condition
def train_and_evaluate_condition(condition, X, y, param_grid, random_state=42):
    particle_size, temperature = condition
    indices = (X['Particle Size'] == particle_size) & (X['Temperature'] == temperature)
    X_condition = X[indices]
    y_condition = y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X_condition[['Shear Rate']], y_condition, test_size=0.2, random_state=random_state)
    
    gb_model = GradientBoostingRegressor(random_state=random_state)
    grid_search = GridSearchCV(gb_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Use the best model to predict on all data (not just test)
    y_pred = best_model.predict(X_condition[['Shear Rate']])
    
    return (particle_size, temperature), best_model, y_pred, indices

# Main function to train and evaluate models
def train_and_evaluate_model(file_path, random_state=42):
    X, y, scaler = load_and_preprocess_data(file_path, random_state)
    unique_conditions = X[['Particle Size', 'Temperature']].drop_duplicates().values
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.004, 0.045, 0.1],
        'subsample': [0.9, 1.0]
    }
    
    # Use joblib to parallelize the training process
    results = Parallel(n_jobs=12)(
        delayed(train_and_evaluate_condition)(condition, X, y, param_grid, random_state)
        for condition in unique_conditions
    )
    
    # Collect results
    y_pred = np.zeros_like(y)
    for (particle_size, temperature), best_model, condition_y_pred, indices in results:
        y_pred[indices] = condition_y_pred
    
    overall_mse = mean_squared_error(y, y_pred)
    overall_rmse = np.sqrt(overall_mse)
    overall_mape = mean_absolute_percentage_error(y, y_pred) * 100
    rsq = r2_score(y, y_pred)
    
    print(f"Overall Mean Squared Error: {overall_mse:.2f}")
    print(f"Overall Root Mean Squared Error: {overall_rmse:.2f}")
    print(f"Overall Mean Absolute Percentage Error: {overall_mape:.2f}%")
    print(f"Overall R-square Value: {rsq}%")
    
    results_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    print(results_df.head(10))
    
    return overall_mse, overall_rmse, overall_mape, results_df

# Example usage
file_path = "C:/Users/laksh/OneDrive/Desktop/Sura/Book1.xlsx"
# file_path = "C:\Users\laksh\OneDrive\Desktop\Sura\dataset_updated.xlsx"  # Replace with the actual file path
random_state = 42
overall_mse, overall_rmse, overall_mape, results_df = train_and_evaluate_model(file_path, random_state)
