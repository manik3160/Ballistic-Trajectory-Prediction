import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate_models(data_path, output_dir):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Features: Velocity, Angle, Gravity
    # Targets: Range (Noisy), Max Height (Noisy)
    X = df[['velocity', 'angle_deg', 'gravity']]
    y_range = df['range_noisy']
    y_height = df['max_height_noisy']
    
    # Split the data
    X_train, X_test, y_range_train, y_range_test, y_height_train, y_height_test = train_test_split(
        X, y_range, y_height, test_size=0.2, random_state=42
    )
    
    metrics = {}
    models = {}
    
    # --- Linear Regression ---
    print("\nTraining Linear Regression models...")
    lr_range = LinearRegression()
    lr_range.fit(X_train, y_range_train)
    
    lr_height = LinearRegression()
    lr_height.fit(X_train, y_height_train)
    
    # Evaluate LR
    lr_range_preds = lr_range.predict(X_test)
    lr_height_preds = lr_height.predict(X_test)
    
    metrics['Linear_Regression'] = {
        'Range': {
            'RMSE': float(np.sqrt(mean_squared_error(y_range_test, lr_range_preds))),
            'MAE': float(mean_absolute_error(y_range_test, lr_range_preds)),
            'R2': float(r2_score(y_range_test, lr_range_preds))
        },
        'Max_Height': {
            'RMSE': float(np.sqrt(mean_squared_error(y_height_test, lr_height_preds))),
            'MAE': float(mean_absolute_error(y_height_test, lr_height_preds)),
            'R2': float(r2_score(y_height_test, lr_height_preds))
        }
    }
    
    models['lr_range'] = lr_range
    models['lr_height'] = lr_height
    
    # --- Random Forest ---
    print("Training Random Forest models...")
    rf_range = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_range.fit(X_train, y_range_train)
    
    rf_height = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_height.fit(X_train, y_height_train)
    
    # Evaluate RF
    rf_range_preds = rf_range.predict(X_test)
    rf_height_preds = rf_height.predict(X_test)
    
    metrics['Random_Forest'] = {
        'Range': {
            'RMSE': float(np.sqrt(mean_squared_error(y_range_test, rf_range_preds))),
            'MAE': float(mean_absolute_error(y_range_test, rf_range_preds)),
            'R2': float(r2_score(y_range_test, rf_range_preds))
        },
        'Max_Height': {
            'RMSE': float(np.sqrt(mean_squared_error(y_height_test, rf_height_preds))),
            'MAE': float(mean_absolute_error(y_height_test, rf_height_preds)),
            'R2': float(r2_score(y_height_test, rf_height_preds))
        }
    }
    
    models['rf_range'] = rf_range
    models['rf_height'] = rf_height
    
    print("\nModel Evaluation Metrics:")
    print(json.dumps(metrics, indent=4))
    
    # Save outputs
    print("\nSaving models and metrics...")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        joblib.dump(model, os.path.join(output_dir, f'{name}.pkl'))
        
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Models and metrics saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__name__))
    data_file = os.path.join(base_dir, 'ballistic_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    train_and_evaluate_models(data_file, models_dir)
