import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json

def plot_results(data_path, models_dir, analysis_dir, plots_dir):
    print("Generating visualizations...")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(data_path)
    
    # Load Models
    lr_range = joblib.load(os.path.join(models_dir, 'lr_range.pkl'))
    rf_range = joblib.load(os.path.join(models_dir, 'rf_range.pkl'))
    
    # 1. Trajectory Plot (Classical Mechanics vs Machine Learning)
    print("  Creating comparison scatter plot...")
    plt.figure(figsize=(10, 6))
    
    # Take a sample for visualization clarity
    sample_df = df.sample(n=200, random_state=42)
    
    X_sample = sample_df[['velocity', 'angle_deg', 'gravity']]
    y_actual = sample_df['range_noisy']
    
    rf_preds = rf_range.predict(X_sample)
    lr_preds = lr_range.predict(X_sample)
    
    plt.scatter(y_actual, rf_preds, alpha=0.6, label='Random Forest', color='blue', marker='o')
    plt.scatter(y_actual, lr_preds, alpha=0.6, label='Linear Regression', color='red', marker='x')
    
    # Ideal line
    min_val = min(y_actual.min(), rf_preds.min(), lr_preds.min())
    max_val = max(y_actual.max(), rf_preds.max(), lr_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Impact Range (m)')
    plt.ylabel('Predicted Impact Range (m)')
    plt.title('Model Predictions vs Actual Physics Simulation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
    plt.close()

    # 2. Residuals Plot
    print("  Creating residuals plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(y_actual - rf_preds, kde=True, color='blue', label='Random Forest', alpha=0.5)
    sns.histplot(y_actual - lr_preds, kde=True, color='red', label='Linear Regression', alpha=0.5)
    plt.axvline(0, color='k', linestyle='dashed', linewidth=1)
    
    plt.xlabel('Prediction Error (m)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Residuals)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_distribution.png'))
    plt.close()

    # 3. Sensitivity Analysis Plot
    print("  Creating sensitivity plot...")
    sens_df = pd.read_csv(os.path.join(analysis_dir, 'sensitivity_results.csv'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(sens_df['angle_error_deg'], sens_df['physics_diff'], 
             'k-', linewidth=2, label='Physics Formula (True Eq)')
    plt.plot(sens_df['angle_error_deg'], sens_df['predicted_diff'], 
             'b--', linewidth=2, label='Random Forest Model')
    
    plt.xlabel('Launch Angle Error (Degrees)')
    plt.ylabel('Change in Impact Range (m)')
    plt.title('Sensitivity to Angles: Physics vs ML Model')
    plt.axhline(0, color='gray', linestyle=':')
    plt.axvline(0, color='gray', linestyle=':')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'angle_sensitivity.png'))
    plt.close()

    print(f"All visualizations saved to {plots_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__name__))
    data_file = os.path.join(base_dir, 'ballistic_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    analysis_dir = os.path.join(base_dir, 'analysis')
    plots_dir = os.path.join(base_dir, 'plots')
    
    plot_results(data_file, models_dir, analysis_dir, plots_dir)
