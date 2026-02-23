import pandas as pd
import numpy as np
import os
import joblib
import json

def calculate_physics_range(velocity, angle_deg, gravity):
    angle_rad = np.radians(angle_deg)
    return (velocity**2 * np.sin(2 * angle_rad)) / gravity

def run_sensitivity_analysis(models_dir, output_dir):
    print("Running sensitivity analysis...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the Random Forest range model (best performing)
    rf_range = joblib.load(os.path.join(models_dir, 'rf_range.pkl'))
    
    # Baseline scenario
    base_v = 800.0 # m/s
    base_a = 45.0  # degrees (max range)
    base_g = 9.81  # m/s^2
    
    base_range_physics = calculate_physics_range(base_v, base_a, base_g)
    
    # Model prediction expects a DataFrame
    input_df = pd.DataFrame({'velocity': [base_v], 'angle_deg': [base_a], 'gravity': [base_g]})
    base_range_pred = rf_range.predict(input_df)[0]
    
    print(f"\nBaseline (V={base_v}, Angle={base_a}, G={base_g}):")
    print(f"  Physics Range: {base_range_physics:.2f} m")
    print(f"  RF Predicted Range: {base_range_pred:.2f} m")
    
    # Analyze angle sensitivity
    angle_errors = np.linspace(-2.0, 2.0, 21) # -2 to +2 degrees error
    
    results = []
    
    for err in angle_errors:
        test_a = base_a + err
        
        phys_range = calculate_physics_range(base_v, test_a, base_g)
        
        # Predict with model
        test_df = pd.DataFrame({'velocity': [base_v], 'angle_deg': [test_a], 'gravity': [base_g]})
        pred_range = rf_range.predict(test_df)[0]
        
        # Calculate impact of error
        phys_diff = phys_range - base_range_physics
        pred_diff = pred_range - base_range_pred
        
        results.append({
            'angle_error_deg': err,
            'test_angle': test_a,
            'physics_range': phys_range,
            'physics_diff': phys_diff,
            'predicted_range': pred_range,
            'predicted_diff': pred_diff
        })
        
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'sensitivity_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\nSaved sensitivity results to {output_path}")
    
    # Summary of findings
    max_err_idx = results_df['physics_diff'].abs().idxmax()
    max_impact = results_df.loc[max_err_idx]
    
    summary = {
        'baseline_physics': base_range_physics,
        'baseline_prediction': base_range_pred,
        'max_impact_of_2_deg_error': {
            'angle_error': max_impact['angle_error_deg'],
            'physics_range_change': max_impact['physics_diff'],
            'predicted_range_change': max_impact['predicted_diff']
        }
    }
    
    with open(os.path.join(output_dir, 'sensitivity_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("\nSensitivity Analysis Summary:")
    print(json.dumps(summary, indent=4))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__name__))
    models_dir = os.path.join(base_dir, 'models')
    analysis_dir = os.path.join(base_dir, 'analysis')
    
    run_sensitivity_analysis(models_dir, analysis_dir)
