import numpy as np
import pandas as pd
import os

def generate_ballistic_data(num_samples=10000, seed=42):
    """
    Generates a synthetic ballistic dataset based on classical mechanics.
    """
    np.random.seed(seed)
    
    # Generate parameters
    # Initial velocity in m/s (e.g., artillery or rifle speeds)
    initial_velocity = np.random.uniform(300, 1000, num_samples)
    
    # Launch angle in degrees, converted to radians for numpy
    launch_angle_deg = np.random.uniform(10, 80, num_samples)
    launch_angle_rad = np.radians(launch_angle_deg)
    
    # Gravity variation (e.g., different altitudes or small anomalies)
    gravity = np.random.uniform(9.78, 9.83, num_samples)
    
    # Calculate Impact Range: R = (v^2 * sin(2 * theta)) / g
    range_actual = (initial_velocity**2 * np.sin(2 * launch_angle_rad)) / gravity
    
    # Calculate Max Height: H = (v^2 * sin^2(theta)) / (2 * g)
    max_height_actual = (initial_velocity**2 * np.sin(launch_angle_rad)**2) / (2 * gravity)
    
    # Add random noise to simulate environmental factors (wind, air resistance, etc.)
    noise_range = np.random.normal(0, range_actual * 0.05, num_samples) # 5% noise
    noise_height = np.random.normal(0, max_height_actual * 0.05, num_samples) # 5% noise
    
    range_noisy = range_actual + noise_range
    max_height_noisy = max_height_actual + noise_height
    
    # Create DataFrame
    df = pd.DataFrame({
        'velocity': initial_velocity,
        'angle_deg': launch_angle_deg,
        'gravity': gravity,
        'range_actual': range_actual,
        'max_height_actual': max_height_actual,
        'range_noisy': range_noisy,
        'max_height_noisy': max_height_noisy
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic ballistic dataset...")
    df = generate_ballistic_data()
    
    output_dir = os.path.dirname(os.path.abspath(__name__))
    output_path = os.path.join(output_dir, 'ballistic_dataset.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} samples and saved to '{output_path}'.")
