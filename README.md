# Ballistic Trajectory Prediction and Impact Analysis using Machine Learning

## Overview
This project simulates ballistic data using principles of classical mechanics and utilizes machine learning models to predict projectile impact distance and maximum trajectory height. It serves as a proof-of-concept for the rapid estimation of trajectory characteristics under varying launch conditions, including simulated environmental noise. A core element is the sensitivity analysis to determine how launch angle errors propagate through physical formulation versus learned model bounds.

This framework aligns with modeling and simulation needs in weapon performance testing, validating trajectory computations, and defense-related analytical applications.

## Key Features
1. **Synthetic Data Generation**: Simulates projectile motion and creates features such as initial velocity, launch angle, and local gravity, adding environmental noise to impact range and maximum height.
2. **Machine Learning Models**: Trains and compares a baseline Linear Regression model against a Random Forest Regressor to capture the non-linear relationship of the variables.
3. **Error Sensitivity Analysis**: Analyzes and quantifies the change in impact range when introducing small errors (e.g., ±2 degrees) in the launch angle.
4. **Visualization & Reporting**: Automatically generates plots comparing model performance and a summary PDF report.

## Project Structure
```
Ballistic-Trajectory-Analysis/
│
├── data_generation.py         # Simulates realistic ballistic data
├── model_training.py          # Trains Linear Regression and Random Forest models
├── sensitivity_analysis.py    # Conducts angle error sensitivity analysis
├── results_visualization.py   # Generates comparative tracking and error plots
├── generate_report.py         # Compiles results into a PDF report
├── README.md                  # Project documentation
│
├── ballistic_dataset.csv      # Generated dataset (output)
├── models/                    # Saved models and metrics (output)
├── analysis/                  # Sensitivity results (output)
├── plots/                     # Generated charts (output)
└── report.pdf                 # Final project report (output)
```

## Setup and Execution

### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn fpdf joblib
  ```

### Running the Project
The pipeline is divided into distinct steps. Execute them in the following order:

1. **Generate Data**
   ```bash
   python data_generation.py
   ```
   *Creates `ballistic_dataset.csv`.*

2. **Train Models**
   ```bash
   python model_training.py
   ```
   *Trains the models, evaluating performance (RMSE, R2), and saves them to the `models/` directory.*

3. **Run Error & Sensitivity Analysis**
   ```bash
   python sensitivity_analysis.py
   ```
   *Analyzes how a ±2 degree shift in launch angle impacts the range. Saves results to the `analysis/` directory.*

4. **Visualize Results**
   ```bash
   python results_visualization.py
   ```
   *Generates performance and residual plots, saving them to the `plots/` directory.*

5. **Generate PDF Report**
   ```bash
   python generate_report.py
   ```
   *Compiles the findings into `report.pdf`.*

## Methodology & Physics Background
The simulation data is derived from the classical mechanics equations for parabolic flight in a vacuum (prior to adding noise):

- **Range ($R$)**: The horizontal distance traveled.
  $$ R = \frac{v^2 \sin(2\theta)}{g} $$
- **Maximum Height ($H$)**: The peak altitude reached.
  $$ H = \frac{v^2 \sin^2(\theta)}{2g} $$

Where:
- $v$ = Initial Velocity (m/s)
- $\theta$ = Launch Angle (radians)
- $g$ = Gravity (m/s²)

Machine learning is introduced to approximate these functions to demonstrate the capability of tree-based models to rapidly emulate non-linear physical functions even with data perturbations (noise).

## Model Comparison
- **Linear Regression**: Showed lower R-squared values, lacking the capability to capture the trigonometric bounds of the trajectory formulation.
- **Random Forest Regressor**: Highly accurate (e.g., $R^2 > 0.98$), closely matching the theoretical physical outputs and adapting to the simulated environmental noise effectively.

## Future Scope in Defense Simulation
- **Atmospheric Drag**: Incorporate aerodynamic drag coefficients ($C_d$) and air density variations based on altitude.
- **Meteorological Factors**: Add wind direction and velocity vectors as dynamic features.
- **Coriolis Effect**: For very long-range artillery simulation, include Earth's rotation effects.
- **Real-time Trajectory Validation**: Use trained, lightweight ML models on edge devices for near-instant impact point verification during field testing against standard firing tables.
