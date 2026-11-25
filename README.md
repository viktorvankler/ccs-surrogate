Subsurface CO₂ Storage Simulator & Machine Learning Surrogate

An Object-Oriented Python pipeline that simulates subsurface carbon dioxide (CO₂) storage capacity and injection pressure buildup using analytical reservoir engineering equations, and benchmarks three Machine Learning surrogate architectures—Random Forest, HistGradientBoosting, and an Artificial Neural Network (MLP)—to predict wellbore pressure buildup instantaneously. 

1. Engineering Problem & Motivation

Numerical reservoir simulations solve coupled fluid-flow equations across geological formations, which becomes computationally expensive when screening thousands of injection scenarios. In Carbon Capture and Storage (CCS), engineers must rapidly evaluate two competing constraints: 

a. Volumetric Storage Capacity: Ensuring sufficient pore volume to trap supercritical CO₂.
b. Wellbore Pressure Buildup (Bar): Ensuring injection pressures remain safely below caprock fracture limits, particularly in low-permeability formations.

This project bridges classical petroleum reservoir physics with data-driven machine learning by generating a synthetic 1,500-reservoir dataset and evaluating ensemble tree models against a multi-layer perceptron neural network to approximate steady-state radial Darcy flow. 

2. Project Architecture

Reservoir_sim.py: Defines the CO2Reservoir Python class encapsulating rock and fluid properties (porosity, permeability_md, thickness_m, area_km2, depth_m, viscosity_cp) alongside analytical methods for volumetric CO₂ capacity, hydrostatic pressure, and steady-state radial Darcy pressure buildup.
generate_dataset.py: Instantiates 1,500 randomized reservoir scenarios across realistic subsurface ranges, executes the physics methods, exports co2_simulation_data.csv, and generates exploratory physics visualizations.
ml_surrogate.py: Splits the simulation dataset (85% training / 15% unseen test set), applies StandardScaler feature normalization for neural network training, and benchmarks three distinct surrogate models using R² and Mean Absolute Error (MAE).

3. Physics Exploration Results

Permeability vs. Wellbore Pressure Buildup

*Consistent with radial Darcy flow (ΔP ∝ 1/k), pressure buildup increases nonlinearly as permeability drops below 50 mD, with higher injection rates amplifying wellbore pressure spikes.* 

Reservoir Thickness vs. Storage Capacity

*Theoretical CO₂ storage capacity scales linearly with reservoir thickness and areal extent.* 

4. Machine Learning Surrogate Benchmark

Model ArchitectureFamilyR² Score (Unseen Test Set)Mean Absolute Error (MAE)
HistGradientBoosting Regressor
Gradient Boosted Trees 0.9076, 0.485 Bar
Random Forest Regressor (n_estimators=200)
Bagging Ensemble 0.8828, 0.570 Bar
Multi-Layer Perceptron (64 × 64)
Artificial Neural Network 0.8784, 0.527 Bar

 

Key Technical Takeaways

Continuous Curve Fit vs. Extreme Outlier Extrapolation: As seen in the side-by-side comparison above, the 64 × 64 MLP Neural Network fits the smooth Darcy physics curve visibly tighter across the core 0–15 Bar regime than tree-based models, achieving a lower MAE (0.527 Bar) than Random Forest (0.570 Bar). However, a single extreme low-permeability outlier (~36 Bar) penalized the MLP's squared-error R² score (0.8784 vs. 0.9076 for HistGradientBoosting).
Future Work: Applying log-uniform sampling across low permeabilities (10–30 mD), scaling the dataset to 10,000+ samples, and extending the analytical solver to 2D transient multiphase flow (CO₂-brine displacement) using Physics-Informed Neural Networks (PINNs).

5. How to Run

bash

pip install numpy pandas matplotlib scikit-learn
python generate_dataset.py
python ml_surrogate.py

Use code with caution.

Copy code text to clipboard.
