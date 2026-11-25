import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# LOADING SIMULATION DATA
df = pd.read_csv('co2_simulation_data.csv')

# DEFINING INPUTS (x) AND TARGET (y)
input_columns = [
    'porosity',
    'permeability_md',
    'thickness_m',
    'area_km2',
    'injection_rate_m3_day',
    'viscosity_cp',
]
x = df[input_columns]
y = df['pressure_buildup_bar']

# SPLITTING INTO STUDY DATA (85%) AND EXAM DATA (15%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=38
)

# SCALING INPUT FEATURES FOR THE NEURAL NETWORK

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. CREATE AND TRAIN ALL THREE MODELS
# Model 1: Random Forest (Bagging Ensemble)
rfr_model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=38)
rfr_model.fit(x_train, y_train)

# Model 2: HistGradientBoosting (Boosting Ensemble)
hgb_model = HistGradientBoostingRegressor(random_state=38)
hgb_model.fit(x_train, y_train)

# Model 3: Multi-Layer Perceptron (Artificial Neural Network: 2 hidden layers of 64 neurons)
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 64), max_iter=2000, random_state=38
)
nn_model.fit(x_train_scaled, y_train)

# 5. TESTING ALL THREE MODELS ON UNSEEN DATA (225 RESERVOIRS)
rfr_predictions = rfr_model.predict(x_test)
hgb_predictions = hgb_model.predict(x_test)
nn_predictions = nn_model.predict(x_test_scaled)

# 6. GRADING ALL THREE MODELS
rfr_r2 = r2_score(y_test, rfr_predictions)
rfr_mae = mean_absolute_error(y_test, rfr_predictions)

hgb_r2 = r2_score(y_test, hgb_predictions)
hgb_mae = mean_absolute_error(y_test, hgb_predictions)

nn_r2 = r2_score(y_test, nn_predictions)
nn_mae = mean_absolute_error(y_test, nn_predictions)

print('=== SURROGATE MODEL BENCHMARK RESULTS ===')
print(
    f'1. Random Forest        -> R² Score: {rfr_r2:.4f} | MAE: {rfr_mae:.3f}'
    ' Bar'
)
print(
    f'2. HistGradientBoosting -> R² Score: {hgb_r2:.4f} | MAE: {hgb_mae:.3f}'
    ' Bar'
)
print(
    f'3. Neural Network (MLP) -> R² Score: {nn_r2:.4f} | MAE: {nn_mae:.3f} Bar'
)

comparison_nn = pd.DataFrame({
    'Actual_Pressure_Buildup_Bar': y_test.values[:10],
    'RFR_Predicted_Bar': rfr_predictions[:10],
    'HGB_Predicted_Bar': hgb_predictions[:10],
    'NN_Predicted_Bar': nn_predictions[:10],
})
print('\nFirst 10 Test Predictions Comparison:')
print(comparison_nn)

# 7. PLOTTING ACTUAL VS. PREDICTED PRESSURE (SIDE-BY-SIDE COMPARISON)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left Plot: HistGradientBoosting
scatter1 = axes[0].scatter(
    y_test,
    hgb_predictions,
    c=x_test['permeability_md'],
    cmap='plasma',
    alpha=0.7,
)
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2,
    label='Perfect Prediction (y = x)',
)
axes[0].set_title(f'HistGradientBoosting (R² = {hgb_r2:.3f}, MAE = {hgb_mae:.2f} Bar)')
axes[0].set_xlabel('Actual Physics Pressure Buildup (Bar)')
axes[0].set_ylabel('Predicted Pressure Buildup (Bar)')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Right Plot: Neural Network (MLP)
scatter2 = axes[1].scatter(
    y_test,
    nn_predictions,
    c=x_test['permeability_md'],
    cmap='plasma',
    alpha=0.7,
)
axes[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2,
    label='Perfect Prediction (y = x)',
)
axes[1].set_title(f'Neural Network MLP (R² = {nn_r2:.3f}, MAE = {nn_mae:.2f} Bar)')
axes[1].set_xlabel('Actual Physics Pressure Buildup (Bar)')
axes[1].set_ylabel('Predicted Pressure Buildup (Bar)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.colorbar(scatter2, ax=axes, label='Permeability (mD)')
plt.savefig('ml_prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()
