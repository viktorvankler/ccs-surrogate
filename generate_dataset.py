import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# importing my reservoir class from 'reservoir_sim'
from Reservoir_sim import CO2Reservoir

# setting a random seed so my results are reproducible
np.random.seed(38)
num_samples = 1500
simulation_results = []

#LOOPING TO CREATE 1,500 RESERVOIR OBJECTS
for i in range(num_samples):
  # Generating realistic random subsurface rock & injection properties
  porosity = np.random.uniform(0.08, 0.32) # 8% to 32%
  permeability = np.random.uniform(10, 500) # 10 mD to 500mD
  thickness = np.random.uniform(15, 100) #15m to 100m
  area = np.random.uniform(5, 50) # 5 km^2 to 50 km^2
  injection_rate = np.random.uniform(500.0, 3000.0)  # 500 to 3,000 m^3/day
  viscosity = np.random.uniform(0.03, 0.08) # 0.03 to 0.08 cp

  # Instantiating a new CO2Reservoir object
  res = CO2Reservoir(
      name = f"Sim_{i}",
      porosity = porosity,
      permeability_md = permeability,
      thickness_m = thickness,
      area_km2 = area,
      depth_m = 1967,
      viscosity_cp = viscosity)
  # Running the class methods
  capacity_mt = res.calculate_storage_capacity_mt()
  pressure_bar = res.calculate_pressure_buildup_bar(injection_rate_m3_day=injection_rate)
  # Storing everything in a dictionary
  simulation_results.append({
      'porosity': porosity,
      'permeability_md': permeability,
      'thickness_m': thickness,
      "area_km2": area,
      "injection_rate_m3_day": injection_rate,
      "storage_capacity_mt": capacity_mt,
      "pressure_buildup_bar": pressure_bar,
      "viscosity_cp": viscosity})

# 2. SAVING TO CSV
df = pd.DataFrame(simulation_results)
df.to_csv("co2_simulation_data.csv", index=False)
print("Saved 1,500 simulations to co2_simulation_data.csv!")
print(df.head())

# 3. PLOTTING THE PHYSICS RELATIONSHIP (Permeability vs. Pressure Buildup)
plt.figure(figsize=(9, 5))
scatter = plt.scatter(
  df["permeability_md"],
  df["pressure_buildup_bar"],
  c=df["injection_rate_m3_day"],
  cmap="viridis",
  alpha=0.7,
)
plt.colorbar(scatter, label="Injection Rate (m³/day)")
plt.title(
  "Subsurface CO₂ Injection: Permeability vs. Wellbore Pressure Buildup"
)
plt.xlabel("Reservoir Permeability (mD)")
plt.ylabel("Pressure Buildup (Bar)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("permeability_vs_pressure.png", dpi=300)
plt.show()

# Plotting another physics relationship (thickness vs storage capacity)
plt.figure(figsize=(9, 5))
scatter = plt.scatter(
    df['thickness_m'],
    df["storage_capacity_mt"],
    c=df["porosity"],
    cmap="plasma",
    alpha=0.7,
)
plt.colorbar(scatter, label="porosity")
plt.title(
  "Porosity: Permeability vs. Wellbore Pressure Buildup"
)
plt.xlabel('Reservoir thickness (m)')
plt.ylabel('storage capacity (Mt)')
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig('thickness_vs_storage.png')
plt.show()