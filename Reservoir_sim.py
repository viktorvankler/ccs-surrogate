import numpy as np

class CO2Reservoir:

  def __init__(self, name, porosity, permeability_md, thickness_m, area_km2, depth_m, viscosity_cp):
    self.name = name
    self.porosity = porosity  # Fraction (e.g., 0.20 for 20%)
    self.permeability = permeability_md  # Millidarcies (mD)
    self.thickness = thickness_m  # Meters (m)
    self.area = area_km2  # Square kilometers (km^2)
    self.depth = depth_m
    self.viscosity = viscosity_cp

  def calculate_storage_capacity_mt(self, co2_density=700, efficiency=0.04):
    """
        Calculates theoretical CO2 storage capacity in Megatonnes (Mt).
        Formula: Area * Thickness * Porosity * CO2_Density * Storage_Efficiency
        """
    area_m2 = self.area * 1e6  # Convert km^2 to m^2
    pore_volume_m3 = area_m2 * self.thickness * self.porosity
    mass_kg = pore_volume_m3 * co2_density * efficiency
    return mass_kg / 1e9  # Convert kg to Megatonnes

  def calculate_pressure_buildup_bar(
      self, injection_rate_m3_day,
  ):
    """
        Uses Steady-State Radial Darcy's Law to estimate pressure buildup at the wellbore.
        """
    # Convert standard field units to SI units for clean math
    q_m3_s = injection_rate_m3_day / 86400  # m^3/day to m^3/s
    k_m2 = self.permeability * 9.869233e-16  # mD to m^2
    mu_pa_s = self.viscosity * 0.001  # cP to Pa.s

    r_e = 5000.0  # Reservoir outer drainage radius (5,000 m)
    r_w = 0.1  # Wellbore radius (0.1 m)

    # Radial Darcy Equation: delta_P = (q * mu * ln(re/rw)) / (2 * pi * k * h)
    delta_p_pa = (q_m3_s * mu_pa_s * np.log(r_e / r_w)) / (
        2 * np.pi * k_m2 * self.thickness
    )
    return delta_p_pa / 1e5  # Convert Pascals to Bar
  def calculate_hydrostatic_pressure_bar(self):
    hydro_p_bar = self.depth * 9.81 * 1000 #the formular for hydrostatic pressure = density*gravity*height
    return hydro_p_bar / 1e5   #convert from pascal to bar


if __name__ == "__main__":

  north_sea_site = CO2Reservoir("Site Alpha", 0.22,150,45,25,2000, viscosity_cp= 0.04)
  tight_sand_site = CO2Reservoir("Site Beta",0.12,25,30,15,1480, viscosity_cp = 0.06)


  for site in [north_sea_site, tight_sand_site]:
    cap = site.calculate_storage_capacity_mt()
    dp = site.calculate_pressure_buildup_bar(injection_rate_m3_day=1500)
    hydp = site.calculate_hydrostatic_pressure_bar()
    print(f"--- {site.name} ---")
    print(f"Estimated CO2 Capacity: {cap:.2f} Megatonnes")
    print(f"Wellbore Pressure Buildup: {dp:.2f} Bar\n")
    print(f"Hydrostatic Pressure: {hydp:.2f} Bar\n")

