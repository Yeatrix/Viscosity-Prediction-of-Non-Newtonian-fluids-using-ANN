import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

# Load the data from an Excel file
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Remove duplicates and ensure the shear_rate is strictly increasing
def preprocess_data(shear_rate, viscosity):
    # Sort the data by shear rate
    sorted_indices = np.argsort(shear_rate)
    shear_rate = shear_rate[sorted_indices]
    viscosity = viscosity[sorted_indices]

    # Remove duplicate shear rate values by averaging corresponding viscosities
    unique_shear_rate, unique_indices = np.unique(shear_rate, return_index=True)
    unique_viscosity = np.array([np.mean(viscosity[shear_rate == sr]) for sr in unique_shear_rate])

    return unique_shear_rate, unique_viscosity

# Train cubic splines for each unique temperature
def train_splines(df, particle_size):
    spline_models = {}
    unique_temperatures = df['temperature'].unique()

    # Train a cubic spline for each temperature at the given particle size
    for temp in unique_temperatures:
        filtered_df = df[(df['temperature'] == temp) & (df['particle_size'] == particle_size)]
        
        if len(filtered_df) < 4:
            continue  # Skip temperatures with insufficient data

        # Extract shear rate and viscosity
        shear_rate = filtered_df['shear_rate'].values
        viscosity = filtered_df['viscosity'].values

        # Preprocess the data to ensure shear_rate is strictly increasing
        shear_rate, viscosity = preprocess_data(shear_rate, viscosity)

        # Fit a cubic spline for this temperature
        spline_model = CubicSpline(shear_rate, viscosity, bc_type='natural')
        spline_models[temp] = spline_model

    return spline_models, unique_temperatures

# Interpolate viscosity for temperatures not in the dataset
def interpolate_viscosity_for_temperature(spline_models, available_temperatures, target_temperature, shear_rate_values, tol=1e-6):
    # Check if the target temperature is within the range of available temperatures
    if target_temperature < min(available_temperatures) or target_temperature > max(available_temperatures):
        raise ValueError("The target temperature is outside the available range.")

    # Get the two nearest temperatures for interpolation
    lower_temp = max([t for t in available_temperatures if t <= target_temperature])
    upper_temp = min([t for t in available_temperatures if t >= target_temperature])

    # If the temperatures are too close, use the spline of one temperature directly
    if abs(upper_temp - lower_temp) < tol:
        print(f"The temperatures {lower_temp} and {upper_temp} are too close. Using the spline for {lower_temp}.")
        return spline_models[lower_temp](shear_rate_values)

    # Get viscosities for both lower and upper temperatures
    lower_viscosities = spline_models[lower_temp](shear_rate_values)
    upper_viscosities = spline_models[upper_temp](shear_rate_values)

    # Linearly interpolate the viscosities for the target temperature
    interp_func = interp1d([lower_temp, upper_temp], [lower_viscosities, upper_viscosities], axis=0)
    interpolated_viscosities = interp_func(target_temperature)

    return interpolated_viscosities

# Plot shear rate vs viscosity for an interpolated temperature
def plot_shear_rate_vs_viscosity_interpolated(df, spline_models, available_temperatures, target_temperature, particle_size):
    # Get a range of shear rates from the dataset
    shear_rates = np.linspace(df['shear_rate'].min(), df['shear_rate'].max(), 100)

    # Interpolate the viscosities for the target temperature
    interpolated_viscosities = interpolate_viscosity_for_temperature(spline_models, available_temperatures, target_temperature, shear_rates)

    # Plot shear rate vs viscosity
    plt.figure(figsize=(8, 6))
    plt.plot(shear_rates, interpolated_viscosities, label=f'Interpolated Temperature={target_temperature}, Particle Size={particle_size}', color='b')
    
    # Scatter the actual data points for available temperatures
    plt.scatter(df['shear_rate'], df['viscosity'], alpha=0.3, color='orange', label='Actual Data')

    # Add labels and title
    plt.xlabel('Shear Rate')
    plt.ylabel('Viscosity')
    plt.title('Shear Rate vs. Viscosity (Interpolated for Temperature)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load the data from your Excel file
    file_path = "C:\\Users\\laksh\\OneDrive\\Desktop\\Sura\\Book1.xlsx"   # Replace with your actual file path
    df = load_data(file_path)

    # Set fixed particle size
    particle_size = float(input("Enter fixed particle size: "))

    # Train splines for the available temperatures
    spline_models, available_temperatures = train_splines(df, particle_size)

    # Ask for a temperature to interpolate (not in the dataset)
    target_temperature = float(input(f"Enter a temperature (between {min(available_temperatures)} and {max(available_temperatures)}): "))

    # Plot the shear rate vs viscosity for the interpolated temperature
    plot_shear_rate_vs_viscosity_interpolated(df, spline_models, available_temperatures, target_temperature, particle_size)
