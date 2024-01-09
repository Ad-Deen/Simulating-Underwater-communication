import numpy as np

def thorpe_attenuation(frequency, distance, temperature=20.0, salinity=35.0):
    """
    Calculate the acoustic attenuation in seawater using the Thorp attenuation model.

    Parameters:
    - frequency (float): Frequency of the acoustic signal in kHz.
    - distance (float): Distance traveled by the signal in meters.
    - temperature (float, optional): Water temperature in degrees Celsius. Default is 20.0°C.
    - salinity (float, optional): Water salinity in parts per thousand (ppt). Default is 35.0 ppt.

    Returns:
    - attenuation (float): Attenuation in dB.
    """

    # Constants for the Thorp model
    a1 = 0.11
    a2 = 8.34e-3
    a3 = 1.7e-4
    a4 = 9.21e-8

    # Calculate absorption coefficient (alpha) based on temperature and salinity
    alpha = a1 * (1.0 + a2 * temperature + a3 * temperature**2 - a4 * temperature**3) * salinity

    # Calculate attenuation using the Thorp model
    attenuation = 20 * np.log10(distance) + 2 * alpha * distance / np.sqrt(frequency)

    return attenuation

# Example usage:
frequency_khz = 10.0  # Frequency in kHz
distance_meters = 1000.0  # Distance in meters
attenuation_db = thorpe_attenuation(frequency_khz, distance_meters)
print(f"Attenuation at {frequency_khz} kHz and {distance_meters} meters: {attenuation_db} dB")
