import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

######################################## Functions defined ##########################################

def thorpe_attenuation(frequency, distance, temperature=30.0, salinity=1.0):
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

def distance_3D(point_from, point_to):
    distance = np.sqrt((point_from[0] - point_to[0])**2 +
                   (point_from[1] - point_to[1])**2 +
                   (point_from[2] - point_to[2])**2)
    return distance

#####################################################################################################

# Constants
tank_length = 10  # Horizontal length of the tank in meters
tank_depth = 5    # Depth of the tank in meters
sound_speed = 1500  # Speed of sound in water in m/s
sampling_rate = 45000  # Sample rate in Hz
simulation_duration = 0.5  # Simulation duration in seconds
tx_freq = 45 # 10kHz transmitter frequency

# Source and receiver positions
source_position = (1, 2, 2)  # (x, y, z) in meters
receiver_position = (9, 2, 2)  # (x, y, z) in meters

# Calculate the distance between the source and receiver
distance = distance_3D(source_position, receiver_position)

# Calculate the time it takes for sound to travel from source to receiver
transit_time = distance / sound_speed
# Create a time vector

time = np.linspace(0, 20*transit_time, int(sampling_rate * transit_time))


# Generate a simple signal to simulate the source
source_signal = 500*np.sin(2 * np.pi * tx_freq * time)

# Simulate the propagation of the signal through the water with attenuation
attenuation_db = thorpe_attenuation(tx_freq, distance)
print(10**(-attenuation_db / 10.0))
received_signal = source_signal * 10**(-attenuation_db / 10.0)

# Plot the transmitted and received signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, source_signal)
plt.title('Transmitted Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(time, received_signal)
plt.title('Received Signal with Attenuation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()

# Create a 3D plot to show the propagation path
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot source and receiver positions
ax.scatter(source_position[0], source_position[1], source_position[2], c='b', marker='o', label='Source')
ax.scatter(receiver_position[0], receiver_position[1], receiver_position[2], c='r', marker='o', label='Receiver')

# Calculate intermediate positions along the propagation path
num_points = 100
x_path = np.linspace(source_position[0], receiver_position[0], num_points)
y_path = np.linspace(source_position[1], receiver_position[1], num_points)
z_path = np.linspace(source_position[2], receiver_position[2], num_points)

# Plot the propagation path
ax.plot(x_path, y_path, z_path, c='g', label='Propagation Path')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Acoustic Signal Propagation Path')
ax.legend()

plt.show()
