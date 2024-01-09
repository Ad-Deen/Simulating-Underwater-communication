import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

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

def noise_sim(freq):
    '''
     Surface motion, caused
    by wind-driven waves (w is the wind speed in m/s) is the major factor contributing to the noise in the frequency region 100 Hz -
    100 kHz (which is the operating region used by the majority of acoustic systems)
    averge wind speed over the water surface w= 4.47 m/s
    '''
    w = 4.47
    noise = 10**((50+7.5*w**(0.5)+20*math.log(freq)-40*math.log(freq+0.4))/10)
    return noise 
    pass
def distance_3D(point_from, point_to):
    # measures distance in 3D geometry
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
time = np.linspace(0, 10*transit_time, int(sampling_rate * transit_time))

# Create a figure for plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.title('Transmitted Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.title('Received Signal with Attenuation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()

# Create a 3D plot to show the propagation path
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Acoustic Signal Propagation Path')

# Initialize empty lists for storing updated received signals
received_signals = []
cur_dist = 0
# Simulate signal propagation over time
for t in time:
    # Update the time vector
    time_t = time - t
    print(time_t)
    cur_dist = cur_dist + distance/len(time)

    # Define the source signal at this time step
    source_signal_t = 500 * np.sin(2 * np.pi * tx_freq * time_t)

    # Simulate the propagation of the signal through the water with attenuation
    attenuation_db = thorpe_attenuation(tx_freq, distance)
    noise_amp = noise_sim(tx_freq)
    print(f"noise = {random.uniform(-noise_amp, noise_amp)}")
    received_signal_t = source_signal_t * 10**(-attenuation_db / 10.0) + 2.0*random.uniform(-noise_amp, noise_amp)

    # Update the transmitted and received signal plots
    plt.subplot(2, 1, 1)
    plt.plot(time_t, source_signal_t, label=f'Transmitted Signal (t = {t:.2f} s)')
    
    plt.subplot(2, 1, 2)
    received_signals.append(received_signal_t)
    plt.plot(time_t, received_signal_t, label=f'Received Signal (t = {t:.2f} s)')

    # Update the propagation path
    num_points = 10
    x_path_t = np.linspace(source_position[0], receiver_position[0], num_points)
    y_path_t = np.linspace(source_position[1], receiver_position[1], num_points)
    z_path_t = np.linspace(source_position[2], receiver_position[2], num_points)

    ax.plot(x_path_t, y_path_t, z_path_t, c='g', label=f'Propagation Path (t = {t:.2f} s)')

    # Pause to allow time for the plot to update
    plt.pause(0.01)

# Show the plots
plt.tight_layout()
plt.show()
