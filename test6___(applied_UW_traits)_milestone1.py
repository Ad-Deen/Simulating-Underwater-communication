import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

######################################## Functions defined ##########################################

def pathloss(frequency, distance, depth , temperature=30.0, salinity=35.0, geometric_spread = 2):
    """
    Calculate the acoustic attenuation in seawater using the Fisher & Simmons model.

    Even though the Fisher & Simmons model allows us to
    model the effects of varying depth, for the model to hold true
    this depth should not be greater than 8km. Furthermore, there
    is a restriction that water salinity must be restricted to the
    global observed average of 35 ppt and the acidy level to a pH
    value of 8.
    Taking into account these restrictions we plot the absoprtion
    coefficient as predicted by the Fisher & Simmons model

    The coefficients A1, A2 and A3 represent the effects
    of temperature on signal absorption, while P1, P2 and P3
    represent the effects of depth and f1 and f2 represent the
    relaxation frequencies introduced due to the absorption caused
    by boric acid and magnesium sulphate
    Returns:
    - attenuation (float): Attenuation in dB.
    """

    # Constants for the Fisher & Simmons model
    l = distance/1000            #killometers
    f = frequency/1000           #kHz
    d = depth/1000               #kilometers meters
    t = temperature         #degree Celsius
    k = geometric_spread    #constant
    a1 = 1.03e-8 + (2.36e-10)*t - (5.22e-12)*t**2               
    a2 = 5.62e-8 + (7.52e-10)*t
    a3 = (55.9 - 2.37*t + (4.77e-2)*t**2 - (3.48e-4)*t**3)*10**(-15)
    f1 = (1.32e3)*(t + 273.1)*math.e**(-1700/(t + 273.1))
    f2 = (1.55e7)*(t + 273.1)*math.e**(-3052/(t + 273.1))
    p1 = 1
    p2 = 1 - (10.3e-5)*d + (3.7e-9)*d**2
    p3 = 1 - (3.84e-5)*d + (7.57e-10)*d**2
    # Calculating absorption for frequency component based on temperature
    # depth and frequency
    frequency_absorption = (a1*p1*f1*f**2/(f1**2+f**2) + a2*p2*f2*f**2/(f2**2+f**2) + a3*p3*f**2)
    
    # Calculating pathloss using Fisher & Simmons model in db/m
    # pathloss = math.exp((k*10*math.log(l)+l*frequency_absorption)/10)
    pathloss = l**(k) * math.exp((l * frequency_absorption)/10)

    return 1+pathloss

def ambient_noise(freq):
    '''
    The
    four sources for ambient noise are turbulence, shipping, waves
    and thermal noise
    
    Each type of noise becomes
    dominant within a certain range of frequencies; turbulence
    noise has an effect in very low frequencies with f < 10 Hz.
    Shipping noise is most prominent in the frequency range of 10
    Hz > f > 100 Hz the shipping factor s represents the amount
    of activity with 0 being none and 1 being very heavy activity.
    In most cases moderate shipping activity can be modelled
    using s = 0.5. Noise caused by wind-driven waves is most
    dominant in the 100 Hz < f < 100kHz  range while thermal
    noise is only effective when f>100 kHz.
    
     Surface motion, caused by wind-driven waves (w is the wind speed in m/s) is the major factor contributing to the noise in the frequency region 100 Hz -
    100 kHz (which is the operating region used by the majority of acoustic systems)
    averge wind speed over the water surface w= 4.47 m/s
    '''
    w = 4.47
    noise = 10**((50+7.5*w**(0.5)+20*math.log(freq)-40*math.log(freq+0.4))/10)
    return noise 

def gradient_decay():
    global initial
    while(initial > 0.05):
        initial -= 0.004
        return initial

def distance_3D(point_from, point_to):
    distance = np.sqrt((point_from[0] - point_to[0])**2 +
                   (point_from[1] - point_to[1])**2 +
                   (point_from[2] - point_to[2])**2)
    return distance

def depth_calc(point_from, point_to):
    depth = np.sqrt((point_from[2] - point_to[2])**2)
    return depth

#####################################################################################################

# Constants
initial = 1.0
tank_length = 10  # Horizontal length of the tank in meters
tank_depth = 5    # Depth of the tank in meters
sound_speed = 1500  # Speed of sound in water in m/s
sampling_rate = 45000  # Sample rate in Hz
simulation_duration = 0.5  # Simulation duration in seconds
tx_freq = 30 # Hz transmitter frequency
noise_amplifier = 1 #Ambient noise amplifier
step = 100       #samples to subdivide

# Source and receiver positions
source_position = (1, 2, 2)  # (x, y, z) in meters
receiver_position = (2000, 2, 2)  # (x, y, z) in meters

# Calculate the distance between the source and receiver
distance = distance_3D(source_position, receiver_position)

# Calculate the time it takes for sound to travel from source to receiver
transit_time = distance / sound_speed
# Create a time vector

time = np.arange(0.0,  transit_time, 1/(sound_speed*step) )
distance_coordinates = []
for t in time:
    distance_coordinates.append(t*1500)

# Generate a simple signal to simulate the source
# dist = 0 
recieved_signal = []
source_signal = []
# Simulate the propagation of the signal through the water with attenuation
for t in time:
    source_pulse = 500 * np.sin(2 * np.pi * tx_freq * t)
    source_signal.append(source_pulse)
    absorption = pathloss(tx_freq, 1500*t , depth_calc(source_position,receiver_position) )
    noise_amp = ambient_noise(tx_freq)
    # print(f"absorption = {100-(1/absorption)*100} % & distance covered {1500*t} m")
    # print(f"ambient = {noise_amp}")
    # print(f"distance covered {dist}")
    external_noise = random.uniform(noise_amplifier*-noise_amp, noise_amplifier*noise_amp)
    recieved_signal.append(source_pulse * (1/absorption) + external_noise)

# Plot the transmitted and received signals
plt.figure(figsize=(20, 6))
plt.subplot(2, 1, 1)
plt.plot(time, source_signal)
plt.title('Transmitted Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(distance_coordinates, recieved_signal)
plt.title('Received Signal with Attenuation')
plt.xlabel('Distance propagated (m)')
plt.ylabel('Amplitude')

plt.tight_layout()

# # Create a 3D plot to show the propagation path
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Plot source and receiver positions
# ax.scatter(source_position[0], source_position[1], source_position[2], c='b', marker='o', label='Source')
# ax.scatter(receiver_position[0], receiver_position[1], receiver_position[2], c='r', marker='o', label='Receiver')

# # Calculate intermediate positions along the propagation path
# num_points = 100
# x_path = np.linspace(source_position[0], receiver_position[0], num_points)
# y_path = np.linspace(source_position[1], receiver_position[1], num_points)
# z_path = np.linspace(source_position[2], receiver_position[2], num_points)

# # Plot the propagation path
# ax.plot(x_path, y_path, z_path, c='g', label='Propagation Path')

# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('Acoustic Signal Propagation Path')
# ax.legend()

plt.show()
