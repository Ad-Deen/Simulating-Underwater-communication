import numpy as np
import matplotlib.pyplot as plt
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

def ambient_noise(freq,shipping_crowd = 0.2):
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
    s = shipping_crowd
    w = 4.47
    noise_wind = 10**((50+7.5*w**(0.5)+20*math.log(freq)-40*math.log(freq+0.4))/10)
    noise_turbulent = math.exp(17-30*math.log(freq))
    noise_surface_wave = math.exp(40 + 20*(s - 0.5) + 26*math.log(freq) - 60*math.log(freq + 0.03))
    noise_thermal = math.exp(-15 + 20*math.log(freq))
    return (noise_wind +  noise_turbulent + noise_surface_wave * noise_thermal)*100000

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

def calculate_reflection_loss(Z1, Z2):
    # Calculate the Fresnel Reflection Coefficient
    R = ((Z2 - Z1) / (Z2 + Z1)) ** 2
    
    # Calculate the reflection loss in decibels (dB)
    reflection_loss_db = -10 * math.log10(R)
    reflection_loss = math.exp(reflection_loss_db)
    return reflection_loss

def is_point_on_point( y, z, point , tolerance= 0.1):
    """
    Check if a point is on a path defined by path_x, path_y, and path_z in a 3D grid.

    Args:
    - x point
    - y point
    - tolerance (float, optional): Tolerance for comparing coordinates. Default is 1.

    Returns:
    - bool: True if the point is on the path, False otherwise.
    """
    # Ensure that the point has the same dimension as the path

    # Check if the point is close to any point on the path within the specified tolerance
    distance = np.sqrt((point[0] - y) ** 2 + (point[1] - z) ** 2)
    if distance < tolerance:
        return True
    return False

def calculate_total_distance(path_x, path_y, path_z):
    """
    Calculate the total distance covered along a 3D path defined by path_x, path_y, and path_z.

    Args:
    - path_x (list): List of x-coordinates of the path.
    - path_y (list): List of y-coordinates of the path.
    - path_z (list): List of z-coordinates of the path.

    Returns:
    - float: The total distance covered by the path.
    """
    total_distance = 0.0

    # Iterate through the coordinates in the path and calculate the distance between consecutive points
    for i in range(1, len(path_x)):
        distance = np.sqrt((path_x[i] - path_x[i - 1])**2 + (path_y[i] - path_y[i - 1])**2 + (path_z[i] - path_z[i - 1])**2)
        total_distance += distance

    return total_distance

def merge_close_elements(input_list, threshold=2.0):
    """
    Merge elements in a list that are close to each other based on a specified threshold.

    Args:
    - input_list (list): The input list of elements.
    - threshold (float, optional): The threshold for merging elements. Default is 0.1.

    Returns:
    - list: A new list where close elements are merged together.
    """
    merged_list = []
    current_group = [input_list[0]]

    for i in range(1, len(input_list)):
        current_element = input_list[i]
        previous_element = current_group[-1]

        if abs(current_element - previous_element) <= threshold:
            current_group.append(current_element)
        else:
            merged_list.append(sum(current_group) / len(current_group))
            current_group = [current_element]

    # Add the last merged group to the result
    if current_group:
        merged_list.append(sum(current_group) / len(current_group))

    return merged_list

def find_highest_lowest_distance(path_x, path_y, path_z):
    if len(path_x) != len(path_y) or len(path_x) != len(path_z):
        raise ValueError("Input lists must have the same length.")

    # Find the highest and lowest points in the path_y list
    highest_point = max(path_y)
    lowest_point = min(path_y)

    # Find the indices of these points
    highest_index = path_y.index(highest_point)
    lowest_index = path_y.index(lowest_point)

    # Calculate the 3D distance between the two points
    distance = np.sqrt((path_x[highest_index] - path_x[lowest_index])**2 +
                      (path_y[highest_index] - path_y[lowest_index])**2 +
                      (path_z[highest_index] - path_z[lowest_index])**2)

    return distance

##########  3D grid peripheral acoustic signal path planner
def multi_path_generator(source_pos, reciever_pos , incident_angle_list = [30,40] , rotation = 50 , rotation_step = 10):
    noise_points = []
    # paths = []
    main_path = []
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    y_offset = abs(reciever_pos[1] - grid_size[1]/2)
    z_offset = abs(reciever_pos[2] - grid_size[2]/2)
    base = abs(source_pos[0] - reciever_pos[0])
    y_rot_offset = y_offset/base
    z_rot_offset = z_offset/base
    counterz_angles = []
    countery_rotations = []
    for angles in incident_angle_list:
        counterz_angles.append(-angles)
    all_angles = incident_angle_list + counterz_angles
    ## plotting the main path
    num_points = int(base * 1/grid_resolution)
    main_x = np.linspace(source_pos[0], reciever_pos[0], num_points)
    main_y = np.linspace(source_pos[1], reciever_pos[1], num_points)
    main_z = np.linspace(source_pos[2], reciever_pos[2], num_points)
    ax.plot(main_x, main_y, main_z, label='Propagation Path')
    main_path.append((main_x,main_y,main_z))
    rotations = np.arange(0,rotation+rotation_step,rotation_step)       #   np.arange(0, 8 , 2) means [0,2,4,6] so we did np.arange(0, 8+2 , 2) = [0,2,4,6,8]
    for j in rotations:
        countery_rotations.append(-j)
    tot_rotations = list(countery_rotations) + list(rotations)[1:]
    # Calculating the multi-path
    for i in all_angles:
        for theta in tot_rotations:
            if i == 0:
                continue  # Skip zero angle to avoid division by zero
            
            path_x = [source_pos[0]]
            path_y = [source_pos[1]]
            path_z = [source_pos[2]]
            
            if i > 0:
                z_rot = math.tan(math.radians(i))
            else:
                z_rot = -math.tan(math.radians(abs(i)))  # Handle negative angles
            
            if theta > 0:
                y_rot = math.tan(math.radians(theta))
            else:
                y_rot = math.tan(math.radians(theta))
            propagation_direction = [1.0, y_rot + y_rot_offset, z_rot + z_rot_offset]
            while True:
                # Calculate the next position based on the propagation direction
                next_x = path_x[-1] + propagation_direction[0] * grid_resolution
                next_y = path_y[-1] + propagation_direction[1] * grid_resolution
                next_z = path_z[-1] + propagation_direction[2] * grid_resolution

                # Check if the next position is within the grid boundaries
                if 0 <= next_x < reciever_pos[0] :
                # Reflect if hitting the Y, or Z walls
                    if next_y < 0 or next_y >= grid_size[1] :
                        propagation_direction[1] *= -1  # Reverse the direction for the corresponding axis
                    if next_z < 0 or next_z >= grid_size[2] :
                        propagation_direction[2] *= -1  # Reverse the direction for the corresponding axis
                    next_x = path_x[-1] + propagation_direction[0] * grid_resolution
                    next_y = path_y[-1] + propagation_direction[1] * grid_resolution
                    next_z = path_z[-1] + propagation_direction[2] * grid_resolution
                        #path update
                    path_x.append(next_x)
                    path_y.append(next_y)
                    path_z.append(next_z)
                    
                else:
                    ax.plot(path_x, path_y, path_z)
                    # paths.append((path_x,path_y,path_z))
                    
                    boundary_distance = find_highest_lowest_distance(path_x, path_y , path_z)

                    interferance_points = []
                    # Checking interfarence points
                    for i in range(len(path_x)):
                        # Point to check
                        point_to_check = (path_y[i], path_z[i])
                        
                        # Check if the point is on the path
                        result = is_point_on_point(main_y[i] , main_z[i] , point_to_check)

                        if result:
                            # if i > 3.0 :
                                # ax.scatter(main_x[i], main_y[i], main_z[i], c='g', marker='o', label='Interfrence Point')
                            interferance_points.append(path_x[i])

                        else:
                            # interferance_points.append(0)
                            pass
                    ## Calculating the distance covered by each path      
                    # interferance_points = merge_close_elements(interferance_points)           
                    total_distance = calculate_total_distance(path_x, path_y, path_z)
                    for i in interferance_points:
                        k= int(i * 10) - 10 
                        ax.scatter(main_x[k], main_y[k], main_z[k], c='g', marker='o', label='Interfrence Points')

                    
                    # Stop propagation when reaching the grid length limit
                    break
            noise_points.append((interferance_points, total_distance , abs(i) , boundary_distance))
            
            
    # Visualize the propagation path
    ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='b', marker='o', label='Source')
    ax.scatter(reciever_pos[0], reciever_pos[1], reciever_pos[2], c='r', marker='o', label='Receiver')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Propagation Path in 3D Grid')

    # Set the aspect ratio to be equal for all three axes
    ax.set_box_aspect([7, 2, 2])  # Adjust the [5, 1, 1] values to control the aspect ratio
    # ax.legend()
    plt.show()
    return noise_points
########Attenuation for acoustic reflection using snell's Law of relection and lamberts law of absorption in boundary medium
def reflected_signal_amplitude(Z_water, theta_i, frequency, distance, attenuation_coefficient):
    """
    Calculate the attenuated signal amplitude of a reflected acoustic signal from the sea surface.

    Args:
    - Z_water (float): Acoustic impedance of the water medium.
    - theta_i (float): Angle of incidence (in degrees) from within the water.
    - frequency (float): Frequency of the acoustic signal (in Hz).
    - distance (float): Distance traveled by the signal (in meters).
    - attenuation_coefficient (float): Attenuation coefficient for the medium.

    Returns:
    - float: Attenuated signal amplitude after reflection.
    """
    # Convert the angle of incidence from degrees to radians
    theta_i = math.radians(theta_i)
    
    # Calculate the angle of refraction using Snell's Law
    n_water = math.sqrt(Z_water)
    sin_theta_t = math.sin(theta_i) / n_water
    
    # Check for total internal reflection
    if sin_theta_t > 1.0:
        return 0.0  # Total internal reflection results in complete attenuation

    # Calculate the reflection coefficient
    cos_theta_i = math.cos(theta_i)
    cos_theta_t = math.sqrt(1.0 - sin_theta_t**2)
    R = ((n_water * cos_theta_i - n_water * cos_theta_t) / (n_water * cos_theta_i + n_water * cos_theta_t))**2

    # Calculate the intensity reflection coefficient
    R_I = R**2

    # Calculate the attenuation due to absorption using Lambert's Law
    attenuation = math.exp(-attenuation_coefficient * distance)
    # print(attenuation)
    # Calculate the attenuated signal amplitude
    signal_amplitude = math.sqrt(1.0 - R_I) * ( attenuation)

    return 1-signal_amplitude

def add_noise_to_1d_signal(signal, noise_indices, noise_amplitude):
    """
    Add noise to a 1D signal at specific indices.

    Args:
        signal (numpy.ndarray): The original 1D signal.
        noise_indices (list): A list of indices where noise should be added.
        noise_amplitude (float): The amplitude of the noise to be added.

    Returns:
        numpy.ndarray: The signal with noise added.
    """
    noisy_signal = (signal.copy())  # Create a copy of the original signal to avoid modifying it directly.
    
    for index in noise_indices:
        # if index < 0 or index >= signal.shape[0]:
        #     raise ValueError("Index out of bounds.")
        noisy_signal[index] += np.random.normal(0.5, noise_amplitude)
    
    return list(noisy_signal)


'#####################################################################################################'
'#########################################Constants###################################################'
# Constants
initial = 1.0
tank_length = 10  # Horizontal length of the tank in meters
tank_depth = 5    # Depth of the tank in meters
sound_speed = 1500  # Speed of sound in water in m/s
sampling_rate = 45000  # Sample rate in Hz
simulation_duration = 0.5  # Simulation duration in seconds
tx_freq = 300 # Hz transmitter frequency
noise_amplifier = 20 #Ambient noise amplifier
step = 100       #samples to subdivide
source_amp = 500


'# Define the grid dimensions and resolution'
grid_size = (70, 40, 40)  # x, y, z dimensions
grid_resolution = 0.1  # Grid spacing
'#####################################################################################################'


'# Create a 3D grid'
x = np.arange(0, grid_size[0], grid_resolution)
y = np.arange(0, grid_size[1], grid_resolution)
z = np.arange(0, grid_size[2], grid_resolution)
X, Y, Z = np.meshgrid(x, y, z)

'# Define the starting point and propagation direction'
source_pos = (1.0, grid_size[1]/2, grid_size[2]/2)
reciever_pos =(grid_size[0]-2 , grid_size[1]/2+5, grid_size[2]/2 - 3)
distance_x = reciever_pos[0] - source_pos[0]
distance = distance_3D(source_pos, reciever_pos)

# transit_time = distance / sound_speed
# Create a time vector

time = np.arange(0.0,  distance_x, grid_resolution )
distance_coordinates = []

for t in time:
    distance_coordinates.append(t)

# Generate a simple signal to simulate the source
# dist = 0 
recieved_signal = []
source_signal = []

for t in time:
    source_pulse = source_amp * np.sin(2 * np.pi * 10 * t/distance_x)
    source_signal.append(source_pulse)
    absorption = pathloss(tx_freq, t , depth_calc(source_pos,reciever_pos) )
    noise_amp = ambient_noise(tx_freq,0.6)
    # print(f"absorption = {100-(1/absorption)*100} % & distance covered {1500*t} m")
    # print(f"ambient = {noise_amp}")
    # print(f"distance covered {dist}")
    external_noise = random.uniform(-noise_amp, noise_amp)
    recieved_signal.append(source_pulse * (1/absorption) +  noise_amplifier * external_noise)



plt.figure(figsize=(20, 6))
plt.subplot(2, 1, 1)
plt.plot(time, source_signal)
plt.title('Transmitted Signal')
plt.xlabel('Distance (m)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(distance_coordinates, recieved_signal)
# plt.plot(distance_coordinates, interfared_signal)
plt.title('Received Signal with Attenuation')
plt.xlabel('Distance propagated (m)')
plt.ylabel('Amplitude')



'########### Here the multi-path channels are included       ###############################################'
z_step = 20
z_resolution = list(np.arange(0,70+1 , z_step)[1:])
# z_resolution = [20,30,40,50]
y_scope = 40
y_resolution = 20
out = multi_path_generator( source_pos , reciever_pos ,z_resolution,y_scope,y_resolution)
'''#################### noise from multi-path channel's interferance points are extracted       #######################'''
### constant for reflective interferance
Z_water = 1.44e6  # Acoustic impedance of water in g/(cm^2*s)
theta_i_deg = 30.0  # Angle of incidence in degrees from within the water  
distance = 10.0  # Distance traveled by the signal in meters
attenuation_coefficient = 0.1  # Attenuation coefficient for the medium
interferance_noise = []
'######      Taking out the noise adder points and amplitude of noise to add         #################################'
for i in out:
    # print(f"{l}st Distance : {i[-1]} and theta {i[2]}")
    amplitude = reflected_signal_amplitude(Z_water, i[2], tx_freq, i[-1] ,0.1)
    interferance_noise.append([i[0],amplitude*source_amp])
    # print(f"{l}st Absorption: {amplitude}")
    # print(f"{l}st Attenuated Signal Amplitude: {amplitude}")
# interfered_signals = []
'##################  Creating empty array of noise which will be added to the recieved signal later      ##########################'
interfered_signal = np.zeros(len(recieved_signal))
for i in interferance_noise:
    indices = [int(j * 10) for j in i[0]]
    amplitude = i[1]  # Use the amplitude value from interferance_noise
    interfered_signal = add_noise_to_1d_signal(interfered_signal, indices, amplitude)
    # interfered_signals.append(interfered_signal)
# summed_interferance = np.sum(interfered_signals, axis=0)
# # Create a larger figure (adjust the figsize as needed)
'###### #####################     Occurance of noise smoothed out     ####################################'
# Define the window size for the moving average
window_size = 10
# Apply the moving average filter
smoothed_array = np.convolve(interfered_signal, np.ones(window_size)/window_size, mode='same')
'''########################    Summing up to get the resultant signal exposed to path loss , ambient noise , interfarance for multi-path periphreral noise ###################'''
noise_stack3 = np.array(recieved_signal) + np.array(interfered_signal)
######### Plotting the final signal     ##########################
plt.figure(figsize=(20, 6))
# plt.plot(distance_coordinates, recieved_signal)
plt.plot(distance_coordinates, noise_stack3)
plt.title('Received Signal with Interfrence')
plt.xlabel('Distance propagated (m)')
plt.ylabel('Amplitude')

# plt.tight_layout()
plt.show()


