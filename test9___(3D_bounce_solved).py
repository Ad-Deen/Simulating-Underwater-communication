import numpy as np
import matplotlib.pyplot as plt
import math

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

def calculate_reflection_loss(Z1, Z2):
    # Calculate the Fresnel Reflection Coefficient
    R = ((Z2 - Z1) / (Z2 + Z1)) ** 2
    
    # Calculate the reflection loss in decibels (dB)
    reflection_loss_db = -10 * math.log10(R)
    reflection_loss = math.exp(reflection_loss_db)
    return reflection_loss

def multi_path_generator(incident_angle_list , rotation = 0 , rotation_step = 0):
    path_x = [start_point[0]]
    path_y = [start_point[1]]
    path_z = [start_point[2]]
    paths = []
    for i in incident_angle_list:
        
        z_rot = math.tan(math.radians(i))
        # y_rot = math.tan(math.radians(i))
        propagation_direction = (1.0, 0.0, z_rot)  # Example: Propagate in the x-direction with given z_rot
        while True:
            # Calculate the next position based on the propagation direction
            next_x = path_x[-1] + propagation_direction[0] * grid_resolution
            next_y = path_y[-1] + propagation_direction[1] * grid_resolution
            next_z = path_z[-1] + propagation_direction[2] * grid_resolution

            # Check if the next position is within the grid boundaries
            if 0 <= next_x < grid_size[0] :
                if 0 <= next_y < grid_size[1] and 0 <= next_z < grid_size[2]:
                    path_x.append(next_x)
                    path_y.append(next_y)
                    path_z.append(next_z)
                elif 0 <= next_y < grid_size[1] and next_z >= grid_size[2]:
                    propagation_direction = (1.0, 0.0, -z_rot)
                    next_x = path_x[-1] + propagation_direction[0] * grid_resolution
                    next_y = path_y[-1] + propagation_direction[1] * grid_resolution
                    next_z = path_z[-1] + propagation_direction[2] * grid_resolution
                    
                    path_x.append(next_x)
                    path_y.append(next_y)
                    path_z.append(next_z)  
                elif  0 <= next_y < grid_size[1] and next_z <= 0:
                    propagation_direction = (1.0, 0.0, z_rot)
                    next_x = path_x[-1] + propagation_direction[0] * grid_resolution
                    next_y = path_y[-1] + propagation_direction[1] * grid_resolution
                    next_z = path_z[-1] + propagation_direction[2] * grid_resolution
                    
                    path_x.append(next_x)
                    path_y.append(next_y)
                    path_z.append(next_z) 
            else:
                paths.append((path_x,path_y,path_z))
                # Stop propagation when reaching the grid limit
                break
    
    return 0

#####################################################################################################
#########################################Constants###################################################
Z_water = 1.5e6  # Acoustic impedance of water in g/(cm^2*s)
Z_second_medium = 0.4e6  # Acoustic impedance of the second medium in g/(cm^2*s)
    

# Define the grid dimensions and resolution
grid_size = (50, 15, 15)  # x, y, z dimensions
grid_resolution = 0.1  # Grid spacing
#####################################################################################################


# Create a 3D grid
x = np.arange(0, grid_size[0], grid_resolution)
y = np.arange(0, grid_size[1], grid_resolution)
z = np.arange(0, grid_size[2], grid_resolution)
X, Y, Z = np.meshgrid(x, y, z)

# Define the starting point and propagation direction
start_point = (1.0, 5.0, 5.0)
propagation_direction = [1.0, 0.839, 0.5773]  # Example: Propagate in the x-direction with upward 30 degree

# Create empty lists to store the coordinates of the propagation path
path_x = [start_point[0]]
path_y = [start_point[1]]
path_z = [start_point[2]]

# Define the maximum number of steps or a stopping condition
max_steps = 5*int((distance_3D((0,0,0),grid_size)/grid_resolution)) + 1 # Adjust as needed

# Simulate the propagation
for step in range(max_steps):
    # Calculate the next position based on the propagation direction
    next_x = path_x[-1] + propagation_direction[0] * grid_resolution
    next_y = path_y[-1] + propagation_direction[1] * grid_resolution
    next_z = path_z[-1] + propagation_direction[2] * grid_resolution

    # Check if the next position is within the grid boundaries
    if 0 <= next_x < grid_size[0] :
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
        # if 0 <= next_y < grid_size[1] and 0 <= next_z < grid_size[2]:
        #     path_x.append(next_x)
        #     path_y.append(next_y)
        #     path_z.append(next_z)
        # if (next_z >= grid_size[2]) :
        #     propagation_direction = (1, 0.839, -0.5773)   #direction update
        #     next_x = path_x[-1] + propagation_direction[0] * grid_resolution
        #     next_y = path_y[-1] + propagation_direction[1] * grid_resolution
        #     next_z = path_z[-1] + propagation_direction[2] * grid_resolution
        #     #path update
        #     path_x.append(next_x)
        #     path_y.append(next_y)
        #     path_z.append(next_z)         
        # if ( next_y >= grid_size[1]):
        #     propagation_direction = (1, -0.839, -0.5773)
        #     next_x = path_x[-1] + propagation_direction[0] * grid_resolution
        #     next_y = path_y[-1] + propagation_direction[1] * grid_resolution
        #     next_z = path_z[-1] + propagation_direction[2] * grid_resolution
        #     #path update
        #     path_x.append(next_x)
        #     path_y.append(next_y)
        #     path_z.append(next_z)
        # if next_z <= 0:
        #     propagation_direction = (1.0, -0.839, 0.5773)
        #     next_x = path_x[-1] + propagation_direction[0] * grid_resolution
        #     next_y = path_y[-1] + propagation_direction[1] * grid_resolution
        #     next_z = path_z[-1] + propagation_direction[2] * grid_resolution
        #     #path update
        #     path_x.append(next_x)
        #     path_y.append(next_y)
        #     path_z.append(next_z)
        # if next_y <= 0 :
        #     propagation_direction = (1.0, 0.839, 0.5773)
        #     next_x = path_x[-1] + propagation_direction[0] * grid_resolution
        #     next_y = path_y[-1] + propagation_direction[1] * grid_resolution
        #     next_z = path_z[-1] + propagation_direction[2] * grid_resolution
        #     #path update
        #     path_x.append(next_x)
        #     path_y.append(next_y)
        #     path_z.append(next_z)
    else:
        # Stop propagation when reaching the grid limit
        break

# Visualize the propagation path
# Create a larger figure (adjust the figsize as needed)
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')
ax.plot(path_x, path_y, path_z, label='Propagation Path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Propagation Path in 3D Grid')

# Set the aspect ratio to be equal for all three axes
ax.set_box_aspect([5, 1, 1])  # Adjust the [5, 1, 1] values to control the aspect ratio

plt.show()
