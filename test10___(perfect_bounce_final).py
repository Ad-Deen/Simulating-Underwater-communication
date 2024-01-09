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

def merge_close_elements(input_list, threshold=0.5):
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
                    ax.plot(path_x, path_y, path_z, label='Propagation Path')
                    # paths.append((path_x,path_y,path_z))
                    interferance_counter = 0
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
                            pass
                    ## Calculating the distance covered by each path      
                    interferance_points = merge_close_elements(interferance_points)           
                    total_distance = calculate_total_distance(path_x, path_y, path_z)
                    for i in interferance_points:
                        k= int(i * 10) - 10 
                        ax.scatter(main_x[k], main_y[k], main_z[k], c='g', marker='o', label='Interfrence Points')

                    
                    # Stop propagation when reaching the grid length limit
                    break
            noise_points.append((interferance_points, total_distance))
            
            
    # Visualize the propagation path
    ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='b', marker='o', label='Source')
    ax.scatter(reciever_pos[0], reciever_pos[1], reciever_pos[2], c='r', marker='o', label='Receiver')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Propagation Path in 3D Grid')

    # Set the aspect ratio to be equal for all three axes
    ax.set_box_aspect([7, 2, 2])  # Adjust the [5, 1, 1] values to control the aspect ratio
    ax.legend()
    plt.show()
    return noise_points

#####################################################################################################
#########################################Constants###################################################
Z_water = 1.5e6  # Acoustic impedance of water in g/(cm^2*s)
Z_second_medium = 0.4e6  # Acoustic impedance of the second medium in g/(cm^2*s)
    

# Define the grid dimensions and resolution
grid_size = (70, 20, 20)  # x, y, z dimensions
grid_resolution = 0.1  # Grid spacing
#####################################################################################################


# Create a 3D grid
x = np.arange(0, grid_size[0], grid_resolution)
y = np.arange(0, grid_size[1], grid_resolution)
z = np.arange(0, grid_size[2], grid_resolution)
X, Y, Z = np.meshgrid(x, y, z)

# Define the starting point and propagation direction
source_pos = (1.0, grid_size[1]/2, grid_size[2]/2)
reciever_pos =(grid_size[0]-2 , grid_size[1]/2+5, grid_size[2]/2 - 3)
z_step = 30
z_resolution = list(np.arange(0,30+1 , z_step)[1:])
# z_resolution = [20,30,40,50]
y_scope = 20
y_resolution = 20
out = multi_path_generator( source_pos , reciever_pos ,z_resolution,y_scope,y_resolution)


# Create a larger figure (adjust the figsize as needed)



