import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sound_speed = 1500      #m/s


# Define grid dimensions
x_min, x_max = 0, 100  # Define your x-axis boundaries
y_min, y_max = 0, 100  # Define your y-axis boundaries
z_min, z_max = 0, 100   # Define your z-axis boundaries

# Define grid resolution
x_resolution = 1  # Adjust based on your needs
y_resolution = 1
z_resolution = 0.1

# Create a 3D grid
x = np.arange(x_min, x_max, x_resolution)
y = np.arange(y_min, y_max, y_resolution)
z = np.arange(z_min, z_max, z_resolution)

# Create a mesh grid
X, Y, Z = np.meshgrid(x, y, z)


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# Visualize properties (e.g., sound speed) at grid points
ax.scatter(X, Y, Z, c=sound_speed, cmap='viridis', marker='.')

# Customize the plot (labels, color bar, etc.)
ax.set_xlabel('Length (m)')
ax.set_ylabel('Breadth (m)')
ax.set_zlabel('Depth (m)')
cbar = plt.colorbar(ax.scatter(X, Y, Z, c=sound_speed, cmap='viridis', marker='.'))
cbar.set_label('Sound Speed (m/s)')

plt.show()