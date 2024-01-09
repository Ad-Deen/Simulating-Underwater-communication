import matplotlib.pyplot as plt
import numpy as np
import random
import time

# Initialize the figure and axis
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data)
line, = ax.plot(x_data, y_data)

# Continuously update the plot
while True:
    # Simulate data updates (replace this with your data)
    y_data = np.sin(x_data + random.uniform(-1, 1))

    # Update the plot
    line.set_ydata(y_data)
    ax.relim()
    ax.autoscale_view()

    plt.draw()
    plt.pause(0.1)
