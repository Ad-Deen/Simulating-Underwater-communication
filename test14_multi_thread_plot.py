import matplotlib.pyplot as plt
import numpy as np
import random
import time
import threading

# Initialize the figure and axes
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

x_data = np.arange(0, 50, 0.1)
y_data1 = np.sin(x_data)
y_data2 = np.cos(x_data)

line1, = ax1.plot(x_data, y_data1)
line2, = ax2.plot(x_data, y_data2)

# Function to add noise to data in a thread
def add_noise_to_data(data, min_noise, max_noise):
    while True:
        # Generate random noise for the data
        noise = np.random.uniform(min_noise, max_noise, len(data))
        data_with_noise = data + noise

        yield data_with_noise  # Return the updated data with noise

# Create threads for adding noise to the data for both graphs
noise_gen1 = add_noise_to_data(y_data1, -0.0, 0.0)
noise_thread1 = threading.Thread(target=lambda: next(noise_gen1))
noise_thread1.daemon = True
noise_thread1.start()

noise_gen2 = add_noise_to_data(y_data2, -0.1, 0.1)
noise_thread2 = threading.Thread(target=lambda: next(noise_gen2))
noise_thread2.daemon = True
noise_thread2.start()

# Set x-axis limits
ax1.set_xlim(0, 30)  # You can adjust the limits based on your data range
ax2.set_xlim(0, 30)

# Continuously update the plots in parallel
while True:
    try:
        y_data1 = next(noise_gen1)  # Get updated data from the generator
        y_data2 = next(noise_gen2)

        # Update the plots
        line1.set_ydata(y_data1)
        line2.set_ydata(y_data2)

        ax1.relim()
        ax2.relim()

        # Autoscale only the y-axes
        ax1.set_ylim(np.min(y_data1), np.max(y_data1))
        ax2.set_ylim(np.min(y_data2), np.max(y_data2))

        plt.draw()
        plt.pause(0.1)
    except StopIteration:
        # Handle thread termination
        break

plt.show()
