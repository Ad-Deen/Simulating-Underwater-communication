# Underwater Communication Simulator

Welcome to the Underwater Communication Simulator project! This simulator is designed to perform real-time multipath propagation simulations in any given 3D underwater environment. The project includes scripts for multipath propagation pathways using Python, visualization with Matplotlib and NumPy, and plans for integrating a 3D CAD marine environment for more realistic simulations.

## Features

- **Multipath Propagation Pathways:** Utilize Python scripts to simulate multipath propagation pathways in real-time within an underwater environment.
- **Visualization:** Use Matplotlib and NumPy to visualize the multipath propagation paths in a 3D environment.
- **Extensibility:** Build a pipeline for automatic simulation in any 3D marine environment by providing CAD data and coordinates for Transmitter (Tx) and Receiver (Rx) devices.
- **AUV Integration:** Plan to integrate with an Autonomous Underwater Vehicle (AUV) for feedback sensor data transmission from the underwater bot to the surface.

## Project Structure

The project is structured as follows:

- `scripts/`: Contains Python scripts for multipath propagation pathways.
- `visualization/`: Matplotlib scripts for visualizing multipath propagation paths.
- `cad_integration/`: Planned directory for integrating 3D CAD marine environments and coordinates for Tx and Rx devices.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/underwater-communication-simulator.git
2. Navigate to the project directory

   ```bash
   cd underwater-communication-simulator
3. Install the required dependencies:
   
   ```bash
   pip install -r requirements.txt
4. Run the simulation script:

   ```bash
   python scripts/multipath_simulation.py

Please check the work log in wiki to see more resources regarding the process. I am recently working with blender python scripts for dynamic real time signal data visualization facility. Because matplotlib visulization is static.There I described my each steps of developing the project in blender with scripts.

## Future Work

### 3D CAD Integration

Develop a pipeline for automatic simulation in any 3D marine environment using CAD data.

### AUV Integration

Connect the simulator with an AUV for real-time feedback sensor data transmission.

### Communication Protocol

Implement a communication protocol for transmitting test signals from Tx and receiving simulated signals in Rx.

### Maximizing Communication Speed

Optimize the communication protocol for maximizing speed and efficiency.

## Contributing

We welcome contributions! If you'd like to contribute to the project, please follow our [Contribution Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
