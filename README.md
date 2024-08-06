# Autonomous Vehicle Software for Formula Student Driverless Simulator

This repository contains the software for an autonomous vehicle developed by Ampera Racing at the beginning of 2022 to run in the Formula Student Driverless Simulator (FSDS). The software integrates various modules for perception, mapping, planning, and control to enable autonomous driving.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/autonomous_vehicle_project.git
    cd autonomous_vehicle_project
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that you have the Formula Student Driverless Simulator installed and set up correctly.

## Usage

To run the autonomous vehicle software, execute the `main.py` script:

```bash
python src/main.py
```

This script will start the threads for perception, mapping, path planning, and control, and will run the autonomous vehicle in the simulator.

![Computer Vision for Object Detection](imgs\car_view.png)

![Track view](imgs\map.png)

![SLAM result](imgs\slam.png)



## Results
The software includes modules for:

- Perception: Using a monocular camera and Lidar to detect cones and obstacles.
- Mapping: Implementing SLAM to create a map of the environment.
- Path Planning: Planning the trajectory using detected cones.
- Control: Controlling the vehicle using a Stanley controller.
- Perception: The perception module uses a monocular camera to detect the positions of cones in the image and Lidar to detect the positions in the real world.