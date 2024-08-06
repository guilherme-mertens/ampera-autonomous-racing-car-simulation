# Architecture Overview

This document describes the architecture of the classes used for autonomous path planning.

## Lidar Class

The `Lidar` class processes data from a Lidar sensor to detect cones. It includes methods for:

- Converting a group of points to a cone's position (`pointgroup_to_cone`)
- Calculating the distance between two points (`distance`)
- Rotating points (`rotate`)
- Filtering angle values (`anglefilter`)
- Extracting point clouds from lidar data (`lidar_sweep`)
- Clustering points into groups representing cones (`first_clustering`)
- Detecting cones using lidar data (`lidar_detect`)

## MonocularCamera Class

The `MonocularCamera` class uses a YOLO model to detect cones in images captured by a monocular camera. It includes methods for:

- Initializing the YOLO model (`init_model`)
- Detecting cones in an image (`detect_cones`)
- Generating the boxes' center coordinates (`get_center_boxes`)
- Detecting and returning the positions of cones in the image (`get_cones_position_in_image`)

## PathPlanning Class (2 Lap)

The `PathPlanning` class uses Delaunay triangulation and spline interpolation to plan paths. It includes methods for:

- Creating the Delaunay triangulation (`delaunay`)
- Finding the path by connecting blue and yellow cones (`find_path`)
- Sorting path points to create a coherent trajectory (`sort_path`)
- Interpolating the path to create a smooth trajectory (`interpolate`)
- Processing cones to generate a trajectory (`process`)
- Updating path planning with new data (`update`)
- Returning the current trajectory (`get_trajectory`)

## PathPlanningLap Class (1 Lap)

The `PathPlanningLap` class plans a single lap path using spline interpolation. It includes methods for:

- Interpolating the path to create a smooth trajectory (`interpolate`)
- Sorting path points to create a coherent trajectory (`sort_path`)
- Processing cones to generate a trajectory (`process`)
- Updating path planning with new data (`update`)
- Returning the current trajectory (`get_trajectory`)
