This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

# Details of Implementaion
Now let's see the details of implementation that I have done for completion of this project.

## Autonomous Car System Architecture
The autonomous car which will run on the track in simulator, has the system architecure as mentioend in the diagram below.
<img src="./imgs/final-project-ros-graph-v2.png" height="100%" width="100%" align="middle" alt="Autonomous Car System Architecture Diagram">
<br>
Now let's see all the modules present in this architecure in details.
## Planning
The planning module is responsible for planning the path  for car movements on the road. This modules creates a path to traverse on the track considering the environments, ex. traffic.
This module in composed of two ROS nodes:
1. Waypoint Loader
2. Waypoint Updater

#### Waypoint Loader
This node is responsible for loading the `waypoints` provided by the simulator and publishing them on `/base_waypoints` topic. Thus, this can be considered as a **Staring Point** of the project. Also, the `frameid` for the `waypoints` published is `/world`.
These `waypoints` are of type `Lane` defined under `styx_msgs/msg/Lane.msg`
<br>
The waypoint contains below data items. All these items are updated with data received from simulator.
1. `position x`: `x` location of the vehicle
2. `position y`: `y` location of the vehicle
3. `position z`: `z` location of the vehicle
4. `orientation`: Car's yaw angle
5. `twist.linear.x`: Car's velocity
<br>

#### Waypoint Updater Node
This node contains `waypoint_updater.py`. This node updates the velocity property of every waypoint based on traffic light and obstacles detected.<br>
This node subscribes to 
1. `/base_waypoints`: Published by `Waypoint Loader`
2. `/current_pose`: Published by `Car/Simulator`
3. `/obstacle_waypoint`: Published by `Obstacle Detection` from `Perception` module
4. `/traffic_waypoint`: Published by `Traffic Light Detection Node` from `Perception` module
<br>

