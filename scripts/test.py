from i2rt.robots.get_robot import get_yam_robot
import numpy as np
# Get a robot instance
robot = get_yam_robot(channel="can0", zero_gravity_mode=True)

# Get the current joint positions
joint_pos = robot.get_joint_pos()

# Command the robot to move to a new joint position
target_pos = np.array([0, 0, 0, 0, 0, 0, 0])

# Command the robot to move to the target position
robot.command_joint_pos(target_pos)

print(robot.get_joint_pos())