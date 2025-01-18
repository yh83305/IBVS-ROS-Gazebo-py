#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
import numpy as np

def model_states_callback(msg):
    try:
        # 获取 black_wall_with_red_dots 和 turtlebot3_burger 的索引
        index_wall = msg.name.index("black_wall_with_red_dots")
        index_turtlebot = msg.name.index("turtlebot3_burger")
        
        # 获取 black_wall_with_red_dots 和 turtlebot3_burger 的位置
        pos_wall = np.array([msg.pose[index_wall].position.x, msg.pose[index_wall].position.y, msg.pose[index_wall].position.z])
        pos_turtlebot = np.array([msg.pose[index_turtlebot].position.x, msg.pose[index_turtlebot].position.y, msg.pose[index_turtlebot].position.z])
        
        # 计算相对平移
        relative_translation = pos_wall - pos_turtlebot
        
        # 输出相对平移
        rospy.loginfo("XYZ: %.2f,%.2f,%.2f meters", relative_translation[0], relative_translation[1], relative_translation[2])
    except ValueError:
        rospy.logwarn("Model not found in model_states")

def main():
    rospy.init_node('relative_translation_calculator')
    rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback)
    rospy.spin()

if __name__ == '__main__':
    main()