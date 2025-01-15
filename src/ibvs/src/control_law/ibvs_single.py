#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist  # 导入 Twist 消息类型

class RGBSubscriber:
    def __init__(self):
        rospy.init_node('rgb_subscriber', anonymous=True)
        self.rgb_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
        # 创建 Twist 消息发布者
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.cv_bridge = CvBridge()
        self.depth_image = None
        self.k = 0.2
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320.5
        self.cy = 240.5
        self.carTcam = np.array([
                                [0.0, 0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0, 0.8],
                                [0.0, 0.0, 0.0, 1.0] 
                                ])
        self.J = np.array([[1, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 1]])
        self.camTcar = self.inverse_pose_transform(self.carTcam)
        self.adjoint_carTcam = self.adjoint(self.carTcam)
        self.adjoint_camTcar = self.adjoint(self.camTcar)
        self.F = np.array([
                          [self.fx, 0],
                          [0, self.fy]
                          ])
        self.F_inv = np.linalg.inv(self.F)
        self.desired_uv = np.array([
                                    [320],
                                    [360]
                                    ])

    def depth_callback(self, data):
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(data)
            # normalized_img = cv2.normalize(self.depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # cv2.imshow("Depth Image", normalized_img)
            # cv2.waitKey(1)  # This is necessary for imshow to work properly
        except Exception as e:
            rospy.logerr("Error converting depth image: %s", str(e))

    def rgb_callback(self, data):
        try:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            
            lower_blue = (80, 100, 0)
            upper_blue = (140, 255, 255)
            
            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
            
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_contour = max(contours, key=cv2.contourArea, default=None)

            if max_contour is not None:
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    u = int(M["m10"] / M["m00"])
                    v = int(M["m01"] / M["m00"])

                    cv2.circle(rgb_image, (u, v), 5, (0, 0, 255), -1)
                    cv2.circle(rgb_image, (self.desired_uv[0, 0], self.desired_uv[1, 0]), 5, (0, 255, 255), -1)

                    cv2.drawContours(rgb_image, [max_contour], -1, (0, 255, 0), 2)

                    if self.depth_image is not None:
                        Z = self.depth_image[v, u]

                        if np.isnan(Z) or Z <= 0:
                            rospy.logwarn("Invalid depth value at pixel (%d, %d), skipping this frame.", u, v)
                            cv2.imshow("Detected Blue Circles", rgb_image)
                            cv2.waitKey(1)
                            return
                        
                        L = self.compute_L(u, v, Z)

                        M = L @ self.adjoint_camTcar @ self.J

                        UV = np.array([[u],[v]]) - self.desired_uv
                        
                        twist_car = -self.k * np.linalg.inv(M) @ self.F_inv @ UV
                        
                        print("depth", Z, "twist_car:", twist_car[0,0], twist_car[1,0], "UV", u, v)
                        self.publish_twist(twist_car)
                        # self.publish_stop()
                     
            cv2.imshow("Detected Blue Circles", rgb_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Error processing the image: %s", str(e))
      
    def compute_L(self, u, v, Z):
        x = (u - self.cx)/self.fx
        y = (v - self.cy)/self.fy
        L = np.zeros((2, 6))
        L[0,0] = -1/Z
        L[1,0] = 0

        L[0,1] = 0
        L[1,1] = -1/Z

        L[0,2] = x/Z
        L[1,2] = y/Z

        L[0,3] = x*y
        L[1,3] = 1+y**2

        L[0,4] = -(1+x**2)
        L[1,4] = -x*y

        L[0,5] = y
        L[1,5] = -x
        return L
    
    def inverse_pose_transform(self, T):

        R = T[:3, :3]
        t = T[:3, 3]
        
        R_inv = R.T
        t_inv = -R_inv @ t
        
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        
        return T_inv

    def skew_symmetric(self, vector):
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])

    def adjoint(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        
        t_skew = self.skew_symmetric(t)

        adj = np.zeros((6, 6))
        adj[:3, :3] = R
        adj[:3, 3:6] = t_skew @ R
        adj[3:6, 3:6] = R
        
        return adj

    def publish_twist(self, twist_car):
        twist_msg = Twist()

        twist_msg.linear.x = np.clip(twist_car[0], -0.5, 0.5)   # v_x
        twist_msg.angular.z = np.clip(twist_car[1], -0.5, 0.5)  # w_z

        self.twist_pub.publish(twist_msg)

    def publish_stop(self):
        twist_msg = Twist()

        twist_msg.angular.z = 0 # w_z
        twist_msg.linear.x = 0  # v_x
        self.twist_pub.publish(twist_msg)

    def run(self): 
        rospy.spin()

if __name__ == '__main__':
    rgb_subscriber = RGBSubscriber()
    rgb_subscriber.run()
