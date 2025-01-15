#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist

np.set_printoptions(threshold=np.inf, edgeitems=5, linewidth=200)

class RGBSubscriber:
    def __init__(self):
        rospy.init_node('rgb_subscriber', anonymous=True)
        self.rgb_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
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
        self.camTcar = self.inverse_pose_transform(self.carTcam)
        

        self.adjoint_carTcam = self.adjoint(self.carTcam)
        self.adjoint_camTcar = self.adjoint(self.camTcar)

        self.J = np.array([[1, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 1]])

        self.F = np.zeros((8, 8))
        for i in range(0, 8, 2): 
            self.F[i, i] = self.fx
        for i in range(1, 8, 2):
            self.F[i, i] = self.fy

        self.F_inv = np.linalg.inv(self.F)

        self.labels = ['A', 'B', 'C', 'D']
        self.desired_uv = np.array([
                                    [254],
                                    [283],

                                    [396],
                                    [283],

                                    [254],
                                    [426],

                                    [398],
                                    [428]
                                    ])
        
        self.x_coords = self.desired_uv[::2, 0]
        self.y_coords = self.desired_uv[1::2, 0]

    def depth_callback(self, data):
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(data)
            # normalized_img = cv2.normalize(self.depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # cv2.imshow("Depth Image", normalized_img)
            # cv2.waitKey(1)  # This is necessary for imshow to work properly
        except Exception as e:
            rospy.logerr("Error converting depth image: %s", str(e))

    def rgb_callback(self, data):
            rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

            lower_red = np.array([170, 120, 70])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

            red_mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((5, 5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            centers = []

            for i in range(min(4, len(contours))):
                contour = contours[i]
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append((cX, cY))

            if centers:
                mean_center = np.mean(centers, axis=0)
                d_u, d_v = int(mean_center[0]), int(mean_center[1])

                if self.depth_image is not None:
                    Z = self.depth_image[d_v, d_u]
                    if np.isnan(Z) or Z <= 0:
                        rospy.logwarn("Invalid depth value at pixel (%d, %d), skipping this frame.", d_u, d_v)
                        return
                        
                    identified_points_vector, identified_points= self.identify_key_points(centers)
                    M_inverse = self.compute_Moore_Penrose(identified_points, Z, self.J, self.adjoint_camTcar)
                    UV = identified_points_vector - self.desired_uv

                    Twist_car = -self.k * M_inverse @ self.F_inv @ UV
                    print(Twist_car)

                    self.publish_twist(Twist_car)

                    # self.publish_stop()
                    # print(self.camTcar)
                    # print(self.adjoint_camTcar)

                    

                    for i, identified_point in enumerate(identified_points):

                        cv2.circle(rgb_image, identified_point[0], 1, (0, 255, 0), -1)
                        cv2.putText(rgb_image, self.labels[i], (identified_point[0][0] + 10, identified_point[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    for i, (x, y) in enumerate(zip(self.x_coords, self.y_coords)):
                        cv2.circle(rgb_image, (x, y), 5, (255, 0, 0), -1)
                        cv2.putText(rgb_image, self.labels[i], (x + 10, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    cv2.imshow("Detected Red Dots", rgb_image)
                    cv2.waitKey(1)


    def compute_Moore_Penrose(self, identified_points, depth, J, X, regularization=1e-6):
        L_matrix = np.vstack([self.compute_L(identified_point[0][0], identified_point[0][1], depth) for identified_point in identified_points])
        L_matrix_selected = L_matrix @ X @ J
        return np.linalg.pinv(L_matrix_selected)
    
    def is_matrix_well_conditioned(self, matrix, threshold=1e6):
        cond_number = np.linalg.cond(matrix)
        return cond_number < threshold
      
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
    
    def inverse_pose_transform(self, T):

        R = T[:3, :3]
        t = T[:3, 3]
        
        R_inv = R.T
        t_inv = -R_inv @ t
        
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        
        return T_inv

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

    def identify_key_points(self, centers):
        # Sort centers to determine A, B, C, D based on x and y coordinates
        A = sorted(centers, key=lambda x: x[0])[:2]
        C = sorted(centers, key=lambda x: x[0], reverse=True)[:2]
        B = sorted(centers, key=lambda x: x[1])[:2]
        D = sorted(centers, key=lambda x: x[1], reverse=True)[:2]

        # Find the intersection points
        point1 = list(set(A) & set(B))
        point4 = list(set(C) & set(D))
        point3 = list(set(A) & set(D))
        point2 = list(set(B) & set(C))

        # If there are no intersections, set the points to None
        if len(point1) == 0: point1 = None
        if len(point4) == 0: point4 = None
        if len(point3) == 0: point3 = None
        if len(point2) == 0: point2 = None

        # Prepare the result in the requested format as a numpy array
        points = []

        # Add points to the array if they exist
        if point1:
            points.append([point1[0][0]])  # x1
            points.append([point1[0][1]])  # y1
        else:
            points.append([None, None])  # if no point, append None

        if point2:
            points.append([point2[0][0]])  # x2
            points.append([point2[0][1]])  # y2
        else:
            points.append([None, None])

        if point3:
            points.append([point3[0][0]])  # x3
            points.append([point3[0][1]])  # y3
        else:
            points.append([None, None])

        if point4:
            points.append([point4[0][0]])  # x4
            points.append([point4[0][1]])  # y4
        else:
            points.append([None, None])

        # Return the points as a numpy array
        return np.array(points), [point1, point2, point3, point4]

    def run(self): 
        rospy.spin()

if __name__ == '__main__':
    rgb_subscriber = RGBSubscriber()
    rgb_subscriber.run()
