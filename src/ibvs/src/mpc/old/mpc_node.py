#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from scipy.optimize import minimize

np.set_printoptions(threshold=np.inf, edgeitems=5, linewidth=200)

class RGBSubscriber:
    def __init__(self):
        rospy.init_node('rgb_subscriber', anonymous=True)
        self.rgb_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.cv_bridge = CvBridge()
        self.depth_image = None

        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320.5
        self.cy = 240.5
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.image_width = 640  # 图像宽度 (像素)
        self.image_height = 480  # 图像高度 (像素)

        self.F = np.zeros((8, 8))
        for i in range(0, 8, 2): 
            self.F[i, i] = self.fx
        for i in range(1, 8, 2):
            self.F[i, i] = self.fy

        self.F_inv = np.linalg.inv(self.F)

        self.Q = np.eye(8)  # 加权矩阵 (8x8单位矩阵)
        self.R_weight = 20000  # 正则化权重
        self.Np = 3
        self.Ts = 0.1
        self.tau_last = np.full((6, 1), 0.1)

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
            # Convert ROS Image message to OpenCV image
            rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Convert the image from BGR to HSV color space
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
                        
                    s, identified_points= self.identify_key_points(centers)

                    tau = self.mpc_controller(s, Z, self.tau_last)
                    
                    twist_wv =  np.array([tau[2],-tau[4]])

                    print(twist_wv)

                    self.publish_twist(twist_wv)

                    self.tau_last = tau

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
    
    
    def mpc_controller(self, s, Z, tau_last):
        # 定义成本函数
        
        # 初始猜测 (6*Np 维向量)
        tau_seq0 = np.tile(tau_last, self.Np)

        constraints = [
            # 等式约束：vx, vy, wx, wz 为零
            {'type': 'eq', 'fun': lambda tau_seq: tau_seq[0::6]},  # vx = 0
            {'type': 'eq', 'fun': lambda tau_seq: tau_seq[1::6]},  # vy = 0
            {'type': 'eq', 'fun': lambda tau_seq: tau_seq[3::6]},  # wx = 0
            {'type': 'eq', 'fun': lambda tau_seq: tau_seq[5::6]},  # wz = 0

            # 不等式约束：vz 和 wy 在 [-0.5, 0.5] 范围内
            {'type': 'ineq', 'fun': lambda tau_seq: 0.5 - tau_seq[2::6]},  # vz <= 0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.5 + tau_seq[2::6]},  # vz >= -0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.3 - tau_seq[4::6]},  # wy <= 0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.3 + tau_seq[4::6]},  # wy >= -0.5
        ]

        # 调用 minimize 求解
        res = minimize(self.cost_function, tau_seq0, args=(s, Z), method='SLSQP', constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})
        # res = minimize(self.cost_function, tau_seq0, args=(s, Z), constraints={'type': 'ineq', 'fun': self.fov_constraints, 'args': (s, Z)}, options={'maxiter': 5000})

        # 提取第一个控制输入
        tau = res.x[:6]

        return tau
    
    def cost_function(self, tau_seq, s, Z):

        # 将 tau_seq 转换为 NumPy 数组
        tau_seq = np.asarray(tau_seq)
        tau_seq = tau_seq.flatten()

        # 定义交互矩阵
        def interaction_matrix(s, Z, K):
            # s: 视觉特征向量 (2Nx1)，形式为 [u1; v1; u2; v2; ...; uN; vN]
            # Z: 特征点的深度 (相机到特征点的距离)
            # K: 相机内参矩阵 (3x3)

            # 提取相机内参
            fx = K[0, 0]  # 焦距 fx
            fy = K[1, 1]  # 焦距 fy
            cx = K[0, 2]  # 图像中心 cx
            cy = K[1, 2]  # 图像中心 cy

            # 提取 u 和 v 坐标
            u = s[::2]  # u 坐标
            v = s[1::2]  # v 坐标

            # 将 u 和 v 转换为归一化的 x 和 y
            x = (u - cx) / fx  # 归一化 x 坐标
            y = (v - cy) / fy  # 归一化 y 坐标

            # 初始化交互矩阵
            Ls = []

            # 计算每个特征点的交互矩阵
            for i in range(len(x)):
                Lsi = np.array([[-1 / Z, 0, x[i,0] / Z, x[i,0] * y[i,0], -(1 + x[i,0] ** 2), y[i,0]],
                                [0, -1 / Z, y[i,0] / Z, 1 + y[i,0] ** 2, -x[i,0] * y[i,0], -x[i,0]]])
                Ls.append(Lsi)

            L = np.vstack(Ls)  # 将每个特征点的交互矩阵堆叠起来
            return L
        
        J1 = 0
        J2 = 0
        s_pred = s
        for i in range(self.Np):
            # 计算交互矩阵
            L = interaction_matrix(s_pred, Z, self.K)
            # 提取当前控制输入
            tau = tau_seq[6*i:6*(i+1)].reshape(-1, 1)  # 第 j 个控制输入
            
            # 更新视觉特征
            s_pred = s_pred + self.Ts * (self.F @ L @ tau)
            # 计算误差
            e = s_pred - self.desired_uv.reshape(-1, 1)
            # 累加成本
            J1 += e.T @ self.Q @ e
            J2 += self.R_weight * (tau.T @ tau)
        
        J = J1 + J2
        return J

    ## 定义视野约束
    # def fov_constraints(self, tau_seq, s, Z):
    #     c = []
    #     s_pred = s
    #     for j in range(self.Np):
    #         # 计算交互矩阵
    #         Ls = self.interaction_matrix(s_pred, Z, self.K)
    #         # 提取当前控制输入
    #         tau = tau_seq[j * 6:(j + 1) * 6]  # 第 j 个控制输入
    #         # 更新视觉特征
    #         s_pred = s_pred + self.Ts * self.F @ Ls @ tau
    #         # 提取特征点坐标 (u, v)
    #         u = s_pred[::2]  # u 坐标
    #         v = s_pred[1::2]  # v 坐标
    #         # 添加视野约束 (确保特征点在图像边界内)
    #         c.extend([-u + 0,  # u >= 10 (留出边界)
    #                 u - (self.image_width - 0),  # u <= image_width - 10
    #                 -v + 0,  # v >= 10
    #                 v - (self.image_height - 0)])  # v <= image_height - 10
    #     c = np.concatenate(c)
    #     return np.array(c)
    
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