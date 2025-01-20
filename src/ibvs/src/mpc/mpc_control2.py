#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from scipy.optimize import minimize
from ibvs.msg import DetectionResult

class MPCController:
    def __init__(self):
        rospy.init_node('mpc_controller', anonymous=True)
        self.detection_sub = rospy.Subscriber('/detection_result', DetectionResult, self.detection_callback)
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320.5
        self.cy = 240.5
        self.image_width = 640
        self.image_height = 480
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.Q = 100 * np.eye(8)
        self.R = 1 * np.eye(2)
        self.K = 1000

        self.Np = 5
        self.Ts = 0.1

        self.tau_seq_last = np.full((self.Np*2, 1), 0.1)
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
        
        self.desired_xy = self.transform_uv_to_xy(self.desired_uv)
        self.s = None
        self.xy_vector = None
        self.Z = None

        self.rate = rospy.Rate(10)  # 10Hz

    def detection_callback(self, msg):
        self.s = np.array(msg.s).reshape(-1, 1)
        self.xy_vector = self.transform_uv_to_xy(self.s)
        self.Z = msg.Z


    def transform_uv_to_xy(self, uv_vector):

        uv_matrix = uv_vector.reshape(-1, 2)

        x = (uv_matrix[:, 0] - self.cx) / self.fx
        y = (uv_matrix[:, 1] - self.cy) / self.fy

        xy_vector = np.column_stack((x, y)).reshape(-1, 1)

        return xy_vector
    
    def transform_xy_to_uv(self, xy_vector):

        xy_matrix = xy_vector.reshape(-1, 2)

        u_vec = xy_matrix[:, 0] * self.fx + self.cx
        v_vec = xy_matrix[:, 1] * self.fy + self.cy

        return u_vec, v_vec

    def interaction_matrix(self, xy_vec, Z):

        xy_matrix = xy_vec.reshape(-1, 2)
        x = xy_matrix[:, 0]
        y = xy_matrix[:, 1]

        Ls = []
        for i in range(len(x)):
            Lsi = np.array([[x[i] / Z, -(1 + x[i] ** 2)],
                            [y[i] / Z, -x[i] * y[i]]])
            Ls.append(Lsi)
        L = np.vstack(Ls)
        return L

    def calculate_squared_difference(self, xy_vec):

        xy_matrix = xy_vec.reshape(-1, 2)
        y = xy_matrix[:, 1]

        diff1 = np.abs(y[0] - y[2])
        diff2 = np.abs(y[1] - y[3])

        result = (1 - diff1/diff2) ** 2

        return result

    def mpc_controller(self):
        if self.xy_vector is None or self.Z is None:
            return
        
        # 定义视野约束
        def fov_constraints(tau_seq, xy_vector, Z):
            tau_seq = np.asarray(tau_seq)
            tau_seq = tau_seq.flatten()

            c = []
            xy_vector_pred = xy_vector
            for j in range(self.Np):
                Ls = self.interaction_matrix(xy_vector_pred, Z)
                tau = tau_seq[2*j:2*(j+1)].reshape(-1, 1)
                xy_vector_pred = xy_vector_pred + self.Ts * (Ls @ tau)
                u_vec, v_vec = self.transform_xy_to_uv(xy_vector_pred)

                c.extend([u_vec + 0,
                        -u_vec + (self.image_width - 10),
                        v_vec + 0,
                        -v_vec + (self.image_height - 10)])
            c = np.concatenate(c)
            return c.flatten()

        # 定义成本函数
        def cost_function(tau_seq, xy_vector, Z):
            tau_seq = np.asarray(tau_seq)
            tau_seq = tau_seq.flatten()

            J1 = 0
            J2 = 0
            xy_vector_pred = xy_vector
            for i in range(self.Np):
                Ls = self.interaction_matrix(xy_vector_pred, Z)
                tau = tau_seq[2*i:2*(i+1)].reshape(-1, 1)
                xy_vector_pred = xy_vector_pred + self.Ts * (Ls @ tau)

                if i == self.Np-1:
                    e = xy_vector_pred - self.desired_xy
                    J1 = e.T @ self.Q @ e
                
                J2 += tau.T @ self.R @ tau
            
            J = J1 + J2
            return J

        # 初始猜测
        tau_seq0 = self.tau_seq_last

        constraints = [
            {'type': 'ineq', 'fun': fov_constraints, 'args': (self.xy_vector, self.Z)},
            {'type': 'ineq', 'fun': lambda tau_seq: 0.5 - tau_seq[0::6]},  # vz <= 0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.5 + tau_seq[0::6]},  # vz >= -0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.3 - tau_seq[1::6]},  # wy <= 0.3
            {'type': 'ineq', 'fun': lambda tau_seq: 0.3 + tau_seq[1::6]},  # wy >= -0.3
        ]

        res = minimize(cost_function, tau_seq0, args=(self.xy_vector, self.Z), method='SLSQP', constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})

        tau = res.x[:2]
        self.tau_seq_last = res.x

        self.publish_twist(tau)
        rospy.loginfo("twist_wv: linear_x = %f, angular_z = %f", tau[0], tau[1])

    def publish_twist(self, twist_car):
        twist_msg = Twist()
        twist_msg.linear.x = np.clip(twist_car[0], -0.5, 0.5)   # v_x
        twist_msg.angular.z = np.clip(-twist_car[1], -0.3, 0.3)  # w_z
        self.twist_pub.publish(twist_msg)

    def run(self):
        while not rospy.is_shutdown():
            try:
                self.mpc_controller()
                self.rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException as e:
                rospy.logwarn("ROS time moved backwards, ignoring: %s", str(e))

if __name__ == '__main__':
    mpc_controller = MPCController()
    mpc_controller.run()