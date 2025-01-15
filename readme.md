### Introduction
本仓库提供了三个仿真，基于turtlebot3+gazebo：
基于控制律的单特征仿真：ibvs_single.py，
基于控制律的多特征仿真：ibvs_multi.py,
基于MPC的多特征仿真：mpc_control2.py、mpc_detect.py

### IBVS by ControlLaw
**单特征**
**(1)**
视觉伺服的目标是让归一化图像坐标系下特征点的误差$\boldsymbol{e}$以指数形式收敛：
\[
\boldsymbol{\dot{e}}=\bigg[ \begin{matrix}
\dot{x}\\
\dot{y}\\
\end{matrix}\bigg]=-k\boldsymbol{e}
=-k\bigg[ \begin{matrix}
x - x_d\\
y - y_d\\
\end{matrix}\bigg]
\]

根据相机模型，相机内参和成像关系为：
\[
x = \frac{u - c_x}{f_x}, \quad y = \frac{v - c_y}{f_y}
\]

代入后得到像素坐标系下特征点的误差变化率：
\[
\boldsymbol{\dot{e}} =
\bigg[ \begin{matrix}
\dot{x}\\
\dot{y}\\
\end{matrix}\bigg]
= \begin{bmatrix} f_x & 0\\
0 & f_y
\end{bmatrix}
\bigg[ \begin{matrix}
\dot{u}\\
\dot{v}\\
\end{matrix}\bigg]=
-k
\begin{bmatrix} f_x & 0\\
0 & f_y
\end{bmatrix}
\bigg[ \begin{matrix}
u-u_d\\
v-v_d\\
\end{matrix}\bigg]=
-k\boldsymbol{F}\boldsymbol{e_{uv}}
\]

**(2)**
图像雅可比的定义:
\[
\boldsymbol{\dot{e}}=\boldsymbol{L}\mathcal{V_{cam}}
\]

其中：
\[
\mathcal{V_{cam}}=
\begin{bmatrix}
v_x\\
v_y\\
v_z\\
\omega_x\\
\omega_y\\
\omega_z\\
\end{bmatrix},
\boldsymbol{L} = \begin{bmatrix}
-\frac{1}{Z} & 0 & \frac{x}{Z} & xy & -(1 + x^2) & y \\
0 & -\frac{1}{Z} & \frac{y}{Z} & 1 + y^2 & -xy & -x
\end{bmatrix}
\]

由相机与底盘的位置关系可知：
\[
\boldsymbol{\dot{e}} = \boldsymbol{L}\mathcal{V_{cam}} = \boldsymbol{L}\boldsymbol{{^{cam}X_{car}}}\mathcal{V_{car}}
 = \boldsymbol{L}\boldsymbol{{^{cam}X_{car}}}\boldsymbol{J} \mathcal{V_{car_2}}
\]

其中$\boldsymbol{J}$用于筛选平面上的运动，即$v_x$和$\omega_z$：
\[
\boldsymbol{J} = \begin{bmatrix}
1 & 0\\
0 & 0\\
0 & 0\\
0 & 0\\
0 & 0\\
0 & 1
\end{bmatrix},
\mathcal{V_{car_2}} =
\begin{bmatrix}
v_x \\
\omega_z
\end{bmatrix}
\]

\[
\boldsymbol{^{cam}X_{car}} = \begin{bmatrix}
\boldsymbol{R} & [\boldsymbol{t}] \boldsymbol{R} \\
\boldsymbol{O} & \boldsymbol{R}
\end{bmatrix}
\]

其中$\boldsymbol{R},\boldsymbol{t}$ 来自 $\boldsymbol{^{cam}T_{car}}$。
令：
\[
\boldsymbol{M} = \boldsymbol{L}\boldsymbol{{^{cam}X_{car}}}\boldsymbol{J} , \boldsymbol{M} \in \mathbb{R}^{2 \times 2}
\]

联合(1)得到最终输出：
\[
\mathcal{V}_{car_2} = -k\boldsymbol{M^{-1}}\boldsymbol{F}\boldsymbol{e_{uv}}
\]

**多特征**
\[
\boldsymbol{L} = \begin{bmatrix}
\boldsymbol{L_1}\\
\boldsymbol{L_2}\\
\boldsymbol{L_3}\\
\boldsymbol{L_4}
\end{bmatrix} \in \mathbb{R}^{8 \times 6}，
\boldsymbol{F} = 
\begin{bmatrix}
f_x & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & f_y & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & f_x & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & f_y & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & f_x & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & f_y & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & f_x & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & f_y
\end{bmatrix}
\]

\[
\boldsymbol{M} = \boldsymbol{L}\boldsymbol{^{cam}X_{car}} \boldsymbol{J} , \boldsymbol{M} \in \mathbb{R}^{8 \times 2}
\]

\[
\mathcal{V}_{car_2} = -k\boldsymbol{M^{+}}\boldsymbol{F}\boldsymbol{e_{uv}}
\]

$\boldsymbol{M^{+}}$为Moore Penrose伪逆:
\[
\boldsymbol{M^{+}} \in \mathbb{R}^{2 \times 8}
\]

### IBVS by MPC

Cost Function:
\[
\min_{u_{k|k}, \dots, u_{k+N_p-1|k}} J(u) = \sum_{i=k+1}^{k+N_p} e_{i|k}^T Q e_{i|k} + u_{i|k}^T R u_{i|k}
\]

subject to:

\[
s_{i+1|k} = s_{i|k} + T_s L u_{i|k}, \quad \forall i = k, \dots, k+N_p-1
\]

\[
e_{i|k} = s_{i|k} - s_d, \quad \forall i = k+1, \dots, k+N_p
\]

\[
s_{\text{min}} \leq s_{i|k} \leq s_{\text{max}}, \quad \forall i = k+1, \dots, k+N_p
\]

\[
u_{\text{min}} \leq u_{i|k} \leq u_{\text{max}}, \quad \forall i = k, \dots, k+N_p-1
\]

### How to use:
**1 创建工作空间**
```
cd src
catkin_init_workspace
cd ..
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```
**2 启动仿真环境**
```
roslaunch visual_servo gazebo_pioneer_servo.launch
```
**3 启动控制器：**
打开另外一个终端:
```
rosrun ibvs ibvs_single.py 
```
或者
```
rosrun ibvs ibvs_multi.py 
```
或者
```
roslaunch ibvs mpc.launch
```