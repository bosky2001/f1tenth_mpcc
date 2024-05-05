#!/usr/bin/env python3
import numpy as np 
import casadi as ca
import math

from dataclasses import dataclass, field
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from utils.track_utils import CentreLine, TrackLine, RaceLine
from utils.mpcc_utils import *


from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry

import time
@dataclass
class mpcc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, yaw, s]
    NU: int = 2 # length of input vector: u = = [steering angle, progress rate]
    TK: int = 12  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MIN_P: float = 0
    MAX_P: float = 4
    MAX_Fn:float = 36.0

    REF_V: float = 1.5      #reference speed set for just lateral control
    S_wt: float = 0.001
    Lag_wt: float = 10
    Contour_wt: float = 20
    Steer_wt: float = 100
    sv_wt: float = 0.001
    accl_wt: float = 0.00001
    P_INIT: float = 5

# just lateral because faster to solve
class MPCCPlanner(Node):
    def __init__(self):
        super().__init__('mpcc_node')
        self.config = mpcc_config()
        self.track_width = None
        self.centre_interpolant = None
        self.left_interpolant, self.right_interpolant = None, None
        self.g, self.obj = None, None
        self.f_max = self.config.MAX_Fn

        self.dt = self.config.DTK
        self.N = self.config.TK
        self.u0 = np.zeros((self.N, self.config.NU))
        self.X0 = np.zeros((self.N + 1, self.config.NXK))

        self.f_max = self.config.MAX_Fn

        self.optimisation_parameters = np.zeros(self.config.NXK + 2 * self.N + 1)
        self.ref_goal_points_ = self.create_publisher(MarkerArray, 'ref_goal_points', 1)

        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.drive_msg_ = AckermannDriveStamped()
        self.is_real = False
        pose_topic = "/pf/viz/inferred_pose" if self.is_real else "/ego_racecar/odom"
        self.pose_sub_ = self.create_subscription(PoseStamped if self.is_real else Odometry, pose_topic, self.pose_callback, 1)
        
        # 
        # self.centre_line = CentreLine("aut")
        
        # print(self.centre_line.path)
        self.mpcc_prob_init()


        self.ref_goal_points_data = self.viz_ref_points()
    
    def mpcc_prob_init(self):
        # self.centre_line = CentreLine(map_name)
        self.centre_line = RaceLine("levine")

        self.centre_interpolant, self.left_interpolant, self.right_interpolant = init_track_interpolants(self.centre_line, 0.4)

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        s = ca.SX.sym('s')

        # [x, y, psi, s]
        states = ca.vertcat(
            x,
            y,
            yaw,
            s
        )
        n_states = states.numel()

        delta = ca.SX.sym('delta')
        # v = ca.SX.sym('v')
        p = ca.SX.sym('p')

        controls = ca.vertcat(
            delta,
            p
        )
        n_controls = controls.numel()

        # setting reference speed
        v = self.config.REF_V

        # dynamics stuff
        RHS = ca.vertcat(v * ca.cos(yaw), 
                         v * ca.sin(yaw), 
                         (v / self.config.WB) * ca.tan(delta), 
                         p)  # dynamic equations of the states

        self.f = ca.Function('f', [states, controls], [RHS])  # nonlinear mapping function f(x,u)
        self.U = ca.MX.sym('U', n_controls, self.N)
        self.X = ca.MX.sym('X', n_states, (self.N + 1))
        self.P = ca.MX.sym('P', n_states + 2 * self.N + 1) # init state and boundaries of the reference path


        '''Initialize upper and lower bounds for state and control variables'''

        # setting lower bounds
        # [x, y, yaw, s]
        lbx = [[-np.inf], [-np.inf], [-10.0], [0]] * (self.N + 1) \
                + [[self.config.MIN_STEER], [self.config.MIN_P]] * self.N
        self.lbx = np.array(lbx)

        # setting upper bounds
        # [x, y, yaw, s]
        ubx = [[np.inf], [np.inf], [10], [300]] * (self.N + 1) + \
                [[self.config.MAX_STEER], [self.config.MAX_P]] * self.N
        self.ubx = np.array(ubx)

        self.init_objective()
        self.init_bounds()
        # self.init_bound_limits()
        
        # solver settings
        optimisation_variables = ca.vertcat(ca.reshape(self.X, n_states * (self.N + 1), 1),
                                ca.reshape(self.U, n_controls * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': optimisation_variables, 'g': self.g, 'p': self.P}
        opts = {"ipopt": {"max_iter": 2000, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def init_objective(self):

        self.obj = 0  # Objective function

        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.centre_interpolant.lut_angle(st_next[3])
            ref_x, ref_y = self.centre_interpolant.lut_x(st_next[3]), self.centre_interpolant.lut_y(st_next[3])
            countour_error = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            lag_error = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + countour_error **2 * self.config.Contour_wt 
            self.obj = self.obj + lag_error **2 * self.config.Lag_wt
            self.obj = self.obj - self.U[1, k] * self.config.S_wt 
            self.obj = self.obj + (self.U[0, k]) ** 2 * self.config.Steer_wt

    def init_bounds(self):
       #Initialise the bounds (g) on the dynamics and track boundaries
        NX = self.config.NXK
        self.g = self.X[:, 0] - self.P[:NX]  # initial condition constraints

        for k in range(self.N):
            st_next = self.X[:, k + 1]
            k1 = self.f(self.X[:, k], self.U[:, k])
            st_next_euler = self.X[:, k] + (self.config.DTK* k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # add dynamics constraint

            # LB<=ax-by<=UB  :represents path boundary constraints
            constraint = self.P[NX + 2 * k] * st_next[0] - self.P[NX + 2 * k + 1] * st_next[1]
            self.g = ca.vertcat(self.g, constraint) 
        
        # init nound limits
        self.lbg = np.zeros((self.g.shape[0], 1))
        self.ubg = np.zeros((self.g.shape[0], 1))
        
    # def init_bound_limits(self):

    #     self.lbg = np.zeros((self.g.shape[0], 1))
    #     self.ubg = np.zeros((self.g.shape[0], 1))



    def plan(self, obs):
        # self.step_counter += 1
        x0 = obs
        self.optimisation_parameters[:self.config.NXK] = x0

        self.set_path_constraints()
        states, controls, solved_status = self.solve()
        if not solved_status:
            self.construct_warm_start_soln(x0) 
            self.set_path_constraints()
            states, controls, solved_status = self.solve()
            if not solved_status:
                print("--> Optimisation has not been solved!!!!!!!!")
                return np.array([0, 1])

        action = np.array([controls[0, 0], self.config.REF_V])

        return action 

    def set_path_constraints(self):

        NX = self.config.NXK

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            right_point = self.right_interpolant.get_point(self.X0[k, 3])
            left_point = self.left_interpolant.get_point(self.X0[k, 3])
            delta_point = right_point - left_point
            delta_point[0] = -delta_point[0]

            self.optimisation_parameters[NX + 2 * k:NX + 2 * k + 2] = delta_point

            right_bound = delta_point[0] * right_point[0] - delta_point[1] * right_point[1]
            left_bound = delta_point[0] * left_point[0] - delta_point[1] * left_point[1]

            self.lbg[NX - 1 + (NX + 1) * (k + 1), 0] = min(left_bound, right_bound)
            self.ubg[NX - 1 + (NX + 1) * (k + 1), 0] = max(left_bound, right_bound)

    def solve(self):

        NX = self.config.NXK
        NU = self.config.NU
        x_init = ca.vertcat(ca.reshape(self.X0.T, NX * (self.N + 1), 1),
                         ca.reshape(self.u0.T, NU * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=self.optimisation_parameters)

        # Get state and control solution
        self.X0 = ca.reshape(sol['x'][0:NX * (self.N + 1)], NX, self.N + 1).T  # get soln trajectory
        u = ca.reshape(sol['x'][NX * (self.N + 1):], NU, self.N).T  # get controls solution

        trajectory = self.X0.full()  # size is (N+1,n_states)
        inputs = u.full()
        solved_status = True
        if self.solver.stats()['return_status'] == 'Infeasible_Problem_Detected':
            solved_status = False

        self.X0 = ca.vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = ca.vertcat(u[1:, :], u[u.size1() - 1, :])

        return trajectory, inputs, solved_status

    def construct_warm_start_soln(self, initial_state):
        NX = self.config.NXK
        NU = self.config.NU
        self.X0 = np.zeros((self.N + 1, NX))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.config.P_INIT * self.config.DTK

            psi_next = self.centre_interpolant.lut_angle(s_next).full()[0, 0]
            x_next, y_next = self.centre_interpolant.lut_x(s_next), self.centre_interpolant.lut_y(s_next)

            # adjusts the centerline angle to be continuous
            psi_diff = self.X0[k-1, 2] - psi_next
            psi_mul = self.X0[k-1, 2] * psi_next
            if (abs(psi_diff) > np.pi and psi_mul < 0) or abs(psi_diff) > np.pi*1.5:
                if psi_diff > 0:
                    psi_next += np.pi * 2
                else:
                    psi_next -= np.pi * 2
            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next, s_next])

    def pose_callback(self, pose_msg):
        
        start = time.time()

        x_state = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        y_state = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y

        curr_orien = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        q = [curr_orien.x, curr_orien.y, curr_orien.z, curr_orien.w]
        yaw_state = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))

        state = np.array([x_state, y_state, yaw_state])
        state = np.hstack((state, self.centre_line.calculate_progress_m(state[:2])))
        # x0 = np.append(obs["pose"], self.centre_line.calculate_progress_m(obs["pose"][0:2]))
        state[2] = normalise_psi(state[2])

        print(state) 

        steer, vel = self.plan(state)

        self.drive_msg_.drive.speed = float(vel)
        self.drive_msg_.drive.steering_angle = float(steer)

        print("solve time is: ", time.time() - start)
        self.drive_pub_.publish(self.drive_msg_)
        print(steer)
        self.ref_goal_points_.publish(self.ref_goal_points_data)
        
    ## Visualization MPC utils
    def viz_ref_points(self):
        ref_points = MarkerArray()
        waypoints = self.centre_line.path
        for i in range(waypoints.shape[0]):
            message = Marker()
            message.header.frame_id="map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.action = Marker.ADD
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.2
            message.scale.y=0.2
            message.scale.z=0.2
            message.color.a=1.0
            message.color.r=1.0
            message.color.b=0.0
            message.color.g=0.0
            message.pose.position.x=float(waypoints[i,0])
            message.pose.position.y=float(waypoints[i,1])
            message.pose.position.z=0.0
            ref_points.markers.append(message)
        return ref_points



def main(args=None):
    
    rclpy.init(args=args)
    print("MPCC Initialized")
    mpc_node = MPCCPlanner()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()