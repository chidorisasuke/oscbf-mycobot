"""Testing the performance of OSCBF for UR5e with different scenarios.
1. Dynamic moving obstacle avoidance.
2. End-effector workspace constraint.

Can switch between Pose (6D) and Position (3D) control via command-line.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.manipulation_env import UR5eTorqueControlEnv, UR5eVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
    PositionTaskTorqueController,
    PositionTaskVelocityController,
)

# ================== KONFIGURASI CBF UNTUK RINTANGAN BERGERAK ==================
@jax.tree_util.register_static
class DynamicCollisionTorqueConfig(OSCBFTorqueConfig):
    def __init__(self, robot: Manipulator, 
                 obstacle_radius: float, 
                 compensate_centrifugal_coriolis: bool,
                 initial_obstacle_pos: ArrayLike):
        self.obstacle_radius = obstacle_radius
        init_pos_tuple = tuple(map(float, initial_obstacle_pos))
        super().__init__(robot, 
                         compensate_centrifugal_coriolis=compensate_centrifugal_coriolis, 
                         init_args=(init_pos_tuple,))

    def h_2(self, z, obstacle_pos, **kwargs):
        q = z[: self.robot.num_joints]
        obstacle_pos = jnp.asarray(obstacle_pos)
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        if robot_collision_pos_rad.size == 0:
            return jnp.array([1.0])
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3]
        center_deltas = robot_collision_positions - obstacle_pos[None, :]
        distances = jnp.linalg.norm(center_deltas, axis=1)
        radii_sums = robot_collision_radii + self.obstacle_radius
        h_collision = distances - radii_sums
        return jnp.array([jnp.min(h_collision)])

    def alpha_2(self, h_2):
        return 10.0 * h_2

@jax.tree_util.register_static
class DynamicCollisionVelocityConfig(OSCBFVelocityConfig):
    def __init__(self, 
                 robot: Manipulator, 
                 obstacle_radius: float,
                 initial_obstacle_pos: ArrayLike):
        self.obstacle_radius = obstacle_radius
        init_pos_tuple = tuple(map(float, initial_obstacle_pos))
        super().__init__(robot, init_args=(init_pos_tuple,)) 

    def h_1(self, z, obstacle_pos, **kwargs):
        q = z[: self.robot.num_joints]
        obstacle_pos = jnp.asarray(obstacle_pos)
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        if robot_collision_pos_rad.size == 0:
            return jnp.array([1.0]) 
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3]
        center_deltas = robot_collision_positions - obstacle_pos[None, :]
        distances = jnp.linalg.norm(center_deltas, axis=1)
        radii_sums = robot_collision_radii + self.obstacle_radius
        h_collision = distances - radii_sums
        return jnp.array([jnp.min(h_collision)])

    def alpha(self, h):
        return 10.0 * h

# ================== KONFIGURASI CBF UNTUK RUANG KERJA (WORKSPACE) ==================
@jax.tree_util.register_static
class EESafeSetTorqueConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        compensate_centrifugal_coriolis: bool,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(
            robot, compensate_centrifugal_coriolis=compensate_centrifugal_coriolis
        )

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

@jax.tree_util.register_static
class EESafeSetVelocityConfig(OSCBFVelocityConfig):
    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

# ================== FUNGSI COMPUTE CONTROL ==================
def compute_pose_torque_control(robot, osc_controller, cbf, compensate, z, z_ee_des, obstacle_pos=None):
    q, qdot = z[:robot.num_joints], z[robot.num_joints:]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    if not compensate: c = jnp.zeros(robot.num_joints)
    nullspace_posture_goal = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    u_nom = osc_controller(q, qdot, ee_tmat[:3,3], ee_tmat[:3,:3], z_ee_des[:3], jnp.reshape(z_ee_des[3:12],(3,3)), z_ee_des[12:15], z_ee_des[15:18], jnp.zeros(3), jnp.zeros(3), nullspace_posture_goal, jnp.zeros(robot.num_joints), J, M, M_inv, g, c)

    if obstacle_pos is not None:
        tau_safe = cbf.safety_filter(z, u_nom, obstacle_pos)
    else:
        tau_safe = cbf.safety_filter(z, u_nom)
    
    q_ddot_safe = M_inv @ (tau_safe - c - g)

    return tau_safe, q_ddot_safe

def compute_pose_velocity_control(robot, osc_controller, cbf, z, z_ee_des, obstacle_pos=None):
    q = z[:robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    des_q = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    u_nom = osc_controller(q, ee_tmat[:3,3], ee_tmat[:3,:3], z_ee_des[:3], jnp.reshape(z_ee_des[3:12],(3,3)), z_ee_des[12:15], z_ee_des[15:18], des_q, J, M_inv)
    
    if obstacle_pos is not None:
        return cbf.safety_filter(q, u_nom, obstacle_pos)
    else:
        return cbf.safety_filter(z, u_nom)

def compute_position_torque_control(robot, osc_controller, cbf, compensate, z, z_ee_des, obstacle_pos=None):
    q, qdot = z[:robot.num_joints], z[robot.num_joints:]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    Jv = J[:3,:]
    if not compensate: c = jnp.zeros(robot.num_joints)
    nullspace_posture_goal = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    u_nom = osc_controller(q, qdot, ee_tmat[:3,3], z_ee_des[:3], z_ee_des[12:15], jnp.zeros(3), nullspace_posture_goal, jnp.zeros(robot.num_joints), Jv, M, M_inv, g, c)
    if obstacle_pos is not None:
        tau_safe = cbf.safety_filter(q, u_nom, obstacle_pos)
    else:
        tau_safe = cbf.safety_filter(z, u_nom)
    
    q_ddot_safe = M_inv @ (tau_safe - c -g)
    return tau_safe, q_ddot_safe

def compute_position_velocity_control(robot, osc_controller, cbf, z, z_ee_des, obstacle_pos=None):
    q = z[:robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    Jv = J[:3,:]
    nullspace_posture_goal = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    u_nom = osc_controller(q, ee_tmat[:3,3], z_ee_des[:3], z_ee_des[12:15], nullspace_posture_goal, Jv, M_inv)
    if obstacle_pos is not None:
        return cbf.safety_filter(q, u_nom, obstacle_pos)
    else:
        return cbf.safety_filter(z, u_nom)
# ========================================================================

def main(control_method="torque", task_type="pose", scenario="obstacle_avoidance"):
    assert control_method in ["torque", "velocity"]
    assert task_type in ["pose", "position"]
    assert scenario in ["obstacle_avoidance", "workspace_constraint"]

    robot = load_ur5e()
    compensate_centrifugal_coriolis = False

    # ================== LINTASAN TARGET (BOLA MERAH) ==================
    ee_traj = SinusoidalTaskTrajectory(
        init_pos=np.array([0.55, 0, 0.45]),
        init_rot=np.eye(3),
        amplitude=(0.25, 0, 0),
        angular_freq=(1.5, 0, 0),
        phase=(np.pi/2, 0, 0)
    )
    
    q_init = (0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0)
    timestep = 1 / 240
    env_kwargs = {
        'q_init': q_init, 
        'traj': ee_traj, 
        'real_time': True, 
        'bg_color': (1,1,1), 
        'load_floor': False, 
        'timestep': timestep, 
        'load_table': True
    }

    # ================== PENGATURAN BERDASARKAN SKENARIO ==================
    if scenario == "obstacle_avoidance":
        print("Running scenario: Obstacle Avoidance")
        obstacle_radius = 0.03
        initial_obstacle_pos = np.array([0.5, 0.0, 0.2])
        obstacle_traj = SinusoidalTaskTrajectory(
            init_pos=np.array([0.55, 0, 0.45]),
            init_rot=np.eye(3),
            amplitude=(0, 0.4, 0),
            angular_freq=(0, 0.75, 0),
            phase=(0, 0, 0)
        )
        torque_cbf = CBF.from_config(DynamicCollisionTorqueConfig(
            robot, obstacle_radius, compensate_centrifugal_coriolis, initial_obstacle_pos
        ))
        velocity_cbf = CBF.from_config(DynamicCollisionVelocityConfig(
            robot, obstacle_radius, initial_obstacle_pos
        ))
        if control_method == "torque":
            env = UR5eTorqueControlEnv(**env_kwargs)
        else:
            env = UR5eVelocityControlEnv(**env_kwargs)
        # Visualisasi rintangan (bola biru)
        vis_id = env.client.createVisualShape(env.client.GEOM_SPHERE, radius=obstacle_radius, rgbaColor=[0, 0, 1, 0.8])
        obstacle_body_id = env.client.createMultiBody(baseVisualShapeIndex=vis_id)

    elif scenario == "workspace_constraint":
        print("Running scenario: Workspace Constraint")
        pos_min = (0.25, -0.35, 0.15)
        pos_max = (0.75, 0.35, 0.65)
        env_kwargs['xyz_min'] = pos_min
        env_kwargs['xyz_max'] = pos_max
        torque_cbf = CBF.from_config(EESafeSetTorqueConfig(
            robot, pos_min, pos_max, compensate_centrifugal_coriolis
        ))
        velocity_cbf = CBF.from_config(EESafeSetVelocityConfig(robot, pos_min, pos_max))
        if control_method == "torque":
            env = UR5eTorqueControlEnv(**env_kwargs)
        else:
            env = UR5eVelocityControlEnv(**env_kwargs)

    env.client.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=(0.4,0,0.3))
    # =======================================================================

    # Gains
    kp_pos = 150.0
    kp_rot = 80.0
    kd_pos = 50.0
    kd_rot = 20.0
    kp_joint = 10.0
    kd_joint = 5.0

    # kp_pos = 25.0 
    # kp_rot = 15.0
    # kd_pos = 15.0 # Ingat, kd seringkali < kp
    # kd_rot = 8.0
    # kp_joint = 5.0
    # kd_joint = 2.0

    # Logika pemilihan kontroler
    if task_type == "pose":
        print("Using control mode: POSE (6D)")
        osc_torque_controller = PoseTaskTorqueController(n_joints=robot.num_joints, kp_task=np.concatenate([kp_pos*np.ones(3),kp_rot*np.ones(3)]), kd_task=np.concatenate([kd_pos*np.ones(3),kd_rot*np.ones(3)]), kp_joint=kp_joint, kd_joint=kd_joint, tau_min=None, tau_max=None)
        osc_velocity_controller = PoseTaskVelocityController(n_joints=robot.num_joints, kp_task=np.array([kp_pos,kp_pos,kp_pos,kp_rot,kp_rot,kp_rot]), kp_joint=kp_joint, qdot_min=None, qdot_max=None)
        compute_torque_control_jit = jax.jit(compute_pose_torque_control, static_argnums=(0,1,2,3))
        compute_velocity_control_jit = jax.jit(compute_pose_velocity_control, static_argnums=(0,1,2))
    elif task_type == "position":
        print("Using control mode: POSITION (3D) + Nullspace Posture")
        osc_torque_controller = PositionTaskTorqueController(n_joints=robot.num_joints, kp_task=kp_pos, kd_task=kd_pos, kp_joint=kp_joint, kd_joint=kd_joint, tau_min=None, tau_max=None)
        osc_velocity_controller = PositionTaskVelocityController(n_joints=robot.num_joints, kp_task=kp_pos, kp_joint=kp_joint, qdot_min=None, qdot_max=None)
        compute_torque_control_jit = jax.jit(compute_position_torque_control, static_argnums=(0,1,2,3))
        compute_velocity_control_jit = jax.jit(compute_position_velocity_control, static_argnums=(0,1,2))

    if control_method == "torque":
        compute_control = compute_torque_control_jit
        cbf = torque_cbf
        osc_controller = osc_torque_controller
    else:
        compute_control = compute_velocity_control_jit
        cbf = velocity_cbf
        osc_controller = osc_velocity_controller

    last_log_time = -1.0
    log_interval = 0.25
    # Main loop
    while True:
        current_time = env.t
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        
        cbf_kwargs = {}
        current_obstacle_pos = None # Default jika tidak ada obstacle
        if scenario == "obstacle_avoidance":
            current_obstacle_pos = obstacle_traj.position(env.t)
            env.client.resetBasePositionAndOrientation(obstacle_body_id, current_obstacle_pos, [0,0,0,1])
            # cbf_kwargs['obstacle_pos'] = current_obstacle_pos

        q_ddot_safe = None
        if control_method == "torque":
            if scenario == "obstacle_avoidance":
                tau, q_ddot_safe = compute_control(robot, osc_controller, cbf, compensate_centrifugal_coriolis, q_qdot, z_zdot_ee_des, current_obstacle_pos)
        else: # velocity
            tau = compute_control(robot, osc_controller, cbf, q_qdot, z_zdot_ee_des)

        env.apply_control(tau)
        env.step()

        if current_time - last_log_time >= log_interval:
            print(f"\n--- LOG @ Waktu: {current_time:.2f} s ---")
            
            # Dapatkan kecepatan aktual (q_dot)
            current_q_dot = q_qdot[robot.num_joints:]
            print(f"Kecepatan Sendi (rad/s): {np.round(current_q_dot, 3)}")
            
            # Cetak percepatan jika ada (hanya untuk torque control)
            if q_ddot_safe is not None:
                print(f"Percepatan Sendi (rad/s^2): {np.round(np.asarray(q_ddot_safe), 3)}")
            else:
                 # Jika velocity control, tau adalah target kecepatan
                 print(f"Target Kecepatan Sendi (rad/s): {np.round(np.asarray(tau), 3)}")

            last_log_time = current_time # Update timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UR5e dynamic motion scenarios.")
    parser.add_argument("--control_method", type=str, choices=["torque", "velocity"], default="torque", help="Pilih metode kontrol: 'torque' untuk kontrol berbasis torsi, 'velocity' untuk kontrol berbasis kecepatan.")
    parser.add_argument("--task_type", type=str, choices=["pose", "position"], default="pose", help="Pilih tipe tugas: 'pose' untuk kontrol 6D (posisi & orientasi), 'position' untuk kontrol 3D (hanya posisi).")
    parser.add_argument("--scenario", type=str, choices=["obstacle_avoidance", "workspace_constraint"], default="obstacle_avoidance", help="Pilih skenario keamanan: 'obstacle_avoidance' (hindari bola biru) atau 'workspace_constraint' (tetap di dalam kotak hijau).")
    args = parser.parse_args()
    main(control_method=args.control_method, task_type=args.task_type, scenario=args.scenario)