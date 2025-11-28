"""UR5e Circular XY Trajectory with Dynamic Obstacle Avoidance

Robot akan:
1. Bergerak dari posisi awal ke titik start lintasan lingkaran
2. Mengikuti lintasan lingkaran di bidang XY
3. Menghindari obstacle bola biru di tengah lintasan
4. Kembali ke lintasan lingkaran setelah menghindari obstacle
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.manipulation_env import UR5eTorqueControlEnv, UR5eVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
    PositionTaskTorqueController,  
    PositionTaskVelocityController,
)
from oscbf.core.ur5e_collision_model import ur5e_collision_data

# ================== CUSTOM CIRCULAR TRAJECTORY CLASS ==================
# ================== CUSTOM CIRCULAR TRAJECTORY CLASS ==================
class CircularXYTrajectory:
    """Lintasan lingkaran di bidang XY dengan fase transisi awal"""
    
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        z_height: float,
        angular_velocity: float,
        transition_time: float = 3.0,
        init_pos: np.ndarray = None
    ):
        self.center = np.array(center)
        self.radius = radius
        self.z_height = z_height
        
        # PERBAIKAN: Ganti nama variabel self.omega menjadi self.ang_vel
        # agar tidak menimpa method omega() di bawah.
        self.ang_vel = angular_velocity 
        
        self.transition_time = transition_time
        
        # Titik start di lingkaran (di sisi kanan pusat)
        self.circle_start = np.array([
            self.center[0] + self.radius,
            self.center[1],
            self.z_height
        ])
        
        if init_pos is not None:
            self.init_pos = np.array(init_pos)
        else:
            self.init_pos = self.circle_start.copy()
    
    def position(self, t: float) -> np.ndarray:
        """Hitung posisi target pada waktu t"""
        if t < self.transition_time:
            alpha = t / self.transition_time
            alpha = 3*alpha**2 - 2*alpha**3
            pos = (1 - alpha) * self.init_pos + alpha * self.circle_start
        else:
            t_circle = t - self.transition_time
            # Update: gunakan self.ang_vel
            theta = self.ang_vel * t_circle
            pos = np.array([
                self.center[0] + self.radius * np.cos(theta),
                self.center[1] + self.radius * np.sin(theta),
                self.z_height
            ])
        return pos
    
    def rotation(self, t: float) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
    
    def velocity(self, t: float) -> np.ndarray:
        """Hitung kecepatan target pada waktu t"""
        if t < self.transition_time:
            alpha = t / self.transition_time
            d_alpha = (6*alpha - 6*alpha**2) / self.transition_time
            vel = d_alpha * (self.circle_start - self.init_pos)
        else:
            t_circle = t - self.transition_time
            # Update: gunakan self.ang_vel
            theta = self.ang_vel * t_circle
            # Update: gunakan self.ang_vel
            vel = np.array([
                -self.radius * self.ang_vel * np.sin(theta),
                self.radius * self.ang_vel * np.cos(theta),
                0.0
            ])
        return vel
    
    def omega(self, t: float) -> np.ndarray:
        """Kecepatan angular (rotasi konstan, jadi omega = 0)"""
        # Method ini sekarang aman dipanggil karena self.omega (float) sudah diganti namanya
        return np.zeros(3)


# ================== CBF UNTUK OBSTACLE AVOIDANCE ==================
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
        return 20.0 * h_2

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


# ================== COMPUTE CONTROL FUNCTIONS ==================
def compute_position_torque_control(robot, osc_controller, cbf, compensate, z, z_ee_des, obstacle_pos=None):
    q, qdot = z[:robot.num_joints], z[robot.num_joints:]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    Jv = J[:3,:]
    if not compensate: c = jnp.zeros(robot.num_joints)
    nullspace_posture_goal = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    u_nom = osc_controller(q, qdot, ee_tmat[:3,3], z_ee_des[:3], z_ee_des[12:15], 
                          jnp.zeros(3), nullspace_posture_goal, jnp.zeros(robot.num_joints), 
                          Jv, M, M_inv, g, c)
    if obstacle_pos is not None:
        tau_safe = cbf.safety_filter(z, u_nom, obstacle_pos)
    else:
        tau_safe = cbf.safety_filter(z, u_nom)
    
    q_ddot_safe = M_inv @ (tau_safe - c - g)
    return tau_safe, q_ddot_safe

def compute_position_velocity_control(robot, osc_controller, cbf, z, z_ee_des, obstacle_pos=None):
    q = z[:robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    Jv = J[:3,:]
    nullspace_posture_goal = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    u_nom = osc_controller(q, ee_tmat[:3,3], z_ee_des[:3], z_ee_des[12:15], 
                          nullspace_posture_goal, Jv, M_inv)
    if obstacle_pos is not None:
        return cbf.safety_filter(q, u_nom, obstacle_pos)
    else:
        return cbf.safety_filter(q, u_nom)


# ================== MAIN FUNCTION ==================
def main(control_method="torque"):
    assert control_method in ["torque", "velocity"]
    
    robot = load_ur5e()
    compensate_centrifugal_coriolis = False
    
    # ================== SETUP TRAJECTORY ==================
    # Parameter lingkaran
    circle_center = np.array([0.4, 0.0])  # Pusat lingkaran di XY
    circle_radius = 0.20  # Radius 20 cm
    z_height = 0.35  # Ketinggian konstan
    angular_velocity = 0.25  # rad/s (sekitar 1 putaran per 15 detik)
    
    # Posisi awal robot (dari q_init)
    q_init = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    init_ee_pos = robot.ee_position(q_init)
    
    # Buat trajectory
    ee_traj = CircularXYTrajectory(
        center=circle_center,
        radius=circle_radius,
        z_height=z_height,
        angular_velocity=angular_velocity,
        transition_time=3.0,  # 3 detik untuk transisi
        init_pos=init_ee_pos
    )
    
    # ================== SETUP OBSTACLE ==================
    obstacle_radius = 0.04
    obstacle_radius_math = obstacle_radius + 0.10
    # Obstacle ditempatkan di tengah lingkaran, sedikit ke atas
    obstacle_pos = np.array([circle_center[0], circle_center[1] + circle_radius, z_height])
    
    # Setup CBF
    torque_cbf = CBF.from_config(DynamicCollisionTorqueConfig(
        robot, obstacle_radius_math, compensate_centrifugal_coriolis, obstacle_pos
    ))
    velocity_cbf = CBF.from_config(DynamicCollisionVelocityConfig(
        robot, obstacle_radius_math, obstacle_pos
    ))
    
    # ================== SETUP ENVIRONMENT ==================
    timestep = 1 / 240
    env_kwargs = {
        'q_init': tuple(q_init), 
        'real_time': True, 
        'bg_color': (1, 1, 1), 
        'load_floor': False, 
        'timestep': timestep, 
        'load_table': True
    }
    
    if control_method == "torque":
        env = UR5eTorqueControlEnv(**env_kwargs)
        cbf = torque_cbf
    else:
        env = UR5eVelocityControlEnv(**env_kwargs)
        cbf = velocity_cbf
    
    # Visualisasi obstacle (bola biru statik)
    vis_id = env.client.createVisualShape(
        env.client.GEOM_SPHERE, 
        radius=obstacle_radius, 
        rgbaColor=[0, 0, 1, 0.8]
    )
    env.client.createMultiBody(
        baseVisualShapeIndex=vis_id, 
        baseCollisionShapeIndex=-1, 
        basePosition=obstacle_pos)

    obstacle_body_id = env.client.createMultiBody(
        baseVisualShapeIndex=vis_id,
        basePosition=obstacle_pos
    )

    # Visualisasi Safety Margin (Bola kawat transparan - Opsional)
    # Ini menunjukkan di mana robot akan mulai menghindar
    # env.client.createMultiBody(
    #     baseVisualShapeIndex=env.client.createVisualShape(
    #         env.client.GEOM_SPHERE, 
    #         radius=obstacle_radius_math, 
    #         rgbaColor=[1,0,0,0.2]), 
    #         basePosition=obstacle_pos
    #         )
    vis_safe_id = env.client.createVisualShape(
        env.client.GEOM_SPHERE, 
        radius=obstacle_radius_math, 
        rgbaColor=[1,0,0,0.1]
        )
    env.client.createMultiBody(
        baseVisualShapeIndex=vis_safe_id, 
        baseCollisionShapeIndex=-1, 
        basePosition=obstacle_pos
        )

    # Visualisasi lintasan lingkaran (garis hijau)
    num_trajectory_points = 50
    trajectory_line_ids = []
    for i in range(num_trajectory_points):
        theta = 2 * np.pi * i / num_trajectory_points
        p1 = [
            circle_center[0] + circle_radius * np.cos(theta),
            circle_center[1] + circle_radius * np.sin(theta),
            z_height
        ]
        theta_next = 2 * np.pi * (i + 1) / num_trajectory_points
        p2 = [
            circle_center[0] + circle_radius * np.cos(theta_next),
            circle_center[1] + circle_radius * np.sin(theta_next),
            z_height
        ]
        line_id = env.client.addUserDebugLine(
            p1, p2, lineColorRGB=[0, 1, 0], lineWidth=2
        )
        trajectory_line_ids.append(line_id)
    
    # Visualisasi target (bola merah kecil)
    target_vis_id = env.client.createVisualShape(
        env.client.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[1, 0, 0, 0.7]
    )
    target_body_id = env.client.createMultiBody(
        baseVisualShapeIndex=target_vis_id
    )
    
    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.2, 
        cameraYaw=45, 
        cameraPitch=-30, 
        cameraTargetPosition=(0.4, 0, 0.3)
    )

    # ================== SETUP VISUALISASI BOLA HIJAU (ROBOT COLLISION) ==================
    robot_sphere_ids = []
    try:
        # Ratakan semua radius menjadi satu list
        all_radii = np.concatenate(ur5e_collision_data['radii'])
        print(f"Membuat visualisasi bola hijau untuk {len(all_radii)} titik kolisi robot...")

        for radius in all_radii:
            vis_shape_id = env.client.createVisualShape(
                shapeType=env.client.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[0.1, 1.0, 0.1, 0.3] # Hijau Transparan
            )
            body_id = env.client.createMultiBody(
                baseVisualShapeIndex=vis_shape_id,
                baseCollisionShapeIndex=-1, 
                basePosition=[0, 0, -1] # Sembunyikan dulu di bawah tanah
            )
            robot_sphere_ids.append(body_id)
    except Exception as e:
        print(f"Gagal membuat bola hijau: {e}")

    # ================== CONTROLLER GAINS ==================
    kp_pos = 400.91
    kd_pos = 77.76
    kp_joint = 5
    kd_joint = 2
    
    # Setup controller
    if control_method == "torque":
        osc_controller = PositionTaskTorqueController(
            n_joints=robot.num_joints, 
            kp_task=kp_pos, 
            kd_task=kd_pos, 
            kp_joint=kp_joint, 
            kd_joint=kd_joint, 
            tau_min=None, 
            tau_max=None
        )
        compute_control_jit = jax.jit(compute_position_torque_control, static_argnums=(0,1,2,3))
    else:
        osc_controller = PositionTaskVelocityController(
            n_joints=robot.num_joints, 
            kp_task=kp_pos, 
            kp_joint=kp_joint, 
            qdot_min=None, 
            qdot_max=None
        )
        compute_control_jit = jax.jit(compute_position_velocity_control, static_argnums=(0,1,2))
    
    # ================== LOGGING ==================
    time_log, h_log, ee_pos_log, ee_pos_des_log = [], [], [], []
    last_log_time = -1.0
    log_interval = 0.5
    
    # ================== MAIN LOOP ==================
    try:
        print(f"Memulai simulasi dengan control method: {control_method}")
        print("Fase 1: Transisi ke lintasan lingkaran (0-3 detik)")
        print("Fase 2: Mengikuti lintasan lingkaran sambil menghindari obstacle")
        simulation_duration = 30  # 30 detik
        
        while env.t < simulation_duration:
            current_time = env.t
            q_qdot = env.get_joint_state()

            # --- UPDATE POSISI BOLA HIJAU (ROBOT BODY) ---
            if robot_sphere_ids:
                local_sphere_positions_per_link = ur5e_collision_data["positions"]
                sphere_id_counter = 0
                
                # Loop setiap link robot
                for i in range(robot.num_joints):
                    try:
                        # Ambil posisi world dari link
                        link_state = env.client.getLinkState(env.robot, i, computeForwardKinematics=1)
                        if link_state is None: continue
                        
                        link_world_pos = link_state[4]
                        link_world_orn = link_state[5]
                        
                        # Ambil offset bola untuk link ini
                        local_positions = np.asarray(local_sphere_positions_per_link[i])
                        if local_positions.ndim == 1: local_positions = local_positions.reshape(1, -1)
                        if local_positions.shape[0] == 0: continue

                        # Transformasi: Lokal -> World
                        rot_matrix = np.array(env.client.getMatrixFromQuaternion(link_world_orn)).reshape(3, 3)
                        
                        for local_pos in local_positions:
                            world_pos = np.array(link_world_pos) + rot_matrix.dot(local_pos)
                            
                            # Pindahkan bola hijau ke posisi tersebut
                            if sphere_id_counter < len(robot_sphere_ids):
                                env.client.resetBasePositionAndOrientation(
                                    bodyUniqueId=robot_sphere_ids[sphere_id_counter],
                                    posObj=world_pos,
                                    ornObj=[0, 0, 0, 1]
                                )
                                sphere_id_counter += 1
                    except: pass
            # ---------------------------------------------
            
            # Hitung posisi target dari trajectory
            target_pos = ee_traj.position(env.t)
            target_vel = ee_traj.velocity(env.t)
            
            # Update visualisasi target
            env.client.resetBasePositionAndOrientation(
                target_body_id, target_pos, [0, 0, 0, 1]
            )
            
            # Buat z_ee_des (format sesuai controller)
            # Untuk position control: [pos(3), rot(9), vel(3), omega(3)]
            z_ee_des = np.concatenate([
                target_pos,
                ee_traj.rotation(env.t).ravel(),
                target_vel,
                ee_traj.omega(env.t)
            ])
            
            # Compute control dengan CBF
            if control_method == "torque":
                current_z = q_qdot
                tau, q_ddot_safe = compute_control_jit(
                    robot, osc_controller, cbf, compensate_centrifugal_coriolis, 
                    q_qdot, z_ee_des, obstacle_pos
                )
                h_value = cbf.h_2(current_z, obstacle_pos)
            else:
                current_z = q_qdot[:robot.num_joints]
                tau = compute_control_jit(
                    robot, osc_controller, cbf, q_qdot, z_ee_des, obstacle_pos
                )
                h_value = cbf.h_1(current_z, obstacle_pos)
            
            env.apply_control(tau)
            env.step()
            
            # Logging
            if h_value is not None:
                time_log.append(current_time)
                h_log.append(np.asarray(h_value))
                q = q_qdot[:robot.num_joints]
                current_ee_pos = robot.ee_position(q)
                ee_pos_log.append(np.asarray(current_ee_pos))
                ee_pos_des_log.append(target_pos)
            
            # Console logging
            if current_time - last_log_time >= log_interval:
                q = q_qdot[:robot.num_joints]
                current_ee_pos = robot.ee_position(q)
                position_error = np.linalg.norm(np.asarray(current_ee_pos) - target_pos)
                h_val_scalar = np.array(h_value)[0]
                dist_to_visual = h_val_scalar + (obstacle_radius_math - obstacle_radius)

                status = "AMAN"
                if h_val_scalar < 0: status = "MENEMBUS SAFETY MARGIN"
                if dist_to_visual < 0: status = "MENABRAK VISUAL"

                print(f"\n--- LOG @ {current_time:.2f}s ---")
                print(f"EE Pos: {np.round(np.asarray(current_ee_pos), 3)}")
                print(f"Target: {np.round(target_pos, 3)}")
                print(f"Error: {position_error:.4f} m")
                print(f"Jarak ke Obstacle (h): {dist_to_visual:.4f} m [{status}]")
                print(f"h(z): {np.asarray(h_value)[0]:.4f}")
                
                if np.asarray(h_value)[0] < 0.05:
                    print("\033[93m[WARNING] Mendekati obstacle!\033[0m")
                
                last_log_time = current_time
    
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh pengguna.")
    
    finally:
        # ================== PLOTTING ==================
        print("\nMembuat visualisasi hasil...")
        
        if time_log and h_log:
            time_np = np.array(time_log)
            h_np = np.array(h_log).flatten()
            ee_pos_np = np.array(ee_pos_log)
            ee_pos_des_np = np.array(ee_pos_des_log)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Barrier Function h(z)
            ax1 = axes[0, 0]
            ax1.plot(time_np, h_np, 'b-', linewidth=2, label='h(z)')
            ax1.axhline(0, color='r', linestyle='--', linewidth=2, label='Batas Aman (h=0)')
            ax1.set_xlabel('Waktu (s)', fontsize=11)
            ax1.set_ylabel('h(z) - Jarak Aman', fontsize=11)
            ax1.set_title('Barrier Function (Jarak ke Obstacle)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Plot 2: Tracking Error
            ax2 = axes[0, 1]
            errors = np.linalg.norm(ee_pos_np - ee_pos_des_np, axis=1)
            ax2.plot(time_np, errors * 1000, 'r-', linewidth=2)  # dalam mm
            ax2.set_xlabel('Waktu (s)', fontsize=11)
            ax2.set_ylabel('Tracking Error (mm)', fontsize=11)
            ax2.set_title('Position Tracking Error', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: XY Trajectory (Top View)
            ax3 = axes[1, 0]
            ax3.plot(ee_pos_np[:, 0], ee_pos_np[:, 1], 'b-', linewidth=2, label='Actual Path', alpha=0.7)
            ax3.plot(ee_pos_des_np[:, 0], ee_pos_des_np[:, 1], 'g--', linewidth=2, label='Desired Circle')
            
            # Plot obstacle
            circle_obstacle = plt.Circle(
                (obstacle_pos[0], obstacle_pos[1]), 
                obstacle_radius, 
                color='blue', 
                alpha=0.3, 
                label='Obstacle'
            )
            ax3.add_patch(circle_obstacle)
            
            # Mark start point
            ax3.plot(ee_pos_np[0, 0], ee_pos_np[0, 1], 'go', markersize=10, label='Start')
            
            ax3.set_xlabel('X (m)', fontsize=11)
            ax3.set_ylabel('Y (m)', fontsize=11)
            ax3.set_title('XY Trajectory (Top View)', fontsize=12, fontweight='bold')
            ax3.axis('equal')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=9)
            
            # Plot 4: 3D Trajectory
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            ax4.plot(ee_pos_np[:, 0], ee_pos_np[:, 1], ee_pos_np[:, 2], 
                    'b-', linewidth=2, label='Actual', alpha=0.7)
            ax4.plot(ee_pos_des_np[:, 0], ee_pos_des_np[:, 1], ee_pos_des_np[:, 2], 
                    'g--', linewidth=2, label='Desired')
            
            # Plot obstacle as sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = obstacle_pos[0] + obstacle_radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = obstacle_pos[1] + obstacle_radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = obstacle_pos[2] + obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax4.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.3)
            
            ax4.set_xlabel('X (m)', fontsize=10)
            ax4.set_ylabel('Y (m)', fontsize=10)
            ax4.set_zlabel('Z (m)', fontsize=10)
            ax4.set_title('3D Trajectory', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print("\n" + "="*50)
            print("STATISTIK HASIL")
            print("="*50)
            print(f"Mean Tracking Error: {np.mean(errors)*1000:.2f} mm")
            print(f"Max Tracking Error: {np.max(errors)*1000:.2f} mm")
            print(f"Min h(z): {np.min(h_np):.4f} m")
            print(f"Collision?: {'YES!' if np.min(h_np) < 0 else 'NO'}")
            print("="*50)
        
        # Disconnect PyBullet
        if 'env' in locals() and hasattr(env, 'client') and env.client.isConnected():
            try:
                env.client.disconnect()
                print("PyBullet disconnected.")
            except Exception as e:
                print(f"Error disconnecting PyBullet: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR5e Circular XY Trajectory with Obstacle Avoidance"
    )
    parser.add_argument(
        "--control_method", 
        type=str, 
        choices=["torque", "velocity"], 
        default="torque",
        help="Control method: 'torque' or 'velocity'"
    )
    args = parser.parse_args()
    main(control_method=args.control_method)