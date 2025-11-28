import numpy as np
import pybullet as p
import pybullet_data
import time
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- Import Library OSCBF ---
from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.manipulation_env import UR5eTorqueControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController
from cbfpy import CBF

# =================================================================================
# 1. KELAS MASS ESTIMATOR
# =================================================================================
class MassEstimator:
    def __init__(self, gripper_mass=0.8, alpha_filter=0.1):
        self.m_g = gripper_mass
        self.m_u = 0.0
        self.alpha = alpha_filter
        self.g_mag = 9.81
        
    def update(self, f_ext_z, p_ddot_z):
        denom = p_ddot_z + self.g_mag 
        if abs(denom) < 0.1: return self.m_u
        raw_estimate = (f_ext_z / denom) - self.m_g
        raw_estimate = np.clip(raw_estimate, 0.0, 5.0)
        self.m_u = (1 - self.alpha) * self.m_u + self.alpha * raw_estimate
        return self.m_u

# =================================================================================
# 2. KONFIGURASI & CONTROLLER
# =================================================================================
@jax.tree_util.register_static
class DynamicCollisionConfig(OSCBFTorqueConfig):
    def __init__(self, robot: Manipulator, obstacle_radius: float, initial_obstacle_pos):
        self.obstacle_radius = obstacle_radius
        init_pos_tuple = tuple(map(float, initial_obstacle_pos))
        super().__init__(robot, compensate_centrifugal_coriolis=False, init_args=(init_pos_tuple,))

    def h_2(self, z, obstacle_pos, **kwargs):
        q = z[:self.robot.num_joints]
        obstacle_pos = jnp.asarray(obstacle_pos)
        robot_coll = self.robot.link_collision_data(q)
        if robot_coll.size == 0: return jnp.array([1.0])
        dist = jnp.linalg.norm(robot_coll[:,:3] - obstacle_pos[None,:], axis=1)
        h = dist - (robot_coll[:,3] + self.obstacle_radius)
        return jnp.array([jnp.min(h)])

    def alpha_2(self, h):
        return 10.0 * h  # Reduced from 15.0 to make CBF less aggressive for better tracking

def compute_control(robot, controller, cbf, z, z_des, obs_pos):
    q, qdot = z[:robot.num_joints], z[robot.num_joints:]
    M, Minv, g, c, J, ee_t = robot.torque_control_matrices(q, qdot)
    q_home = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    u_nom = controller(q, qdot, ee_t[:3,3], ee_t[:3,:3], 
                       z_des[:3], jnp.reshape(z_des[3:12],(3,3)), 
                       z_des[12:15], z_des[15:18], 
                       jnp.zeros(3), jnp.zeros(3), 
                       q_home, jnp.zeros(6), 
                       J, M, Minv, g, c)
    tau_safe = cbf.safety_filter(z, u_nom, obs_pos)
    return tau_safe, M, c, g

# =================================================================================
# 3. MAIN LOOP
# =================================================================================
def main():
    # --- Setup Awal ---
    robot_loader = load_ur5e()
    obs_pos = np.array([0.5, 0.0, 0.2])
    obs_radius = 0.05
    
    # Controller Setup
    cbf_config = DynamicCollisionConfig(robot_loader, obs_radius, obs_pos)
    cbf_controller = CBF.from_config(cbf_config)
    
    osc_controller = PoseTaskTorqueController(
        n_joints=6,
        kp_task=np.concatenate([200.0*np.ones(3), 100.0*np.ones(3)]),  # Increased position gains, moderate orientation gains
        kd_task=np.concatenate([25.0*np.ones(3), 15.0*np.ones(3)]),    # Adjusted velocity gains
        kp_joint=15.0, kd_joint=2.0,                                  # Increased joint gains for better tracking
        tau_min=None, tau_max=None
    )
    
    compute_control_jit = jax.jit(compute_control)
    
    # Environment PyBullet (FORCE REAL TIME FALSE AGAR KITA KONTROL WAKTUNYA)
    env = UR5eTorqueControlEnv(
        q_init=(0, -1.57, 1.57, -1.57, -1.57, 0),
        real_time=False, 
        timestep=0.008, 
        load_table=True
    )
    
    # Visuals
    vis_obs = env.client.createVisualShape(p.GEOM_SPHERE, radius=obs_radius, rgbaColor=[0,0,1,0.8])
    env.client.createMultiBody(baseVisualShapeIndex=vis_obs, basePosition=obs_pos)

    # Cari Joint Aktif
    moving_joint_indices = []
    for i in range(p.getNumJoints(env.robot, physicsClientId=env.client._client)):
        info = p.getJointInfo(env.robot, i, physicsClientId=env.client._client)
        if info[2] != p.JOINT_FIXED:
            moving_joint_indices.append(i)
            
    # Estimator & Log (simplified - removing mass estimation)
    log_t, log_err = [], []
    
    t = 0.0
    dt = 0.008
    duration = 8.0  # Increased to see more of the circular motion
    
    print("\n" + "="*50)
    print("SIMULASI SIAP DIJALANKAN")
    print("Melanjutkan otomatis...")
    print("="*50)
    # input()  # Removed to allow automatic execution

    print(f"{'WAKTU (s)':<10} | {'ERROR (m)':<10} | {'MASS EST':<15} | {'STATUS'}")
    print("-" * 60)

    try:
        while t < duration:
            loop_start = time.time()
            
            # 1. Get State
            q_state = env.get_joint_state()
            q_curr, dq_curr = q_state[:6], q_state[6:]
            
            # 2. Trajectory Target - Circular motion in XY plane with gradual start
            radius = 0.15  # Circle radius in meters
            center_pos = np.array([0.5, 0.0, 0.4])  # Center point of the circle
            
            # Gradually increase the radius for smoother start
            effective_radius = radius * min(1.0, t / 2.0)  # Gradually reach full radius over 2 seconds
            target_pos = center_pos + np.array([effective_radius * np.cos(t*0.8), effective_radius * np.sin(t*0.8), 0.0])
            
            # Calculate target velocity for smooth motion
            target_vel = np.array([-effective_radius * 0.8 * np.sin(t*0.8), effective_radius * 0.8 * np.cos(t*0.8), 0.0])
            
            # Keep orientation constant (identity rotation matrix)
            target_rot = np.eye(3).flatten()
            z_des = np.concatenate([target_pos, target_rot, target_vel, np.zeros(3)])
            
            # 3. Compute Control
            tau_safe, M, c, g_nominal = compute_control_jit(robot_loader, osc_controller, cbf_controller, q_state, z_des, obs_pos)
            tau_safe, M, c, g_nominal = np.array(tau_safe), np.array(M), np.array(c), np.array(g_nominal)
            
            # 4. Optimized torque to velocity control (removing mass adaptation)
            # Use a smoother integration with a damping factor to prevent jerky movements
            q_ddot = np.linalg.inv(M) @ tau_safe
            # Apply a smoothing factor for more controlled movement
            alpha = 0.1  # Smoothing factor
            dq_target = (1 - alpha) * dq_curr + alpha * np.clip(dq_curr + q_ddot * dt, -1.5, 1.5)
            
            # 5. Apply Command with optimized force limits
            env.client.setJointMotorControlArray(
                env.robot, moving_joint_indices, p.VELOCITY_CONTROL, 
                targetVelocities=dq_target, forces=[100]*6  # Reduced forces for smoother motion
            )
            
            env.step()
            
            # Logging & Real-time Print
            curr_ee = robot_loader.ee_position(q_curr)
            err = np.linalg.norm(curr_ee - target_pos)
            log_t.append(t)
            log_err.append(err)
            
            # PRINT SETIAP 0.1 DETIK (MENGGUNAKAN FLUSH=TRUE)
            if int(t/dt) % 12 == 0: # approx every 0.1s
                status = "ACTIVE"  # Simplified status
                print(f"{t:6.2f}     | {err:6.4f}     | {'N/A':<15} | {status}", flush=True)

            t += dt
            
            # Manual Sleep untuk sinkronisasi waktu nyata
            elapsed = time.time() - loop_start
            if dt > elapsed:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nBerhenti Manual.")
        
    finally:
        print("\nMenampilkan Grafik...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(log_t, log_err, 'b'); ax.set_title("Tracking Error (m)"); ax.grid(True)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
        plt.tight_layout(); plt.show()
        if hasattr(env, 'client'): env.client.disconnect()

if __name__ == "__main__":
    main()