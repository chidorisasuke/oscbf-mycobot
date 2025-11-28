"""Testing the performance of OSCBF in highly-constrained settings

We consider a cluttered tabletop environment with many randomized obstacles,
each represented as a sphere. We then enforce collision avoidance with 
all of the obstacles, and all of the collision bodies on the robot

Tahap 2: Tuning Stabilitas & Smoothness (PositionTaskTorqueController)
- Menggunakan kontrol posisi 3D (x,y,z) agar memiliki nullspace untuk postur.
- Menggunakan parameter gain yang disesuaikan untuk kehalusan gerak.
"""

import argparse
import time 

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_mycobot
from oscbf.core.manipulation_env import MyCobotTorqueControlEnv, MyCobotVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
# PENTING: Gunakan PositionTask (3D) bukan PoseTask (6D) untuk stabilitas robot 6-DOF
from oscbf.core.controllers import PositionTaskTorqueController, PositionTaskVelocityController
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from oscbf.core.myco_collision_model import mycobot_collision_data
from oscbf.core.myco_collision_model import print_collision_model_info


np.random.seed(0)


@jax.tree_util.register_static
class CollisionsConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)

        self.singularity_tol = 1e-4

        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        manipulability_index = self.robot.manipulability(q)
        h_singularity = jnp.array([manipulability_index - self.singularity_tol])

        return jnp.concatenate([h_collision, h_table, h_singularity])

    def alpha(self, h):
        return 25.0 * h

    def alpha_2(self, h_2):
        return 25.0 * h_2


@jax.tree_util.register_static
class CollisionsVelocityConfig(OSCBFVelocityConfig):

    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        return jnp.concatenate([h_collision, h_table])

    def alpha(self, h):
        return 25.0 * h

    def alpha_2(self, h_2):
        return 25.0 * h_2


# FUNGSI BARU: Menggunakan PositionTaskTorqueController (3D)
# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PositionTaskTorqueController, # Tipe kontroler diubah
    cbf: CBF,
    compensate_centrifugal_coriolis:bool,
    z: ArrayLike,
    z_ee_des: ArrayLike,
    q_des_nullspace: ArrayLike # Target postur untuk nullspace
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    
    # Ambil Jacobian Linear (3 baris pertama) untuk Position Control
    Jv = J[:3, :] 
    
    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)

    # u_nom untuk Position Control (3D)
    # Ini memberikan kebebasan pada orientasi (nullspace) untuk menghindari singularitas
    u_nom = osc_controller(
        q,
        qdot,
        pos=ee_tmat[:3, 3],
        # rot tidak diperlukan untuk position control
        des_pos=z_ee_des[:3],
        des_vel=z_ee_des[12:15],
        des_accel=jnp.zeros(3),
        des_q=q_des_nullspace, # Postur yang diinginkan (misal: tegak)
        des_qdot=jnp.zeros(robot.num_joints),
        Jv=Jv, # Gunakan Jv (3xN), bukan J (6xN)
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )
    return cbf.safety_filter(z, u_nom)


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_velocity_control(
    robot: Manipulator,
    osc_controller: PositionTaskVelocityController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    Jv = J[:3, :] # Linear Jacobian
    pos = ee_tmat[:3, 3]
    
    # Dummy nullspace target
    des_q = jnp.zeros(robot.num_joints)
    
    u_nom = osc_controller(
        q, pos, z_ee_des[:3], z_ee_des[12:15], des_q, Jv, M_inv
    )
    return cbf.safety_filter(q, u_nom)


def main(control_method="torque", num_bodies=25):
    assert control_method in ["torque", "velocity"]

    robot = load_mycobot()
    z_min = 0.1

    time_log = []
    h_log = []

    max_num_bodies = 50

    # Sample a lot of collision bodies
    all_collision_pos = np.random.uniform(
        low=[0.2, -0.4, 0.1], high=[0.8, 0.4, 0.3], size=(max_num_bodies, 3)
    )
    all_collision_radii = np.random.uniform(low=0.01, high=0.1, size=(max_num_bodies,))
    # Only use a subset of them based on the desired quantity
    collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
    collision_radii = all_collision_radii[:num_bodies]
    collision_data = {"positions": collision_pos, "radii": collision_radii}

    torque_config = CollisionsConfig(robot, z_min, collision_pos, collision_radii)
    torque_cbf = CBF.from_config(torque_config)
    velocity_config = CollisionsVelocityConfig(
        robot, z_min, collision_pos, collision_radii
    )
    velocity_cbf = CBF.from_config(velocity_config)

    # --- SETTING POSISI AWAL TEGAK ---
    # Posisi ini jauh dari singularitas lipatan, bagus untuk start
    mycobot_q_init = (0, 0, 0, 0, 0, 0) 
    # ---------------------------------

    timestep = 1 / 240
    bg_color = (1, 1, 1)
    
    if control_method == "torque":
        env = MyCobotTorqueControlEnv(
            q_init=mycobot_q_init,
            real_time=False,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
            collision_data=collision_data,
            load_table=True,
        )
    else:
        env = MyCobotVelocityControlEnv(
            real_time=False,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
            collision_data=collision_data,
            load_table=True,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.40,
        cameraYaw=104.40,
        cameraPitch=-37,
        cameraTargetPosition=(0.20, 0.07, -0.09),
    )

    # --- TUNING PARAMETER (SMOOTH & STABIL) ---
    # Parameter yang Anda request:
    kp_pos = 40.0   # Cukup kuat untuk bergerak
    kd_pos = 10.0   # Peredam
    
    # Parameter Rotasi (Tidak dipakai di PositionTask, tapi didefinisikan saja)
    kp_rot = 20.0
    kd_rot = 5.0
    
    # Parameter Nullspace (PENTING untuk menjaga postur)
    kp_joint = 5.0  # Kekakuan untuk kembali ke posisi tegak
    kd_joint = 1.5  # Peredam getaran tubuh
    
    # Gunakan PositionTaskTorqueController (3D Task)
    # Ini kunci untuk menghindari guncangan pada robot 6-DOF
    osc_torque_controller = PositionTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=kp_pos, # Scalar untuk 3D (posisi)
        kd_task=kd_pos, # Scalar untuk 3D (posisi)
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        tau_min=None,
        tau_max=None,
    )

    osc_velocity_controller = PositionTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=kp_pos,
        kp_joint=kp_joint,
        qdot_min=None,
        qdot_max=None,
    )

    # Target Nullspace: Usahakan tetap tegak/netral
    q_des_nullspace = jnp.array(mycobot_q_init)

    @jax.jit
    def compute_torque_control_jit(z, z_ee_des):
        return compute_torque_control(
            robot, osc_torque_controller, torque_cbf, True, z, z_ee_des, q_des_nullspace
        )

    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des):
        return compute_velocity_control(
            robot, osc_velocity_controller, velocity_cbf, z, z_ee_des
        )

    if control_method == "torque":
        compute_control = compute_torque_control_jit
    elif control_method == "velocity":
        compute_control = compute_velocity_control_jit
    else:
        raise ValueError(f"Invalid control method: {control_method}")

    last_print_time = 0
    time_log, h_log, torque_log, velocity_log, position_log, singularity_log = [], [], [], [], [], []
    
    # --- VISUALISASI AWAL & PAUSE UNTUK TUNING ---
    print("\n" + "="*60)
    print("MODE TUNING MYCOBOT (Position Control)")
    print("------------------------------------------------------------")
    print("1. Robot akan spawn di posisi tegak (0,0,0,0,0,0).")
    print("2. Environment otomatis menampilkan bola hijau (Collision Spheres).")
    print("3. Cek apakah bola hijau terlalu besar/jauh dari body robot.")
    print("   -> Jika ya, edit 'radii' di file 'myco_collision_model.py'.")
    print("------------------------------------------------------------")
    
    input("Tekan [ENTER] di terminal untuk memulai simulasi...")
    print("Simulasi berjalan...")
    print("="*60 + "\n")

    try:
        while True:
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            
            tau = compute_control(q_qdot, z_zdot_ee_des)
            env.apply_control(tau)
            env.step()

            if (env.t - last_print_time) >= 0.5:
                q_aktual = q_qdot[:robot.num_joints]
                ee_pos_aktual = robot.ee_position(q_aktual)
                posisi_bola_merah = z_zdot_ee_des[:3]
                selisih = np.linalg.norm(posisi_bola_merah - ee_pos_aktual)

                print(f"--- Waktu: {env.t:.2f} s ---")
                print(f"Posisi End-Effector:   {np.round(ee_pos_aktual, 2)}")
                print(f"Posisi Bola Merah:     {np.round(posisi_bola_merah, 2)}")
                print(f"Selisih Jarak:         {selisih:.3f} m")
                print("-" * 20)

                last_print_time = env.t

            time_log.append(env.t)

            q = q_qdot[:robot.num_joints]
            q_dot = q_qdot[robot.num_joints:]
            position_log.append(q)
            velocity_log.append(q_dot)

            if control_method == "torque":
                h_values = torque_config.h_2(q_qdot)
                torque_log.append(tau)
                singularity_value = h_values[-1] + torque_config.singularity_tol
                singularity_log.append(singularity_value)
            else: 
                h_values = velocity_config.h_1(q_qdot)
                torque_log.append(np.zeros_like(q))
                singularity_log.append(0)
            
            h_log.append(h_values)

    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh pengguna.")

    finally:
        print("Simulation finished. Plotting data...")
        min_len = min(len(time_log), len(h_log), len(torque_log), len(position_log), len(velocity_log), len(singularity_log))
        
        time_log = time_log[:min_len]
        h_log = np.array(h_log[:min_len])
        torque_log = np.array(torque_log[:min_len])
        position_log = np.array(position_log[:min_len])
        velocity_log = np.array(velocity_log[:min_len])
        singularity_log = np.array(singularity_log[:min_len])

        if 'env' in locals() and hasattr(env, 'client'):
            try:
                env.client.disconnect()
                print("PyBullet disconnected.")
            except Exception as e:
                print(f"Error disconnecting PyBullet: {e}")

        fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)
        fig.suptitle('Analisis Simulasi Robot Mycobot 280 (Stabilized)', fontsize=16)

        # Plot H(z)
        if len(time_log) > 0:
            for i in range(h_log.shape[1]):
                if i == 0:
                    axs[0].plot(time_log, h_log[:, i], alpha=0.7, label=f'Batasan Tabrakan ({h_log.shape[1]} total)')
                else:
                    axs[0].plot(time_log, h_log[:, i], alpha=0.7)

        axs[0].axhline(0, color='r', linestyle='--', label='Batas Aman (h=0)')
        axs[0].set_title('Evolusi Batasan Keamanan (h(z))')
        axs[0].set_ylabel('Nilai h(z)')
        axs[0].grid(True)
        axs[0].legend(fontsize='small')

        # Plot Torsi
        start_index = 10 
        if len(time_log) > start_index:
            for i in range(torque_log.shape[1]):
                axs[1].plot(time_log[start_index:], torque_log[start_index:, i], label=f'Sendi {i+1}')
        axs[1].set_title('Perintah Torsi Aman (Γ*)')
        axs[1].set_ylabel('Torsi (Nm)')
        axs[1].grid(True)
        axs[1].legend(fontsize='small')

        # Plot Kecepatan
        for i in range(velocity_log.shape[1]):
            axs[2].plot(time_log, velocity_log[:, i], label=f'Sendi {i+1}')
        axs[2].set_title('Kecepatan Sendi (q_dot)')
        axs[2].set_ylabel('Kecepatan (rad/s)')
        axs[2].grid(True)
        axs[2].legend(fontsize='small')

        # Plot Posisi
        for i in range(position_log.shape[1]):
            axs[3].plot(time_log, position_log[:, i], label=f'Sendi {i+1}')
        axs[3].set_title('Posisi Sendi (q)')
        axs[3].set_ylabel('Posisi (rad)')
        axs[3].set_xlabel('Waktu (s)')
        axs[3].grid(True)
        axs[3].legend(fontsize='small')

        # Plot Singularitas
        if len(time_log) > 0:
            axs[4].plot(time_log, singularity_log, label='Manipulability Index (μ)', color='purple')
            axs[4].axhline(torque_config.singularity_tol, color='r', linestyle='--', label=f'Batas Aman (μ={torque_config.singularity_tol})')
        axs[4].set_title('Evolusi Batasan Singularitas (Manipulability)')
        axs[4].set_ylabel('Nilai Manipulabilitas (μ)')
        axs[4].set_yscale('log')
        axs[4].grid(True)
        axs[4].legend(fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run highly-constrained collision avoidance experiment."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="torque",
        help="Control method to use (default: torque)",
    )
    parser.add_argument(
        "--num_bodies",
        type=int,
        default=25,
        help="Number of collision bodies to simulate (default: 25)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method, num_bodies=args.num_bodies)