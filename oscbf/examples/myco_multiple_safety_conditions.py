"""Testing the performance of OSCBF under many different safety constraints, namely:

1. End-effector set containment
2. Joint limit avoidance
3. Singularity avoidance
4. Collision avoidance
5. Whole-body set containment

While there are many other safety constraints that we could also account for, this
should give a good view of the controller's performance under common situations
encountered in practice.
"""

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from cbfpy import CBF

from oscbf.core.manipulator import Manipulator, load_mycobot
from oscbf.core.manipulation_env import MyCobotTorqueControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController


@jax.tree_util.register_static
class CombinedConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        assert len(collision_positions) == len(collision_radii)
        self.num_collision_bodies = len(collision_positions)
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # EE Set Containment
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole-body Set Containment
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()
        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate(
            [
                h_ee_safe_set,
                h_joint_limits,
                h_singularity,
                h_collision,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    # Set nullspace desired joint position
    nullspace_posture_goal = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 2,
            0.0,
            jnp.pi / 3,
        ]
    )

    # Compute nominal control
    u_nom = osc_controller(
        q,
        qdot,
        pos=ee_tmat[:3, 3],
        rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3],
        des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15],
        des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3),
        des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal,
        des_qdot=jnp.zeros(robot.num_joints),
        J=J,
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )
    # Apply the CBF safety filter
    return cbf.safety_filter(z, u_nom)


def main():
    robot = load_mycobot()
    ee_pos_min = np.array([0.15, -0.25, 0.25])
    ee_pos_max = np.array([0.75, 0.25, 0.75])
    wb_pos_min = np.array([-0.5, -0.5, 0.0])
    wb_pos_max = np.array([0.75, 0.5, 1.0])
    collision_pos = np.array([[0.5, 0.5, 0.5]])
    collision_radii = np.array([0.3])
    collision_data = {"positions": collision_pos, "radii": collision_radii}
    config = CombinedConfig(
        robot,
        ee_pos_min,
        ee_pos_max,
        collision_pos,
        collision_radii,
        wb_pos_min,
        wb_pos_max,
    )
    cbf = CBF.from_config(config)
    env = MyCobotTorqueControlEnv(
        xyz_min=config.pos_min,
        xyz_max=config.pos_max,
        collision_data=collision_data,
        wb_xyz_min=wb_pos_min,
        wb_xyz_max=wb_pos_max,
        load_floor=False,
        bg_color=(1, 1, 1),
        real_time=True,
    )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraPitch=-27.80,
        cameraYaw=36.80,
        cameraTargetPosition=(0.08, 0.49, -0.04),
    )

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    osc_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        # Note: torque limits will be enforced via the QP. We'll set them to None here
        # because we don't want to clip the values before the QP
        tau_min=None,
        tau_max=None,
    )

    @jax.jit
    def compute_control_jit(z, z_des):
        return compute_control(robot, osc_controller, cbf, z, z_des)
    
    time_log, h_log, torque_log, velocity_log, position_log = [], [], [], [], []

    try:
        while True:
            joint_state = env.get_joint_state()
            ee_state_des = env.get_desired_ee_state()
            tau = compute_control_jit(joint_state, ee_state_des)
            env.apply_control(tau)
            env.step()

            time_log.append(env.t)

            q = joint_state[:robot.num_joints]
            q_dot = joint_state[robot.num_joints:]
            position_log.append(q)
            velocity_log.append(q_dot)
            torque_log.append(tau)

            # Hitung nilai h(z) saat ini
            h_values = config.h_2(joint_state)
            h_log.append(h_values)
            
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh pengguna.")
    
    finally:
        # Di dalam blok `finally:`

        print("Simulasi selesai. Mempersiapkan data plot...")
        # === TAMBAHKAN SELURUH BLOK DI BAWAH INI ===

        # Konversi list ke numpy array
        h_log = np.array(h_log)
        torque_log = np.array(torque_log)
        position_log = np.array(position_log)
        velocity_log = np.array(velocity_log)

        # Buat 4 subplot
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        # Ganti judul agar sesuai dengan file ini
        fig.suptitle('Analisis Multi-Kondisi Aman (MyCobot)', fontsize=16)

        # --- Plot 1: Evolusi Batasan Keamanan (h(z)) ---
        # PENYESUAIAN PENTING: Beri label pada setiap jenis batasan
        num_ee_constraints = 6  # 3 max, 3 min
        num_joint_constraints = robot.num_joints * 2
        num_singularity_constraints = 1
        
        # Buat label untuk setiap batasan agar plot lebih informatif
        labels = (
            [f"EE limit {i}" for i in range(num_ee_constraints)] +
            [f"Joint limit {i}" for i in range(num_joint_constraints)] +
            ["Singularity"] +
            # Sisa batasan adalah tabrakan dan batas seluruh tubuh
            [f"Other {i}" for i in range(h_log.shape[1] - num_ee_constraints - num_joint_constraints - num_singularity_constraints)]
        )

        if len(time_log) > 0:
            for i in range(h_log.shape[1]):
                # Beri label hanya untuk beberapa batasan penting agar legenda tidak penuh
                label_to_show = None
                if labels[i] in ["EE limit 0", "Joint limit 0", "Singularity"]:
                    label_to_show = labels[i]
                axs[0].plot(time_log, h_log[:, i], alpha=0.7, label=label_to_show)

        axs[0].axhline(0, color='r', linestyle='--', label='Batas Aman (h=0)')
        axs[0].set_title('Evolusi Batasan Keamanan (h(z))')
        axs[0].set_ylabel('Nilai h(z)')
        axs[0].grid(True)
        axs[0].legend(fontsize='small')
        
        # --- Plot 2: Torsi Sendi (Torque) ---
        start_index = 10 # Abaikan lonjakan awal
        if len(time_log) > start_index:
            for i in range(torque_log.shape[1]):
                axs[1].plot(time_log[start_index:], torque_log[start_index:, i], label=f'Sendi {i+1}')
        axs[1].set_title('Perintah Torsi Aman (Γ*)')
        axs[1].set_ylabel('Torsi (Nm)')
        axs[1].grid(True)
        axs[1].legend(fontsize='small')

        # --- Plot 3: Kecepatan Sendi (Velocity) ---
        if len(time_log) > start_index:
            for i in range(velocity_log.shape[1]):
                axs[2].plot(time_log[start_index:], velocity_log[start_index:, i], label=f'Sendi {i+1}')
        axs[2].set_title('Kecepatan Sendi (q_dot)')
        axs[2].set_ylabel('Kecepatan (rad/s)')
        axs[2].grid(True)
        axs[2].legend(fontsize='small')

        # --- Plot 4: Posisi Sendi (Position) ---
        if len(time_log) > 0:
            for i in range(position_log.shape[1]):
                axs[3].plot(time_log, position_log[:, i], label=f'Sendi {i+1}')
        axs[3].set_title('Posisi Sendi (q)')
        axs[3].set_ylabel('Posisi (rad)')
        axs[3].set_xlabel('Waktu (s)')
        axs[3].grid(True)
        axs[3].legend(fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



if __name__ == "__main__":
    main()
