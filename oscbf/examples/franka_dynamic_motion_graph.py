"""Testing the performance of OSCBF during dynamic motions and input constraints

In general, we will command a rapid motion of the end-effector into the unsafe set,
and observe the controller's behavior under velocity control and torque control.

The reduced-order (velocity-control) OSCBF has no lower-level understanding of torque
limits, so the full-order (torque-control) OSCBF should perform better in this case.
"""

import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv, FrankaVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
)

DATA_DIR = "oscbf/experiments/data/"
SAVE_DATA = False
PAUSE_FOR_PICTURES = False
RECORD_VIDEO = False
PICTURE_IDXS = [1000, 1250, 1600, 1900, 2200]


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

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2, 3))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    compensate_centrifugal_coriolis: bool,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)

    nullspace_posture_goal = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
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


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_velocity_control(
    robot: Manipulator,
    osc_controller: PoseTaskVelocityController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    pos = ee_tmat[:3, 3]
    rot = ee_tmat[:3, :3]
    des_pos = z_ee_des[:3]
    des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
    des_vel = z_ee_des[12:15]
    des_omega = z_ee_des[15:18]

    des_q = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
        ]
    )
    
    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )
    return cbf.safety_filter(q, u_nom)


def main(control_method="torque"):
    assert control_method in ["torque", "velocity"]

    robot = load_panda()
    # pos untuk Franka
    pos_min = (0.25, -0.25, 0.25)
    pos_max = (0.65, 0.25, 0.65)

    # NOTE: This term has a noticeable impact on the performance for this demo.
    # It's often neglected due to computational demands and model error
    compensate_centrifugal_coriolis = True

    torque_config = EESafeSetTorqueConfig(
        robot,
        pos_min,
        pos_max,
        compensate_centrifugal_coriolis=compensate_centrifugal_coriolis,
    )
    torque_cbf = CBF.from_config(torque_config)
    velocity_config = EESafeSetVelocityConfig(robot, pos_min, pos_max)
    velocity_cbf = CBF.from_config(velocity_config)
    traj = SinusoidalTaskTrajectory(
        init_pos=(0.55, 0, 0.45), #Franka
        init_rot=np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ),
        amplitude=(0.25, 0, 0), #Franka
        angular_freq=(5, 0, 0),
        phase=(0, 0, 0),
    )
    timestep = 1 / 1000
    bg_color = (1, 1, 1)
    if control_method == "torque":
        env = FrankaTorqueControlEnv(
            torque_config.pos_min,
            torque_config.pos_max,
            traj=traj,
            real_time=False,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )
    else:
        env = FrankaVelocityControlEnv(
            velocity_config.pos_min,
            velocity_config.pos_max,
            traj=traj,
            real_time=False,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.00,
        cameraYaw=12,
        cameraPitch=-2.6,
        cameraTargetPosition=(0.44, 0.16, 0.28),
    )

    # Franka
    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0

    osc_torque_controller = PoseTaskTorqueController(
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

    osc_velocity_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        # Note: velocity limits will be enforced via the QP
        # because we don't want to clip the values before the QP
        qdot_min=None,
        qdot_max=None,
    )

    @jax.jit
    def compute_torque_control_jit(z, z_ee_des):
        return compute_torque_control(
            robot,
            osc_torque_controller,
            torque_cbf,
            compensate_centrifugal_coriolis,
            z,
            z_ee_des,
        )

    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des):
        return compute_velocity_control(
            robot, osc_velocity_controller, velocity_cbf, z, z_ee_des
        )
    time_log, h_log, torque_log, position_log, velocity_log = [], [], [], [], []

    if control_method == "torque":
        compute_control = compute_torque_control_jit
    elif control_method == "velocity":
        compute_control = compute_velocity_control_jit
    else:
        raise ValueError(f"Invalid control method: {control_method}")
    

    try:
        simulation_duration = 20
        while env.t < simulation_duration:
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            tau = compute_control(q_qdot, z_zdot_ee_des)
            env.apply_control(tau)
            env.step()

            current_time = env.t
            

            q = q_qdot[:robot.num_joints]
            q_qdot = q_qdot[robot.num_joints:]
            position_log.append(q)
            velocity_log.append(q_qdot)

            if control_method == "torque":
                h_values = torque_config.h_2(q_qdot)
                torque_log.append(tau)
            else: 
                h_values = velocity_config.h_1(q_qdot)
                torque_log.append(np.zeros_like(q))
            
            if h_values is not None:
                h_value_np = np.asarray(h_values)
                time_log.append(current_time)
                h_log.append(h_value_np)
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh pengguna.")
    finally:
        print("Simulation finished. Plotting data...")
        
        if not time_log or not h_log:
            print("Tidak ada data h(z) yang tercatat untuk diplot.")
        else:
            time_np = np.array(time_log)
            h_np = np.vstack(h_log) # Tumpuk list of arrays h menjadi (N, 6)

            fig, ax = plt.subplots(1, 1, figsize=(12, 4)) # Ukuran disesuaikan

            if h_np.ndim == 2 and h_np.shape[1] == 6:
                labels = ["X max - x", "Y max - y", "Z max - z", 
                        "x - X min", "y - Y min", "z - Z min"]
                # Gunakan colormap agar warna berbeda
                colors = plt.cm.viridis(np.linspace(0, 1, 6)) 
                for i in range(6):
                    ax.plot(time_np, h_np[:, i], label=labels[i], color=colors[i], alpha=0.9, linewidth=1.5)
            else: 
                print(f"[WARNING] Unexpected h(z) data shape: {h_np.shape}. Plotting as generic lines.")
                # Fallback plotting jika shape tidak (N, 6)
                num_lines = h_np.shape[1] if h_np.ndim == 2 else 1
                h_plot = h_np if h_np.ndim == 2 else h_np.reshape(-1,1)
                for i in range(num_lines):
                    ax.plot(time_np, h_plot[:,i], label=f'h_{i+1}')

            # Garis batas aman h=0
            ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Batas Aman (h=0)')

            # Pengaturan Plot
            ax.set_xlabel("Waktu (s)")
            ax.set_ylabel("Nilai h(z)")
            ax.set_title(f"Evolusi Batasan Keamanan (Franka Panda - {control_method.title()} Control)")
            ax.grid(True)
            ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5)) # Legenda di luar

            # Atur batas Y agar fokus di dekat 0, tapi beri ruang lihat osilasi
            min_h = np.min(h_np) if h_np.size > 0 else -0.1
            max_h = np.max(h_np) if h_np.size > 0 else 1.0
            ax.set_ylim(min(min_h - 0.1, -0.2), max(max_h + 0.2, 0.6)) 

            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Beri ruang untuk legenda
            plt.show() # Tampilkan plot

        # Disconnect PyBullet
        if 'env' in locals() and hasattr(env, 'client') and env.client.isConnected():
            try:
                env.client.disconnect()
                print("PyBullet disconnected.")
            except Exception as e:
                print(f"Error disconnecting PyBullet: {e}")
        
#         # Konversi list ke numpy array
#         h_log = np.array(h_log)
#         torque_log = np.array(torque_log)
#         position_log = np.array(position_log)
#         velocity_log = np.array(velocity_log)

#         # Buat 4 subplot
#         fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
#         fig.suptitle('Analisis Simulasi Robot Franka Panda', fontsize=16)

#         # Plot 1: Evolusi Batasan Keamanan (h(z))
#         constraint_labels = ["X max", "Y max", "Z max", "X min", "Y min", "Z min"]
        
#         # Kustomisasi plot agar terlihat seperti gambar
#         # Menggunakan colormap Biru untuk garis
#         num_lines = h_log.shape[1]
#         colors = plt.cm.Blues(np.linspace(0.4, 1, num_lines))

#         for i in range(num_lines):
#             axs[0].plot(time_log, h_log[:, i], color=colors[i])

#         # Menambahkan garis putus-putus hitam di h=0
#         axs[0].axhline(0, color='k', linestyle='--')

#         # Mengarsir wilayah tidak aman (h<0) dengan warna merah
#         ymin, _ = axs[0].get_ylim()
#         axs[0].axhspan(ymin, 0, facecolor='mistyrose', alpha=0.5, zorder=0)

#         axs[0].set_title('Evolusi Batasan Keamanan (h(z))')
#         axs[0].set_ylabel('Nilai h(z)')
#         axs[0].grid(False)

#         # Plot 2: Torsi Sendi (Torque)
#         for i in range(torque_log.shape[1]):
#             axs[1].plot(time_log, torque_log[:, i], label=f'Sendi {i+1}')
#         axs[1].set_title('Perintah Torsi Aman (Γ*)')
#         axs[1].set_ylabel('Torsi (Nm)')
#         axs[1].grid(True)
#         axs[1].legend(fontsize='small')

#         # Plot 3: Kecepatan Sendi (Velocity)
#         for i in range(velocity_log.shape[1]):
#             axs[2].plot(time_log, velocity_log[:, i], label=f'Sendi {i+1}')
#         axs[2].set_title('Kecepatan Sendi (q_dot)')
#         axs[2].set_ylabel('Kecepatan (rad/s)')
#         axs[2].grid(True)
#         axs[2].legend(fontsize='small')

#         # Plot 4: Posisi Sendi (Position)
#         for i in range(position_log.shape[1]):
#             axs[3].plot(time_log, position_log[:, i], label=f'Sendi {i+1}')
#         axs[3].set_title('Posisi Sendi (q)')
#         axs[3].set_ylabel('Posisi (rad)')
#         axs[3].set_xlabel('Waktu (s)')
#         axs[3].grid(True)
#         axs[3].legend(fontsize='small')

#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#         plt.show()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-effector safe-set containment experiment."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="torque",
        help="Control method to use (default: torque)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method)
