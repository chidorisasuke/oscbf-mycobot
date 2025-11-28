"""
Menjalankan simulasi UR5e HANYA dengan kontrol nominal (OSC).
TIDAK ADA SAFETY CONTROL (CBF dinonaktifkan).
Digunakan untuk membandingkan perilaku dengan dan tanpa filter keamanan.
"""

import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# --- Impor CBF dan Config DIHAPUS ---
# from cbfpy import CBF
# from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
# ------------------------------------

from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.manipulation_env import UR5eTorqueControlEnv, UR5eVelocityControlEnv
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
)

# --- Kelas Config CBF (EESafeSetTorqueConfig, dll.) DIHAPUS ---

# --- Fungsi Kontrol Dimodifikasi (MENGEMBALIKAN u_nom) ---
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    # cbf: CBF, # Dihapus
    compensate_centrifugal_coriolis: bool,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)

    nullspace_posture_goal = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])

    # Hitung perintah nominal
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
    # Kembalikan perintah nominal SECARA LANGSUNG
    return u_nom


def compute_velocity_control(
    robot: Manipulator,
    osc_controller: PoseTaskVelocityController,
    # cbf: CBF, # Dihapus
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

    des_q = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    
    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )
    # Kembalikan perintah nominal SECARA LANGSUNG
    return u_nom


def main(control_method="torque"):
    assert control_method in ["torque", "velocity"]

    robot = load_ur5e()
    
    # --- Konfigurasi CBF DIHAPUS ---
    compensate_centrifugal_coriolis = False
    # -----------------------------

    traj = SinusoidalTaskTrajectory(
        init_pos=(0.55, 0, 0.45),
        init_rot=np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ),
        amplitude=(0.25, 0, 0),
        angular_freq=(1, 0, 0),
        phase=(0, 0, 0),
    )
    timestep = 1 / 1000
    bg_color = (1, 1, 1)

    # Gunakan posisi awal yang aman (tidak singular)
    ur5e_q_init = (0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0)

    if control_method == "torque":
        env = UR5eTorqueControlEnv(
            # xyz_min dan xyz_max dihapus (tidak ada kotak hijau)
            q_init=ur5e_q_init,
            traj=traj,
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )
    else:
        env = UR5eVelocityControlEnv(
            # xyz_min dan xyz_max dihapus
            q_init=ur5e_q_init,
            traj=traj,
            real_time=True,
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

    # Gains (Anda bisa menaikkan ini untuk melihat perilaku agresif)
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
        tau_min=None,
        tau_max=None,
    )

    osc_velocity_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        qdot_min=None,
        qdot_max=None,
    )

    # --- Fungsi JIT Dimodifikasi (tanpa argumen cbf) ---
    @jax.jit
    def compute_torque_control_jit(z, z_ee_des):
        return compute_torque_control(
            robot,
            osc_torque_controller,
            # torque_cbf, # Dihapus
            compensate_centrifugal_coriolis,
            z,
            z_ee_des,
        )

    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des):
        return compute_velocity_control(
            robot, 
            osc_velocity_controller, 
            # velocity_cbf, # Dihapus
            z, 
            z_ee_des
        )

    if control_method == "torque":
        compute_control = compute_torque_control_jit
    elif control_method == "velocity":
        compute_control = compute_velocity_control_jit
    else:
        raise ValueError(f"Invalid control method: {control_method}")

    while True:
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        
        # tau sekarang adalah perintah nominal (u_nom)
        tau = compute_control(q_qdot, z_zdot_ee_des) 
        
        env.apply_control(tau)
        env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-effector containment experiment for UR5e (NOMINAL CONTROL - NO SAFETY)."
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