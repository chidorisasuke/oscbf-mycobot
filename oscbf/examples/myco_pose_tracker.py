"""
Implementasi dan Tuning OSCBF dengan PoseTaskTorqueController untuk MyCobot 6-DOF.
Tujuan: Melacak target pose (posisi + orientasi) 6D secara stabil dan aman.
"""

import argparse
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
from oscbf.utils.trajectory import SinusoidalTaskTrajectory

# Konfigurasi keamanan untuk menjaga end-effector di dalam sebuah kotak
@jax.tree_util.register_static
class EESafeSetTorqueConfig(OSCBFTorqueConfig):
    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot, compensate_centrifugal_coriolis=True)

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h): return 100.0 * h
    def alpha_2(self, h_2): return 100.0 * h_2

# Fungsi perhitungan kontrol utama
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q, qdot = z[:robot.num_joints], z[robot.num_joints:]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    # Postur nullspace default (akan diabaikan karena non-redundant, tapi tetap harus ada)
    nullspace_posture_goal = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    u_nom = osc_controller(
        q, qdot, 
        pos=ee_tmat[:3, 3], rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3], des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15], des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3), des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal, des_qdot=jnp.zeros(robot.num_joints),
        J=J, M=M, M_inv=M_inv, g=g, c=c
    )
    return cbf.safety_filter(z, u_nom)


def main(control_method="torque"):
    robot = load_mycobot()

    # ================== PARAMETER UNTUK DI-Tuning ==================
    # 1. Gain Kontroler (mulai dari nilai ini, lalu naikkan perlahan)
    kp_pos = 40.0   # Gain Proportional untuk Posisi
    kp_rot = 25.0   # Gain Proportional untuk Rotasi
    kd_pos = 15.0   # Gain Derivative untuk Posisi (peredam)
    kd_rot = 10.0   # Gain Derivative untuk Rotasi (peredam)
    kp_joint = 0.0  # Gain Nullspace (diabaikan untuk MyCobot, set ke 0)
    kd_joint = 0.0  # Gain Nullspace (diabaikan untuk MyCobot, set ke 0)

    # 2. Parameter Lintasan (Trajectory)
    init_pos = (0.2, 0, 0.15)
    amplitude = (0.08, 0.08, 0.05)
    angular_freq = (0.5, 0.5, 0.3)

    # 3. Posisi Awal Robot
    mycobot_q_init = (0, 0, 0, 0, 0, 0)
    # =============================================================
    
    pos_min = (-0.1, -0.2, 0.1)
    pos_max = (0.35, 0.2, 0.35)
    
    config = EESafeSetTorqueConfig(robot, pos_min, pos_max)
    cbf = CBF.from_config(config)

    # traj = SinusoidalTaskTrajectory(
    #     init_pos=init_pos,
    #     init_rot=np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    #     amplitude=amplitude,
    #     angular_freq=angular_freq,
    #     phase=(0, 0, 0)
    # )
    
    env = MyCobotTorqueControlEnv(
        xyz_min=pos_min, xyz_max=pos_max,
        q_init=mycobot_q_init, 
        # traj=traj,
        real_time=True, # Gunakan True untuk observasi visual
        timestep=1 / 240
    )

    osc_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        tau_min=None, tau_max=None
    )

    compute_control_jit = jax.jit(lambda z, z_des: compute_control(robot, osc_controller, cbf, z, z_des))

    print("Menjalankan simulasi dengan PoseTaskTorqueController...")
    try:
        while True:
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            tau = compute_control_jit(q_qdot, z_zdot_ee_des)
            env.apply_control(tau)
            env.step()
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan.")


if __name__ == "__main__":
    main()