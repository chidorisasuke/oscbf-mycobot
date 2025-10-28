"""
Implementasi dan Tuning OSCBF dengan PoseTaskTorqueController untuk UR5e 6-DOF.
Tujuan: Melacak target pose (posisi + orientasi) 6D secara stabil dan aman.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF
# ================== IMPORT UNTUK UR5e ==================
from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.manipulation_env import UR5eTorqueControlEnv
# =======================================================
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController
from oscbf.utils.trajectory import SinusoidalTaskTrajectory

# Konfigurasi keamanan untuk menjaga end-effector di dalam sebuah kotak
@jax.tree_util.register_static
class EESafeSetTorqueConfig(OSCBFTorqueConfig):
    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        # Aktifkan kompensasi Coriolis untuk performa lebih baik
        super().__init__(robot, compensate_centrifugal_coriolis=True)

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        # Batasan area kerja 6D
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    # Gunakan nilai alpha yang lebih kuat sebagai awal untuk robot besar
    def alpha(self, h): return 100.0 * h
    def alpha_2(self, h_2): return 100.0 * h_2

# # Fungsi perhitungan kontrol utama (tetap generik, tapi nullspace_posture_goal disesuaikan)
# def compute_control(
#     robot: Manipulator,
#     osc_controller: PoseTaskTorqueController,
#     cbf: CBF,
#     z: ArrayLike,
#     z_ee_des: ArrayLike,
# ):
#     q, qdot = z[:robot.num_joints], z[robot.num_joints:]
#     # --- DEBUG: Periksa Input ---
#     jax.debug.print("DEBUG compute_control: q = {x}", x=q)
#     jax.debug.print("DEBUG compute_control: qdot = {x}", x=qdot)

#     # --- DEBUG: Periksa Hasil Perhitungan Dinamika ---
#     M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

#     # Postur nullspace default (6-DOF) - Akan diabaikan karena non-redundant
#     nullspace_posture_goal = jnp.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])

#     u_nom = osc_controller(
#         q, qdot,
#         pos=ee_tmat[:3, 3], rot=ee_tmat[:3, :3],
#         des_pos=z_ee_des[:3], des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
#         des_vel=z_ee_des[12:15], des_omega=z_ee_des[15:18],
#         des_accel=jnp.zeros(3), des_alpha=jnp.zeros(3),
#         des_q=nullspace_posture_goal, des_qdot=jnp.zeros(robot.num_joints),
#         J=J, M=M, M_inv=M_inv, g=g, c=c
#     )
#     return cbf.safety_filter(z, u_nom)

# Fungsi perhitungan kontrol utama
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q, qdot = z[:robot.num_joints], z[robot.num_joints:]
    
    # --- DEBUG: Periksa Input ---
    jax.debug.print("DEBUG compute_control: q = {x}", x=q)
    jax.debug.print("DEBUG compute_control: qdot = {x}", x=qdot)

    # --- DEBUG: Periksa Hasil Perhitungan Dinamika ---
    try:
        M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
        
        # Cek apakah ada NaN di hasil perhitungan
        jax.debug.print("DEBUG compute_control: M contains NaN? {x}", x=jnp.isnan(M).any())
        jax.debug.print("DEBUG compute_control: g contains NaN? {x}", x=jnp.isnan(g).any())
        jax.debug.print("DEBUG compute_control: c contains NaN? {x}", x=jnp.isnan(c).any())
        jax.debug.print("DEBUG compute_control: J contains NaN? {x}", x=jnp.isnan(J).any())
        # Anda bisa print nilainya langsung jika perlu, tapi cek NaN lebih efisien
        # jax.debug.print("DEBUG compute_control: M = {x}", x=M)
        # jax.debug.print("DEBUG compute_control: J = {x}", x=J)

    except Exception as e:
        # Tangkap jika error terjadi di dalam torque_control_matrices
        jax.debug.print("!!! ERROR saat menghitung torque_control_matrices: {e}", e=e)
        # Kembalikan nilai nol agar tidak crash (meskipun ini tidak benar)
        return jnp.zeros(robot.num_joints)

    # --- Perhitungan u_nom (seperti sebelumnya) ---
    nullspace_posture_goal = jnp.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    u_nom = osc_controller(
        q, qdot,
        pos=ee_tmat[:3, 3], rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3], des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15], des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3), des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal, des_qdot=jnp.zeros(robot.num_joints),
        J=J, M=M, M_inv=M_inv, g=g, c=c
    )
    
    # --- DEBUG: Periksa u_nom ---
    jax.debug.print("DEBUG compute_control: u_nom contains NaN? {x}", x=jnp.isnan(u_nom).any())
    # jax.debug.print("DEBUG compute_control: u_nom = {x}", x=u_nom) # Jika perlu lihat nilainya

    # --- Filter CBF (seperti sebelumnya) ---
    safe_command = cbf.safety_filter(z, u_nom)
    
    # --- DEBUG: Periksa hasil akhir ---
    jax.debug.print("DEBUG compute_control: safe_command (tau) contains NaN? {x}", x=jnp.isnan(safe_command).any())

    return safe_command


def main(control_method="torque"):
    # ================== GUNAKAN ROBOT UR5e ==================
    robot = load_ur5e()
    # =======================================================

    # ================== PARAMETER UNTUK DI-Tuning (AWAL UNTUK UR5e) ==================
    # 1. Gain Kontroler (gunakan nilai mirip Franka sebagai awal)
    # kp_pos = 50.0   # Gain Proportional untuk Posisi
    # kp_rot = 20.0   # Gain Proportional untuk Rotasi
    # kd_pos = 20.0   # Gain Derivative untuk Posisi (peredam)
    # kd_rot = 10.0   # Gain Derivative untuk Rotasi (peredam)
    # kp_joint = 0.0  # Gain Nullspace (diabaikan untuk 6-DOF)
    # kd_joint = 0.0  # Gain Nullspace (diabaikan untuk 6-DOF)

    kp_pos = 1.0
    kp_rot = 0.5
    kd_pos = 0.2
    kd_rot = 0.1
    kp_joint = 0.0
    kd_joint = 0.0

    # 2. Parameter Lintasan (Trajectory) - Sesuaikan dengan jangkauan UR5e
    init_pos = (0.5, 0, 0.5) # Di tengah area kerja UR5e
    amplitude = (0.2, 0.2, 0.1) # Gerakan lebih besar
    angular_freq = (0.5, 0.5, 0.3) # Gerakan lebih lambat untuk mulai

    # 3. Posisi Awal Robot UR5e
    # ur5e_q_init = (0, -np.pi/2, 0, -np.pi/2, 0, 0) # Posisi tegak standar
    ur5e_q_init = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # ==============================================================================

    # Definisikan area kerja aman (kotak hijau) - Sesuaikan dengan jangkauan UR5e
    pos_min = (0.2, -0.5, 0.1)
    pos_max = (0.8, 0.5, 0.8)

    config = EESafeSetTorqueConfig(robot, pos_min, pos_max)
    cbf = CBF.from_config(config)

    traj = SinusoidalTaskTrajectory(
        init_pos=init_pos,
        init_rot=np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        amplitude=amplitude,
        angular_freq=angular_freq,
        phase=(0, 0, 0)
    )

    # ================== GUNAKAN ENVIRONMENT UR5e ==================
    env = UR5eTorqueControlEnv(
        xyz_min=pos_min, xyz_max=pos_max,
        q_init=ur5e_q_init, traj=traj,
        real_time=False, # Set False untuk pengumpulan data cepat
        timestep=1 / 240 # Tingkatkan frekuensi kontrol jika mungkin
    )
    # ===========================================================

    osc_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        tau_min=None, tau_max=None
    )

    compute_control_jit = jax.jit(lambda z, z_des: compute_control(robot, osc_controller, cbf, z, z_des))

    # --- Persiapan untuk Plotting ---
    time_log, h_log, torque_log, position_log, velocity_log = [], [], [], [], []

    print("Menjalankan simulasi UR5e dengan PoseTaskTorqueController...")
    try:
        simulation_duration = 30 # Jalankan selama 20 detik waktu simulasi
        while env.t < simulation_duration:
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            tau = compute_control_jit(q_qdot, z_zdot_ee_des)

            # --- TAMBAHKAN PRINT UNTUK DEBUGGING ---
            print(f"DEBUG: Torsi yang dikirim: {np.round(tau, 3)}")
            # ----------------------------------------

            env.apply_control(tau)
            env.step()

            # --- Pengumpulan Data ---
            time_log.append(env.t)
            q = q_qdot[:robot.num_joints]
            q_dot = q_qdot[robot.num_joints:]
            position_log.append(q)
            velocity_log.append(q_dot)
            torque_log.append(tau)
            h_values = config.h_2(q_qdot)
            h_log.append(h_values)

    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\n!!! TERJADI ERROR YANG MENGHENTIKAN SIMULASI !!!")
        print(f"Jenis Error: {type(e).__name__}")
        print(f"Pesan Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Kode Plotting ---
        print("\nSimulation finished or stopped. Plotting data...")

        if not time_log: # Cek jika tidak ada data
             print("Tidak ada data untuk di-plot.")
             return # Keluar jika tidak ada data

        min_len = min(len(time_log), len(h_log), len(torque_log), len(position_log), len(velocity_log))
        time_log = time_log[:min_len]
        h_log = np.array(h_log[:min_len])
        torque_log = np.array(torque_log[:min_len])
        position_log = np.array(position_log[:min_len])
        velocity_log = np.array(velocity_log[:min_len])

        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle('Analisis Simulasi Robot UR5e', fontsize=16) # Ganti Judul

        # Plot 1: Evolusi Batasan Keamanan (h(z))
        constraint_labels = ["X max", "Y max", "Z max", "X min", "Y min", "Z min"]
        if h_log.shape[1] >= 6: # Pastikan ada cukup data batasan
            for i in range(6): # Plot hanya 6 batasan area kerja
                axs[0].plot(time_log, h_log[:, i], label=constraint_labels[i])
        axs[0].axhline(0, color='r', linestyle='--', label='Batas Aman (h=0)')
        axs[0].set_title('Evolusi Batasan Keamanan (h(z)) - Area Kerja')
        axs[0].set_ylabel('Nilai h(z)')
        axs[0].grid(True)
        axs[0].legend(fontsize='small')

        # Plot 2: Torsi Sendi (Torque)
        start_index = 10
        if min_len > start_index:
            for i in range(torque_log.shape[1]):
                axs[1].plot(time_log[start_index:], torque_log[start_index:, i], label=f'Sendi {i+1}')
        axs[1].set_title('Perintah Torsi Aman (Γ*)')
        axs[1].set_ylabel('Torsi (Nm)')
        axs[1].grid(True)
        axs[1].legend(fontsize='small')

        # Plot 3: Kecepatan Sendi (Velocity)
        if min_len > start_index:
            for i in range(velocity_log.shape[1]):
                axs[2].plot(time_log[start_index:], velocity_log[start_index:, i], label=f'Sendi {i+1}')
        axs[2].set_title('Kecepatan Sendi (q_dot)')
        axs[2].set_ylabel('Kecepatan (rad/s)')
        axs[2].grid(True)
        axs[2].legend(fontsize='small')

        # Plot 4: Posisi Sendi (Position)
        if min_len > 0:
            for i in range(position_log.shape[1]):
                axs[3].plot(time_log, position_log[:, i], label=f'Sendi {i+1}')
        axs[3].set_title('Posisi Sendi (q)')
        axs[3].set_ylabel('Posisi (rad)')
        axs[3].set_xlabel('Waktu (s)')
        axs[3].grid(True)
        axs[3].legend(fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.show() # Tampilkan jendela plot
        plt.savefig('ur5e_simulation_results.png') # Atau simpan ke file
        print("Grafik disimpan sebagai ur5e_simulation_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UR5e pose tracking experiment.")
    # Argumen control_method bisa dihapus karena kita hanya pakai torque
    # parser.add_argument("--control_method", type=str, choices=["torque"], default="torque")
    args = parser.parse_args()
    main() # Panggil main tanpa argumen