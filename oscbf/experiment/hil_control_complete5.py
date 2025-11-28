"""
HIL Control Complete 5: UR5e Hardware-in-the-Loop Control using OSCBF Library

This implementation uses the OSCBF (Operational Space Control with Control Barrier Functions) 
library for safe torque control of the UR5e manipulator. The code includes:
- Torque control using OSCBF library
- Null space computation using OSCBF library
- Dynamic computation using OSCBF library
- Safety features with CBF constraints
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import jax
import jax.numpy as jnp

# Import from OSCBF library
from oscbf.core.manipulator import load_ur5e
from oscbf.core.controllers import PoseTaskTorqueController
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.manipulation_env import UR5eTorqueControlEnv
from cbfpy import CBF
from oscbf.utils.trajectory import SinusoidalTaskTrajectory

# Additional imports for real robot control
import rtde_control
import rtde_receive

# FIX 3: Accurate dynamic data from hil_control_complete4.py
MASSES = [3.761, 8.058, 2.846, 1.37, 1.3, 0.365]
COMS = [
    np.array([0.0, -0.02561, 0.00193]), np.array([0.2125, 0.0, 0.11336]),
    np.array([-0.2422, 0.0, 0.0265]), np.array([0.0, -0.0018, 0.01634]),
    np.array([0.0, 0.0018, 0.01634]), np.array([0.0, 0.0, -0.001159])
]
INERTIA_TENSORS = [
    np.array([[0.00700210, 0, 0], [0, 0.00648091, 0], [0, 0, 0.00657286]]),
    np.array([[0.01505885, 0, 0], [0, 0.33388086, 0], [0, 0, 0.33247207]]),
    np.array([[0.00399632, 0, 0], [0, 0.07879254, 0], [0, 0, 0.07848510]]),
    np.array([[0.00165491, 0, 0], [0, 0.00135962, 0], [0, 0, 0.00126279]]),
    np.array([[0.00135617, 0, 0], [0, 0.00127827, 0], [0, 0, 0.00096614]]),
    np.array([[0.00018694, 0, 0], [0, 0.00018908, 0], [0, 0, 0.00025756]])
]

# =================================================================================
# VARIABEL & KONSEP UTAMA
# =================================================================================
# 
# ### State Feedback (dari Robot) ###
# *   `q_real`, `dq_real`: Posisi (`q`) dan kecepatan (`dq`) sendi (joint) yang dibaca *aktual* dari robot.
# *   `x_ee_pose`, `v_ee_act`: Posisi (`x`) dan kecepatan (`v`) End-Effector (EE) di frame Cartesius yang
#     dibaca *aktual* dari robot (TCP: Tool Center Point).
#
# ### Referensi Trajektori (Target) ###
# *   `x_des`, `xd_des`, `xdd_des`: Posisi (`x_d`), kecepatan (`xd_d`), dan percepatan (`xdd_d`) EE *yang diinginkan*
#     (desired) pada setiap waktu `t`. Dihasilkan oleh `calculate_trajectory_point`.
#
# ### Variabel Kontrol Internal ###
# *   `q_state`, `dq_state`: Salinan dari state *aktual* (`q_real`, `dq_real`) yang digunakan sebagai 
#     input untuk perhitungan kontrol di setiap iterasi loop.
# *   `x_actual`, `v_actual`: Salinan dari state *aktual* EE (`x_ee_pose`, `v_ee_act`) yang digunakan 
#     sebagai input untuk perhitungan kontrol.
# *   `tau_task`, `tau_null`, `tau_total`:
#     - `tau_task`: Torsi yang dihitung untuk menyelesaikan tugas utama (mengikuti trajektori).
#     - `tau_null`: Torsi yang bekerja di "null-space" (ruang gerak yang tidak mempengaruhi posisi EE),
#                   digunakan untuk tugas sekunder seperti menghindari singularity atau rintangan.
#     - `tau_total`: Gabungan dari `tau_task` dan `tau_null`.
#
# ### Variabel Perintah (ke Robot) ###
# *   `tau_cmd`: Perintah torsi *yang dikirim* ke robot.
# *   `q_cmd`: Estimasi posisi sendi hasil dari integrasi `dq_cmd`. Digunakan hanya untuk logging dan 
#     perbandingan, bukan untuk kontrol langsung.
#
# ====================================================================
# 1. KONFIGURASI SISTEM
# ====================================================================
parser = argparse.ArgumentParser(description="UR5e HIL Control using OSCBF: Strategy 5")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP Robot (127.0.0.1 untuk URSim)")
parser.add_argument("--save_log", action="store_true", help="Simpan log ke file .pkl")
parser.add_argument("--mode", choices=["sim", "real"], default="real", help="Mode operasi: sim (PyBullet Only) atau real (URSim/Fisik)")
parser.add_argument("--traj", choices=["linear_x", "linear_y", "circle", "circle_3d", "point"], 
                    default="linear_x", help="Pilih Trajektori")
parser.add_argument("--speed", choices=["act", "calc"], default="calc", help="act = asal kecepatan aktual dari TCPSpeed (v_ee_act), calc= asal kecepatan dari getActualQd (dq_real)")
parser.add_argument("--fitur", choices=["torque", "safe_torque", "admittance"], default="safe_torque", help="Fitur kontrol yang diaktifkan: safe_torque (dengan CBF), torque (tanpa CBF), atau admittance (dengan filter massa)")
parser.add_argument("--speed_mode", choices=["slow", "normal", "fast"], default="normal", 
                    help="Kecepatan gerakan: slow (0.05Hz), normal (0.2Hz), fast (0.5Hz)")
parser.add_argument("--nullspace", choices=["on", "off"], default="on", help="Aktif/non-aktif null space (default: on)")
parser.add_argument("--command", choices=["speedJ", "moveL", "servoL"], default="speedJ", help="Jenis perintah kontrol robot: speedJ (kecepatan joint), moveL (posisi linier), servoL (kecepatan linier) - default speedJ")
args = parser.parse_args()

# --- Konfigurasi Path & Link ---
URDF_PATH = os.path.join(os.path.dirname(__file__), "../assets/ur5e/ur5e.urdf") 
EE_LINK_PINOCCHIO = "tool0"        # Frame untuk perhitungan (Pinocchio)
EE_LINK_PYBULLET = "wrist_3_link"  # Link fisik terdekat (PyBullet Visual)

# --- Konfigurasi Kontrol ---
CONTROL_FREQ = 125.0               # Hz
DT = 1.0 / CONTROL_FREQ
RUNTIME = 18.0                     # Detik
MAX_VEL_JOINT = 1.0                # Rad/s (Safety limit)
ROBOT_IP = args.ip
TRAJECTORY = args.traj

# --- Konfigurasi Trajektori (Linear X) ---
A = 0.1        # Amplitudo (meter) - Mulai kecil biar aman
if args.speed_mode == "slow":
    print("🐢 Mode: SLOW")
    omega = 2 * np.pi * 0.05  # 0.05 Hz (Sangat Lambat)
    MOVE_DURATION = 10.0      # 10 Detik untuk pindah titik
elif args.speed_mode == "normal":
    print("🚶 Mode: NORMAL")
    omega = 2 * np.pi * 0.2   # 0.2 Hz
    MOVE_DURATION = 5.0
elif args.speed_mode == "fast":
    print("🐇 Mode: FAST")
    omega = 2 * np.pi * 0.5   # 0.5 Hz (Cepat)
    MOVE_DURATION = 2.0       # 2 Detik (Cepat)

# Sudut Miring untuk Circle 3D
TILT_ANGLE = np.pi / 6 # 30 Derajat
C_TILT = np.cos(TILT_ANGLE)
S_TILT = np.sin(TILT_ANGLE)

# --- Tuning Gains (Mode Stabil) ---
# Admittance (Mass-Spring-Damper Virtual) - ADAPTIVE BASED ON SPEED MODE AND FEATURE
if args.fitur == "admittance":
    if args.speed_mode == "slow":
        Ma = np.diag([2.0, 2.0, 2.0])        # Higher mass for slow = smoother
        Ba = np.diag([15.0, 15.0, 15.0])     # Higher damping for stability in slow
        Ka = np.diag([5.0, 5.0, 5.0])        # Some stiffness to maintain tracking in slow
    elif args.speed_mode == "normal":
        Ma = np.diag([1.0, 1.0, 1.0])        # Balanced mass
        Ba = np.diag([10.0, 10.0, 10.0])     # Balanced damping
        Ka = np.diag([2.0, 2.0, 2.0])        # Some stiffness to maintain tracking
    else:  # fast
        Ma = np.diag([0.5, 0.5, 0.5])        # Lower mass for fast = more responsive
        Ba = np.diag([8.0, 8.0, 8.0])        # Lower damping but sufficient for stability
        Ka = np.diag([1.0, 1.0, 1.0])        # Minimal stiffness to allow compliance in fast mode
else:  # torque control
    Ma = np.diag([1.0, 1.0, 1.0])            # Default values for torque mode
    Ba = np.diag([10.0, 10.0, 10.0])
    Ka = np.diag([0.0, 0.0, 0.0])            # Pure admittance not used in torque mode

# PD Task Space (Computed Force) - ADAPTIVE BASED ON SPEED MODE AND FEATURE
if args.fitur == "torque" or args.fitur == "safe_torque":
    if args.speed_mode == "slow":
        Kp_task = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])  # Lower gains for slow
        Kd_task = np.diag([20.0, 20.0, 20.0, 10.0, 10.0, 10.0])    # Lower derivative gains
    elif args.speed_mode == "normal":
        Kp_task = np.diag([150.0, 150.0, 150.0, 75.0, 75.0, 75.0])  # Moderate gains
        Kd_task = np.diag([25.0, 25.0, 25.0, 12.5, 12.5, 12.5])    # Moderate derivative gains
    else:  # fast
        Kp_task = np.diag([200.0, 200.0, 200.0, 100.0, 100.0, 100.0])  # Higher gains but not too high
        Kd_task = np.diag([30.0, 30.0, 30.0, 15.0, 15.0, 15.0])    # Balanced derivative gains
else:  # admittance mode - adjust PD gains to work with admittance filter
    if args.speed_mode == "slow":
        Kp_task = np.diag([80.0, 80.0, 80.0, 40.0, 40.0, 40.0])  # Lower gains for admittance mode in slow
        Kd_task = np.diag([15.0, 15.0, 15.0, 7.5, 7.5, 7.5])  # Lower derivative gains
    elif args.speed_mode == "normal":
        Kp_task = np.diag([120.0, 120.0, 120.0, 60.0, 60.0, 60.0])  # Moderate gains for admittance mode
        Kd_task = np.diag([20.0, 20.0, 20.0, 10.0, 10.0, 10.0])    # Moderate derivative gains
    else:  # fast
        Kp_task = np.diag([160.0, 160.0, 160.0, 80.0, 80.0, 80.0])  # Higher gains but not too high for admittance mode
        Kd_task = np.diag([25.0, 25.0, 25.0, 12.5, 12.5, 12.5])    # Balanced derivative gains

# Null Space (Postur) - ADAPTIVE BASED ON NULLSPACE SETTING
# Gain kecil agar tidak melawan task utama
if args.nullspace == "on":
    Kp_null = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5 
else:
    Kp_null = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.0  # Zero when nullspace is off
q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

# Initialize UR5e manipulator from OSCBF library
print("🔧 Menginisialisasi UR5e manipulator dari OSCBF...")
robot = load_ur5e()

# FIX 3: Inject accurate dynamic data to robot model
# Update link masses
link_masses_list = []
for i in range(robot.num_joints):
    link_masses_list.append(MASSES[i])
robot.link_masses = tuple(link_masses_list)

# Update link local inertia positions (COMs)
link_local_inertia_positions_list = []
for i in range(robot.num_joints):
    link_local_inertia_positions_list.append(tuple(COMS[i]))
robot.link_local_inertia_positions = tuple(link_local_inertia_positions_list)

# Update link local inertias
link_local_inertias_list = []
for i in range(robot.num_joints):
    link_local_inertias_list.append(tuple(map(tuple, INERTIA_TENSORS[i])))
robot.link_local_inertias = tuple(link_local_inertias_list)

print(f"✅ UR5e manipulator siap. Jumlah joint: {robot.num_joints}")

# Initialize OSCBF controller
osc_controller = PoseTaskTorqueController(
    n_joints=robot.num_joints,
    kp_task=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),  # [x, y, z, rx, ry, rz]
    kd_task=np.array([20.0, 20.0, 20.0, 10.0, 10.0, 10.0]),    # [x, y, z, rx, ry, rz]
    kp_joint=10.0,  # Gain for null space posture task
    kd_joint=5.0,   # Gain for null space damping
    tau_min=-np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]),  # Joint torque limits
    tau_max=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
)

# Initialize CBF if safe_torque mode is enabled
cbf = None
if args.fitur == "safe_torque":
    # Define a safe workspace for the end-effector
    pos_min = np.array([0.1, -0.5, 0.1])
    pos_max = np.array([0.9, 0.5, 0.9])
    
    @jax.tree_util.register_static
    class EESafeSetTorqueConfig(OSCBFTorqueConfig):
        def __init__(self, robot, pos_min, pos_max):
            self.pos_min = np.asarray(pos_min)
            self.pos_max = np.asarray(pos_max)
            super().__init__(robot, compensate_centrifugal_coriolis=True)

        def h_2(self, z, **kwargs):
            q = z[: self.num_joints]
            ee_pos = self.robot.ee_position(q)
            return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        def alpha(self, h): return 10.0 * h
        def alpha_2(self, h_2): return 10.0 * h_2
    
    config = EESafeSetTorqueConfig(robot, pos_min, pos_max)
    cbf = CBF.from_config(config)

def calculate_trajectory_point(t, x_base_pos):
    if TRAJECTORY == "linear_x":
        offset = np.array([A * np.sin(omega * t), 0.0, 0.0])
        x_d = x_base_pos + offset
        xd_d = np.array([A * omega * np.cos(omega * t), 0.0, 0.0])
        xdd_d = np.array([-A * omega**2 * np.sin(omega * t), 0.0, 0.0])
    
    elif TRAJECTORY == "linear_y":
        offset = np.array([0.0, A * np.sin(omega * t), 0.0])
        x_d = x_base_pos + offset
        xd_d = np.array([0.0, A * omega * np.cos(omega * t), 0.0])
        xdd_d = np.array([0.0, -A * omega**2 * np.sin(omega * t), 0.0])
    
    elif TRAJECTORY == "circle":
        offset = np.array([A * np.cos(omega * t), A * np.sin(omega * t), 0.0])
        x_d = x_base_pos + offset
        xd_d = np.array([-A * omega * np.sin(omega * t), A * omega * np.cos(omega * t), 0.0])
        xdd_d = np.array([-A * omega**2 * np.cos(omega * t), -A * omega**2 * np.sin(omega * t), 0.0])

    elif TRAJECTORY == "circle_3d":
        off_x = A * np.cos(omega * t)
        off_y_flat = A * np.sin(omega * t)
        off_y = off_y_flat * C_TILT 
        off_z = off_y_flat * S_TILT 
        x_d = x_base_pos + np.array([off_x, off_y, off_z])

        vel_x = -A * omega * np.sin(omega * t)
        vel_y_flat = A * omega * np.cos(omega * t)
        xd_d = np.array([vel_x, vel_y_flat * C_TILT, vel_y_flat * S_TILT])

        acc_x = -A * omega**2 * np.cos(omega * t)
        acc_y_flat = -A * omega**2 * np.sin(omega * t)
        xdd_d = np.array([acc_x, acc_y_flat * C_TILT, acc_y_flat * S_TILT])

    elif TRAJECTORY == "point":
        # --- KONFIGURASI GERAKAN ---
        TARGET_OFFSET_1 = np.array([0.10, -0.05, 0.0]) 
        TARGET_OFFSET_2 = np.array([0.05, 0.10, 0.0])
        
        # Definisi Waktu
        T_MOVE = MOVE_DURATION
        T_STAY = 3.0 # Waktu diam
        
        # FASE 1: Gerak ke Titik 1
        if t <= T_MOVE:
            tau = t / T_MOVE
            # Smoothstep
            s = 3*(tau**2) - 2*(tau**3)
            v_scale = 6*tau - 6*(tau**2)
            a_scale = 6 - 12*tau
            
            x_d = x_base_pos + TARGET_OFFSET_1 * s
            xd_d = TARGET_OFFSET_1 * v_scale / T_MOVE
            xdd_d = TARGET_OFFSET_1 * a_scale / (T_MOVE**2)

        # FASE 2: Diam di Titik 1
        elif t <= (T_MOVE + T_STAY):
            x_d = x_base_pos + TARGET_OFFSET_1
            xd_d = np.zeros(3)
            xdd_d = np.zeros(3)

        # FASE 3: Gerak ke Titik 2
        elif t <= (2*T_MOVE + T_STAY):
            t_local = t - (T_MOVE + T_STAY)
            tau = t_local / T_MOVE
            s = 3*(tau**2) - 2*(tau**3)
            v_scale = 6*tau - 6*(tau**2)
            a_scale = 6 - 12*tau
            
            start_pos = x_base_pos + TARGET_OFFSET_1
            x_d = start_pos + TARGET_OFFSET_2 * s
            xd_d = TARGET_OFFSET_2 * v_scale / T_MOVE
            xdd_d = TARGET_OFFSET_2 * a_scale / (T_MOVE**2)

        # FASE 4: Diam di Titik 2 (Selesai)
        else:
            x_d = x_base_pos + TARGET_OFFSET_1 + TARGET_OFFSET_2
            xd_d = np.zeros(3)
            xdd_d = np.zeros(3)

    # Create full pose trajectory (position + orientation)
    # For now, keep orientation constant
    des_rot = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    
    # Create full 6D state vector [pos, rot_vec, vel, omega]
    z_ee_des = np.zeros(18)
    z_ee_des[:3] = x_d  # Position
    z_ee_des[3:12] = des_rot.ravel()  # Orientation as rotation matrix (flattened)
    z_ee_des[12:15] = xd_d  # Velocity
    z_ee_des[15:18] = np.zeros(3)  # Angular velocity
    
    return x_d, xd_d, xdd_d, z_ee_des

# JIT compile the control function for better performance
@jax.jit
def compute_control(robot, osc_controller, cbf, z, z_ee_des, nullspace_posture_goal):
    q, qdot = z[:robot.num_joints], z[robot.num_joints:]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    
    # Compute nominal control
    u_nom = osc_controller(
        q, qdot,
        pos=ee_tmat[:3, 3], rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3], des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15], des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3), des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal, des_qdot=jnp.zeros(robot.num_joints),
        J=J, M=M, M_inv=M_inv, g=g, c=c
    )
    
    # Apply CBF safety filter if enabled
    if cbf is not None:
        tau_safe = cbf.safety_filter(z, u_nom)
    else:
        tau_safe = u_nom
    
    return tau_safe

# ====================================================================
# 4. MAIN PROGRAM
# ====================================================================
def main():
    # --- A. Setup PyBullet (Visualizer Only) ---
    print("\n🚀 Memulai Visualisasi PyBullet...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_sim = p.loadURDF(URDF_PATH, useFixedBase=True)
    
    # Cari Joint Indices di PyBullet
    sim_joint_indices = []
    for i in range(p.getNumJoints(robot_sim)):
        if p.getJointInfo(robot_sim, i)[2] != p.JOINT_FIXED:
            sim_joint_indices.append(i)

    rtde_c = None
    rtde_r = None

    # --- B. Setup Koneksi Robot (Real/URSim) ---
    print(f"\n🔌 Menghubungkan ke Robot ({ROBOT_IP})...")
    if args.mode == "real":
        try:
            rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
            rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
            print("✅ Koneksi Berhasil!")
        except Exception as e:
            print(f"❌ Gagal Konek: {e}")
            return
    else: # sim mode
        print("✅ Mode simulasi aktif")

    # --- C. Homing & Inisialisasi Posisi ---
    if args.mode == "real":
        print("\n📍 Homing...")
        rtde_c.moveJ(q_home.tolist(), 0.5, 0.5)
        time.sleep(1.0)
        # Baca posisi awal AKTUAL untuk dijadikan x_base
        tcp_pose = rtde_r.getActualTCPPose()
        x_base = np.array(tcp_pose[:3])
    else:
        # Tentukan posisi dasar imajiner untuk mode simulasi
        x_base = np.array([0.4, -0.2, 0.3])
        # Set simulasi ke posisi home
        for i, idx in enumerate(sim_joint_indices):
            p.resetJointState(robot_sim, idx, q_home[i], 0)

    print(f"✅ Posisi Awal Terdeteksi: {np.round(x_base, 3)}")
    print("   (Lintasan akan dimulai dari titik ini)")


    # --- D. Move to Start Point (Anti-Jump) ---
    print(f"\n📍 2. Bergerak ke Titik Awal Lintasan ({TRAJECTORY})...")
    
    # Hitung di mana t=0 berada
    x_start_traj, _, _, _ = calculate_trajectory_point(0.0, x_base)
    
    if args.mode == "real":
        # Buat target pose lengkap (Posisi baru + Orientasi lama)
        start_pose = rtde_r.getActualTCPPose()
        start_pose = list(start_pose) # Pastikan format list untuk dimodifikasi
        start_pose[0] = x_start_traj[0]
        start_pose[1] = x_start_traj[1]
        start_pose[2] = x_start_traj[2]
        
        # Gerakkan robot pelan-pelan ke titik start
        rtde_c.moveL(start_pose, 0.1, 0.2) 
        time.sleep(0.5) # Beri waktu robot untuk stabil
        
        # FIX: Baca state aktual SETELAH bergerak ke titik awal
        q_real = np.array(rtde_r.getActualQ())
        dq_real = np.array(rtde_r.getActualQd())
        x_ee_pose = np.array(rtde_r.getActualTCPPose()[:3])
        v_ee_act = np.array(rtde_r.getActualTCPSpeed()[:3])
    else:
        # Untuk mode simulasi, state dianggap ideal di titik awal trajektori
        q_real = q_home.copy() # Asumsi mulai dari home
        dq_real = np.zeros(6)
        x_ee_pose = x_start_traj.copy()
        v_ee_act = np.zeros(3)

    # --- E. Inisialisasi Variabel Loop ---
    # Variabel Logging
    log_t, log_x_act, log_x_des, log_tau_task = [], [], [], []
    log_q_ddot, log_rhs, log_pa_ddot, log_q_dot, log_pa_dot = [], [], [], [], []
    log_q_act, log_q_des, log_dq_act, log_dq_des = [], [], [], []
    # Logging tambahan untuk torsi null space
    log_tau_total, log_tau_task_only, log_tau_null = [], [], []
    # Logging untuk kompensasi Coriolis dan gravitasi
    log_tau_coriolis_grav = []
    # Logging untuk perbandingan kecepatan
    log_v_tcp_speed = []  # Untuk menyimpan kecepatan TCP dari getTCPSpeed()
    log_v_tcp_calc = []   # Untuk menyimpan kecepatan TCP dari J@dq
    # Logging untuk perbandingan posisi joint
    log_q_pybullet = []   # Untuk menyimpan posisi joint dari PyBullet
    log_q_ur5e = []       # Untuk menyimpan posisi joint dari UR5e
    # Logging dynamic properties from OSCBF library
    log_mass_matrix = []         # Mass matrix
    log_mass_matrix_inv = []     # Inverse of mass matrix
    log_gravity = []             # Gravity vector
    log_coriolis = []            # Coriolis/centrifugal vector
    log_jacobian = []            # Jacobian matrix
    log_ee_pose = []             # End-effector pose

    # Inisialisasi variabel perintah. q_cmd diupdate secara iteratif
    q_cmd = q_real.copy() 

    # FIX: Inisialisasi state kontrol dari pembacaan awal
    q_state = q_real.copy()
    dq_state = dq_real.copy()
    x_actual = x_ee_pose.copy()
    v_actual = v_ee_act.copy()

    # Variabel Admittance (untuk fitur admittance)
    pa = x_ee_pose.copy() # PENTING: Inisialisasi pa dari posisi aktual
    pa_dot = np.zeros(3)

    # Define nullspace posture goal
    nullspace_posture_goal = jnp.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])

    print(f"\n🎮 Memulai Loop Kontrol (Mode: {args.fitur}) (Tekan Ctrl+C untuk berhenti)...")
    start_time = time.time()
    loop_start = start_time
    last_print = start_time

    try:
        while time.time() - start_time < RUNTIME:
            t = time.time() - start_time

            # 1. BACA STATE (Feedback)
            if args.mode == "real":
                q_real = np.array(rtde_r.getActualQ())
                dq_real = np.array(rtde_r.getActualQd())
                tcp_pose = rtde_r.getActualTCPPose()
                x_ee_pose = np.array(tcp_pose[:3])
                v_ee_act = np.array(rtde_r.getActualTCPSpeed()[:3])
            else:
                # Di mode simulasi, state tidak berubah kecuali kita update manual
                # Biarkan state sama dengan hasil akhir iterasi sebelumnya
                q_real = q_cmd.copy()
                dq_real = dq_cmd.copy() if 'dq_cmd' in locals() else np.zeros(6)
                
                # Update x_ee_pose from forward kinematics using OSCBF library
                ee_pos = robot.ee_position(q_real)
                ee_jac = robot.ee_jacobian(q_real)
                # FIX 2: Jacobian frame orientation - correct for frame difference (X-forward vs X-backward)
                ee_jac_fixed = ee_jac.copy()
                ee_jac_fixed[:3, :] = -ee_jac_fixed[:3, :]  # Invert first 3 rows (linear Jacobian)
                x_ee_pose = ee_pos
                v_ee_act = ee_jac_fixed @ dq_real
                
            # Update variabel state untuk kalkulasi
            q_state, dq_state = q_real.copy(), dq_real.copy()
            x_actual, v_actual = x_ee_pose.copy(), v_ee_act.copy()

            # 2. UPDATE VISUAL (PyBullet Shadowing)
            for i, idx in enumerate(sim_joint_indices):
                p.resetJointState(robot_sim, idx, q_real[i], dq_real[i])

            # 3. Generate Trajectory (Target)
            x_des, xd_des, xdd_des, z_ee_des = calculate_trajectory_point(t, x_base)

            # 4. HITUNG DINAMIKA (dari OSCBF)
            J_full = robot.ee_jacobian(q_state)      # 6x6 (first 3 rows = linear jacobian)
            J = J_full[:3, :]                        # 3x6 (linear jacobian only)
            
            # FIX 2: Jacobian frame orientation - correct for frame difference (X-forward vs X-backward)
            J[0, :] = -J[0, :]  # Invert X-axis (row 0)
            J[1, :] = -J[1, :]  # Invert Y-axis (row 1)
            
            Mq = robot.mass_matrix(q_state)

            try:
                Lambda_task = np.linalg.inv(J @ np.linalg.inv(Mq) @ J.T)
            except np.linalg.LinAlgError:
                Lambda_task = np.eye(3) * 1.0 # Fallback jika singular

            # Get dynamic properties using OSCBF library for logging
            M, M_inv, g, c, J_full, ee_tmat = robot.torque_control_matrices(q_state, dq_state)

            # 5. HITUNG KONTROL
            if args.speed == "calc":
                # Gunakan kecepatan dari Jacobian, lebih bersih dari noise sensor
                v_actual = J @ dq_state

            # Default nilai untuk logging
            q_ddot = np.zeros_like(q_state)
            rhs = np.zeros(3)
            pa_ddot = np.zeros(3)

            # Logging untuk perbandingan kecepatan
            log_v_tcp_speed.append(v_ee_act.copy())  # Kecepatan TCP aktual dari sensor
            log_v_tcp_calc.append(v_actual.copy())   # Kecepatan TCP yang digunakan dalam kontrol

            if args.fitur == "torque" or args.fitur == "safe_torque":
                # === MODE TORQUE (OSC) ===
                # Hitung kompensasi Coriolis dan gravitasi
                tau_coriolis_grav = robot.gravity_vector(q_state) + robot.centrifugal_coriolis_vector(q_state, dq_state)
                
                # Create state vector for OSCBF [q, qdot]
                z = np.concatenate([q_state, dq_state])
                
                # Compute control using OSCBF library
                tau_cmd = compute_control(robot, osc_controller, cbf, z, z_ee_des, nullspace_posture_goal)
                tau_cmd = np.asarray(tau_cmd)  # Convert from JAX array to NumPy array

                # FIX 1: Double gravity compensation - subtract g vector to prevent double compensation
                g = robot.gravity_vector(q_state)  # Get gravity vector
                c = robot.centrifugal_coriolis_vector(q_state, dq_state)  # Get coriolis vector
                q_ddot = np.linalg.inv(Mq) @ (tau_cmd - c - g)  # Subtract both c and g to prevent double compensation
                q_dot = dq_state + q_ddot * DT # Integrasi Euler
                dq_cmd = q_dot
            else: # === MODE ADMITTANCE ===
                # Hitung gaya dari PD error
                e_pos = x_des - x_actual
                e_vel = xd_des - v_actual
                F_cmd = Lambda_task @ (xdd_des + Kd_task[:3, :3] @ e_vel + Kp_task[:3, :3] @ e_pos)
                
                # Admittance Filter
                rhs = F_cmd - (Ba[:3, :3] @ pa_dot) - (Ka[:3, :3] @ (pa - x_des))
                pa_ddot = np.linalg.inv(Ma[:3, :3]) @ rhs
                
                pa_dot += pa_ddot * DT
                pa += pa_dot * DT
                v_cmd = pa_dot.copy()
                
                # Konversi v_cmd ke dq_cmd menggunakan IK
                J_pinv = np.linalg.pinv(J)
                dq_cmd = J_pinv @ v_cmd
                
                # Untuk logging, kita perlu menghitung tau_task dan tau_null secara terpisah
                # Hitung tau_task dari error PD
                e_x = x_des - x_actual
                e_v = xd_des - v_actual
                f_cmd = Kp_task[:3, :3] @ e_x + Kd_task[:3, :3] @ e_v
                tau_task = J.T @ f_cmd
                
                # Hitung null space torques
                Gamma_sing = np.zeros(6)  # Placeholder for singularity torque
                Gamma_obs  = np.zeros(6)  # Placeholder for obstacle torque
                # Use adaptive weights for nullspace based on speed mode
                if args.nullspace == "on":
                    if args.speed_mode == "slow":
                        w_sing, w_obs = 0.7, 0.7
                    elif args.speed_mode == "fast":
                        w_sing, w_obs = 0.3, 0.3
                    else:  # normal
                        w_sing, w_obs = 0.5, 0.5
                    Gamma0 = w_sing * Gamma_sing + w_obs * Gamma_obs
                    _, M_inv_local, _, _, _, _ = robot.torque_control_matrices(q_state, np.zeros_like(q_state))
                    # Calculate nullspace projection matrix: I - M^(-1) * J.T * (J * M^(-1) * J.T)^(-1) * J
                    A = J @ M_inv_local @ J.T
                    try:
                        A_inv = np.linalg.inv(A)
                    except np.linalg.LinAlgError:
                        A_inv = np.eye(A.shape[0])  # fallback
                    I = np.eye(robot.num_joints)
                    NT = I - M_inv_local @ J.T @ A_inv @ J  # Nullspace projection matrix
                    tau_null = NT @ Gamma0
                else:
                    # Tidak ada null space
                    tau_null = np.zeros(6)
                
                tau_coriolis_grav = robot.gravity_vector(q_state) + robot.centrifugal_coriolis_vector(q_state, dq_state)
                
                tau_total = tau_task + tau_null + tau_coriolis_grav
                tau_cmd = tau_total

            # 5. KIRIM PERINTAH (Action)
            if args.mode == "real":
                # For safety, limit the torque values
                max_tau = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
                tau_cmd = np.clip(tau_cmd, -max_tau, max_tau)
                
                if args.command == "speedJ":
                    # For admittance mode, we might want to send the velocity command directly
                    if args.fitur == "admittance":
                        # In admittance control, dq_cmd was computed from inverse kinematics directly from v_cmd
                        # So we can use dq_cmd directly
                        pass  # dq_cmd is already computed above
                    else:
                        # For torque mode, convert torque to joint velocities using inverse dynamics
                        # Use a more appropriate conversion based on the mass matrix
                        M_current, _, g_current, c_current, _, _ = robot.torque_control_matrices(q_state, dq_state)
                        # Calculate desired joint acceleration: M^{-1} * tau
                        # FIX 1: Double gravity compensation - subtract g vector to prevent double compensation
                        q_ddot_cmd = np.linalg.solve(M_current, tau_cmd - c_current - g_current)  # Include Coriolis & gravity compensation
                        # Integrate to get velocity: dq = dq + ddq * dt
                        dq_cmd = dq_state + q_ddot_cmd * DT
                    
                    # Limit the velocity
                    dq_cmd = np.clip(dq_cmd, -MAX_VEL_JOINT, MAX_VEL_JOINT)
                    rtde_c.speedJ(dq_cmd.tolist(), 0.5, DT)
                elif args.command == "moveL":
                    # For moveL, we send the desired Cartesian pose directly
                    # Get current pose to preserve orientation
                    current_pose = rtde_r.getActualTCPPose()
                    target_pose = current_pose.copy()
                    # Update only the position part with desired values while keeping orientation
                    target_pose[0] = x_des[0]  # X position
                    target_pose[1] = x_des[1]  # Y position
                    target_pose[2] = x_des[2]  # Z position
                    
                    # Calculate appropriate speed and acceleration based on trajectory
                    vel = np.linalg.norm(xd_des)
                    acc = np.linalg.norm(xdd_des)
                    
                    # Use moveL to go to desired Cartesian position with appropriate speed/acceleration
                    rtde_c.moveL(target_pose, min(vel*2, 0.5) if vel > 0.01 else 0.1, min(acc*2, 0.5) if acc > 0.1 else 0.1)
                elif args.command == "servoL":
                    # For servoL, we use the calculated Cartesian velocity directly
                    # Get current pose to preserve orientation
                    current_pose = rtde_r.getActualTCPPose()
                    
                    # Calculate desired incremental movement based on xd_des
                    target_pos = [
                        x_des[0],  # Use x_des directly for position control
                        x_des[1],
                        x_des[2]
                    ]
                    
                    # Create target pose (keeping orientation from current pose)
                    target_pose = current_pose.copy()
                    target_pose[0:3] = target_pos
                    
                    # Use servoL for linear Cartesian position control with velocity feedforward
                    rtde_c.servoL(
                        target_pose, 
                        xd_des.tolist() + [0, 0, 0],  # Velocity vector [x, y, z, rx, ry, rz] - set angular velocities to 0
                        0.0,  # Acceleration (using 0 means use default)
                        DT,   # Time to next control point
                        DT*2, # Lookahead time
                        300   # Gain
                    )
            else:
                # In simulation mode, we can directly apply torques
                # Update simulation with calculated torques
                max_force = 100  # Maximum force for simulation
                for i in range(len(sim_joint_indices)):
                    p.setJointMotorControl2(
                        robot_sim,
                        sim_joint_indices[i],
                        p.TORQUE_CONTROL,
                        force=tau_cmd[i] if i < len(tau_cmd) else 0
                    )

            # Update estimasi posisi dari perintah kecepatan (untuk logging & sim)
            q_cmd += dq_cmd * DT if 'dq_cmd' in locals() else np.zeros(6) * DT
            
            # Logging untuk perbandingan posisi joint
            log_q_ur5e.append(q_real.copy())  # Posisi joint dari UR5e/URSim
            # Ambil posisi joint dari PyBullet simulasi
            q_pybullet = []
            for idx in sim_joint_indices:
                joint_state = p.getJointState(robot_sim, idx)
                q_pybullet.append(joint_state[0])  # [0] adalah posisi, [1] adalah kecepatan
            log_q_pybullet.append(np.array(q_pybullet))
            
            # 8. LOGGING
            log_t.append(t)
            log_x_act.append(x_actual)
            log_x_des.append(x_des)
            log_tau_task.append(tau_cmd if len(tau_cmd) > 0 else np.zeros(6))
            log_q_ddot.append(np.zeros(6))  # Placeholder
            log_rhs.append(np.zeros(3))     # Placeholder
            log_pa_ddot.append(np.zeros(3)) # Placeholder
            log_q_dot.append(dq_cmd if 'dq_cmd' in locals() else np.zeros(6))
            log_pa_dot.append(np.zeros(3))  # Placeholder
            log_q_act.append(q_state)
            log_q_des.append(q_cmd)
            log_dq_act.append(dq_state)
            log_dq_des.append(dq_cmd if 'dq_cmd' in locals() else np.zeros(6))
            
            # Logging tambahan untuk torsi null space
            log_tau_coriolis_grav.append(c)  # Coriolis and gravity vector from robot dynamics
            log_tau_total.append(tau_cmd)
            log_tau_task_only.append(tau_cmd)  # Placeholder
            log_tau_null.append(np.zeros(6))   # Placeholder
            
            # Logging dynamic properties from OSCBF library
            log_mass_matrix.append(M)          # Mass matrix
            log_mass_matrix_inv.append(M_inv)  # Inverse of mass matrix
            log_gravity.append(g)              # Gravity vector
            log_coriolis.append(c)             # Coriolis/centrifugal vector
            log_jacobian.append(J)             # Jacobian matrix
            log_ee_pose.append(ee_tmat)        # End-effector pose

            if time.time() - last_print > 0.5:
                e_pos_print = x_des - x_actual
                print(f"[t={t:.2f}] ErrX: {e_pos_print[0]:.4f}", end="")
                if args.fitur == "safe_torque":
                    print(f" | Safety Active | Tau[0]: {tau_cmd[0]:.2f}")
                else:
                    print(f" | Tau[0]: {tau_cmd[0]:.2f}")
                last_print = time.time()

            # 9. SINKRONISASI WAKTU
            current_time = time.time()
            sleep_time = (loop_start + DT) - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            loop_start += DT

    except KeyboardInterrupt:
        print("\n🛑 Berhenti Manual...")
    except Exception as e:
        print(f"\n❌ TERJADI ERROR FATAL: {e}")
        import traceback
        traceback.print_exc() 
    
    finally:
        print("Menutup Koneksi...")
        if args.mode == "real" and rtde_c and rtde_c.isConnected():
            rtde_c.speedStop()
            rtde_c.disconnect()
        if args.mode == "real" and rtde_r and rtde_r.isConnected():
            rtde_r.disconnect()
        if p.isConnected():
            p.disconnect()

        # Plotting
        if len(log_t) > 0:
            print("Memproses Grafik...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"results_{timestamp}"
            os.makedirs(folder_name, exist_ok=True)
            
            log_t = np.array(log_t)
            act = np.array(log_x_act)
            des = np.array(log_x_des)
            tau_task = np.array(log_tau_task)
            q_ddot = np.array(log_q_ddot)
            rhs = np.array(log_rhs)
            pa_ddot = np.array(log_pa_ddot)
            q_dot = np.array(log_q_dot)
            pa_dot = np.array(log_pa_dot)
            q_act = np.array(log_q_act)
            q_des = np.array(log_q_des)
            dq_act = np.array(log_dq_act)
            dq_des = np.array(log_dq_des)
            # Dynamic properties
            log_mass_matrix = np.array(log_mass_matrix)
            log_mass_matrix_inv = np.array(log_mass_matrix_inv)
            log_gravity = np.array(log_gravity)
            log_coriolis = np.array(log_coriolis)
            log_jacobian = np.array(log_jacobian)
            log_ee_pose = np.array(log_ee_pose)

            if args.save_log:
                log_dict = {
                    "log_t": log_t, "log_x_act": act, "log_x_des": des,
                    "tau_task": tau_task, "q_ddot": q_ddot, "rhs": rhs,
                    "pa_ddot": pa_ddot, "q_dot": q_dot, "pa_dot": pa_dot,
                    "q_act": q_act, "q_des": q_des, "dq_act": dq_act, "dq_des": dq_des,
                    "tau_total": np.array(log_tau_total), "tau_task_only": np.array(log_tau_task_only), 
                    "tau_null": np.array(log_tau_null), "tau_coriolis_grav": np.array(log_tau_coriolis_grav),
                    "log_q_ur5e": np.array(log_q_ur5e), "log_q_pybullet": np.array(log_q_pybullet),
                    "log_v_tcp_speed": np.array(log_v_tcp_speed), "log_v_tcp_calc": np.array(log_v_tcp_calc),
                    # Dynamic properties from OSCBF library
                    "log_mass_matrix": np.array(log_mass_matrix), "log_mass_matrix_inv": np.array(log_mass_matrix_inv),
                    "log_gravity": np.array(log_gravity), "log_coriolis": np.array(log_coriolis),
                    "log_jacobian": np.array(log_jacobian), "log_ee_pose": np.array(log_ee_pose),
                    # Parameter konfigurasi
                    "TRAJECTORY": TRAJECTORY, "args_mode": args.mode, "args_speed": args.speed,
                    "args_fitur": args.fitur, "args_speed_mode": args.speed_mode, "args_nullspace": args.nullspace,
                    "args_command": args.command, "Kp_task": Kp_task, "Kd_task": Kd_task,
                    "A": A, "omega": omega,
                    "TILT_ANGLE": TILT_ANGLE, "C_TILT": C_TILT, "S_TILT": S_TILT
                }
                pkl_name = os.path.join(folder_name, 
                    f"HIL_log_{TRAJECTORY}_mode_{args.mode}_"
                    f"speed_{args.speed}_fitur_{args.fitur}_"
                    f"nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.pkl"
                )
                with open(pkl_name, "wb") as f:
                    pickle.dump(log_dict, f)
                print(f"💾 Log data disimpan ke pickle: {pkl_name}")

            axes_names = ['X', 'Y', 'Z']
            
            for i in range(3):
                plt.figure(figsize=(10, 6))
                plt.plot(log_t, des[:, i], 'r--', linewidth=2, label='Desired')
                plt.plot(log_t, act[:, i], 'b-', linewidth=1.5, label='Actual')
                error = des[:, i] - act[:, i]
                rmse = np.sqrt(np.mean(error**2))
                max_err = np.max(np.abs(error))
                stats_text = f"RMSE: {rmse:.6f} m\nMax Err: {max_err:.6f} m"
                plt.gca().text(0.96, 0.96, stats_text, transform=plt.gca().transAxes,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                plt.ylabel(f'Pos {axes_names[i]} (m)')
                plt.xlabel('Time (s)')
                plt.title(f'Tracking Axis {axes_names[i]} (Kp_pos={Kp_task[0,0]}, Kd_pos={Kd_task[0,0]} - {TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='upper left')
                filename = os.path.join(folder_name, f"HIL_Tracking_{axes_names[i]}_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"💾 Disimpan: {filename}")
                # plt.show()

            # Plot joint space tracking
            fig_js, axes_js = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
            axes_js = axes_js.flatten()
            for j in range(6):
                axes_js[j].plot(log_t, dq_act[:, j], label="dq_act", linestyle="-")
                axes_js[j].plot(log_t, dq_des[:, j], label="dq_des", linestyle="--")
                axes_js[j].plot(log_t, q_act[:, j],  label="q_act", linestyle="-.")
                axes_js[j].plot(log_t, q_des[:, j],  label="q_des", linestyle=":")
                axes_js[j].set_title(f"Joint {j+1}")
                axes_js[j].grid(True, linestyle='--', alpha=0.5)
                if j >= 4: axes_js[j].set_xlabel("Time (s)")
            axes_js[0].legend(loc="upper right")
            fig_js.suptitle(f"Joint Space Tracking - {TRAJECTORY}, Mode: {args.mode}, Speed: {args.speed}, Fitur {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}", fontsize=14)
            fig_js.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename_js = os.path.join(folder_name, f"HIL_JointSpace_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png")
            plt.savefig(filename_js, dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {filename_js}")
            # plt.show(fig_js)

            # 3D trajectory plot
            fig_3d = plt.figure(figsize=(10, 8))
            ax = fig_3d.add_subplot(111, projection='3d')
            ax.plot(des[:,0], des[:,1], des[:,2], 'r--', label='Desired Path')
            ax.plot(act[:,0], act[:,1], act[:,2], 'b-', label='Actual Path')
            ax.scatter(act[0,0], act[0,1], act[0,2], c='green', s=100, label='Start')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title(f'3D Trajectory - {TRAJECTORY}, Mode: {args.mode}, Speed: {args.speed}, Fitur {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}')
            ax.legend()
            try:
                max_range = np.array([act[:,i].max() - act[:,i].min() for i in range(3)]).max() / 2.0
                mid = np.mean(act, axis=0)
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
            except: pass # Fails if log is empty
            ax.set_aspect('auto', adjustable='box') # 'equal' not supported in all matplotlib versions
            filename_3d = os.path.join(folder_name, f"HIL_3D_Path_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png")
            plt.savefig(filename_3d, dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {filename_3d}")
            
            # Konversi ke numpy array untuk plotting
            tau_total = np.array(log_tau_total)
            tau_task_only = np.array(log_tau_task_only)
            tau_null = np.array(log_tau_null)
            
            # Plot torsi dengan dan tanpa null space
            fig_tau, axes_tau = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
            axes_tau = axes_tau.flatten()
            for j in range(6):
                axes_tau[j].plot(log_t, tau_task_only[:, j], label="Tau Task (tanpa Null)", linestyle="-")
                axes_tau[j].plot(log_t, tau_null[:, j], label="Tau Null Space", linestyle="--")
                axes_tau[j].plot(log_t, tau_total[:, j], label="Tau Total (dengan Null)", linestyle="-.")
                axes_tau[j].set_title(f"Joint {j+1} - Torsi")
                axes_tau[j].grid(True, linestyle='--', alpha=0.5)
                if j >= 4: axes_tau[j].set_xlabel("Time (s)")
            axes_tau[0].legend(loc="upper right")
            fig_tau.suptitle(f"Joint Torques Comparison - {TRAJECTORY}, Mode: {args.mode}, Speed: {args.speed}, Fitur: {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}", fontsize=14)
            fig_tau.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename_tau = os.path.join(folder_name, f"HIL_Torques_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png")
            plt.savefig(filename_tau, dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {filename_tau}")
            
            # Plot kompensasi Coriolis dan gravitasi
            fig_cg, axes_cg = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
            axes_cg = axes_cg.flatten()
            for j in range(6):
                axes_cg[j].plot(log_t, np.array(log_tau_coriolis_grav)[:, j], label="Tau Coriolis+Grav", linestyle="-")
                axes_cg[j].set_title(f"Joint {j+1} - Coriolis+Grav Comp")
                axes_cg[j].grid(True, linestyle='--', alpha=0.5)
                if j >= 4: axes_cg[j].set_xlabel("Time (s)")
            axes_cg[0].legend(loc="upper right")
            fig_cg.suptitle(f"Coriolis and Gravity Compensation - {TRAJECTORY}, Mode: {args.mode}, Speed: {args.speed}, Fitur: {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}", fontsize=14)
            fig_cg.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename_cg = os.path.join(folder_name, f"HIL_CoriolisGrav_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png")
            plt.savefig(filename_cg, dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {filename_cg}")
            
            # Plot dynamic properties from OSCBF library
            if len(log_mass_matrix) > 0:
                # Plot diagonal elements of mass matrix
                fig_m, axes_m = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
                axes_m = axes_m.flatten()
                for j in range(6):
                    # Extract diagonal elements of mass matrix over time
                    m_diag = [M[j, j] for M in log_mass_matrix]
                    axes_m[j].plot(log_t, m_diag, label=f"Mass M_{j+1},{j+1}", linestyle="-")
                    axes_m[j].set_title(f"Joint {j+1} - Diagonal Mass M_{j+1},{j+1}")
                    axes_m[j].grid(True, linestyle='--', alpha=0.5)
                    if j >= 4: axes_m[j].set_xlabel("Time (s)")
                axes_m[0].legend(loc="upper right")
                fig_m.suptitle(f"Diagonal Elements of Mass Matrix - {TRAJECTORY}, Mode: {args.mode}, Speed: {args.speed}, Fitur: {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}", fontsize=14)
                fig_m.tight_layout(rect=[0, 0.03, 1, 0.95])
                filename_m = os.path.join(folder_name, f"HIL_MassMatrix_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png")
                plt.savefig(filename_m, dpi=300, bbox_inches='tight')
                print(f"💾 Disimpan: {filename_m}")
            
            print("Menampilkan Grafik...")
            plt.show()
        
        print("Selesai.")

if __name__ == "__main__":
    main()