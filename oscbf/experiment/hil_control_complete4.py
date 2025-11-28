import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
import pinocchio as pin
import rtde_control
import rtde_receive
import pickle

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
# *   `pa`, `pa_dot`, `pa_ddot`: Variabel untuk *Admittance Control*. Ini merepresentasikan state dari
#     massa virtual. Gaya (`F_cmd`) yang dihitung akan menggerakkan massa virtual ini, dan hasilnya 
#     (posisi `pa`, kecepatan `pa_dot`) menjadi referensi kecepatan baru untuk robot.
#
# ### Variabel Perintah (ke Robot) ###
# *   `dq_cmd`: Perintah kecepatan sendi *yang dikirim* ke robot (`speedJ`).
# *   `q_cmd`: Estimasi posisi sendi hasil dari integrasi `dq_cmd`. Digunakan hanya untuk logging dan 
#     perbandingan, bukan untuk kontrol langsung.
#
# ====================================================================
# 1. KONFIGURASI SISTEM
# ====================================================================
parser = argparse.ArgumentParser(description="UR5e HIL Control: Strategy 2 (Calculator)")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP Robot (127.0.0.1 untuk URSim)")
parser.add_argument("--save_log", action="store_true", help="Simpan log ke file .pkl")
parser.add_argument("--mode", choices=["sim", "real"], default="real", help="Mode operasi: sim (PyBullet Only) atau real (URSim/Fisik)")
parser.add_argument("--traj", choices=["linear_x", "linear_y", "circle", "circle_3d", "point"], 
                    default="linear_x", help="Pilih Trajektori")
parser.add_argument("--speed", choices=["act", "calc"], default="calc", help="act = asal kecepatan aktual dari TCPSpeed (v_ee_act), calc= asal kecepatan dari getActualQd (dq_real)")
parser.add_argument("--fitur", choices=["torque", "admittance"], default="admittance", help="Fitur kontrol yang diaktifkan")
parser.add_argument("--speed_mode", choices=["slow", "normal", "fast"], default="normal", 
                    help="Kecepatan gerakan: slow (0.05Hz), normal (0.2Hz), fast (0.5Hz)")
parser.add_argument("--nullspace", choices=["on", "off"], default="on", help="Aktif/non-aktif null space (default: on)")
parser.add_argument("--command", choices=["speedJ", "moveL", "servoL"], default="speedJ", help="Jenis perintah kontrol robot: speedJ (kecepatan joint), moveL (posisi linier), servoL (kecepatan linier) - default speedJ")
args = parser.parse_args()

# --- Konfigurasi Path & Link ---
# Pastikan file ur5e.urdf berada di folder yang sama atau sesuaikan path
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
if args.fitur == "torque":
    if args.speed_mode == "slow":
        Kp_task = np.diag([100.0, 100.0, 100.0]) # Proportional (Pegas ke target) - lower for slow
        Kd_task = np.diag([20.0, 20.0, 20.0])    # Derivative (Peredam error) - lower for slow
    elif args.speed_mode == "normal":
        Kp_task = np.diag([150.0, 150.0, 150.0]) # Proportional (Pegas ke target)
        Kd_task = np.diag([25.0, 25.0, 25.0])    # Derivative (Peredam error)
    else:  # fast
        Kp_task = np.diag([200.0, 200.0, 200.0]) # Proportional (Pegas ke target) - higher for fast
        Kd_task = np.diag([30.0, 30.0, 30.0])    # Derivative (Peredam error) - higher for fast
else:  # admittance mode - adjust PD gains to work with admittance filter
    if args.speed_mode == "slow":
        Kp_task = np.diag([80.0, 80.0, 80.0])  # Lower gains for admittance mode in slow
        Kd_task = np.diag([15.0, 15.0, 15.0])  # Lower derivative gains
    elif args.speed_mode == "normal":
        Kp_task = np.diag([120.0, 120.0, 120.0]) # Moderate gains for admittance mode
        Kd_task = np.diag([20.0, 20.0, 20.0])    # Moderate derivative gains
    else:  # fast
        Kp_task = np.diag([160.0, 160.0, 160.0]) # Higher gains but not too high for admittance mode
        Kd_task = np.diag([25.0, 25.0, 25.0])    # Balanced derivative gains

# Null Space (Postur) - ADAPTIVE BASED ON NULLSPACE SETTING
# Gain kecil agar tidak melawan task utama
if args.nullspace == "on":
    Kp_null = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5 
else:
    Kp_null = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.0  # Zero when nullspace is off
q_home = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

# ====================================================================
# 2. DATA DINAMIKA (YAML INJECTION)
# ====================================================================
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

# ====================================================================
# 3. KELAS KALKULATOR (PINOCCHIO)
# ====================================================================
class PinocchioDynamicsCalculator:
    def __init__(self, urdf_path, ee_link_name):
        print("🔧 Menginisialisasi Pinocchio...")
        self.model = pin.buildModelFromUrdf(urdf_path)
        
        # Inject Data Dinamika YAML
        for i in range(self.model.nv):
            # joint_id 1 corresponds to the first moving link
            self.model.inertias[i+1] = pin.Inertia(MASSES[i], COMS[i], INERTIA_TENSORS[i])
            
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_link_name)
        print(f"✅ Pinocchio Siap. Target Frame: {ee_link_name} (ID: {self.ee_frame_id})")

    def get_jacobian(self, q):
        """
        Menghitung Jacobian dan membaliknya agar sesuai dengan frame RTDE
        RTDE X+ adalah Belakang, Pinocchio X+ adalah Depan.
        """
        pin.forwardKinematics(self.model, self.data, q)
        J_world = pin.computeFrameJacobian(
            self.model, 
            self.data, 
            q, 
            self.ee_frame_id, 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        J_base = J_world[:3, :].copy()
        J_base[0, :] = -J_base[0, :] # Flip X
        J_base[1, :] = -J_base[1, :] # Flip Y
        return J_base

    def get_mass_matrix(self, q):
        pin.crba(self.model, self.data, q)
        return self.data.M.copy()
    
    def get_nullspace(self, q, J):
        M = self.get_mass_matrix(q)
        M_inv = np.linalg.inv(M)
        try:
            Lambda = np.linalg.inv(J @ M_inv @ J.T)
        except: 
            A = J @ M_inv @ J.T
            Lambda = np.linalg.pinv(A)
        # J_bar = M^-1 * J.T * (J * M^-1 * J.T)^-1
        J_bar = M_inv @ J.T @ Lambda
        eye = np.eye(self.model.nv) # Identity Matrix
        NT = eye - (J.T @ J_bar.T)
        return NT, M_inv
    
    def get_coriolis_gravity(self, q, dq):
        """
        Menghitung vektor Coriolis dan gravitasi
        """
        # Set gravity
        g = np.array([0, 0, -9.81])
        # Set acceleration to zero for gravity/Coriolis compensation
        ddq = np.zeros_like(dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        # RNEA: Recursive Newton-Euler Algorithm
        # Arguments: model, data, q (position), dq (velocity), ddq (acceleration)
        pin.rnea(self.model, self.data, q, dq, ddq)
        # Set gravity to get gravity term
        pin.computeGeneralizedGravity(self.model, self.data, q)
        # The gravity term is already included in rnea result, but we can also add it explicitly
        # tau = C(q,dq)*dq + g(q), where C is Coriolis matrix and g is gravity vector
        # The rnea function already computes both terms together
        return self.data.tau.copy()  # tau berisi [C(q,dq)*dq + g(q)]

def manipulability(J):
    """
    Menghitung manipulability scalar μ(q) = sqrt(det(J J^T))
    J: Jacobian translasi 3x6
    """
    A = J @ J.T          # 3x3
    detA = np.linalg.det(A)
    # clamp supaya gak negatif karena error numerik
    detA = max(detA, 1e-12)
    return np.sqrt(detA)

def compute_singularity_torque(q, dyn, k_sing=1.0, eps=1e-4):
    """
    Bangun torsi Null Space untuk menghindari singularity:
    Gamma_sing = k_sing * ∂μ/∂q

    q   : joint posisi (6,)
    dyn : instance PinocchioDynamicsCalculator (punya get_jacobian)
    """
    n = q.shape[0]
    grad_mu = np.zeros(n)

    # finite difference dua sisi
    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = eps

        Jp = dyn.get_jacobian(q + dq)
        Jm = dyn.get_jacobian(q - dq)

        mu_p = manipulability(Jp)
        mu_m = manipulability(Jm)

        grad_mu[i] = (mu_p - mu_m) / (2.0 * eps)

    Gamma_sing = k_sing * grad_mu  # shape (6,)
    return Gamma_sing
    
class SimpleObstacle:
    def __init__(self, center, safe_dist, k_obs):
        self.center = np.asarray(center).reshape(3,)
        self.safe_dist = safe_dist
        self.k_obs = k_obs

    def get_point_and_jacobian(self, q, dyn):
        """
        Contoh paling sederhana:
        - p  : posisi end-effector (ambil dari Pinocchio)
        - Jp : Jacobian translasi end-effector 3x6
        """
        # pakai forwardKinematics + frame placement Pinocchio
        pin.forwardKinematics(dyn.model, dyn.data, q)
        pin.updateFramePlacements(dyn.model, dyn.data)
        oMf = dyn.data.oMf[dyn.ee_frame_id]    # SE3
        p = oMf.translation                    # np.array(3,)

        Jp = dyn.get_jacobian(q)               # 3x6 translasi
        return p, Jp
    
def compute_obstacle_torque(q, dyn, obstacles):
    """
    Bangun torsi Null Space untuk obstacle avoidance.
    Penjumlahan semua obstacle.
    Hasil: Gamma_obs shape (6,)
    """
    Gamma_obs = np.zeros_like(q)  # (6,)

    for obs in obstacles:
        p, Jp = obs.get_point_and_jacobian(q, dyn)  # p:3, Jp:3x6
        diff = p - obs.center
        d = np.linalg.norm(diff) + 1e-12

        if d >= obs.safe_dist:
            # di luar jarak aman, tidak ada gaya
            continue

        # Potential U(d) = 0.5 * k_obs * (1/d - 1/d_safe)^2
        # dU/dd = k_obs * (1/d - 1/d_safe) * (-1/d^2)
        term = (1.0 / d - 1.0 / obs.safe_dist)
        dU_dd = obs.k_obs * term * (-1.0 / (d**2))

        # dd/dq = (diff^T / d) * Jp
        dd_dq = (diff / d) @ Jp      # shape (6,)

        # gradU = dU/dd * dd/dq
        gradU = dU_dd * dd_dq        # ∂U/∂q

        # tau = -∂U/∂q
        Gamma_obs -= gradU

    return Gamma_obs

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

    return x_d, xd_d, xdd_d

def compute_tau_total_with_nullspace(
    q,
    dq,
    x_act,
    v_act,
    x_des,
    v_des,
    dyn,
    obstacles,
    Kp_task,
    Kd_task,
    w_sing=1.0,
    w_obs=1.0,
):
    """
    Bangun tau_task dari OSC translasi,
    tambahkan Null Space torque (sing + obstacle),
    dan gabungkan jadi tau_total.

    q, dq    : joint posisi & kecepatan (6,)
    x_act    : posisi EE aktual (3,)
    v_act    : kecepatan EE aktual (3,)
    x_des    : posisi EE desired (3,) -> dari traj / admittance (pa)
    v_des    : kecepatan EE desired (3,) -> dari traj / admittance (pa_dot)
    dyn      : PinocchioDynamicsCalculator
    obstacles: list of obstacle object (lihat SimpleObstacle)
    Kp_task  : 3x3
    Kd_task  : 3x3
    """

    # 1) Jacobian & Null Space projector
    J = dyn.get_jacobian(q)          # 3x6
    NT, M_inv = dyn.get_nullspace(q, J)  # 6x6, 6x6

    # 2) Task-space PD (ini bisa sudah termasuk admittance/mass-comp
    e_x = x_des - x_act
    e_v = v_des - v_act

    f_cmd = Kp_task @ e_x + Kd_task @ e_v   # R^3
    tau_task = J.T @ f_cmd                  # R^6

    # 3) Null Space torques
    Gamma_sing = compute_singularity_torque(q, dyn, k_sing=w_sing)
    Gamma_obs  = compute_obstacle_torque(q, dyn, obstacles)

    Gamma0 = w_sing * Gamma_sing + w_obs * Gamma_obs    # (6,)

    # 4) Proyeksikan ke Null Space (level torsi)
    tau_null = NT @ Gamma0   # (6,)

    # 5) Gabungkan
    tau_total = tau_task + tau_null

    return tau_total, tau_task, tau_null


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
    dyn_calc = None

    # --- B. Setup Koneksi Robot (Real/URSim) ---
    print(f"\n🔌 Menghubungkan ke Robot ({ROBOT_IP})...")
    if args.mode == "real":
        try:
            rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
            rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
            dyn_calc = PinocchioDynamicsCalculator(URDF_PATH, EE_LINK_PINOCCHIO)
            print("✅ Koneksi Berhasil!")
        except Exception as e:
            print(f"❌ Gagal Konek: {e}")
            return
    else: # sim mode
        # Jika simulasi, buat instance kalkulator saja tanpa koneksi robot
        dyn_calc = PinocchioDynamicsCalculator(URDF_PATH, EE_LINK_PINOCCHIO)


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
    x_start_traj, _, _ = calculate_trajectory_point(0.0, x_base)
    
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
    # Variabel Admittance
    pa = x_ee_pose.copy() # PENTING: Inisialisasi pa dari posisi aktual
    pa_dot = np.zeros(3)
    
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

    # Inisialisasi variabel perintah. q_cmd diupdate secara iteratif
    q_cmd = q_real.copy() 

    # FIX: Inisialisasi state kontrol dari pembacaan awal
    q_state = q_real.copy()
    dq_state = dq_real.copy()
    x_actual = x_ee_pose.copy()
    v_actual = v_ee_act.copy()

    # Obstacle
    # obstacles = [
    #     SimpleObstacle(center=[0.5, 0.0, 0.3], safe_dist=0.15, k_obs=10.0),
    # ]
    obstacles = []

    print("\n🎮 Memulai Loop Kontrol (Tekan Ctrl+C untuk berhenti)...")
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
                
                # Update x_ee_pose dari forward kinematics simulasi
                pin.forwardKinematics(dyn_calc.model, dyn_calc.data, q_real)
                pin.updateFramePlacements(dyn_calc.model, dyn_calc.data)
                x_ee_pose = dyn_calc.data.oMf[dyn_calc.ee_frame_id].translation
                v_ee_act = dyn_calc.get_jacobian(q_real) @ dq_real
                
            # Update variabel state untuk kalkulasi
            q_state, dq_state = q_real.copy(), dq_real.copy()
            x_actual, v_actual = x_ee_pose.copy(), v_ee_act.copy()

            # 2. UPDATE VISUAL (PyBullet Shadowing)
            for i, idx in enumerate(sim_joint_indices):
                p.resetJointState(robot_sim, idx, q_real[i], dq_real[i])

            # 3. Generate Trajectory (Target)
            x_des, xd_des, xdd_des = calculate_trajectory_point(t, x_base)

            # 4. HITUNG DINAMIKA (Pinocchio)
            J = dyn_calc.get_jacobian(q_state)
            Mq = dyn_calc.get_mass_matrix(q_state)

            try:
                Lambda = np.linalg.inv(J @ np.linalg.inv(Mq) @ J.T)
            except np.linalg.LinAlgError:
                Lambda = np.eye(3) * 1.0 # Fallback jika singular

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

            if args.fitur == "torque":
                # === MODE TORQUE (OSC) ===
                # Hitung kompensasi Coriolis dan gravitasi
                tau_coriolis_grav = dyn_calc.get_coriolis_gravity(q_state, dq_state)
                
                if args.nullspace == "on":
                    # Gunakan fungsi dengan null space
                    _, tau_task, tau_null = compute_tau_total_with_nullspace(
                        q=q_state, dq=dq_state,
                        x_act=x_actual, v_act=v_actual,
                        x_des=x_des, v_des=xd_des,
                        dyn=dyn_calc, obstacles=obstacles,
                        Kp_task=Kp_task, Kd_task=Kd_task,
                    )
                    # Use adaptive weights for nullspace based on speed mode
                    if args.speed_mode == "slow":
                        tau_null = tau_null * 0.7  # Reduce nullspace effect for slow
                    elif args.speed_mode == "fast":
                        tau_null = tau_null * 0.3  # Further reduce nullspace effect for fast
                    tau_total = tau_task + tau_null + tau_coriolis_grav
                else:
                    # Hanya gunakan task space, tanpa null space
                    J = dyn_calc.get_jacobian(q_state)          # 3x6
                    e_x = x_des - x_actual
                    e_v = xd_des - v_actual
                    f_cmd = Kp_task @ e_x + Kd_task @ e_v   # R^3
                    tau_task = J.T @ f_cmd                  # R^6
                    tau_null = np.zeros(6)  # Tidak ada null space
                    tau_total = tau_task + tau_coriolis_grav
                    
                q_ddot = np.linalg.inv(Mq) @ tau_total
                q_dot = dq_state + q_ddot * DT # Integrasi Euler
                dq_cmd = q_dot
            else: # === MODE ADMITTANCE ===
                # Hitung gaya dari PD error
                e_pos = x_des - x_actual
                e_vel = xd_des - v_actual
                F_cmd = Lambda @ (xdd_des + Kd_task @ e_vel + Kp_task @ e_pos)
                
                # Admittance Filter
                rhs = F_cmd - (Ba @ pa_dot) - (Ka @ (pa - x_des))
                pa_ddot = np.linalg.inv(Ma) @ rhs
                
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
                f_cmd = Kp_task @ e_x + Kd_task @ e_v
                tau_task = J.T @ f_cmd
                
                if args.nullspace == "on":
                    # Hitung null space torques
                    Gamma_sing = compute_singularity_torque(q_state, dyn_calc, k_sing=1.0)
                    Gamma_obs  = compute_obstacle_torque(q_state, dyn_calc, obstacles)
                    Gamma0 = 1.0 * Gamma_sing + 1.0 * Gamma_obs
                    try:
                        Lambda_null = np.linalg.inv(J @ np.linalg.inv(Mq) @ J.T)
                    except np.linalg.LinAlgError:
                        Lambda_null = np.eye(3) * 1.0
                    # J_bar = M^-1 * J.T * Lambda
                    J_bar = np.linalg.inv(Mq) @ J.T @ Lambda_null
                    # Null space projector: I - J.T * J_bar.T
                    NT = np.eye(6) - J.T @ J_bar.T
                    tau_null = NT @ Gamma0
                else:
                    # Tidak ada null space
                    tau_null = np.zeros(6)
                
                tau_coriolis_grav = dyn_calc.get_coriolis_gravity(q_state, dq_state)
                
                tau_total = tau_task + tau_null + tau_coriolis_grav

            # Safety Clamp - ADAPTIVE BASED ON SPEED MODE
            if args.speed_mode == "slow":
                MAX_VEL_JOINT_ADAPTIVE = MAX_VEL_JOINT * 0.7  # Reduce max velocity for slow mode
            elif args.speed_mode == "fast":
                MAX_VEL_JOINT_ADAPTIVE = MAX_VEL_JOINT * 1.2  # Slightly increase for fast mode (but stay within safety)
            else:
                MAX_VEL_JOINT_ADAPTIVE = MAX_VEL_JOINT  # Normal mode
                
            dq_cmd = np.clip(dq_cmd, -MAX_VEL_JOINT_ADAPTIVE, MAX_VEL_JOINT_ADAPTIVE)

            # 7. KIRIM PERINTAH (Action)
            if args.mode == "real":
                if args.command == "speedJ":
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

                    speed = min(vel*2, 0.5) if vel > 0.01 else 0.1
                    acc = min(acc*2, 0.5) if acc > 0.1 else 0.1
                    
                    # Use moveL to go to desired Cartesian position with appropriate speed/acceleration
                    rtde_c.moveL(target_pose, speed, acc, asynchronous=True)
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

                    vel_scalar = np.linalg.norm(xd_des)
                    acc_scalar = np.linalg.norm(xdd_des)
                    
                    # Use servoL for linear Cartesian position control with velocity feedforward
                    rtde_c.servoL(
                        target_pose, 
                        vel_scalar,  # Velocity vector [x, y, z, rx, ry, rz] - set angular velocities to 0
                        acc_scalar,  # Acceleration (using 0 means use default)
                        DT,   # Time to next control point
                        0.1, # Lookahead time
                        300   # Gain
                    )

            # Update estimasi posisi dari perintah kecepatan (untuk logging & sim)
            q_cmd += dq_cmd * DT
            
            # Logging untuk perbandingan posisi joint
            log_q_ur5e.append(q_real.copy())  # Posisi joint dari UR5e/URSim
            # Ambil posisi joint dari PyBullet simulasi
            q_pybullet = []
            for idx in sim_joint_indices:
                joint_state = p.getJointState(robot_sim, idx)
                q_pybullet.append(joint_state[0])  # [0] adalah posisi, [1] adalah kecepatan
            log_q_pybullet.append(np.array(q_pybullet))
            
            # Logging untuk perbandingan command - tambahkan command ke dalam log
            if 'command_log' not in locals():
                command_log = []
            command_log.append(args.command)
            
            # 8. LOGGING
            log_t.append(t)
            log_x_act.append(x_actual)
            log_x_des.append(x_des)
            log_tau_task.append(tau_task if 'tau_task' in locals() else np.zeros(6))
            log_q_ddot.append(q_ddot)
            log_rhs.append(rhs)
            log_pa_ddot.append(pa_ddot)
            log_q_dot.append(dq_cmd) # q_dot adalah dq_cmd
            log_pa_dot.append(pa_dot.copy())
            log_q_act.append(q_state)
            log_q_des.append(q_cmd)
            log_dq_act.append(dq_state)
            log_dq_des.append(dq_cmd)
            
            # Logging tambahan untuk torsi null space
            if args.fitur == "torque":
                # Hitung kompensasi Coriolis dan gravitasi
                tau_coriolis_grav = dyn_calc.get_coriolis_gravity(q_state, dq_state)
                log_tau_coriolis_grav.append(tau_coriolis_grav)
                
                # Dalam mode torque, kita sudah memiliki tau_task dan tau_null
                log_tau_total.append(tau_total)
                log_tau_task_only.append(tau_task)  # Torsi tanpa null space
                log_tau_null.append(tau_null)       # Torsi dari null space saja
            else: # mode admittance
                # Dalam mode admittance, kita perlu menghitung tau_task dan tau_null secara terpisah
                # Hitung tau_task dari error PD
                e_x = x_des - x_actual
                e_v = xd_des - v_actual
                f_cmd = Kp_task @ e_x + Kd_task @ e_v
                tau_task = J.T @ f_cmd
                
                if args.nullspace == "on":
                    # Hitung null space torques
                    Gamma_sing = compute_singularity_torque(q_state, dyn_calc, k_sing=1.0)
                    Gamma_obs  = compute_obstacle_torque(q_state, dyn_calc, obstacles)
                    Gamma0 = 1.0 * Gamma_sing + 1.0 * Gamma_obs
                    _, M_inv = dyn_calc.get_nullspace(q_state, J)
                    NT = np.eye(6) - (J.T @ np.linalg.pinv(J @ M_inv @ J.T) @ J @ M_inv)  # Null space projector
                    tau_null = NT @ Gamma0
                else:
                    # Tidak ada null space
                    tau_null = np.zeros(6)
                
                tau_total = tau_task + tau_null
                # Hitung dan log kompensasi Coriolis dan gravitasi
                tau_coriolis_grav = dyn_calc.get_coriolis_gravity(q_state, dq_state)
                log_tau_coriolis_grav.append(tau_coriolis_grav)
                
                log_tau_total.append(tau_total)
                log_tau_task_only.append(tau_task)  # Torsi tanpa null space
                log_tau_null.append(tau_null)       # Torsi dari null space saja

            if time.time() - last_print > 0.5:
                e_pos_print = x_des - x_actual
                print(f"[t={t:.2f}] ErrX: {e_pos_print[0]:.4f}", end="")
                if args.fitur == "admittance":
                    print(f" | vCmdX: {v_cmd[0]:.4f} | PaDotX: {pa_dot[0]:.4f}")
                else:
                    print(f" | TskTrq0: {tau_task[0]:.2f} | qDotCmd0: {dq_cmd[0]:.4f}")
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
            from datetime import datetime
            import os
            
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
                    "command_log": command_log if 'command_log' in locals() else [],
                    # Parameter konfigurasi
                    "TRAJECTORY": TRAJECTORY, "args_mode": args.mode, "args_speed": args.speed,
                    "args_fitur": args.fitur, "args_speed_mode": args.speed_mode, "args_nullspace": args.nullspace,
                    "args_command": args.command, "Kp_task": Kp_task, "Kd_task": Kd_task,
                    "Ma": Ma, "Ba": Ba, "Ka": Ka, "A": A, "omega": omega,
                    "TILT_ANGLE": TILT_ANGLE, "C_TILT": C_TILT, "S_TILT": S_TILT
                }
                pkl_name = (
                    f"HIL_log_{TRAJECTORY}_mode_{args.mode}_"
                    f"speed_{args.speed}_fitur_{args.fitur}_"
                    f"nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.pkl"
                )
                with open(os.path.join(folder_name, pkl_name), "wb") as f:
                    pickle.dump(log_dict, f)
                print(f"💾 Log data disimpan ke pickle: {os.path.join(folder_name, pkl_name)}")

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
                plt.title(f'Tracking Axis {axes_names[i]} (Kp={Kp_task[0,0]}, Kd={Kd_task[0,0]} - {TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='upper left')
                filename = f"HIL_Tracking_{axes_names[i]}_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png"
                plt.savefig(os.path.join(folder_name, filename), dpi=300, bbox_inches='tight')
                print(f"💾 Disimpan: {os.path.join(folder_name, filename)}")
                # plt.show()

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
            filename_js = f"HIL_JointSpace_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png"
            plt.savefig(os.path.join(folder_name, filename_js), dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {os.path.join(folder_name, filename_js)}")
            # plt.show(fig_js)

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
            filename_3d = f"HIL_3D_Path_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png"
            plt.savefig(os.path.join(folder_name, filename_3d), dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {os.path.join(folder_name, filename_3d)}")
            
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
            fig_tau.suptitle(f"Joint Torques Comparison - {TRAJECTORY}, Mode: {args.mode}, Speed: {args.speed}, Fitur {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}", fontsize=14)
            fig_tau.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename_tau = f"HIL_Torques_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png"
            plt.savefig(os.path.join(folder_name, filename_tau), dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {os.path.join(folder_name, filename_tau)}")
            
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
            filename_cg = f"HIL_CoriolisGrav_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png"
            plt.savefig(os.path.join(folder_name, filename_cg), dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {os.path.join(folder_name, filename_cg)}")
            
            # Plot perbandingan kecepatan - dibuat dalam 1 jendela dengan 4 subplot
            fig_compare, axes_compare = plt.subplots(2, 2, figsize=(15, 12))
            fig_compare.suptitle(f'Perbandingan Kecepatan - Mode: {args.mode}, Speed: {args.speed}, Fitur: {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}', fontsize=16)

            # 1. Perbandingan kecepatan TCP dari TCPSpeed vs kecepatan dari getActualQd (melalui Jacobian)
            axes_compare[0,0].plot(log_t, np.array(log_v_tcp_speed)[:, 0], label="TCPSpeed X", linestyle="-", color='blue')
            axes_compare[0,0].plot(log_t, np.array(log_v_tcp_calc)[:, 0], label="J@dq X", linestyle="--", color='red')
            axes_compare[0,0].set_title("Kecepatan TCP - Arah X")
            axes_compare[0,0].set_xlabel("Time (s)")
            axes_compare[0,0].set_ylabel("Kecepatan (m/s)")
            axes_compare[0,0].grid(True, linestyle='--', alpha=0.6)
            axes_compare[0,0].legend()

            axes_compare[0,1].plot(log_t, np.array(log_v_tcp_speed)[:, 1], label="TCPSpeed Y", linestyle="-", color='blue')
            axes_compare[0,1].plot(log_t, np.array(log_v_tcp_calc)[:, 1], label="J@dq Y", linestyle="--", color='red')
            axes_compare[0,1].set_title("Kecepatan TCP - Arah Y")
            axes_compare[0,1].set_xlabel("Time (s)")
            axes_compare[0,1].set_ylabel("Kecepatan (m/s)")
            axes_compare[0,1].grid(True, linestyle='--', alpha=0.6)
            axes_compare[0,1].legend()

            axes_compare[1,0].plot(log_t, np.array(log_v_tcp_speed)[:, 2], label="TCPSpeed Z", linestyle="-", color='blue')
            axes_compare[1,0].plot(log_t, np.array(log_v_tcp_calc)[:, 2], label="J@dq Z", linestyle="--", color='red')
            axes_compare[1,0].set_title("Kecepatan TCP - Arah Z")
            axes_compare[1,0].set_xlabel("Time (s)")
            axes_compare[1,0].set_ylabel("Kecepatan (m/s)")
            axes_compare[1,0].grid(True, linestyle='--', alpha=0.6)
            axes_compare[1,0].legend()

            # 2. Perbandingan kecepatan sendi aktual
            for j in range(6):
                if j < 3:
                    axes_compare[1,1].plot(log_t, np.array(log_dq_act)[:, j], label=f"Joint {j+1} Act", alpha=0.6)
                else:
                    axes_compare[1,1].plot(log_t, np.array(log_dq_act)[:, j], label=f"Joint {j+1} Act", alpha=0.6)
            axes_compare[1,1].set_title("Kecepatan Sendi Aktual (getActualQd)")
            axes_compare[1,1].set_xlabel("Time (s)")
            axes_compare[1,1].set_ylabel("Kecepatan (rad/s)")
            axes_compare[1,1].grid(True, linestyle='--', alpha=0.6)
            axes_compare[1,1].legend()

            plt.tight_layout()
            filename_compare = f"HIL_SpeedComparison_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_{timestamp}.png"
            plt.savefig(os.path.join(folder_name, filename_compare), dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {os.path.join(folder_name, filename_compare)}")

            # Plot perbandingan posisi joint - dibuat dalam 1 jendela dengan 2 subplot per joint
            fig_joint, axes_joint = plt.subplots(3, 2, figsize=(15, 12))
            fig_joint.suptitle(f'Perbandingan Posisi Joint - Mode: {args.mode}, Speed: {args.speed}, Fitur: {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}', fontsize=16)

            for j in range(6):
                row = j // 2
                col = j % 2
                axes_joint[row, col].plot(log_t, np.array(log_q_ur5e)[:, j], label=f"Joint {j+1} UR5e", linestyle="-", color='blue')
                axes_joint[row, col].plot(log_t, np.array(log_q_pybullet)[:, j], label=f"Joint {j+1} PyBullet", linestyle="--", color='red', alpha=0.7)
                axes_joint[row, col].set_title(f"Joint {j+1} Position")
                axes_joint[row, col].set_xlabel("Time (s)")
                axes_joint[row, col].set_ylabel("Position (rad)")
                axes_joint[row, col].grid(True, linestyle='--', alpha=0.6)
                axes_joint[row, col].legend()

            plt.tight_layout()
            filename_joint = f"HIL_JointPosComparison_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_{timestamp}.png"
            plt.savefig(os.path.join(folder_name, filename_joint), dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {os.path.join(folder_name, filename_joint)}")

            # Plot perbandingan command - menampilkan karakteristik ketiga jenis command
            fig_cmd, axes_cmd = plt.subplots(2, 2, figsize=(15, 12))
            fig_cmd.suptitle(f'Perbandingan Command - Mode: {args.mode}, Speed: {args.speed}, Fitur: {args.fitur}, Nullspace: {args.nullspace}, SpeedMode: {args.speed_mode}, Command: {args.command}', fontsize=16)

            # 1. Tracking error posisi X
            pos_error_x = np.array(log_x_des)[:, 0] - np.array(log_x_act)[:, 0]
            axes_cmd[0,0].plot(log_t, pos_error_x, label=f"Pos X Error - {args.command}", color='red')
            axes_cmd[0,0].set_title(f"Posisi X Error - {args.command}")
            axes_cmd[0,0].set_xlabel("Time (s)")
            axes_cmd[0,0].set_ylabel("Error (m)")
            axes_cmd[0,0].grid(True, linestyle='--', alpha=0.6)
            axes_cmd[0,0].legend()

            # 2. Tracking error posisi Y
            pos_error_y = np.array(log_x_des)[:, 1] - np.array(log_x_act)[:, 1]
            axes_cmd[0,1].plot(log_t, pos_error_y, label=f"Pos Y Error - {args.command}", color='green')
            axes_cmd[0,1].set_title(f"Posisi Y Error - {args.command}")
            axes_cmd[0,1].set_xlabel("Time (s)")
            axes_cmd[0,1].set_ylabel("Error (m)")
            axes_cmd[0,1].grid(True, linestyle='--', alpha=0.6)
            axes_cmd[0,1].legend()

            # 3. Tracking error posisi Z
            pos_error_z = np.array(log_x_des)[:, 2] - np.array(log_x_act)[:, 2]
            axes_cmd[1,0].plot(log_t, pos_error_z, label=f"Pos Z Error - {args.command}", color='blue')
            axes_cmd[1,0].set_title(f"Posisi Z Error - {args.command}")
            axes_cmd[1,0].set_xlabel("Time (s)")
            axes_cmd[1,0].set_ylabel("Error (m)")
            axes_cmd[1,0].grid(True, linestyle='--', alpha=0.6)
            axes_cmd[1,0].legend()

            # 4. Norm error posisi
            pos_des = np.array(log_x_des)
            pos_act = np.array(log_x_act)
            pos_error_norm = np.linalg.norm(pos_des - pos_act, axis=1)
            axes_cmd[1,1].plot(log_t, pos_error_norm, label=f"Pos Error Norm - {args.command}", color='purple')
            axes_cmd[1,1].set_title(f"Norm Error Posisi - {args.command}")
            axes_cmd[1,1].set_xlabel("Time (s)")
            axes_cmd[1,1].set_ylabel("Error Norm (m)")
            axes_cmd[1,1].grid(True, linestyle='--', alpha=0.6)
            axes_cmd[1,1].legend()

            plt.tight_layout()
            filename_cmd = f"HIL_CommandComparison_{TRAJECTORY}_mode_{args.mode}_speed_{args.speed}_fitur_{args.fitur}_nullspace_{args.nullspace}_speedMode_{args.speed_mode}_command_{args.command}_{timestamp}.png"
            plt.savefig(os.path.join(folder_name, filename_cmd), dpi=300, bbox_inches='tight')
            print(f"💾 Disimpan: {os.path.join(folder_name, filename_cmd)}")

            print("Menampilkan Grafik...")
            plt.show()
        
        print("Selesai.")

if __name__ == "__main__":
    main()