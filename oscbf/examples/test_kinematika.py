# File: test_kinematika.py (Versi Perbaikan)
import pybullet as p
import numpy as np
import jax
import jax.numpy as jnp
from oscbf.core.manipulator import load_ur5e

print("Memulai tes kinematika...")

# 1. Muat model OSCBF (Matematika)
robot_oscbf = load_ur5e()
print("Model OSCBF (Manipulator) berhasil dimuat.")

# 2. Muat model PyBullet (Simulasi)
try:
    p.connect(p.DIRECT)
    robot_pb = p.loadURDF("oscbf/assets/ur5e/ur5e_simple.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    print("Model PyBullet (Simulasi) berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat URDF di PyBullet: {e}")
    exit()

# Tentukan link index end-effector di PyBullet
# Kita perlu mencari link 'tool0' atau 'wrist_3_link'
num_joints_pb = p.getNumJoints(robot_pb)
ee_link_name = "wrist_3_link" # Coba 'tool0' dulu, jika gagal ganti ke 'wrist_3_link'
ee_link_index = -1
for i in range(num_joints_pb):
    info = p.getJointInfo(robot_pb, i)
    link_name = info[12].decode('utf-8')
    if link_name == ee_link_name:
        ee_link_index = i
        break

if ee_link_index == -1:
    print(f"ERROR: Tidak dapat menemukan link '{ee_link_name}' di PyBullet. Mencoba 'wrist_3_link'...")
    # Coba cari wrist_3_link
    for i in range(num_joints_pb):
        info = p.getJointInfo(robot_pb, i)
        if info[12].decode('utf-8') == 'wrist_3_link':
            ee_link_index = i
            break
    if ee_link_index == -1:
         print("ERROR: Tidak dapat menemukan link end-effector. Menggunakan link terakhir.")
         ee_link_index = num_joints_pb - 1 # Fallback

print(f"Menggunakan End-Effector Link Index: {ee_link_index}")


# 3. Pilih satu konfigurasi sendi (q)
q_test = (0.1, -0.5, 0.2, -0.1, 0.3, 0.4)

# 4. Hitung Forward Kinematics (FK) dari OSCBF
ee_pos_oscbf = robot_oscbf.ee_position(jnp.array(q_test))
print(f"Posisi EE (OSCBF): {ee_pos_oscbf}")

# 5. Hitung FK dari PyBullet
# Atur sendi PyBullet ke pose q_test
# Pastikan kita hanya mengatur 6 sendi yang relevan
joint_indices_oscbf = [i for i in range(num_joints_pb) if p.getJointInfo(robot_pb, i)[2] != p.JOINT_FIXED]
if len(joint_indices_oscbf) != len(q_test):
     print(f"PERINGATAN: Jumlah sendi bergerak di PyBullet ({len(joint_indices_oscbf)}) tidak cocok dengan q_test ({len(q_test)})")

for i in range(len(q_test)):
    p.resetJointState(robot_pb, joint_indices_oscbf[i], targetValue=q_test[i])

# Dapatkan posisi link dari PyBullet
link_state_pb = p.getLinkState(robot_pb, ee_link_index)
ee_pos_pybullet_tuple = link_state_pb[0] # Ini adalah tuple

# ================== PERBAIKAN TypeError DI SINI ==================
ee_pos_pybullet = np.array(ee_pos_pybullet_tuple) 
# ===============================================================

print(f"Posisi EE (PyBullet): {ee_pos_pybullet}")

# 6. Bandingkan Keduanya
selisih = np.linalg.norm(ee_pos_oscbf - ee_pos_pybullet)
print(f"==================================================")
print(f"SELISIH POSISI (ERROR): {selisih:.4f} meter")
print(f"==================================================")

p.disconnect()

if selisih > 0.01: # Toleransi 1 cm
    print("\nKESIMPULAN: Model Mismatch TERBUKTI.")
    print("Parser OSCBF dan PyBullet menghitung posisi yang berbeda untuk robot yang sama.")
else:
    print("\nKESIMPULAN: Model cocok. Masalahnya ada di tempat lain.")