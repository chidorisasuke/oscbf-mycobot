import pybullet as p
import time
import numpy as np

# Ganti dengan path URDF Anda yang benar
urdf_path = "oscbf/assets/ur5e/ur5e.urdf" 

# Coba ganti antara GUI dan DIRECT
physicsClient = p.connect(p.GUI) 
# physicsClient = p.connect(p.DIRECT) 

p.setGravity(0, 0, -9.81)

# Posisi awal (coba semua nol dulu)
q_init = [0.0] * 6 

try:
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    print(f"URDF berhasil dimuat. Jumlah joints terdeteksi: {num_joints}")

    # Atur posisi awal
    for i in range(num_joints):
         # Dapatkan info joint untuk memastikan index valid
         info = p.getJointInfo(robot_id, i)
         joint_type = info[2]
         # Hanya atur joint revolute/prismatic
         if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
             # Pastikan index tidak melebihi q_init
             if i < len(q_init):
                  p.resetJointState(robot_id, i, targetValue=q_init[i], targetVelocity=0.0)
             else:
                  print(f"Peringatan: Joint index {i} melebihi panjang q_init ({len(q_init)})")


    # Jalankan simulasi beberapa langkah dan cek state
    for step in range(100):
        p.stepSimulation()
        joint_states = p.getJointStates(robot_id, range(num_joints))
        q = [state[0] for state in joint_states[:len(q_init)]] # Ambil posisi sebanyak q_init
        qdot = [state[1] for state in joint_states[:len(q_init)]]

        print(f"Step {step}: q={np.round(q, 3)}, qdot={np.round(qdot, 3)}")

        # Periksa NaN
        if np.isnan(q).any() or np.isnan(qdot).any():
            print("!!! NaN terdeteksi di joint states PyBullet !!!")
            break

        if p.getConnectionInfo()['connectionMethod'] == p.GUI:
             time.sleep(1./240.)

except Exception as e:
    print(f"Error saat memuat atau menjalankan simulasi PyBullet: {e}")
finally:
    p.disconnect()