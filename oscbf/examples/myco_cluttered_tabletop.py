"""Testing the performance of OSCBF in highly-constrained settings

We consider a cluttered tabletop environment with many randomized obstacles,
each represented as a sphere. We then enforce collision avoidance with 
all of the obstacles, and all of the collision bodies on the robot

There are likely "smarter" ways to filter out the collision pairs that are
least likely to cause a collision, but for now, this test just tries to see
how much we can scale up the collision avoidance while retaining real-time
performance.

oscbf/examples/myco_cluttered_tabletop.py
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_mycobot
from oscbf.core.manipulation_env import MyCobotTorqueControlEnv, MyCobotVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.core.controllers import PositionTaskTorqueController, PositionTaskVelocityController
from oscbf.core.controllers import PoseTaskTorqueController, PoseTaskVelocityController
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


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    compensate_centrifugal_coriolis:bool, #NEW
    z: ArrayLike,
    z_ee_des: ArrayLike,
    q_des_target: ArrayLike # <-- Tambahkan argumen ini
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    Jv = J[:3, :] #Ambil hanya Jacobian linear (untuk posisi)
    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)
    # Set nullspace desired joint position
    nullspace_posture_goal = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
        ]
    )

    # Compute nominal control
    # u_nom = osc_controller(
    #     q,
    #     qdot,
    #     pos=ee_tmat[:3, 3],
    #     des_pos=z_ee_des[:3],
    #     des_vel=z_ee_des[12:15],
    #     des_accel=jnp.zeros(3),
    #     des_q=q_des_target, # <-- Gunakan pose target sebagai tujuan nullspace
    #     des_qdot=jnp.zeros(robot.num_joints),
    #     Jv=Jv,
    #     M=M,
    #     M_inv=M_inv,
    #     g=g,
    #     c=c,
    # )

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
    # jax.debug.print("Perintah Nominal (u_nom): {x}", x=u_nom)
    return cbf.safety_filter(z, u_nom)
    # return u_nom


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
    # Set nullspace desired joint position
    des_q = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
        ]
    )
    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )
    return cbf.safety_filter(q, u_nom)

# TAMBAHKAN di myco_cluttered_tabletop.py
# Letakkan setelah pembuatan environment, sebelum main loop

# def visualize_link_frames(env, robot, duration=5.0):
#     """
#     Visualisasi frame koordinat setiap link untuk membantu tuning posisi bola
    
#     Args:
#         env: Environment PyBullet
#         robot: Robot manipulator
#         duration: Durasi tampilan garis (detik)
#     """
#     num_links = env.num_joints
#     line_length = 0.05  # Panjang garis axis (meter)
    
#     print("\n" + "="*60)
#     print("INFORMASI LINK FRAMES")
#     print("="*60)
    
#     for link_idx in range(num_links):
#         try:
#             # Dapatkan info link
#             link_info = env.client.getJointInfo(env.robot, link_idx)
#             link_name = link_info[12].decode('utf-8')
            
#             # Dapatkan state link
#             link_state = env.client.getLinkState(
#                 env.robot, 
#                 link_idx,
#                 computeForwardKinematics=1
#             )
            
#             link_world_pos = link_state[4]
#             link_world_orn = link_state[5]
            
#             # Dapatkan rotation matrix
#             rot_mat = np.array(
#                 env.client.getMatrixFromQuaternion(link_world_orn)
#             ).reshape(3, 3)
            
#             # Gambar axis X (merah), Y (hijau), Z (biru)
#             colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#             axis_names = ['X', 'Y', 'Z']
            
#             for i in range(3):
#                 axis_end = np.array(link_world_pos) + rot_mat[:, i] * line_length
#                 env.client.addUserDebugLine(
#                     link_world_pos,
#                     axis_end.tolist(),
#                     lineColorRGB=colors[i],
#                     lineWidth=3,
#                     lifeTime=duration
#                 )
            
#             # Print info
#             print(f"\nLink {link_idx}: {link_name}")
#             print(f"  Posisi World: {np.round(link_world_pos, 4)}")
#             print(f"  Orientasi (quaternion xyzw): {np.round(link_world_orn, 4)}")
            
#             # Hitung posisi COM jika ada
#             if link_info[3] > 0:  # Jika ada massa
#                 com_local = link_info[3]  # Local COM position
#                 print(f"  Center of Mass (local): {com_local}")
                
#         except Exception as e:
#             print(f"[WARNING] Gagal visualisasi link {link_idx}: {e}")
    
#     print("="*60)
#     print("Garis merah = X axis, Hijau = Y axis, Biru = Z axis")
#     print("="*60 + "\n")


# def debug_collision_spheres(env, robot):
#     """
#     Print informasi detail tentang bola kolisi yang sedang digunakan
#     """
#     print("\n" + "="*60)
#     print("INFORMASI BOLA KOLISI SAAT INI")
#     print("="*60)
    
#     if not hasattr(env, 'robot_collision_offsets'):
#         print("Tidak ada data bola kolisi!")
#         return
    
#     total_spheres = 0
#     for link_idx, offsets in env.robot_collision_offsets:
#         radii = env.robot_collision_radii[
#             [i for i, (idx, _) in enumerate(env.robot_collision_offsets) if idx == link_idx][0]
#         ]
        
#         if len(offsets) == 0:
#             continue
        
#         print(f"\nLink {link_idx}:")
#         print(f"  Jumlah bola: {len(offsets)}")
        
#         # Dapatkan posisi link saat ini
#         try:
#             link_state = env.client.getLinkState(
#                 env.robot, 
#                 link_idx,
#                 computeForwardKinematics=1
#             )
#             link_world_pos = np.array(link_state[4])
#             link_world_orn = link_state[5]
#             rot_mat = np.array(
#                 env.client.getMatrixFromQuaternion(link_world_orn)
#             ).reshape(3, 3)
            
#             # Hitung posisi world setiap bola
#             world_offsets = offsets @ rot_mat.T
#             sphere_world_positions = link_world_pos + world_offsets
            
#             for i, (offset, radius, world_pos) in enumerate(
#                 zip(offsets, radii, sphere_world_positions)
#             ):
#                 print(f"  Bola {i+1}:")
#                 print(f"    Offset (link frame): {np.round(offset, 4)}")
#                 print(f"    Radius: {radius:.4f} m")
#                 print(f"    Posisi (world frame): {np.round(world_pos, 4)}")
#                 total_spheres += 1
                
#         except Exception as e:
#             print(f"  [ERROR] Gagal mendapatkan posisi: {e}")
    
#     print(f"\n{'='*60}")
#     print(f"Total bola kolisi: {total_spheres}")
#     print("="*60 + "\n")


# def interactive_sphere_tuning(env, robot):
#     """
#     Mode interaktif untuk tuning posisi bola kolisi
#     Tekan 'i' untuk melihat info, 'q' untuk keluar
#     """
#     print("\n" + "="*60)
#     print("MODE INTERAKTIF TUNING BOLA KOLISI")
#     print("="*60)
#     print("Perintah:")
#     print("  'i' - Tampilkan informasi bola dan link frames")
#     print("  'd' - Debug detail bola kolisi")
#     print("  'v' - Toggle visualisasi link frames")
#     print("  'q' - Keluar dari mode tuning")
#     print("="*60 + "\n")
    
#     show_frames = False
    
#     while True:
#         # Tampilkan frames jika diaktifkan
#         if show_frames:
#             visualize_link_frames(env, robot, duration=0.1)
        
#         env.step()
        
#         # Check keyboard input (simplified - dalam praktik gunakan pybullet keyboard events)
#         keys = env.client.getKeyboardEvents()
        
#         if ord('i') in keys and keys[ord('i')] & env.client.KEY_WAS_TRIGGERED:
#             visualize_link_frames(env, robot, duration=5.0)
            
#         if ord('d') in keys and keys[ord('d')] & env.client.KEY_WAS_TRIGGERED:
#             debug_collision_spheres(env, robot)
            
#         if ord('v') in keys and keys[ord('v')] & env.client.KEY_WAS_TRIGGERED:
#             show_frames = not show_frames
#             print(f"Visualisasi frames: {'ON' if show_frames else 'OFF'}")
            
#         if ord('q') in keys and keys[ord('q')] & env.client.KEY_WAS_TRIGGERED:
#             print("Keluar dari mode tuning...")
#             break

# ==============================================================================
# FUNGSI HELPER UNTUK DEBUGGING VISUALISASI BOLA KOLISI
# ==============================================================================

def visualize_link_frames(env, duration=5.0):
    """
    Visualisasi frame koordinat (X,Y,Z) setiap link aktif di PyBullet.
    Merah=X, Hijau=Y, Biru=Z.
    
    Args:
        env: Objek environment PyBullet (ManipulationEnv).
        duration (float): Berapa lama garis axis ditampilkan (detik). 0 untuk permanen.
    """
    num_links = env.num_joints # Asumsi link aktif = jumlah sendi
    line_length = 0.05  # Panjang garis axis (5 cm)
    
    print("\n" + "="*60)
    print("VISUALISASI LINK FRAMES (Origin & Orientasi)")
    print("Merah=X, Hijau=Y, Biru=Z")
    print("="*60)
    
    # Base link (biasanya index -1, tapi originnya di base_pos env)
    try:
         base_pos, base_orn = env.client.getBasePositionAndOrientation(env.robot)
         base_rot_mat = np.array(env.client.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
         print(f"\nBase Link (-1):")
         print(f"  Posisi World: {np.round(base_pos, 4)}")
         for i in range(3):
            axis_end = np.array(base_pos) + base_rot_mat[:, i] * line_length
            env.client.addUserDebugLine(base_pos, axis_end.tolist(), 
                                        lineColorRGB=[1 if j==i else 0 for j in range(3)], 
                                        lineWidth=3, lifeTime=duration)
    except Exception as e:
         print(f"[WARNING] Gagal visualisasi Base Link: {e}")

    # Link-link bergerak (index 0 sampai num_joints-1)
    for link_idx in range(num_links):
        try:
            link_state = env.client.getLinkState(
                env.robot, 
                link_idx,
                computeForwardKinematics=1 # Penting! Dapatkan state LINK FRAME
            )
            if link_state is None: continue # Skip jika link tidak valid

            link_name_info = env.client.getJointInfo(env.robot, link_idx)[12].decode('utf-8')
            link_world_pos = link_state[4] # World position of the Link frame
            link_world_orn = link_state[5] # World orientation of the Link frame
            
            rot_mat = np.array(env.client.getMatrixFromQuaternion(link_world_orn)).reshape(3, 3)
            
            print(f"\nLink {link_idx} ({link_name_info}):")
            print(f"  Posisi World: {np.round(link_world_pos, 4)}")
            
            # Gambar axis X, Y, Z
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for i in range(3):
                axis_end = np.array(link_world_pos) + rot_mat[:, i] * line_length
                env.client.addUserDebugLine(
                    link_world_pos,
                    axis_end.tolist(),
                    lineColorRGB=colors[i],
                    lineWidth=3,
                    lifeTime=duration
                )
        except Exception as e:
            print(f"[WARNING] Gagal visualisasi Link {link_idx}: {e}")
    print("="*60 + "\n")


def debug_collision_spheres(env):
    """
    Print informasi detail tentang bola kolisi (offset & posisi world saat ini).
    """
    print("\n" + "="*70)
    print("DEBUG INFORMASI BOLA KOLISI SAAT INI")
    print("="*70)
    
    if not hasattr(env, 'robot_collision_offsets') or not env.robot_collision_offsets:
        print("Data bola kolisi (robot_collision_offsets) tidak ditemukan di environment.")
        return
        
    total_spheres_debug = 0
    # Dapatkan data model kolisi (asumsi sudah ada di env)
    try:
        from oscbf.core.myco_collision_model import mycobot_collision_data # Sesuaikan jika nama file/var beda
        loaded_collision_data = mycobot_collision_data
    except ImportError:
        print("[ERROR] Gagal memuat data asli dari myco_collision_model.py")
        return

    # Iterasi berdasarkan struktur data di environment
    for link_idx, offsets_np in env.robot_collision_offsets:
        try:
            # Ambil radii yang sesuai (agak rumit karena struktur data terpisah)
            # Cari indeks di list radii yang cocok dengan link_idx ini
            radii_idx = -1
            for r_idx, (r_link_idx, _) in enumerate(env.robot_collision_offsets):
                 if r_link_idx == link_idx:
                      radii_idx = r_idx
                      break
            if radii_idx == -1:
                 print(f"[WARNING] Tidak ditemukan data radius untuk link {link_idx}")
                 continue
            radii_np = env.robot_collision_radii[radii_idx]
            
            link_name_info = env.client.getJointInfo(env.robot, link_idx)[12].decode('utf-8')
            print(f"\nLink {link_idx} ({link_name_info}):")
            print(f"  Jumlah bola di model: {len(offsets_np)}")
            
            # Dapatkan state link saat ini
            link_state = env.client.getLinkState(env.robot, link_idx, computeForwardKinematics=1)
            if link_state is None: 
                print("  [ERROR] Gagal mendapatkan state link.")
                continue
            link_world_pos = np.array(link_state[4])
            link_world_orn = link_state[5]
            rot_mat = np.array(env.client.getMatrixFromQuaternion(link_world_orn)).reshape(3, 3)
            
            # Hitung posisi world setiap bola
            world_offsets = offsets_np @ rot_mat.T
            sphere_world_positions = link_world_pos + world_offsets
            
            # Ambil data asli dari file model untuk perbandingan
            original_offsets = loaded_collision_data['positions'][link_idx]
            original_radii = loaded_collision_data['radii'][link_idx]

            for i in range(len(offsets_np)):
                print(f"  Bola {i+1}:")
                # Tampilkan offset asli dari file model
                print(f"    Offset Model   (link frame): {np.round(original_offsets[i], 4)}") 
                print(f"    Radius Model   : {original_radii[i]:.4f} m")
                # Tampilkan posisi world yang dihitung
                print(f"    Posisi Aktual (world frame): {np.round(sphere_world_positions[i], 4)}") 
                total_spheres_debug += 1
                
        except Exception as e:
            print(f"  [ERROR] Gagal memproses link {link_idx}: {e}")
            import traceback
            traceback.print_exc() # Tampilkan error detail
    
    print(f"\n{'='*70}")
    print(f"Total bola kolisi yang diproses: {total_spheres_debug}")
    print("="*70 + "\n")


def interactive_sphere_tuning(env, robot, control_func):
    """
    Mode interaktif untuk membantu tuning posisi bola kolisi.
    """
    print("\n" + "="*60)
    print("MODE INTERAKTIF TUNING BOLA KOLISI")
    print("="*60)
    print("Instruksi:")
    print(" - Pindahkan robot ke berbagai pose menggunakan bola merah.")
    print(" - Amati bola hijau dan link frames (jika diaktifkan).")
    print(" - Edit file 'myco_collision_model.py' untuk menyesuaikan offset (x,y,z).")
    print(" - Simpan file model, lalu restart simulasi ini untuk melihat hasilnya.")
    print("\nPerintah Keyboard di Jendela Simulasi:")
    print("  'f' - Toggle visualisasi link frames (Merah=X, Hijau=Y, Biru=Z)")
    print("  'd' - Cetak informasi detail posisi bola saat ini ke terminal")
    print("  'q' - Keluar dari mode tuning dan akhiri simulasi")
    print("="*60 + "\n")
    
    show_frames = False
    frame_lines = [] # Simpan ID garis frame agar bisa dihapus

    while True:
        # Hapus garis frame sebelumnya jika ada
        if not show_frames:
            for line_id in frame_lines:
                try: env.client.removeUserDebugItem(line_id)
                except: pass # Abaikan jika sudah terhapus
            frame_lines = []

        # Tampilkan frames jika diaktifkan (sementara)
        if show_frames:
             # Hapus dulu yang lama
             for line_id in frame_lines:
                 try: env.client.removeUserDebugItem(line_id)
                 except: pass
             frame_lines = []
             # Gambar yang baru
             num_links = env.num_joints
             line_length = 0.05
             # Base
             try:
                 base_pos, base_orn = env.client.getBasePositionAndOrientation(env.robot)
                 base_rot_mat = np.array(env.client.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
                 for i in range(3):
                    axis_end = np.array(base_pos) + base_rot_mat[:, i] * line_length
                    line_id = env.client.addUserDebugLine(base_pos, axis_end.tolist(), 
                                                lineColorRGB=[1 if j==i else 0 for j in range(3)], 
                                                lineWidth=3, lifeTime=0) # lifeTime 0 = permanen (akan dihapus manual)
                    frame_lines.append(line_id)
             except: pass
             # Links
             for link_idx in range(num_links):
                 try:
                     link_state = env.client.getLinkState(env.robot, link_idx, computeForwardKinematics=1)
                     if link_state is None: continue
                     link_world_pos = link_state[4]
                     link_world_orn = link_state[5]
                     rot_mat = np.array(env.client.getMatrixFromQuaternion(link_world_orn)).reshape(3, 3)
                     colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                     for i in range(3):
                         axis_end = np.array(link_world_pos) + rot_mat[:, i] * line_length
                         line_id = env.client.addUserDebugLine(link_world_pos, axis_end.tolist(),
                                                     lineColorRGB=colors[i], lineWidth=3, lifeTime=0)
                         frame_lines.append(line_id)
                 except: continue

        # Jalankan satu step simulasi (termasuk update bola hijau)
        # Ambil perintah kontrol (meskipun tidak dipakai, agar simulasi jalan)
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        # Asumsikan compute_control ada dan berfungsi
        tau = control_func(q_qdot, z_zdot_ee_des) 
        env.apply_control(tau)
        env.step() # Ini akan memanggil update bola hijau di dalamnya
        
        # Periksa input keyboard
        keys = env.client.getKeyboardEvents()
            
        if ord('f') in keys and keys[ord('f')] & env.client.KEY_WAS_TRIGGERED:
            show_frames = not show_frames
            print(f"Visualisasi frames: {'AKTIF' if show_frames else 'NONAKTIF'}")
            
        if ord('d') in keys and keys[ord('d')] & env.client.KEY_WAS_TRIGGERED:
            debug_collision_spheres(env) # Cetak info detail ke terminal
            
        if ord('q') in keys and keys[ord('q')] & env.client.KEY_WAS_TRIGGERED:
            print("Keluar dari mode tuning interaktif...")
            # Hapus garis frame sebelum keluar
            for line_id in frame_lines:
                 try: env.client.removeUserDebugItem(line_id)
                 except: pass
            break # Keluar dari loop while

# ==============================================================================


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

    # traj = SinusoidalTaskTrajectory(
    #     init_pos=(0.15, 0.0, 0.2),  # Posisi tengah lintasan di atas meja
    #     init_rot=np.eye(3),
    #     amplitude=(0.25, 0.25, 0.1), # Amplitudo gerakan (x, y, z)
    #     angular_freq=(0.3, 0.3, 0.1),   # Kecepatan gerakan
    #     phase=(0, 0, 0),
    # )

    mycobot_q_init = (0, -0.446, -0.071, -0.041, 0, 0) # Contoh: semua sendi di posisi nol

    timestep = 1 / 240  #  1 / 1000
    bg_color = (1, 1, 1)
    if control_method == "torque":
        env = MyCobotTorqueControlEnv(
            # traj=traj,
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
            # traj=traj,
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

    robot_sphere_ids = []
    try:
        all_radii = np.concatenate(mycobot_collision_data['radii'])
        print(f"Membuat visualisasi untuk {len(all_radii)} bola tabrakan robot...")

        for radius in all_radii:
            vis_shape_id = env.client.createVisualShape(
                shapeType=env.client.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[0.1, 1.0, 0.1, 0.3] # Hijau transparan
            )
            body_id = env.client.createMultiBody(
                baseVisualShapeIndex=vis_shape_id,
                baseCollisionShapeIndex=-1, # Tidak perlu deteksi tabrakan (hanya visual)
                basePosition=[0, 0, -1] # Mulai di suatu tempat (di bawah lantai)
            )
            robot_sphere_ids.append(body_id)
            
    except ImportError:
        print("PERINGATAN: Gagal memuat mycobot_collision_data. Bola pelindung robot tidak akan ditampilkan.")

    kp_pos = 20.91
    kp_rot = 22.76
    kd_pos = 77.76
    kd_rot = 8.21
    kp_joint = 30.22
    kd_joint = 1.64

    # kp_pos = 30.0
    # kp_rot = 64.4
    # kd_pos = 4.282
    # kd_rot = 9.055
    # kp_joint = 60.0
    # kd_joint = 6.552
    osc_torque_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        # kp_task=kp_pos,
        # kd_task=kd_pos,
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
        # kp_task=kp_pos,
        kp_joint=kp_joint,
        # Note: velocity limits will be enforced via the QP
        qdot_min=None,
        qdot_max=None,
    )

    q_des_rviz = jnp.array([0.586, -1.210, -0.212, 0.397, 0.0, -0.323])
    @jax.jit
    def compute_torque_control_jit(z, z_ee_des):
        return compute_torque_control(
            robot, osc_torque_controller, torque_cbf, True, z, z_ee_des, q_des_rviz
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

    # === TAMBAHKAN SALAH SATU DARI INI ===
    
    # Opsi 1: Lihat frame sekali saja (5 detik)
    # visualize_link_frames(env, robot, duration=5.0)
    
    # Opsi 2: Debug info bola saat ini
    # print_collision_model_info()
    # debug_collision_spheres(env, robot)
    
    # Opsi 3: Mode interaktif (recommended!)
    # interactive_sphere_tuning(env, robot, compute_control)

    last_print_time = 0
    time_log, h_log, torque_log, velocity_log, position_log, singularity_log = [], [], [], [], [], []
    try:
        # simulation_duration = 30
        # while env.t < simulation_duration:
        while True:
            q_qdot = env.get_joint_state()
            z_zdot_ee_des = env.get_desired_ee_state()
            # print(f"Target Posisi (Bola Merah): {np.round(z_zdot_ee_des[:3], 3)}")
            tau = compute_control(q_qdot, z_zdot_ee_des)
            env.apply_control(tau)
            env.step()

            if robot_sphere_ids: # Hanya jika bola berhasil dibuat
                q = q_qdot[:robot.num_joints]
                # Dapatkan posisi bola tabrakan robot saat ini di koordinat dunia
                robot_collision_data_world = robot.link_collision_data(q)
                robot_sphere_positions = robot_collision_data_world[:, :3]

                # Perbarui posisi setiap bola visual satu per satu
                if len(robot_sphere_ids) == len(robot_sphere_positions):
                    for i in range(len(robot_sphere_ids)):
                        env.client.resetBasePositionAndOrientation(
                            bodyUniqueId=robot_sphere_ids[i],
                            posObj=robot_sphere_positions[i],
                            ornObj=[0, 0, 0, 1] # Orientasi tidak penting untuk bola
                        )

            if (env.t - last_print_time) >= 0.5:
                # Dapatkan posisi aktual dari end-effector
                q_aktual = q_qdot[:robot.num_joints]
                ee_pos_aktual = robot.ee_position(q_aktual)
                
                # Dapatkan posisi target (bola merah)
                posisi_bola_merah = z_zdot_ee_des[:3]

                # Hitung selisih (error)
                selisih = np.linalg.norm(posisi_bola_merah - ee_pos_aktual)

                # Cetak log ke terminal (dibulatkan agar mudah dibaca)
                print(f"--- Waktu: {env.t:.2f} s ---")
                print(f"Posisi Sendi (q):      {np.round(q_aktual, 2)}")
                print(f"Posisi End-Effector:   {np.round(ee_pos_aktual, 2)}")
                print(f"Posisi Bola Merah:     {np.round(posisi_bola_merah, 2)}")
                print(f"Selisih Jarak (Error): {selisih:.3f} m")
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
        # Tangani jika pengguna menekan Ctrl+C
        print("\nSimulasi dihentikan oleh pengguna.")

    finally:
        print("Simulation finished. Plotting data...")
        # Konversi list ke numpy array
        h_log = np.array(h_log)
        torque_log = np.array(torque_log)
        position_log = np.array(position_log)
        velocity_log = np.array(velocity_log)
        # TAMBAHKAN BARIS INI (jika belum ada):
        if 'env' in locals() and hasattr(env, 'client'):
            try:
                env.client.disconnect()
                print("PyBullet disconnected.")
            except Exception as e:
                print(f"Error disconnecting PyBullet: {e}")

        # Buat 4 subplot
        # fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)
        # fig.suptitle('Analisis Simulasi Robot Mycobot 280', fontsize=16)

        # Plot 1: Evolusi Batasan Keamanan (h(z))
        # constraint_labels = ["X max", "Y max", "Z max", "X min", "Y min", "Z min"]
        # for i in range(h_log.shape[1]):
        #     axs[0].plot(time_log, h_log[:, i], label=constraint_labels[i])
        # for i in range(h_log.shape[1]):
        #     axs[0].plot(time_log, h_log[:, i])
        # axs[0].plot([], [], label='Batasan Tabrakan')
        # axs[0].axhline(0, color='r', linestyle='--', label='Batas Aman (h=0)')
        # if len(time_log) > 0: # Pastikan ada data untuk di-plot
        # # Plot garis batasan pertama dengan label
        #     axs[0].plot(time_log, h_log[:, 0], color='c', alpha=0.7, label=f'Batasan Tabrakan ({h_log.shape[1]} total)')
        # # Plot sisa garisnya tanpa label agar legenda tidak ramai
        # for i in range(1, h_log.shape[1]):
        #     axs[0].plot(time_log, h_log[:, i], color='c', alpha=0.7)
        # BLOK BARU

        #KOMEN
        # if len(time_log) > 0:
        #     # Loop melalui setiap kolom di h_log (setiap batasan)
        #     for i in range(h_log.shape[1]):
        #         # Biarkan Matplotlib memilih warna secara otomatis
        #         # Kita hanya akan memberi label pada garis pertama agar legenda tidak terlalu ramai
        #         if i == 0:
        #             axs[0].plot(time_log, h_log[:, i], alpha=0.7, label=f'Batasan Tabrakan ({h_log.shape[1]} total)')
        #         else:
        #             axs[0].plot(time_log, h_log[:, i], alpha=0.7)


        # axs[0].axhline(0, color='r', linestyle='--', label='Batas Aman (h=0)')
        # axs[0].set_title('Evolusi Batasan Keamanan (h(z))')
        # axs[0].set_ylabel('Nilai h(z)')
        # axs[0].grid(True)
        # axs[0].legend(fontsize='small')

        # # Plot 2: Torsi Sendi (Torque)
        # start_index = 10 
        # if len(time_log) > start_index:
        #     for i in range(torque_log.shape[1]):
        #         axs[1].plot(time_log[start_index:], torque_log[start_index:, i], label=f'Sendi {i+1}')
        # # Atau, Anda bisa secara manual mengatur batas sumbu Y jika lonjakan terjadi di tengah
        # # axs[1].set_ylim(-15, 15) # Batas -15 sampai 15 Nm (sesuaikan jika perlu)
        # axs[1].set_title('Perintah Torsi Aman (Γ*)')
        # axs[1].set_ylabel('Torsi (Nm)')
        # axs[1].grid(True)
        # axs[1].legend(fontsize='small')

        # # Plot 3: Kecepatan Sendi (Velocity)
        # for i in range(velocity_log.shape[1]):
        #     axs[2].plot(time_log, velocity_log[:, i], label=f'Sendi {i+1}')
        # axs[2].set_title('Kecepatan Sendi (q_dot)')
        # axs[2].set_ylabel('Kecepatan (rad/s)')
        # axs[2].grid(True)
        # axs[2].legend(fontsize='small')

        # # Plot 4: Posisi Sendi (Position)
        # for i in range(position_log.shape[1]):
        #     axs[3].plot(time_log, position_log[:, i], label=f'Sendi {i+1}')
        # axs[3].set_title('Posisi Sendi (q)')
        # axs[3].set_ylabel('Posisi (rad)')
        # axs[3].set_xlabel('Waktu (s)')
        # axs[3].grid(True)
        # axs[3].legend(fontsize='small')

        # # Plot 2: Evolusi Batasan Singularitas
        # if len(time_log) > 0:
        #     axs[4].plot(time_log, singularity_log, label='Manipulability Index (μ)', color='purple')
        #     # Gambar garis batas toleransi singularitas
        #     axs[4].axhline(torque_config.singularity_tol, color='r', linestyle='--', label=f'Batas Aman (μ={torque_config.singularity_tol})')
        # axs[4].set_title('Evolusi Batasan Singularitas (Manipulability)')
        # axs[4].set_ylabel('Nilai Manipulabilitas (μ)')
        # axs[4].set_yscale('log') # Gunakan skala logaritmik agar lebih terlihat
        # axs[4].grid(True)
        # axs[4].legend(fontsize='small')

        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.show()


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
