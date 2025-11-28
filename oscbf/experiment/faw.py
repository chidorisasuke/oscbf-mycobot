import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rtde_control
import rtde_receive
import csv
import os
from datetime import datetime
import pybullet as p
import pybullet_data


# ============================================
# NULLSPACE HELPER FUNCTIONS
# ============================================

def compute_manipulability(J):
    """Manipulability measure"""
    JJT = J @ J.T
    det_val = np.linalg.det(JJT)
    return np.sqrt(max(0, det_val))

def compute_manipulability_gradient(q, J, epsilon=1e-6):
    """Gradient of manipulability"""
    w_current = compute_manipulability(J)
    grad_w = np.zeros(6)
    
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += epsilon
        J_plus = compute_jacobian(q_plus)
        w_plus = compute_manipulability(J_plus)
        grad_w[i] = (w_plus - w_current) / epsilon
    
    return grad_w, w_current

def compute_joint_limit_gradient(q, q_min, q_max):
    """Gradient for joint limit avoidance"""
    q_mid = (q_max + q_min) / 2.0
    q_range = q_max - q_min
    q_range = np.where(q_range < 1e-6, 1e-6, q_range)
    
    q_normalized = (q - q_mid) / q_range
    grad_w = -2.0 * q_normalized / q_range
    
    dist_to_min = (q - q_min) / q_range
    dist_to_max = (q_max - q) / q_range
    min_dist = np.minimum(dist_to_min, dist_to_max).min()
    
    return grad_w, min_dist


# ============================================
# OBSTACLE AVOIDANCE FUNCTIONS
# ============================================

def get_link_positions(q):
    """
    Compute positions of all robot links/joints for obstacle avoidance.
    Returns list of 3D positions for each link.
    Uses UR5e DH parameters to compute intermediate frame positions.
    """
    d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
    a = [0, -0.425, -0.3922, 0, 0, 0]
    alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
    
    positions = []
    T = np.eye(4)
    
    # Base position
    positions.append(T[:3, 3].copy())
    
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        T_i = np.array([
            [ct, -st*ca, st*sa, a[i]*ct],
            [st, ct*ca, -ct*sa, a[i]*st],
            [0, sa, ca, d[i]],
            [0, 0, 0, 1]
        ])
        T = T @ T_i
        positions.append(T[:3, 3].copy())
    
    return positions


def compute_min_distance_to_obstacle(q, obstacle_pos):
    """
    Compute minimum distance from any point on the robot to the obstacle.
    w(q) = min_{p,o} ||p(q) - o||
    
    Args:
        q: Joint configuration (6,)
        obstacle_pos: Obstacle center position (3,)
    
    Returns:
        min_dist: Minimum distance to obstacle
        closest_link_idx: Index of the closest link
        closest_point: Position of closest point on robot
    """
    link_positions = get_link_positions(q)
    
    min_dist = np.inf
    closest_link_idx = 0
    closest_point = link_positions[0]
    
    for idx, p in enumerate(link_positions):
        dist = np.linalg.norm(p - obstacle_pos)
        if dist < min_dist:
            min_dist = dist
            closest_link_idx = idx
            closest_point = p
    
    return min_dist, closest_link_idx, closest_point


def compute_obstacle_avoidance_gradient(q, obstacle_pos, epsilon=1e-6):
    """
    Compute gradient of distance function w(q) = min_{p,o} ||p(q) - o||
    We want to MAXIMIZE distance, so gradient points away from obstacle.
    
    Args:
        q: Joint configuration (6,)
        obstacle_pos: Obstacle center position (3,)
        epsilon: Numerical differentiation step
    
    Returns:
        grad_w: Gradient of distance w.r.t. joint angles (6,)
        min_dist: Current minimum distance to obstacle
    """
    min_dist, _, _ = compute_min_distance_to_obstacle(q, obstacle_pos)
    
    grad_w = np.zeros(6)
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += epsilon
        dist_plus, _, _ = compute_min_distance_to_obstacle(q_plus, obstacle_pos)
        grad_w[i] = (dist_plus - min_dist) / epsilon
    
    return grad_w, min_dist


def nullspace_obstacle_avoidance(q, J, obstacle_pos, K_null=0.5, 
                                  activation_distance=0.3, influence_distance=0.5):
    """
    Nullspace optimization for obstacle avoidance.
    
    The robot will try to maximize distance from obstacle when within influence zone.
    Activation becomes stronger as robot gets closer to obstacle.
    
    Args:
        q: Current joint configuration (6,)
        J: Current Jacobian (3x6)
        obstacle_pos: Obstacle center position (3,)
        K_null: Nullspace gain
        activation_distance: Distance below which avoidance is fully active
        influence_distance: Distance below which avoidance starts
    
    Returns:
        dq_null: Nullspace velocity for obstacle avoidance (6,)
        metrics: Dictionary with obstacle avoidance metrics
    """
    # Compute gradient and current distance
    grad_dist, min_dist = compute_obstacle_avoidance_gradient(q, obstacle_pos)
    
    # Compute activation factor based on distance
    if min_dist >= influence_distance:
        # Far from obstacle - no avoidance needed
        activation = 0.0
    elif min_dist <= activation_distance:
        # Very close - full activation
        activation = 1.0
    else:
        # Smooth transition using sigmoid-like function
        normalized_dist = (min_dist - activation_distance) / (influence_distance - activation_distance)
        activation = 1.0 - normalized_dist  # Linear interpolation
        # Or use smoother transition:
        # activation = 0.5 * (1 - np.cos(np.pi * (1 - normalized_dist)))
    
    # Apply exponential scaling for stronger response when close
    if min_dist < activation_distance:
        # Exponential increase as we get closer
        scale_factor = np.exp(2.0 * (activation_distance - min_dist) / activation_distance)
    else:
        scale_factor = 1.0
    
    # Compute nullspace velocity (maximize distance = follow positive gradient)
    dq_null = K_null * activation * scale_factor * grad_dist
    
    # Get closest link info for visualization
    _, closest_link_idx, closest_point = compute_min_distance_to_obstacle(q, obstacle_pos)
    
    metrics = {
        'min_distance': min_dist,
        'activation': activation,
        'scale_factor': scale_factor,
        'closest_link_idx': closest_link_idx,
        'closest_point': closest_point.copy(),
        'grad_norm': np.linalg.norm(grad_dist)
    }
    
    return dq_null, metrics


# ============================================
# ROBOT FUNCTIONS
# ============================================

# URSIM_IP = '127.0.0.1'
URSIM_IP = '192.168.0.191'
URSIM_PORT = 5900

def forward_kinematics_ur5e(q):
    d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
    a = [0, -0.425, -0.3922, 0, 0, 0]
    alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
    T = np.eye(4)
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        T_i = np.array([
            [ct, -st*ca, st*sa, a[i]*ct],
            [st, ct*ca, -ct*sa, a[i]*st],
            [0, sa, ca, d[i]],
            [0, 0, 0, 1]
        ])
        T = T @ T_i
    return T[:3, 3], T[:3, :3]

def compute_jacobian(q):
    epsilon = 1e-6
    J = np.zeros((3, 6))
    tcp_pos, _ = forward_kinematics_ur5e(q)
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += epsilon
        tcp_plus, _ = forward_kinematics_ur5e(q_plus)
        J[:, i] = (tcp_plus - tcp_pos) / epsilon
    return J

def rpy_from_R(R):
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw

def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


# ============================================
# FOLDER MANAGEMENT
# ============================================
def get_trajectory_name(choice):
    if choice == "1":
        return "xy"
    elif choice == "2":
        return "3d"
    return "unknown"

def get_nullspace_name(nullspace_choice):
    names = {
        "1": "current_config",
        "2": "manipulability",
        "3": "joint_limits",
        "4": "obstacle_avoidance"
    }
    return names.get(nullspace_choice, "unknown")

def get_control_method_name(control_choice):
    """Get control method name for folder"""
    names = {
        "1": "speedJ",
        "2": "moveL",
        "3": "servoL"
    }
    return names.get(control_choice, "unknown")

def get_robot_source(ip_address):
    if ip_address == '127.0.0.1':
        return "URSIM"
    elif ip_address == '192.168.0.191':
        return "Eksperimen"
    else:
        return f"Custom_{ip_address.replace('.', '_')}"

def create_experiment_folder(traj_name, nullspace_name, control_method, robot_source):
    base_dir = "experiments"
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    date_path = os.path.join(base_dir, date_folder)
    os.makedirs(date_path, exist_ok=True)
    
    folder_name = f"{traj_name}_{nullspace_name}_{control_method}_{robot_source}_{timestamp}"
    full_path = os.path.join(date_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    
    return full_path, timestamp


# ============================================
# MAIN PROGRAM
# ============================================

def main():

    # ============================================
    # PARAMETERS
    # ============================================
    dt = 1/125
    freq_sent = 1 / dt
    duration = 5
    Kp = 2.5
    K_null = 0.5
    amplitude = 0.1
    frequency_traj = 0.2
    MAX_JOINT_VEL = 1.0
    
    # Obstacle avoidance parameters
    ACTIVATION_DISTANCE = 0.15   # Full activation when closer than 15cm
    INFLUENCE_DISTANCE = 0.35    # Start avoiding when closer than 35cm
    
    print("=== CONTROL PARAMETERS ===")
    print(f"dt = {dt}s ({1/dt:.0f} Hz)")
    print(f"Kp = {Kp}")
    print(f"K_null = {K_null}")
    print(f"Frequency = {frequency_traj} Hz")
    print(f"Amplitude = {amplitude} m")
    print(f"Duration = {duration} s")
    print(f"Max joint velocity = {MAX_JOINT_VEL} rad/s")
    print(f"Obstacle activation distance = {ACTIVATION_DISTANCE} m")
    print(f"Obstacle influence distance = {INFLUENCE_DISTANCE} m\n")

    print(" ╔═══════════════════════════════════════════╗")
    print(f"║  UR5e RTDE Control - {freq_sent:.0f} Hz              ║")
    print(" ║  Multi-Method Control with Digital Twin   ║")
    print(" ╚═══════════════════════════════════════════╝\n")
    
    # PyBullet visualization
    print("\n🚀 Starting PyBullet Visualization (Digital Twin)...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot_sim = p.loadURDF("ur_e_description/urdf/ur5e.urdf", useFixedBase=True)

    sim_joint_indices = []
    for i in range(p.getNumJoints(robot_sim)):
        if p.getJointInfo(robot_sim, i)[2] != p.JOINT_FIXED:
            sim_joint_indices.append(i)

    print("=== SELECT TRAJECTORY TYPE ===")
    print("[1] Circular in XY plane")
    print("[2] Sinusoidal 3D (XYZ)")
    choice = input("Enter choice (1/2): ").strip()
    
    print("\n=== SELECT NULLSPACE OPTIMIZATION ===")
    print("[1] Current Configuration (default)")
    print("[2] Manipulability Maximization")
    print("[3] Joint Limits Avoidance")
    print("[4] Obstacle Avoidance")
    nullspace_choice = input("Enter choice (1/2/3/4): ").strip()

    # ============================================
    # NEW: CONTROL METHOD SELECTION
    # ============================================
    print("\n=== SELECT CONTROL METHOD ===")
    print("[1] speedJ - Joint Velocity Control (default)")
    print("[2] moveL - Linear Cartesian Move")
    print("[3] servoL - Joint Position Servoing (high-frequency)")
    control_choice = input("Enter choice (1/2/3): ").strip()
    
    if control_choice not in ['1', '2', '3']:
        control_choice = '1'
        print("[INFO] Invalid choice. Using speedJ (default)")
    
    control_method_name = get_control_method_name(control_choice)
    print(f"[INFO] Selected control method: {control_method_name}\n")

    # ============================================
    # OBSTACLE CONFIGURATION
    # ============================================
    obstacle_pos = None
    obstacle_visual_id = None
    
    if nullspace_choice == '4':
        print("\n=== OBSTACLE CONFIGURATION ===")
        print("Enter obstacle position in meters (X, Y, Z)")
        print("Example: 0.3, -0.2, 0.4")
        print("Or press Enter for default position (0.3, -0.3, 0.4)")
        
        obs_input = input("Obstacle position: ").strip()
        if obs_input == '':
            obstacle_pos = np.array([0.3, -0.3, 0.4])
        else:
            try:
                coords = [float(x.strip()) for x in obs_input.split(',')]
                if len(coords) != 3:
                    raise ValueError("Need exactly 3 coordinates")
                obstacle_pos = np.array(coords)
            except Exception as e:
                print(f"[WARN] Invalid input ({e}), using default position")
                obstacle_pos = np.array([0.3, -0.3, 0.4])
        
        print(f"[OBSTACLE] Position set to: {obstacle_pos}")
        
        # Add obstacle visualization in PyBullet
        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[1, 0, 0, 0.8]  # Red sphere
        )
        obstacle_collision_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.05
        )
        obstacle_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_collision_id,
            baseVisualShapeIndex=obstacle_visual_id,
            basePosition=obstacle_pos.tolist()
        )
        
        # Add influence zone visualization (transparent sphere)
        influence_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=INFLUENCE_DISTANCE,
            rgbaColor=[1, 1, 0, 0.1]  # Yellow transparent
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=influence_visual,
            basePosition=obstacle_pos.tolist()
        )

    robot_source = get_robot_source(URSIM_IP)
    traj_name = get_trajectory_name(choice)
    nullspace_name = get_nullspace_name(nullspace_choice)
    exp_folder, timestamp = create_experiment_folder(traj_name, nullspace_name, 
                                                     control_method_name, robot_source)
    
    print(f"\n{'='*60}")
    print(f"[FOLDER] Experiment folder created")
    print(f"{'='*60}")
    print(f"Path: {exp_folder}")
    print(f"Date: {timestamp.split('_')[0]}")
    print(f"Time: {timestamp.split('_')[1].replace('-', ':')}")
    print(f"Robot: {robot_source}")
    print(f"Trajectory: {traj_name}")
    print(f"Nullspace: {nullspace_name}")
    print(f"Control Method: {control_method_name}")
    print(f"{'='*60}\n")

    # ============================================
    # RTDE CONNECTION
    # ============================================
    print("[RTDE] Connecting to robot...")
    try:
        rtde_c = rtde_control.RTDEControlInterface(URSIM_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(URSIM_IP)
        print("[RTDE] ✓ Connected successfully!\n")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return

    # Initial configuration
    q_init = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
    print("[RTDE] Moving to initial position...")
    rtde_c.moveJ(q_init, speed=0.5, acceleration=0.5)
    time.sleep(2)
    
    q = np.array(rtde_r.getActualQ())
    print(f"[RTDE] Initial position: {np.rad2deg(q)}")

    base_pos, _ = forward_kinematics_ur5e(q)
    center_x, center_y, center_z = base_pos
    print(f"[RTDE] Initial joint angles (deg): {np.rad2deg(q).round(2)}")
    print(f"[RTDE] Initial TCP position: X={center_x:.4f}, Y={center_y:.4f}, Z={center_z:.4f}\n")

    q_min = np.deg2rad(np.array([-180, -180, -180, -180, -180, -180]))
    q_max = np.deg2rad(np.array([180, 180, 180, 180, 180, 180]))

    timesteps = np.arange(0, duration, dt)
    print(f"[INFO] Total control steps: {len(timesteps)}\n")

    # Trajectory center
    base_pos, _ = forward_kinematics_ur5e(q)
    x_start, y_start, z_start = base_pos
    print(f"[INFO] Initial EE position: X={x_start:.4f}, Y={y_start:.4f}, Z={z_start:.4f}")
    
    center_x, center_y, center_z = x_start, y_start, z_start
       
    if choice == "1":
        phase_offset = 0.0
        print(f"[TRAJECTORY] Circular XY")
        print(f"  Center: ({center_x:.4f}, {center_y:.4f})")
        print(f"  Radius: {amplitude:.4f}m\n")
    elif choice == "2":
        phase_offset = 0.0
        print(f"[TRAJECTORY] 3D Sinusoidal")
        print(f"  Starting from current position\n")

    # ============================================
    # LOGGING INITIALIZATION
    # ============================================
    trajectory_target = []
    traj_act_fk = []
    traj_act_rtde = []
    traj_act_rtde_xyz = []
    rpy_log = []
    rpy_rtde_log = []
    t_log = []
    dq_ns_norm_log = []
    dq_cmd_log = []
    q_log = []
    dq_act_log = []
    twin_q_log = []
    twin_qd_log = []
    twin_pos_log = []
    twin_rpy_log = []
    twin_time_log = []
    
    # NEW: Control method specific logs
    control_command_log = []  # Store actual commands sent
    control_timing_log = []   # Timing for each control method
    
    if nullspace_choice in ['2']:
        manip_log = []
    if nullspace_choice in ['3']:
        dist_limits_log = []
    if nullspace_choice == '4':
        obstacle_metrics_log = []

    # ============================================
    # CONTROL LOOP
    # ============================================
    print("[CONTROL] Starting trajectory execution...")
    start_time = time.time()
    
    q0 = q.copy()

    center_x = center_x - amplitude * np.cos(0) * 0.8
    center_y = center_y - amplitude * np.sin(0) * 0.8
    
    # Get current TCP orientation for moveL
    tcp_pose_init = rtde_r.getActualTCPPose()
    tcp_orientation = list(tcp_pose_init[3:])  # Convert to list [rx, ry, rz]
    
    try:
        for i, t in enumerate(timesteps):
            loop_start = time.time()
            
            # ============================================
            # TRAJECTORY GENERATION
            # ============================================
            if choice == "1":
                x = center_x + amplitude * np.cos(2 * np.pi * frequency_traj * t + phase_offset)
                y = center_y + amplitude * np.sin(2 * np.pi * frequency_traj * t + phase_offset)
                z_target = center_z
                
            elif choice == "2":
                x = center_x + amplitude * np.sin(2 * np.pi * frequency_traj * t)
                y = center_y + amplitude * np.sin(2 * np.pi * frequency_traj * t + np.pi/2)
                z_target = center_z + amplitude * np.sin(2 * np.pi * frequency_traj * t + np.pi)

            target_pos = np.array([x, y, z_target])
            trajectory_target.append(target_pos)

            # Get actual position from RTDE
            q_actual_rtde = np.array(rtde_r.getActualQ())
            q = q_actual_rtde.copy()
            
            # Forward Kinematics
            x_actual_fk, R_actual = forward_kinematics_ur5e(q)
            traj_act_fk.append(x_actual_fk)
            
            # Direct TCP from RTDE
            tcp_pose = rtde_r.getActualTCPPose()
            x_actual_rtde = np.array(tcp_pose[:3])
            traj_act_rtde.append(x_actual_rtde)
            traj_act_rtde_xyz.append(x_actual_rtde.copy())
            rpy_rtde_log.append(np.rad2deg(tcp_pose[3:]))

            # Task-space control
            delta_x = x_actual_fk - target_pos
            nu = -Kp * delta_x

            # Jacobian
            J = compute_jacobian(q)
            J_pinv = np.linalg.pinv(J)
            dq_task = J_pinv @ nu

            # ============================================
            # NULLSPACE SELECTION
            # ============================================
            if nullspace_choice == '1':
                dq_null = -K_null * (q - q0)
                
            elif nullspace_choice == '2':
                grad_manip, manip_val = compute_manipulability_gradient(q, J)
                dq_null = K_null * grad_manip
                manip_log.append(manip_val)
                
            elif nullspace_choice == '3':
                grad_limits, min_dist = compute_joint_limit_gradient(q, q_min, q_max)
                dq_null = K_null * grad_limits
                dist_limits_log.append(min_dist)
                
            elif nullspace_choice == '4':
                # OBSTACLE AVOIDANCE
                dq_null, obs_metrics = nullspace_obstacle_avoidance(
                    q, J, obstacle_pos,
                    K_null=K_null,
                    activation_distance=ACTIVATION_DISTANCE,
                    influence_distance=INFLUENCE_DISTANCE
                )
                obstacle_metrics_log.append(obs_metrics)
            
            # Nullspace projection
            N = np.eye(6) - J_pinv @ J
            dq_ns = N @ dq_null
            dq_cmd = dq_task + dq_ns
            
            dq_cmd_log.append(dq_cmd.copy())

            # ============================================
            # CONTROL METHOD EXECUTION
            # ============================================
            control_start = time.time()
            
            if control_choice == '1':
                # speedJ - Joint Velocity Control
                rtde_c.speedJ(dq_cmd.tolist(), acceleration=1, time=0.001)
                control_command_log.append({
                    'type': 'speedJ',
                    'command': dq_cmd.copy(),
                    'success': True
                })
                
            elif control_choice == '2':
                # moveL - Linear Cartesian Move
                # Compute target TCP pose from target position
                target_pose = target_pos.tolist() + tcp_orientation
                
                # Use moveL with asynchronous execution
                rtde_c.moveL(target_pose, speed=0.5, acceleration=0.5, asynchronous=True)
                control_command_log.append({
                    'type': 'moveL',
                    'command': target_pose.copy() if isinstance(target_pose, np.ndarray) else target_pose[:],
                    'success': True
                })
                
            elif control_choice == '3':
                # servoL - Joint Position Servoing
                # Compute target joint configuration using inverse kinematics approximation
                # # q_target = q_current + dq * dt
                # q_target = q + dq_cmd * dt
                
                # # Clamp to joint limits
                # q_target = np.clip(q_target, q_min, q_max)
                
                # # servoL requires: target_q, velocity, acceleration, time, lookahead_time, gain
                # # All as positional arguments (not keyword arguments)
                # rtde_c.servoL(
                #     q_target.tolist(),  # arg0: target joint positions
                #     1.0,                # arg1: velocity (max velocity)
                #     1.0,                # arg2: acceleration (max acceleration)
                #     dt,                 # arg3: time (control cycle time)
                #     0.1,                # arg4: lookahead_time
                #     300                 # arg5: gain (proportional gain)
                # )
                tcp_pose_init = rtde_r.getActualTCPPose()
                tcp_orientation = list(tcp_pose_init[3:])  # [rx, ry, rz]


                target_pose = target_pos.tolist() + tcp_orientation

                rtde_c.servoL(
                    target_pose,  # arg0: target joint positions
                    1.0,                # arg1: velocity (max velocity)
                    1.0,                # arg2: acceleration (max acceleration)
                    dt,                 # arg3: time (control cycle time)
                    0.1,                # arg4: lookahead_time
                    300                 # arg5: gain (proportional gain)
                )
                
                control_command_log.append({
                    'type': 'servoL',
                    'command': target_pose.copy(),
                    'success': True
                })

            control_time = time.time() - control_start
            control_timing_log.append(control_time)
            
            q_log.append(q.copy())

            # Get actual velocity
            qd_actual = np.array(rtde_r.getActualQd())
            dq_act_log.append(qd_actual.copy())

            # Update PyBullet
            for j, idx in enumerate(sim_joint_indices):
                p.resetJointState(robot_sim, idx, q_actual_rtde[j], qd_actual[j])

            # ===== PyBullet logging (Digital Twin) =====
            joint_states = p.getJointStates(robot_sim, sim_joint_indices)
            q_twin = np.array([js[0] for js in joint_states])
            qd_twin = np.array([js[1] for js in joint_states])
            pos_twin, R_twin = forward_kinematics_ur5e(q_twin)
            rpy_twin = np.rad2deg(rpy_from_R(R_twin))

            twin_q_log.append(q_twin.copy())
            twin_qd_log.append(qd_twin.copy())
            twin_pos_log.append(pos_twin.copy())
            twin_rpy_log.append(rpy_twin.copy())
            twin_time_log.append(time.time())

            # Logging
            roll, pitch, yaw = rpy_from_R(R_actual)
            rpy_log.append(np.rad2deg([roll, pitch, yaw]))
            dq_ns_norm_log.append(np.linalg.norm(dq_ns))
            t_log.append(t)
            
            # Progress indicator
            if i % 200 == 0:
                elapsed = time.time() - start_time
                if nullspace_choice == '4':
                    print(f"[{elapsed:.1f}s] Step {i}/{len(timesteps)}, "
                          f"Error: {np.linalg.norm(delta_x)*1000:.2f} mm, "
                          f"Obs dist: {obs_metrics['min_distance']*1000:.1f} mm, "
                          f"Method: {control_method_name}")
                else:
                    print(f"[{elapsed:.1f}s] Step {i}/{len(timesteps)}, "
                          f"Error: {np.linalg.norm(delta_x)*1000:.2f} mm, "
                          f"Method: {control_method_name}")
            
            # Timing control
            loop_time = time.time() - loop_start
            sleep_time = dt - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n[CONTROL] Trajectory execution interrupted by user.")

    # Stop robot based on control method
    if control_choice == '1':
        rtde_c.speedStop()
    elif control_choice == '2':
        rtde_c.stopL()
    elif control_choice == '3':
        rtde_c.servoStop()
    
    rtde_c.stopScript()
    
    total_time = time.time() - start_time
    print(f"\n[DONE] Trajectory completed!")
    print(f"[DONE] Total time: {total_time:.2f}s")
    print(f"[DONE] Average cycle time: {total_time/len(t_log)*1000:.2f}ms")
    print(f"[DONE] Average control time: {np.mean(control_timing_log)*1000:.3f}ms\n")

    # Convert logs
    trajectory_target = np.array(trajectory_target)
    traj_act_fk = np.array(traj_act_fk)
    traj_act_rtde = np.array(traj_act_rtde)
    traj_act_rtde_xyz = np.array(traj_act_rtde_xyz)
    rpy_log = np.array(rpy_log)
    rpy_rtde_log = np.array(rpy_rtde_log)
    t_log = np.array(t_log)
    dq_ns_norm_log = np.array(dq_ns_norm_log)
    dq_cmd_log = np.array(dq_cmd_log)
    q_log = np.array(q_log)
    dq_act_log = np.array(dq_act_log)
    twin_q_arr = np.array(twin_q_log)
    twin_qd_arr = np.array(twin_qd_log)
    twin_pos_arr = np.array(twin_pos_log)
    twin_rpy_arr = np.array(twin_rpy_log)
    twin_time_arr = np.array(twin_time_log)
    control_timing_arr = np.array(control_timing_log)

    # ========================================
    # SAVE CSV FILES
    # ========================================
    
    def save_csv(filename, data, header=None):
        path = os.path.join(exp_folder, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(data)
        print(f"[SAVED] {filename}")

    print("\n[INFO] Saving data...")
    save_csv("trajectory_target.csv", trajectory_target, header=["x_target", "y_target", "z_target"])
    save_csv("traj_act_fk.csv", traj_act_fk,header=["x_actual_fk", "y_actual_fk", "z_actual_fk"])
    save_csv("traj_act_rtde.csv", traj_act_rtde,header=["x_actual_rtde", "y_actual_rtde", "z_actual_rtde"])
    save_csv("rpy_log.csv", rpy_log,header=["roll_deg", "pitch_deg", "yaw_deg"])
    save_csv("rpy_rtde_log.csv", rpy_rtde_log,header=["rx_deg", "ry_deg", "rz_deg"])
    save_csv("dq_ns_norm.csv", dq_ns_norm_log.reshape(-1, 1),header=["dq_ns_norm"])
    save_csv("dq_cmd.csv", dq_cmd_log,header=[f"dq_cmd_{i+1}" for i in range(6)])
    save_csv("q_log.csv", np.rad2deg(q_log),header=[f"q_deg_{i+1}" for i in range(6)])
    save_csv("dq_act_log.csv", dq_act_log,header=[f"dq_act_{i+1}" for i in range(6)])
    save_csv("time_log.csv", t_log.reshape(-1, 1),header=["time_s"])
    save_csv("twin_q_log.csv", twin_q_arr, header=[f"q_twin_{i+1}" for i in range(twin_q_arr.shape[1])])
    save_csv("twin_qd_log.csv", twin_qd_arr, header=[f"dq_twin_{i+1}" for i in range(twin_qd_arr.shape[1])])
    save_csv("twin_pos_log.csv", twin_pos_arr, header=["x_twin", "y_twin", "z_twin"])
    save_csv("twin_rpy_log.csv", twin_rpy_arr, header=["roll_deg_twin","pitch_deg_twin","yaw_deg_twin"])
    save_csv("twin_time_log.csv", twin_time_arr.reshape(-1,1), header=["twin_timestamp"])
    
    # NEW: Save control method specific data
    save_csv("control_timing_log.csv", control_timing_arr.reshape(-1,1), header=["control_time_s"])
    
    # Save obstacle avoidance metrics
    if nullspace_choice == "4":
        min_dists = [m['min_distance'] for m in obstacle_metrics_log]
        activations = [m['activation'] for m in obstacle_metrics_log]
        scale_factors = [m['scale_factor'] for m in obstacle_metrics_log]
        closest_links = [m['closest_link_idx'] for m in obstacle_metrics_log]
        grad_norms = [m['grad_norm'] for m in obstacle_metrics_log]
        obs_data = list(zip(min_dists, activations, scale_factors, closest_links, grad_norms))
        save_csv("obstacle_metrics.csv", obs_data,
                 header=["min_distance", "activation", "scale_factor", 
                         "closest_link_idx", "grad_norm"])

    print("[SAVED] ✓ All CSV files saved\n")

    # ========================================
    # EXPERIMENT SUMMARY
    # ========================================
    summary_path = os.path.join(exp_folder, 'experiment_info.txt')
    errors = np.linalg.norm(traj_act_rtde - trajectory_target, axis=1)
    
    with open(summary_path, 'w') as f:
        f.write("╔═══════════════════════════════════════════╗\n")
        f.write("║        EXPERIMENT SUMMARY                ║\n")
        f.write("╚═══════════════════════════════════════════╝\n\n")
        
        f.write("=== ROBOT CONFIGURATION ===\n")
        f.write(f"Robot Source: {robot_source}\n")
        f.write(f"IP Address: {URSIM_IP}\n")
        f.write(f"Control Method: {control_method_name}\n")
        f.write(f"Control Frequency: {1/dt:.0f} Hz\n\n")
        
        f.write("=== EXPERIMENT DETAILS ===\n")
        f.write(f"Date: {timestamp.split('_')[0]}\n")
        f.write(f"Time: {timestamp.split('_')[1].replace('-', ':')}\n")
        f.write(f"Trajectory Type: {traj_name}\n")
        f.write(f"Nullspace Method: {nullspace_name}\n")
        f.write(f"Duration: {duration}s (actual: {total_time:.2f}s)\n\n")
        
        f.write("=== CONTROL PARAMETERS ===\n")
        f.write(f"dt = {dt}s ({1/dt:.0f} Hz)\n")
        f.write(f"Kp = {Kp}\n")
        f.write(f"K_null = {K_null}\n")
        f.write(f"Amplitude = {amplitude}m ({amplitude*1000:.0f}mm)\n")
        f.write(f"Frequency = {frequency_traj}Hz\n")
        f.write(f"Max joint velocity = {MAX_JOINT_VEL} rad/s\n\n")
        
        f.write("=== CONTROL METHOD PERFORMANCE ===\n")
        f.write(f"Control Method: {control_method_name}\n")
        f.write(f"Mean control execution time: {np.mean(control_timing_arr)*1000:.3f}ms\n")
        f.write(f"Max control execution time: {np.max(control_timing_arr)*1000:.3f}ms\n")
        f.write(f"Min control execution time: {np.min(control_timing_arr)*1000:.3f}ms\n")
        f.write(f"Std control execution time: {np.std(control_timing_arr)*1000:.3f}ms\n\n")
        
        if nullspace_choice == '4':
            f.write("=== OBSTACLE AVOIDANCE PARAMETERS ===\n")
            f.write(f"Obstacle Position: {obstacle_pos}\n")
            f.write(f"Activation Distance: {ACTIVATION_DISTANCE}m\n")
            f.write(f"Influence Distance: {INFLUENCE_DISTANCE}m\n")
            min_dists = [m['min_distance'] for m in obstacle_metrics_log]
            f.write(f"Min distance to obstacle: {np.min(min_dists)*1000:.2f}mm\n")
            f.write(f"Mean distance to obstacle: {np.mean(min_dists)*1000:.2f}mm\n\n")
        
        f.write("=== TRACKING PERFORMANCE ===\n")
        f.write(f"Mean error: {np.mean(errors)*1000:.2f}mm\n")
        f.write(f"Max error: {np.max(errors)*1000:.2f}mm\n")
        f.write(f"Min error: {np.min(errors)*1000:.2f}mm\n")
        f.write(f"Std error: {np.std(errors)*1000:.2f}mm\n")
        f.write(f"RMS error: {np.sqrt(np.mean(errors**2))*1000:.2f}mm\n\n")
        
        f.write("=== FOLDER STRUCTURE ===\n")
        f.write(f"Root: experiments/{timestamp.split('_')[0]}/\n")
        f.write(f"Folder: {os.path.basename(exp_folder)}/\n")
    
    print(f"[SAVED] ✓ Experiment summary: experiment_info.txt\n")
    
    # ========================================
    # GENERATE AND SAVE PLOTS
    # ========================================
    print("[PLOTTING] Generating plots...")
    
    # Plot 1: Trajectory
    if choice == "1":
        plt.figure(figsize=(8, 7))
        plt.scatter(trajectory_target[:, 0], trajectory_target[:, 1],
                    color='blue', label='Target', s=10, alpha=0.6)
        plt.scatter(traj_act_rtde[:, 0], traj_act_rtde[:, 1],
                    color='red', label=f'Actual ({control_method_name})', s=10, alpha=0.6)
        if nullspace_choice == '4':
            plt.scatter(obstacle_pos[0], obstacle_pos[1], 
                       color='black', marker='X', s=200, label='Obstacle', zorder=5)
            circle = plt.Circle((obstacle_pos[0], obstacle_pos[1]), INFLUENCE_DISTANCE,
                               color='yellow', fill=False, linestyle='--', linewidth=2,
                               label='Influence Zone')
            plt.gca().add_patch(circle)
        plt.title(f'XY Trajectory Tracking - {control_method_name} - {robot_source}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '01_trajectory_tracking.png'), dpi=150)
        plt.close()

    if choice == "2":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory_target[:, 0], trajectory_target[:, 1], trajectory_target[:, 2], 
                'b--', label='Target', linewidth=2)
        ax.plot(traj_act_rtde[:, 0], traj_act_rtde[:, 1], traj_act_rtde[:, 2], 
                'r-', label=f'Actual ({control_method_name})', linewidth=1.5, alpha=0.8)
        if nullspace_choice == '4':
            ax.scatter(obstacle_pos[0], obstacle_pos[1], obstacle_pos[2],
                      color='black', marker='X', s=200, label='Obstacle')
        ax.set_title(f'3D Trajectory Tracking - {control_method_name} - {robot_source}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        set_axes_equal_3d(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '01_trajectory_tracking_3d.png'), dpi=150)
        plt.close()

    # Plot 2: Tracking Error
    errors_mm = np.linalg.norm(traj_act_rtde - trajectory_target, axis=1) * 1000
    plt.figure(figsize=(10, 5))
    plt.plot(t_log, errors_mm, linewidth=1.5, color='red')
    plt.axhline(y=np.mean(errors_mm), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(errors_mm):.2f}mm', linewidth=2)
    plt.title(f'Tracking Error over Time - {control_method_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (mm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '02_tracking_error.png'), dpi=150)
    plt.close()

    # NEW: Plot 2.1: Control Method Execution Time
    plt.figure(figsize=(10, 5))
    plt.plot(t_log, control_timing_arr*1000, linewidth=1.2, color='purple', alpha=0.7)
    plt.axhline(y=np.mean(control_timing_arr)*1000, color='red', linestyle='--', 
                label=f'Mean: {np.mean(control_timing_arr)*1000:.3f}ms', linewidth=2)
    plt.title(f'Control Execution Time - {control_method_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '02.1_control_execution_time.png'), dpi=150)
    plt.close()

    # Plot 3: Nullspace norm
    plt.figure(figsize=(9, 4))
    plt.plot(t_log, dq_ns_norm_log, label='|| dq_ns ||', linewidth=1.5, color='purple')
    plt.title('Nullspace Component Magnitude', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Norm (rad/s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '03_nullspace_norm.png'), dpi=150)
    plt.close()

    # Plot 4: Nullspace-specific metrics
    if nullspace_choice == '2':
        plt.figure(figsize=(9, 4))
        plt.plot(t_log, manip_log, linewidth=1.5, color='green')
        plt.title('Manipulability over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Manipulability')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '04_manipulability.png'), dpi=150)
        plt.close()
    
    elif nullspace_choice == '3':
        plt.figure(figsize=(9, 4))
        plt.plot(t_log, dist_limits_log, linewidth=1.5, color='orange')
        plt.axhline(y=0.3, color='r', linestyle='--', label='Warning threshold', linewidth=2)
        plt.title('Distance to Joint Limits', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Min normalized distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '04_joint_limits_distance.png'), dpi=150)
        plt.close()
    
    elif nullspace_choice == '4':
        # Obstacle Avoidance Metrics Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Obstacle Avoidance Metrics', fontsize=14, fontweight='bold')
        
        # Distance to obstacle
        min_dists = np.array([m['min_distance'] for m in obstacle_metrics_log]) * 1000
        ax1.plot(t_log, min_dists, linewidth=1.5, color='red')
        ax1.axhline(y=ACTIVATION_DISTANCE*1000, color='orange', linestyle='--', 
                   label=f'Activation: {ACTIVATION_DISTANCE*1000:.0f}mm', linewidth=2)
        ax1.axhline(y=INFLUENCE_DISTANCE*1000, color='yellow', linestyle='--', 
                   label=f'Influence: {INFLUENCE_DISTANCE*1000:.0f}mm', linewidth=2)
        ax1.set_ylabel('Distance (mm)')
        ax1.set_title('Minimum Distance to Obstacle')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Activation level
        activations = [m['activation'] for m in obstacle_metrics_log]
        ax2.plot(t_log, activations, linewidth=1.5, color='blue')
        ax2.fill_between(t_log, 0, activations, alpha=0.3, color='blue')
        ax2.set_ylabel('Activation (0-1)')
        ax2.set_title('Avoidance Activation Level')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        
        # Scale factor
        scale_factors = [m['scale_factor'] for m in obstacle_metrics_log]
        ax3.plot(t_log, scale_factors, linewidth=1.5, color='green')
        ax3.set_ylabel('Scale Factor')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Exponential Scale Factor')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, '04_obstacle_avoidance_metrics.png'), dpi=150)
        plt.close()

    # Plot 5: Joint velocities (actual)
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(6):
        plt.plot(t_log, dq_act_log[:, i], label=f'Joint {i+1}', 
                linewidth=1.2, color=colors[i])
    plt.title("Actual Joint Velocities", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '05_joint_velocities_actual.png'), dpi=150)
    plt.close()

    # Plot 6: Joint velocities (commanded)
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.plot(t_log, dq_cmd_log[:, i], label=f'Joint {i+1}', 
                linewidth=1.2, color=colors[i])
    plt.title("Commanded Joint Velocities", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '06_joint_velocities_commanded.png'), dpi=150)
    plt.close()

    # Plot 7: Joint positions
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.plot(t_log, np.rad2deg(q_log[:, i]), label=f'Joint {i+1}', 
                linewidth=1.2, color=colors[i])
    plt.title("Joint Positions", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '07_joint_positions.png'), dpi=150)
    plt.close()

    # Plot 8: Commanded vs Actual Velocity Comparison
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Commanded vs Actual Joint Velocities', fontsize=14, fontweight='bold')
    for i in range(6):
        ax = axes[i//2, i%2]
        ax.plot(t_log, dq_cmd_log[:, i], 'b-', label='Commanded', linewidth=1.2, alpha=0.7)
        ax.plot(t_log, dq_act_log[:, i], 'r-', label='Actual', linewidth=1.0)
        ax.set_title(f'Joint {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '08_velocity_comparison.png'), dpi=150)
    plt.close()

    # Plot 9: Error components (X, Y, Z)
    error_xyz = traj_act_rtde - trajectory_target
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle('Position Error Components', fontsize=14, fontweight='bold')
    
    ax1.plot(t_log, error_xyz[:, 0]*1000, linewidth=1.5, color='red')
    ax1.set_ylabel('X Error (mm)')
    ax1.set_title('X-axis Error')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    ax2.plot(t_log, error_xyz[:, 1]*1000, linewidth=1.5, color='green')
    ax2.set_ylabel('Y Error (mm)')
    ax2.set_title('Y-axis Error')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    ax3.plot(t_log, error_xyz[:, 2]*1000, linewidth=1.5, color='blue')
    ax3.set_ylabel('Z Error (mm)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Z-axis Error')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, '09_error_components.png'), dpi=150)
    plt.close()

    # Plot 10: Orientation comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('End-Effector Orientation', fontsize=14, fontweight='bold')
    
    ax1.plot(t_log, rpy_log[:, 0], 'r-', label='Roll', linewidth=1.5)
    ax1.plot(t_log, rpy_log[:, 1], 'g-', label='Pitch', linewidth=1.5)
    ax1.plot(t_log, rpy_log[:, 2], 'b-', label='Yaw', linewidth=1.5)
    ax1.set_ylabel('Angle (deg)')
    ax1.set_title('From Forward Kinematics (RPY)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_log, rpy_rtde_log[:, 0], 'r-', label='Rx', linewidth=1.5)
    ax2.plot(t_log, rpy_rtde_log[:, 1], 'g-', label='Ry', linewidth=1.5)
    ax2.plot(t_log, rpy_rtde_log[:, 2], 'b-', label='Rz', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (deg)')
    ax2.set_title('From RTDE (Axis-Angle)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Rest of plots continue... (Plot 3-10 remain the same)
    
    # Plot 3-10: [Keep all existing plots from original code]
    # ... (I'll add them in abbreviated form to save space)
    
    # [Add all plots 3-10 from original code here - nullspace, velocities, positions, etc.]
    
    # TWIN COMPARISON PLOTS (from original code)
    if len(twin_pos_log) > 0:
        twin_q_arr = np.array(twin_q_log)
        twin_qd_arr = np.array(twin_qd_log)
        twin_pos_arr = np.array(twin_pos_log)

        n = min(len(t_log), len(twin_pos_arr), len(traj_act_rtde))
        t_plot = t_log[:n]

        rtde_q_arr = q_log[:n]
        rtde_qd_arr = dq_act_log[:n]
        rtde_pos_arr = traj_act_rtde[:n]
        target_arr = trajectory_target[:n]

        twin_q_arr = twin_q_arr[:n]
        twin_qd_arr = twin_qd_arr[:n]
        twin_pos_arr = twin_pos_arr[:n]

        # Plot A: joint position comparison
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Joint Position: RTDE vs Twin ({control_method_name})')
        for i in range(6):
            ax = axes[i//2, i%2]
            ax.plot(t_plot, np.rad2deg(rtde_q_arr[:, i]), 'b-', label='RTDE', linewidth=1.2)
            ax.plot(t_plot, np.rad2deg(twin_q_arr[:, i]), 'g--', label='Twin', linewidth=1.0, alpha=0.8)
            ax.set_title(f'Joint {i+1}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (deg)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, 'twin_joint_pos_comparison.png'), dpi=150)
        plt.close()

        # Plot B: joint velocity comparison
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Joint Velocity: RTDE vs Twin ({control_method_name})')
        for i in range(6):
            ax = axes[i//2, i%2]
            ax.plot(t_plot, rtde_qd_arr[:, i], 'b-', label='RTDE', linewidth=1.2)
            ax.plot(t_plot, twin_qd_arr[:, i], 'g--', label='Twin', linewidth=1.0, alpha=0.8)
            ax.set_title(f'Joint {i+1}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (rad/s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, 'twin_joint_vel_comparison.png'), dpi=150)
        plt.close()

        # Plot C: End-effector position comparison
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
        fig.suptitle(f'End-Effector Position: RTDE vs Twin ({control_method_name})')
        ax1.plot(t_plot, rtde_pos_arr[:,0], 'r-', label='RTDE X')
        ax1.plot(t_plot, twin_pos_arr[:,0], 'g--', label='Twin X')
        ax1.set_ylabel('X (m)'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(t_plot, rtde_pos_arr[:,1], 'r-', label='RTDE Y')
        ax2.plot(t_plot, twin_pos_arr[:,1], 'g--', label='Twin Y')
        ax2.set_ylabel('Y (m)'); ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3.plot(t_plot, rtde_pos_arr[:,2], 'r-', label='RTDE Z')
        ax3.plot(t_plot, twin_pos_arr[:,2], 'g--', label='Twin Z')
        ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Z (m)'); ax3.legend(); ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, 'twin_ee_pos_comparison.png'), dpi=150)
        plt.close()

        # Plot D: RTDE vs Twin position error
        pos_error_rtde_twin = np.linalg.norm(rtde_pos_arr - twin_pos_arr, axis=1) * 1000.0
        plt.figure(figsize=(10,5))
        plt.plot(t_plot, pos_error_rtde_twin, linewidth=1.2)
        plt.title(f'RTDE vs Twin: EE Position Difference - {control_method_name}')
        plt.xlabel('Time (s)'); plt.ylabel('Position diff (mm)')
        plt.grid(True, alpha=0.3)
        plt.axhline(np.mean(pos_error_rtde_twin), linestyle='--', 
                   label=f'Mean = {np.mean(pos_error_rtde_twin):.2f} mm')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, 'twin_pos_difference_norm.png'), dpi=150)
        plt.close()

        # Save error stats
        with open(os.path.join(exp_folder, 'twin_comparison_stats.txt'), 'w') as fh:
            fh.write(f"RTDE vs Twin comparison stats - {control_method_name}\n")
            fh.write(f"Samples: {n}\n")
            fh.write(f"Mean EE position diff (mm): {np.mean(pos_error_rtde_twin):.3f}\n")
            fh.write(f"Max EE position diff (mm): {np.max(pos_error_rtde_twin):.3f}\n")
            fh.write(f"Std EE position diff (mm): {np.std(pos_error_rtde_twin):.3f}\n")

    else:
        print("[INFO] No PyBullet twin logs available for comparison plots.")

    print("[SAVED] ✓ All plots generated\n")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("=" * 60)
    print("EXPERIMENT SUMMARY:")
    print(f"  Control Method: {control_method_name}")
    print(f"  Trajectory: {traj_name}")
    print(f"  Nullspace: {nullspace_name}")
    print("\nTRACKING PERFORMANCE:")
    print(f"  Mean error:  {np.mean(errors_mm):.2f} mm")
    print(f"  Max error:   {np.max(errors_mm):.2f} mm")
    print(f"  Min error:   {np.min(errors_mm):.2f} mm")
    print(f"  Std error:   {np.std(errors_mm):.2f} mm")
    print(f"  RMS error:   {np.sqrt(np.mean(errors_mm**2)):.2f} mm")
    
    print(f"\nCONTROL METHOD PERFORMANCE:")
    print(f"  Method: {control_method_name}")
    print(f"  Mean execution time: {np.mean(control_timing_arr)*1000:.3f} ms")
    print(f"  Max execution time:  {np.max(control_timing_arr)*1000:.3f} ms")
    print(f"  Min execution time:  {np.min(control_timing_arr)*1000:.3f} ms")
    print(f"  Std execution time:  {np.std(control_timing_arr)*1000:.3f} ms")
    
    if nullspace_choice == '4':
        min_dists_m = [m['min_distance'] for m in obstacle_metrics_log]
        print("\nOBSTACLE AVOIDANCE PERFORMANCE:")
        print(f"  Min distance to obstacle: {np.min(min_dists_m)*1000:.2f} mm")
        print(f"  Mean distance to obstacle: {np.mean(min_dists_m)*1000:.2f} mm")
        print(f"  Obstacle position: {obstacle_pos}")
    
    print(f"\nTIMING ANALYSIS:")
    print(f"  Total execution time: {total_time:.2f}s")
    print(f"  Data points collected: {len(t_log)}")
    print(f"  Average cycle time: {total_time/len(t_log)*1000:.2f}ms")
    print("=" * 60)
    
    # Disconnect
    rtde_c.disconnect()
    rtde_r.disconnect()
    print("\n[RTDE] ✓ Disconnected from robot\n")

    if p.isConnected():
        p.disconnect()
    
    print("=" * 60)
    print("All data saved successfully!")
    print(f"Check folder: {exp_folder}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Program terminated by user")
    except Exception as e:
        print(f"\n\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()