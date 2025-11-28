import numpy as np
from typing import Tuple, Dict, List, Optional, Callable, Union
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
import time
import pybullet as p
import argparse
import json
from datetime import datetime

from oscbf.core.manipulator import Manipulator, load_mycobot, load_ur5e
from oscbf.core.manipulation_env import MyCobotTorqueControlEnv
from oscbf.core.controllers import PoseTaskTorqueController
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.utils.trajectory import SinusoidalTaskTrajectory
from cbfpy import CBF
import jax
import jax.numpy as jnp

@dataclass
class PDGains:
    """Store PD gains for task and joint space"""
    kp_pos: float
    kd_pos: float
    kp_rot: float
    kd_rot: float
    kp_joint: np.ndarray
    kd_joint: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert gains to dictionary for saving"""
        return {
            'kp_pos': float(self.kp_pos),
            'kd_pos': float(self.kd_pos),
            'kp_rot': float(self.kp_rot),
            'kd_rot': float(self.kd_rot),
            'kp_joint': self.kp_joint.tolist(),
            'kd_joint': self.kd_joint.tolist()
        }

@dataclass
class ObstacleConfig:
    """Configuration for obstacles in the workspace"""
    position: np.ndarray  # Center position [x, y, z]
    radius: float  # Obstacle radius
    type: str = "sphere"  # "sphere", "cylinder", "box"
    dimensions: Optional[np.ndarray] = None  # For non-spherical obstacles

@dataclass
class MyCobotConfig:
    """Enhanced configuration for MyCobot with obstacle settings"""
    n_joints: int = 6
    control_freq: float = 240.0
    
    # Joint limits
    joint_velocity_limits: np.ndarray = field(default_factory=lambda: np.array([2.5]*6))
    joint_torque_limits: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.15]))
    
    # CBF parameters - Enhanced for collision avoidance
    cbf_alpha_obstacle: float = 15.0  # Increased for better obstacle avoidance
    cbf_alpha_joint: float = 10.0
    cbf_alpha_singularity: float = 8.0
    cbf_safety_margin: float = 0.05
    obstacle_safety_margin: float = 0.08  # 8cm safety margin for obstacles
    singularity_tol: float = 1e-4
    
    # Workspace limits
    workspace_radius: float = 0.28
    
    # Obstacles in workspace
    obstacles: List[ObstacleConfig] = field(default_factory=list)
    
    # Link dimensions for collision checking (MyCobot specific)
    link_radii: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.015, 0.015, 0.01, 0.01]))

@dataclass
class UR5eConfig:
    """Configuration specific to UR5e simulation"""
    n_joints: int = 6
    control_freq: float = 240.0 

    # UR5e Joint limits (CHECK THESE VALUES - examples below)
    joint_velocity_limits: np.ndarray = field(default_factory=lambda: np.array([np.pi]*6)) # ~3.14 rad/s
    # Example Torque Limits (Nm) - VERIFY THESE for UR5e
    joint_torque_limits: np.ndarray = field(default_factory=lambda: np.array([150, 150, 150, 28, 28, 28]))

    # CBF parameters (Can start similar to MyCobot, might need tuning)
    cbf_alpha_obstacle: float = 15.0
    cbf_alpha_joint: float = 10.0
    cbf_alpha_singularity: float = 8.0
    cbf_safety_margin: float = 0.05
    obstacle_safety_margin: float = 0.08 # Safety margin for obstacles
    singularity_tol: float = 1e-4

    # Workspace & Obstacles (Adjust for UR5e's larger workspace)
    obstacles: List[ObstacleConfig] = field(default_factory=lambda: [
        ObstacleConfig(position=np.array([0.5, 0.2, 0.4]), radius=0.1),
        ObstacleConfig(position=np.array([0.6, -0.2, 0.5]), radius=0.08),
    ])

    # Link radii for simplified collision check (NEEDS UR5e SPECIFIC VALUES)
    # Placeholder - You MUST define approximate radii for UR5e links
    link_radii: np.ndarray = field(default_factory=lambda: np.array([0.06, 0.05, 0.05, 0.04, 0.04, 0.03]))

@dataclass
class TuningConfig:
    """Enhanced tuning configuration with collision-aware metrics"""
    # Search bounds - refined for MyCobot
    kp_pos_range: Tuple[float, float] = (20.0, 150.0)
    kd_pos_range: Tuple[float, float] = (2.0, 25.0)
    kp_rot_range: Tuple[float, float] = (15.0, 100.0)
    kd_rot_range: Tuple[float, float] = (1.5, 12.0)
    kp_joint_range: Tuple[float, float] = (5.0, 50.0)
    kd_joint_range: Tuple[float, float] = (0.5, 6.0)
    
    # Performance weights - adjusted for collision avoidance
    weight_error: float = 1.0
    weight_smoothness: float = 0.4
    weight_overshoot: float = 0.15
    weight_settling_time: float = 0.1
    weight_cbf_violation: float = 3.0  # Increased for safety
    weight_collision_proximity: float = 2.5  # New: penalize near-collisions
    weight_energy: float = 0.2  # Energy efficiency
    
    # Test parameters
    test_duration: float = 10.0
    n_test_points: int = 2400
    
    # Tuning strategy
    use_adaptive_weights: bool = True  # Adapt weights based on performance
    early_stopping_patience: int = 5  # Stop if no improvement
    convergence_tol: float = 0.0001

@dataclass
class UR5eTuningConfig(TuningConfig): # Inherit default weights, etc.
    """Tuning specific bounds for UR5e"""
    # Search bounds - Adjust for UR5e (Likely higher gains possible)
    kp_pos_range: Tuple[float, float] = (50.0, 300.0) # Higher kp feasible
    kd_pos_range: Tuple[float, float] = (10.0, 60.0) # Higher kd needed
    kp_rot_range: Tuple[float, float] = (40.0, 200.0)
    kd_rot_range: Tuple[float, float] = (5.0, 30.0)
    kp_joint_range: Tuple[float, float] = (10.0, 100.0) # Nullspace gains
    kd_joint_range: Tuple[float, float] = (1.0, 20.0)

    # Performance weights might need adjustment based on UR5e behavior
    # weight_smoothness: float = 0.3 # Maybe less critical than MyCobot
    # weight_energy: float = 0.1 # UR5e might be more robust

    test_duration: float = 10.0
    n_test_points: int = int(test_duration * UR5eConfig.control_freq) # Use UR5e freq

class EnhancedPDTuner:
    """
    Enhanced PD tuner with comprehensive collision avoidance and CBF integration
    """
    
    def __init__(self, 
                 robot_config: Union[MyCobotConfig, UR5eConfig] = None,
                 tuning_config: Union[TuningConfig, UR5eTuningConfig] = None,
                 simulation_func: Optional[Callable] = None,
                 verbose: bool = True):
        
        if robot_config is None:
            self.robot_config = UR5eConfig()
        else:
            self.robot_config = robot_config

        if tuning_config is None:
            self.tuning_config = UR5eTuningConfig() if isinstance(self.robot_config, UR5eConfig) else TuningConfig()
        else:
            self.tuning_config = tuning_config
        
        self.simulation_func = simulation_func
        self.verbose = verbose
        
        # Add default obstacles if none specified
        if not self.robot_config.obstacles:
            self.robot_config.obstacles = [
                ObstacleConfig(position=np.array([0.1, 0.1, 0.15]), radius=0.05),
                ObstacleConfig(position=np.array([0.2, -0.1, 0.2]), radius=0.04),
            ]
        
        try:
            if isinstance(self.robot_config, UR5eConfig):
                print("Memuat model UR5e Manipulator...")
                self.robot_model = load_ur5e() # Panggil fungsi load_ur5e()
            else: # Asumsi MyCobot
                print("Memuat model MyCobot Manipulator...")
                self.robot_model = load_mycobot()
            print("Model Manipulator berhasil dimuat.")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model Manipulator: {e}")
            print("       Pastikan path URDF benar dan file model kolisi ada.")
            self.robot_model = None # Tandai bahwa model gagal dimuat

        # Initialize best gains with physically motivated values
        self.best_gains = self._initialize_gains()
        
        # Performance tracking
        self.tuning_history = []
        self.best_performance = float('inf')
        self.iteration_count = 0
        self.no_improvement_count = 0
        
    def _initialize_gains(self) -> PDGains:
        """Initialize gains using physics-based heuristics"""
        # Estimate based on robot mass and workspace
        estimated_mass = 1.0  # kg (approximate for MyCobot)
        natural_freq = 2 * np.pi * 1.0  # 1 Hz natural frequency
        damping_ratio = 0.7  # Slightly underdamped
        
        kp_pos = estimated_mass * natural_freq**2
        kd_pos = 2 * damping_ratio * estimated_mass * natural_freq
        
        return PDGains(
            kp_pos=np.clip(kp_pos * 10, *self.tuning_config.kp_pos_range),
            kd_pos=np.clip(kd_pos * 10, *self.tuning_config.kd_pos_range),
            kp_rot=np.clip(kp_pos * 5, *self.tuning_config.kp_rot_range),
            kd_rot=np.clip(kd_pos * 5, *self.tuning_config.kd_rot_range),
            kp_joint=np.clip(kp_pos * 2, *self.tuning_config.kp_joint_range) * np.ones(self.robot_config.n_joints),
            kd_joint=np.clip(kd_pos * 2, *self.tuning_config.kd_joint_range) * np.ones(self.robot_config.n_joints)
        )
    
    def simulate_with_full_cbf(self, gains: PDGains, 
                               target_trajectory: np.ndarray,
                               initial_state: Dict = None) -> Dict:
        """
        Simulasi internal MENGGUNAKAN kelas Manipulator untuk fisika yang akurat.
        """
        if self.robot_model is None:
             print("[ERROR] Model Manipulator tidak dimuat. Tidak dapat menjalankan simulasi internal.")
             # Kembalikan hasil dummy dengan cost sangat tinggi
             dummy_len = self.tuning_config.n_test_points
             return {'errors': np.full(dummy_len, 1e6), 'torques': np.zeros((dummy_len, self.robot_config.n_joints)), 
                     'velocities': np.zeros((dummy_len, self.robot_config.n_joints)), 'accelerations': np.zeros(dummy_len),
                     'avg_error': 1e6, 'max_error': 1e6, 'min_obstacle_distance': 0.0, 
                     'avg_cbf_violation': 1e6, 'total_energy': 1e6, 'error_std': 0.0, 'jerk_values': np.zeros(dummy_len-1)}


        dt = 1.0 / self.robot_config.control_freq
        n_steps = min(len(target_trajectory), self.tuning_config.n_test_points) # Pastikan tidak melebihi n_test_points

        # --- Inisialisasi State (Gunakan JAX Array) ---
        if initial_state is None:
            q = jnp.zeros(self.robot_config.n_joints)
            qdot = jnp.zeros(self.robot_config.n_joints)
        else:
            q = jnp.array(initial_state.get('q', jnp.zeros(self.robot_config.n_joints)))
            qdot = jnp.array(initial_state.get('qdot', jnp.zeros(self.robot_config.n_joints)))
        
        # --- Setup Controller (Gunakan implementasi OSCBF asli) ---
        # Asumsi Task Type Pose untuk tuning internal
        osc_controller = PoseTaskTorqueController(
            n_joints=self.robot_model.num_joints,
            kp_task=np.concatenate([gains.kp_pos * np.ones(3), gains.kp_rot * np.ones(3)]),
            kd_task=np.concatenate([gains.kd_pos * np.ones(3), gains.kd_rot * np.ones(3)]),
            kp_joint=gains.kp_joint,
            kd_joint=gains.kd_joint,
            tau_min=-self.robot_config.joint_torque_limits, # Gunakan batas torsi
            tau_max=self.robot_config.joint_torque_limits
        )

        # --- Setup CBF (Hanya obstacle & singularity untuk internal) ---
        # Kita perlu kelas CBF Config sederhana di sini
        @jax.tree_util.register_static
        class InternalSimCBFConfig(OSCBFTorqueConfig):
            def __init__(self, robot, obstacles, obstacle_radius, singularity_tol, initial_obstacle_pos_list):
                 self.obstacles = obstacles # List ObstacleConfig
                 self.obstacle_radius = obstacle_radius # Asumsi semua rintangan sama
                 self.singularity_tol = singularity_tol
                 # init_args harus tuple berisi semua posisi rintangan awal
                 init_args_tuple = tuple(tuple(map(float, pos)) for pos in initial_obstacle_pos_list)
                 super().__init__(robot, init_args=init_args_tuple) # Kirim tuple of tuples

            def h_2(self, z, *obstacle_positions_tuple, **kwargs):
                 q = z[:self.robot.num_joints]
                 h_all = []
                 # Hitung CBF untuk setiap rintangan
                 for i, obs_config in enumerate(self.obstacles):
                      # obstacle_pos diambil dari *obstacle_positions_tuple sesuai urutan
                      obstacle_pos = jnp.asarray(obstacle_positions_tuple[i])
                      robot_coll = self.robot.link_collision_data(q)
                      if robot_coll.size > 0:
                           positions = robot_coll[:, :3]
                           radii = robot_coll[:, 3]
                           deltas = positions - obstacle_pos[None, :]
                           dists = jnp.linalg.norm(deltas, axis=1)
                           h_obs = dists - (radii + self.obstacle_radius) # Radius obstacle SAMA
                           h_all.append(jnp.min(h_obs))
                      else: h_all.append(1.0) # Aman jika tak ada collision data
                 
                 # Singularity
                 h_sing = self.robot.manipulability(q) - self.singularity_tol
                 h_all.append(h_sing)

                 # Joint Limits (jika perlu) - bisa ditambahkan
                 
                 return jnp.array(h_all) # Kembalikan array 1D berisi semua nilai h min

        initial_obs_pos_list = [obs.position for obs in self.robot_config.obstacles]
        sim_cbf_config = InternalSimCBFConfig(self.robot_model, 
                                            self.robot_config.obstacles, 
                                            0.05, # Asumsi radius obstacle sama (sesuaikan jika perlu)
                                            self.robot_config.singularity_tol,
                                            initial_obs_pos_list)
        sim_cbf = CBF.from_config(sim_cbf_config)

        # --- JIT Compile Fungsi Kontrol ---
        @jax.jit
        def compute_internal_control(z_state, z_ee_target, *obstacle_pos_args):
            q_j, qdot_j = z_state[:self.robot_model.num_joints], z_state[self.robot_model.num_joints:]
            M, M_inv, g, c, J, ee_tmat = self.robot_model.torque_control_matrices(q_j, qdot_j)
            
            # Target EE (asumsi hanya posisi relevan untuk traj 3D)
            target_ee_pos = z_ee_target[:3]
            # Untuk Pose controller, kita perlu target rotasi dummy
            target_ee_rot = jnp.eye(3) 
            target_ee_vel = jnp.zeros(3) # Asumsi target diam sesaat
            target_ee_omg = jnp.zeros(3)

            u_nom_j = osc_controller(q_j, qdot_j, ee_tmat[:3,3], ee_tmat[:3,:3], 
                                   target_ee_pos, target_ee_rot, 
                                   target_ee_vel, target_ee_omg, 
                                   jnp.zeros(3), jnp.zeros(3), q_j, jnp.zeros_like(q_j), # Nullspace target = q saat ini
                                   J, M, M_inv, g, c)
            
            tau_safe_j = sim_cbf.safety_filter(z_state, u_nom_j, *obstacle_pos_args)
            qddot_safe_j = M_inv @ (tau_safe_j - c - g)
            return tau_safe_j, qddot_safe_j

        # --- Loop Simulasi ---
        errors, velocities, accelerations, torques_log = [], [], [], []
        collision_proximities = []
        energy_consumption = []
        jerk_values = []
        qddot_prev = jnp.zeros_like(qdot)

        for i in range(n_steps):
            z = jnp.concatenate([q, qdot])
            target_pos = jnp.array(target_trajectory[i, :3]) # Ambil target posisi
            
            # Dapatkan posisi rintangan saat ini (asumsi statis untuk simulasi internal)
            current_obs_pos_list = initial_obs_pos_list # Gunakan posisi awal
            current_obs_pos_tuple = tuple(current_obs_pos_list) # Ubah ke tuple untuk JIT

            # Hitung kontrol dan percepatan
            try:
                # Perlu unpack tuple posisi rintangan saat memanggil fungsi JIT
                tau_safe, qddot_safe = compute_internal_control(z, target_pos, *current_obs_pos_tuple)
            except Exception as e:
                 print(f"[ERROR] Gagal menghitung kontrol di iterasi {i}: {e}")
                 # Hentikan simulasi jika kontrol gagal
                 break

            # Update state menggunakan integrasi Euler
            qdot_next = qdot + qddot_safe * dt
            q_next = q + qdot_next * dt # Gunakan qdot_next untuk semi-implicit Euler

            # Hitung metrik SEBELUM update state
            current_ee_pos = self.robot_model.ee_position(q)
            error_norm = jnp.linalg.norm(current_ee_pos - target_pos)
            errors.append(float(error_norm))
            velocities.append(float(jnp.linalg.norm(qdot)))
            accelerations.append(float(jnp.linalg.norm(qddot_safe)))
            torques_log.append(np.array(tau_safe)) # Simpan torsi sebagai numpy
            
            # Hitung jarak minimum ke rintangan
            min_dist_to_obs = float('inf')
            robot_coll_data = self.robot_model.link_collision_data(q)
            if robot_coll_data.size > 0:
                 robot_positions = robot_coll_data[:,:3]
                 robot_radii = robot_coll_data[:,3]
                 for obs_idx, obs_pos in enumerate(current_obs_pos_list):
                      obs_pos_jnp = jnp.asarray(obs_pos)
                      deltas = robot_positions - obs_pos_jnp[None, :]
                      dists = jnp.linalg.norm(deltas, axis=1)
                      # Jarak aktual (pusat bola ke pusat obstacle) dikurangi radius robot
                      actual_distances = dists - robot_radii 
                      min_dist_to_obs = min(min_dist_to_obs, float(jnp.min(actual_distances)))

            collision_proximities.append(min_dist_to_obs)
            
            # Hitung Jerk
            jerk = jnp.linalg.norm(qddot_safe - qddot_prev) / dt
            jerk_values.append(float(jerk))
            
            # Hitung Energi
            energy = jnp.sum(jnp.abs(tau_safe * qdot)) * dt # P = τ ⋅ ω
            energy_consumption.append(float(energy))

            # Update state untuk iterasi berikutnya
            q, qdot = q_next, qdot_next
            qddot_prev = qddot_safe

        # Handle jika loop berhenti lebih awal
        num_actual_steps = len(errors)
        if num_actual_steps < n_steps:
             # Isi sisa data dengan nilai penalti
             remaining_steps = n_steps - num_actual_steps
             errors.extend([1e6] * remaining_steps)
             # ... isi sisa list lainnya dengan nilai penalti/nol ...


        # Kembalikan metrics (konversi ke NumPy array)
        metrics = {
            'errors': np.array(errors),
            'velocities': np.array(velocities),
            'accelerations': np.array(accelerations),
            'torques': np.array(torques_log), # Kembalikan torsi juga
            'cbf_violations': np.zeros(num_actual_steps), # Asumsi CBF selalu dipenuhi oleh safety_filter
            'collision_proximities': np.array(collision_proximities),
            'energy_consumption': np.array(energy_consumption),
            'jerk_values': np.array(jerk_values[1:]), # Jerk dimulai dari step ke-2
            'final_error': errors[-1] if errors else float('inf'),
            'avg_error': np.mean(errors) if errors else float('inf'),
            'max_error': np.max(errors) if errors else float('inf'),
            'error_std': np.std(errors) if errors else float('inf'),
            'avg_cbf_violation': 0.0,
            'min_obstacle_distance': np.min(collision_proximities) if collision_proximities else 0,
            'total_energy': np.sum(energy_consumption),
            'avg_jerk': np.mean(jerk_values[1:]) if len(jerk_values) > 1 else 0,
            # 'final_state': {'q': np.array(q), 'qdot': np.array(qdot)} # Opsional
        }
        return metrics
    
    def evaluate_collision_aware_performance(self, trajectory_data: Dict) -> float:

        """

        Enhanced performance evaluation with collision awareness

        """

        # Extract metrics

        errors = trajectory_data['errors']

        # cbf_violations = trajectory_data['cbf_violations']

        collision_proximities = trajectory_data['collision_proximities']

        energy = trajectory_data['total_energy']

        jerk = trajectory_data['avg_jerk']

        

        # Adaptive weight calculation

        if self.tuning_config.use_adaptive_weights:

            # Increase collision weight if we're getting too close to obstacles

            min_distance = trajectory_data['min_obstacle_distance']

            if min_distance < self.robot_config.obstacle_safety_margin * 2:

                collision_weight = self.tuning_config.weight_collision_proximity * 2

            else:

                collision_weight = self.tuning_config.weight_collision_proximity

        else:

            collision_weight = self.tuning_config.weight_collision_proximity

        

        # Calculate collision penalty (exponential increase as we get closer)

        collision_penalty = 0

        for dist in collision_proximities:

            if dist < self.robot_config.obstacle_safety_margin * 3:

                collision_penalty += np.exp(-dist / self.robot_config.obstacle_safety_margin)

        collision_penalty /= len(collision_proximities)

        

        # Combined cost function

        cost = (

            self.tuning_config.weight_error * trajectory_data['avg_error'] +

            self.tuning_config.weight_error * 0.2 * trajectory_data['error_std'] +  # Consistency

            self.tuning_config.weight_smoothness * jerk +

            # self.tuning_config.weight_cbf_violation * trajectory_data['avg_cbf_violation'] + # Not needed with new sim

            collision_weight * collision_penalty +

            self.tuning_config.weight_energy * energy / len(errors)

        )

        

        # Add heavy penalty for actual collisions

        if trajectory_data['min_obstacle_distance'] < self.robot_config.obstacle_safety_margin:

            cost *= 10  # Heavy penalty for violation

        

        return cost

    

    def generate_collision_test_trajectories(self) -> List[np.ndarray]:

        """

        Generate test trajectories that specifically test collision avoidance

        """

        trajectories = []

        t = np.linspace(0, self.tuning_config.test_duration, self.tuning_config.n_test_points)

        

        # Adjust trajectories for UR5e workspace

        # 1. Trajectory passing near first obstacle

        traj1 = np.zeros((len(t), 3))

        obstacle1 = self.robot_config.obstacles[0]

        # Create path that would collide without avoidance

        traj1[:, 0] = obstacle1.position[0] - 0.1 + 0.2 * np.sin(2 * np.pi * 0.3 * t)

        traj1[:, 1] = obstacle1.position[1] + 0.3 * np.cos(2 * np.pi * 0.3 * t)  # Pass near obstacle

        traj1[:, 2] = obstacle1.position[2] + 0.1 * np.sin(2 * np.pi * 0.2 * t)

        trajectories.append(traj1)

        

        # 2. Weaving between obstacles

        traj2 = np.zeros((len(t), 3))

        traj2[:, 0] = 0.4 + 0.2 * np.sin(2 * np.pi * 0.4 * t)

        traj2[:, 1] = 0.0 + 0.3 * np.sin(2 * np.pi * 0.3 * t)  # Weave motion

        traj2[:, 2] = 0.3 + 0.15 * np.cos(2 * np.pi * 0.25 * t)

        trajectories.append(traj2)

        

        # 3. Vertical motion near obstacle

        traj3 = np.zeros((len(t), 3))

        traj3[:, 0] = obstacle1.position[0] + 0.15  # Fixed X near obstacle

        traj3[:, 1] = obstacle1.position[1]  # Fixed Y at obstacle

        traj3[:, 2] = 0.2 + 0.3 * (1 + np.sin(2 * np.pi * 0.3 * t))  # Vertical motion

        trajectories.append(traj3)

        

        # 4. Safe trajectory (baseline)

        traj4 = np.zeros((len(t), 3))

        traj4[:, 0] = 0.3

        traj4[:, 1] = -0.3

        traj4[:, 2] = 0.5 + 0.05 * np.sin(2 * np.pi * 0.5 * t)

        trajectories.append(traj4)

        

        return trajectories

    

    def adaptive_bayesian_optimization(self, test_trajectories: List[np.ndarray],

                                      n_iterations: int = 50,

                                      n_initial_samples: int = 20) -> PDGains:

        """

        Enhanced Bayesian optimization with adaptive sampling

        """

        print("\n" + "="*60)

        print("Starting Adaptive Bayesian Optimization for Collision-Aware PD Tuning")

        print("="*60)

        

        # Store trajectories for objective function

        self._internal_test_trajectories = test_trajectories

        

        # Define search space

        bounds = [

            self.tuning_config.kp_pos_range,

            self.tuning_config.kd_pos_range,

            self.tuning_config.kp_rot_range,

            self.tuning_config.kd_rot_range,

            self.tuning_config.kp_joint_range,

            self.tuning_config.kd_joint_range

        ]

        

        # Initialize with Sobol sampling for better coverage

        from scipy.stats import qmc

        sampler = qmc.Sobol(d=6, scramble=True)

        initial_samples = sampler.random(n_initial_samples)

        

        # Scale samples to bounds

        for i, (low, high) in enumerate(bounds):

            initial_samples[:, i] = low + (high - low) * initial_samples[:, i]

        

        # Evaluate initial samples

        print(f"\nEvaluating {n_initial_samples} initial samples...")

        initial_results = []

        for i, sample in enumerate(initial_samples):

            gains = self._unpack_gains(sample)

            cost = self._comprehensive_objective(sample)

            initial_results.append((sample, cost))

            

            if self.verbose and i % 5 == 0:

                print(f"  Sample {i+1}/{n_initial_samples}: Cost = {cost:.4f}")

        

        # Sort by cost

        initial_results.sort(key=lambda x: x[1])

        best_initial = initial_results[0]

        print(f"\nBest initial sample: Cost = {best_initial[1]:.4f}")

        

        # Run differential evolution with warm start

        print("\nRunning differential evolution optimization with collision awareness...")



        result = differential_evolution(

            self._comprehensive_objective,

            bounds,

            maxiter=n_iterations,

            popsize=15,

            tol=self.tuning_config.convergence_tol,

            seed=42,

            disp=True,

            workers=1,  # Serial for stability with complex objective

            init=np.array([s[0] for s in initial_results[:15]]),  # Warm start

            callback=self._optimization_callback

        )

        

        print(f"\nOptimization complete! Final cost: {result.fun:.4f}")

        return self.best_gains

    

    def _unpack_gains(self, x: np.ndarray) -> PDGains:

        """Unpack optimization vector to gains"""

        return PDGains(

            kp_pos=x[0],

            kd_pos=x[1],

            kp_rot=x[2],

            kd_rot=x[3],

            kp_joint=x[4] * np.ones(self.robot_config.n_joints),

            kd_joint=x[5] * np.ones(self.robot_config.n_joints)

        )

    

    def _comprehensive_objective(self, x: np.ndarray) -> float:

        """

        Comprehensive objective function with collision awareness

        """

        gains = self._unpack_gains(x)

        

        total_cost = 0

        collision_events = 0

        

        # Evaluate on all test trajectories

        for traj_idx, trajectory_or_config in enumerate(self._internal_test_trajectories):

            if self.simulation_func is not None:

                # Use external simulation

                result = self.simulation_func(gains, trajectory_or_config)

            else:

                # Use internal simulation with full CBF

                result = self.simulate_with_full_cbf(gains, trajectory_or_config)

            

            # Calculate cost with collision awareness

            cost = self.evaluate_collision_aware_performance(result)

            total_cost += cost

            

            # Track collision events

            if result.get('min_obstacle_distance', float('inf')) < self.robot_config.obstacle_safety_margin:

                collision_events += 1

        

        avg_cost = total_cost / len(self._internal_test_trajectories)

        

        # Add penalty for collision events

        if collision_events > 0:

            avg_cost *= (1 + collision_events * 0.5)

        

        # Update tracking

        self.iteration_count += 1

        self.tuning_history.append({

            'iteration': self.iteration_count,

            'gains': gains,

            'avg_cost': avg_cost,

            'collision_events': collision_events

        })

        

        # Update best if improved

        if avg_cost < self.best_performance:

            self.best_performance = avg_cost

            self.best_gains = gains

            self.no_improvement_count = 0

            if self.verbose:

                print(f"  ✓ New best! Cost: {avg_cost:.4f}, Collisions: {collision_events}")

        else:

            self.no_improvement_count += 1

        

        return avg_cost

    

    def _optimization_callback(self, xk, convergence=0):

        """Callback for optimization progress"""

        if self.no_improvement_count >= self.tuning_config.early_stopping_patience:

            print(f"\nEarly stopping triggered after {self.tuning_config.early_stopping_patience} iterations without improvement")

            return True

        return False

    

    def validate_gains(self, gains: PDGains, validation_trajectories: List[np.ndarray]) -> Dict:

        """

        Validate tuned gains on new trajectories

        """

        print("\n" + "="*50)

        print("Validating Tuned Gains")

        print("="*50)

        

        validation_results = []

        

        for i, trajectory in enumerate(validation_trajectories):

            if self.simulation_func is not None:

                result = self.simulation_func(gains, trajectory)

            else:

                result = self.simulate_with_full_cbf(gains, trajectory)

            

            validation_results.append(result)

            

            print(f"\nTrajectory {i+1}:")

            print(f"  Avg Error: {result['avg_error']*1000:.2f} mm")

            print(f"  Max Error: {result['max_error']*1000:.2f} mm")

            print(f"  Min Obstacle Distance: {result['min_obstacle_distance']*100:.1f} cm")

            # print(f"  Avg CBF Violations: {result['avg_cbf_violation']:.4f}")

            print(f"  Energy Consumption: {result['total_energy']:.2f}")

        

        # Aggregate statistics

        avg_metrics = {

            'mean_error': np.mean([r['avg_error'] for r in validation_results]) * 1000,

            'max_error_overall': np.max([r['max_error'] for r in validation_results]) * 1000,

            'min_distance_overall': np.min([r['min_obstacle_distance'] for r in validation_results]) * 100,

            'collision_free': all(r['min_obstacle_distance'] > self.robot_config.obstacle_safety_margin 

                                 for r in validation_results),

            'avg_energy': np.mean([r['total_energy'] for r in validation_results])

        }

        

        print("\n" + "-"*50)

        print("Overall Validation Results:")

        print(f"  Mean Tracking Error: {avg_metrics['mean_error']:.2f} mm")

        print(f"  Max Tracking Error: {avg_metrics['max_error_overall']:.2f} mm")

        print(f"  Min Obstacle Distance: {avg_metrics['min_distance_overall']:.1f} cm")

        print(f"  Collision-Free: {'✓ Yes' if avg_metrics['collision_free'] else '✗ No'}")

        print(f"  Avg Energy: {avg_metrics['avg_energy']:.2f}")

        

        return avg_metrics

    

    def plot_enhanced_results(self):

        """

        Comprehensive visualization with collision metrics

        """

        if not self.tuning_history:

            print("No tuning history to plot")

            return

        

        fig = plt.figure(figsize=(16, 12))

        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        

        # Extract data

        iterations = range(len(self.tuning_history))

        costs = [h['avg_cost'] for h in self.tuning_history]

        collision_events = [h.get('collision_events', 0) for h in self.tuning_history]

        

        # 1. Cost and Collision Evolution

        ax1 = fig.add_subplot(gs[0, :])

        ax1.plot(iterations, costs, 'b-', linewidth=2, label='Cost Function')

        ax1.set_xlabel('Iteration')

        ax1.set_ylabel('Cost', color='b')

        ax1.tick_params(axis='y', labelcolor='b')

        ax1.grid(True, alpha=0.3)

        

        ax1_twin = ax1.twinx()

        ax1_twin.bar(iterations, collision_events, alpha=0.3, color='r', label='Collision Events')

        ax1_twin.set_ylabel('Collision Events', color='r')

        ax1_twin.tick_params(axis='y', labelcolor='r')

        

        ax1.set_title('Optimization Progress: Cost Reduction and Collision Avoidance')

        ax1.legend(loc='upper left')

        ax1_twin.legend(loc='upper right')

        

        # 2. Gain Evolution - Position

        ax2 = fig.add_subplot(gs[1, 0])

        kp_pos = [h['gains'].kp_pos for h in self.tuning_history]

        kd_pos = [h['gains'].kd_pos for h in self.tuning_history]

        ax2.plot(iterations, kp_pos, 'g-', linewidth=2, label='Kp_pos')

        ax2.plot(iterations, kd_pos, 'g--', linewidth=1.5, label='Kd_pos')

        ax2.set_xlabel('Iteration')

        ax2.set_ylabel('Gain Value')

        ax2.set_title('Position Gains Evolution')

        ax2.legend()

        ax2.grid(True, alpha=0.3)

        

        # 3. Gain Evolution - Rotation

        ax3 = fig.add_subplot(gs[1, 1])

        kp_rot = [h['gains'].kp_rot for h in self.tuning_history]

        kd_rot = [h['gains'].kd_rot for h in self.tuning_history]

        ax3.plot(iterations, kp_rot, 'b-', linewidth=2, label='Kp_rot')

        ax3.plot(iterations, kd_rot, 'b--', linewidth=1.5, label='Kd_rot')

        ax3.set_xlabel('Iteration')

        ax3.set_ylabel('Gain Value')

        ax3.set_title('Rotation Gains Evolution')

        ax3.legend()

        ax3.grid(True, alpha=0.3)

        

        # 4. Gain Evolution - Joint

        ax4 = fig.add_subplot(gs[1, 2])

        kp_joint = [h['gains'].kp_joint[0] for h in self.tuning_history]

        kd_joint = [h['gains'].kd_joint[0] for h in self.tuning_history]

        ax4.plot(iterations, kp_joint, 'm-', linewidth=2, label='Kp_joint')

        ax4.plot(iterations, kd_joint, 'm--', linewidth=1.5, label='Kd_joint')

        ax4.set_xlabel('Iteration')

        ax4.set_ylabel('Gain Value')

        ax4.set_title('Joint Gains Evolution')

        ax4.legend()

        ax4.grid(True, alpha=0.3)

        

        # 5. Final Gains Summary

        ax5 = fig.add_subplot(gs[2, :])

        final_gains = self.best_gains

        categories = ['Kp_pos', 'Kd_pos', 'Kp_rot', 'Kd_rot', 'Kp_joint', 'Kd_joint']

        values = [

            final_gains.kp_pos,

            final_gains.kd_pos,

            final_gains.kp_rot,

            final_gains.kd_rot,

            final_gains.kp_joint[0],

            final_gains.kd_joint[0]

        ]

        colors = ['green', 'lightgreen', 'blue', 'lightblue', 'magenta', 'pink']

        bars = ax5.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)

        ax5.set_ylabel('Gain Value')

        ax5.set_title('Final Optimized Gains')

        ax5.grid(True, alpha=0.3, axis='y')

        

        # Add value labels on bars

        for bar, val in zip(bars, values):

            height = bar.get_height()

            ax5.text(bar.get_x() + bar.get_width()/2., height,

                    f'{val:.2f}', ha='center', va='bottom')

        

        # 6. Convergence Analysis

        ax6 = fig.add_subplot(gs[3, 0])

        if len(costs) > 1:

            improvements = -np.diff(costs)

            ax6.plot(improvements, 'o-', markersize=4)

            ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)

            ax6.set_xlabel('Iteration')

            ax6.set_ylabel('Cost Improvement')

            ax6.set_title('Convergence Rate')

            ax6.grid(True, alpha=0.3)

        

        # 7. Collision Event Distribution

        ax7 = fig.add_subplot(gs[3, 1])

        if sum(collision_events) > 0:

            collision_iterations = [i for i, c in enumerate(collision_events) if c > 0]

            collision_counts = [c for c in collision_events if c > 0]

            ax7.scatter(collision_iterations, collision_counts, c='red', s=50, alpha=0.6)

            ax7.set_xlabel('Iteration')

            ax7.set_ylabel('Number of Collisions')

            ax7.set_title('Collision Events During Optimization')

            ax7.grid(True, alpha=0.3)

        else:

            ax7.text(0.5, 0.5, 'No Collision Events', ha='center', va='center', 

                    transform=ax7.transAxes, fontsize=12)

            ax7.set_title('Collision Events During Optimization')

        

        # 8. Parameter Space Exploration

        ax8 = fig.add_subplot(gs[3, 2])

        kp_pos_normalized = [(kp - self.tuning_config.kp_pos_range[0]) / 

                             (self.tuning_config.kp_pos_range[1] - self.tuning_config.kp_pos_range[0]) 

                             for kp in kp_pos]

        kd_pos_normalized = [(kd - self.tuning_config.kd_pos_range[0]) / 

                             (self.tuning_config.kd_pos_range[1] - self.tuning_config.kd_pos_range[0]) 

                             for kd in kd_pos]

        

        scatter = ax8.scatter(kp_pos_normalized, kd_pos_normalized, 

                             c=costs, cmap='viridis', s=20, alpha=0.6)

        ax8.scatter([kp_pos_normalized[-1]], [kd_pos_normalized[-1]], 

                   c='red', s=100, marker='*', label='Final')

        ax8.set_xlabel('Normalized Kp_pos')

        ax8.set_ylabel('Normalized Kd_pos')

        ax8.set_title('Parameter Space Exploration')

        plt.colorbar(scatter, ax=ax8, label='Cost')

        ax8.legend()

        ax8.grid(True, alpha=0.3)

        

        plt.suptitle('Enhanced PD Tuning Results with Collision Avoidance', fontsize=14, fontweight='bold')

        plt.tight_layout()

        plt.show()

    

    def save_results(self, filename: str = None):

        """Save tuning results to file"""

        if filename is None:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"ur5e_tuning_results_{timestamp}"

        

        # Save gains as NumPy array

        np.savez(f"{filename}.npz",

                kp_task=np.concatenate([

                    self.best_gains.kp_pos * np.ones(3),

                    self.best_gains.kp_rot * np.ones(3)

                ]),

                kd_task=np.concatenate([

                    self.best_gains.kd_pos * np.ones(3),

                    self.best_gains.kd_rot * np.ones(3)

                ]),

                kp_joint=self.best_gains.kp_joint,

                kd_joint=self.best_gains.kd_joint)

        

        # Save detailed results as JSON

        results = {

            'best_gains': self.best_gains.to_dict(),

            'best_performance': float(self.best_performance),

            'robot_config': {

                'n_joints': self.robot_config.n_joints,

                'control_freq': self.robot_config.control_freq,

                'cbf_alpha_obstacle': self.robot_config.cbf_alpha_obstacle,

                'cbf_alpha_joint': self.robot_config.cbf_alpha_joint,

                'obstacle_safety_margin': self.robot_config.obstacle_safety_margin

            },

            'tuning_config': {

                'weight_error': self.tuning_config.weight_error,

                'weight_smoothness': self.tuning_config.weight_smoothness,

                # 'weight_cbf_violation': self.tuning_config.weight_cbf_violation,

                'weight_collision_proximity': self.tuning_config.weight_collision_proximity

            },

            'optimization_iterations': self.iteration_count

        }

        

        with open(f"{filename}.json", 'w') as f:

            json.dump(results, f, indent=2)

        

        print(f"\nResults saved to {filename}.npz and {filename}.json")



# ==============================================================================

# BLOK UTAMA UNTUK MENJALANKAN ENHANCED AUTO-TUNER

# ==============================================================================

if __name__ == "__main__":

    

    # --- Setup Argumen Parser untuk CLI ---

    parser = argparse.ArgumentParser(description="Enhanced Auto-Tuning untuk Gain PD UR5e dengan Collision Avoidance.")

    parser.add_argument(

        '--use-pybullet', 

        action='store_true', 

        help="Gunakan simulasi PyBullet eksternal (belum diadaptasi sepenuhnya untuk UR5e)."

    )

    parser.add_argument(

        '--iterations',

        type=int,

        default=25,

        help="Jumlah iterasi untuk proses optimisasi."

    )

    args = parser.parse_args()



    # --- Langkah 1: Konfigurasi untuk UR5e ---

    print("1. Mengkonfigurasi lingkungan dan target tuning untuk UR5e...")



    ur5e_config = UR5eConfig() # <-- Gunakan config UR5e

    tuning_config = UR5eTuningConfig() # <-- Gunakan config tuning UR5e



    # --- Langkah 2: Inisialisasi Tuner Sesuai Mode ---

    print("\n2. Menginisialisasi Auto-Tuner...")

    

    tuner = None

    if args.use_pybullet:

        print("\n[ERROR] Mode PyBullet belum sepenuhnya diadaptasi untuk UR5e di contoh ini.")

        print("        Jalankan tanpa flag '--use-pybullet' untuk menggunakan simulasi internal.")

        exit()

    else:

        print("   MODE: Menjalankan TANPA simulasi PyBullet (menggunakan simulasi internal akurat).")

        # Menggunakan simulasi internal 'simulate_with_full_cbf'

        tuner = EnhancedPDTuner(

            robot_config=ur5e_config, 

            tuning_config=tuning_config,

            simulation_func=None # Ini akan membuat tuner memakai simulasi internalnya

        )



    # --- Langkah 3: Buat Trajektori Uji ---

    print("\n3. Membuat trajektori uji untuk mengetes collision avoidance...")

    test_trajectories = tuner.generate_collision_test_trajectories()

    print(f"   Dibuat {len(test_trajectories)} trajektori uji.")



    # --- Langkah 4: Jalankan Proses Optimisasi ---

    print("\n4. Memulai proses optimisasi gain PD...")

    

    optimized_gains = tuner.adaptive_bayesian_optimization(

        test_trajectories=test_trajectories,

        n_iterations=args.iterations

    )



    # --- Langkah 5: Validasi Hasil (Opsional tapi Direkomendasikan) ---

    if optimized_gains:

        print("\n5. Memvalidasi gain yang telah dioptimalkan...")

        validation_trajectories = tuner.generate_collision_test_trajectories() # Gunakan set data baru

        tuner.validate_gains(optimized_gains, validation_trajectories)



    # --- Langkah 6: Tampilkan dan Simpan Hasil ---

    print("\n6. Menampilkan dan menyimpan hasil akhir...")

    if optimized_gains:

        # Menyimpan hasil ke file .npz dan .json

        tuner.save_results()

        

        # Menampilkan grafik hasil tuning yang lengkap

        tuner.plot_enhanced_results()

    else:

        print("\nOptimisasi tidak menemukan solusi yang lebih baik dari nilai awal.")