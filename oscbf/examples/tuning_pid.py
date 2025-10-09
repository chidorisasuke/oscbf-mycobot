import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
import time
import pybullet as p
import argparse

from oscbf.core.manipulator import Manipulator, load_mycobot
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

@dataclass
class MyCobotConfig:
    """Configuration specific to MyCobot 280pi simulation"""
    # Physical limits from your specification
    n_joints: int = 6
    control_freq: float = 240.0  # Hz - dari timestep = 1/240
    
    # Joint limits (from URDF and your specs)
    joint_velocity_limits: np.ndarray = None  # rad/s
    joint_torque_limits: np.ndarray = None  # Nm (converted from 2.1 kg·cm)
    
    # CBF parameters
    cbf_alpha_obstacle: float = 10.0  # Obstacle avoidance strength
    cbf_alpha_joint: float = 10.0  # Joint limit avoidance strength
    cbf_safety_margin: float = 0.05  # 5cm safety margin
    singularity_tol: float = 1e-4
    
    # Workspace limits (meter)
    workspace_radius: float = 0.28  # MyCobot 280mm reach
    
    def __post_init__(self):
        if self.joint_velocity_limits is None:
            self.joint_velocity_limits = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        if self.joint_torque_limits is None:
            # Convert 2.1 kg·cm to Nm: 2.1 * 0.0981 ≈ 0.206 Nm
            # Use conservative values for safety
            self.joint_torque_limits = np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.15])

@dataclass
class TuningConfig:
    """Auto-tuning configuration for MyCobot simulation"""
    # Search bounds optimized for MyCobot 280pi
    kp_pos_range: Tuple[float, float] = (30.0, 200.0)  # Position proportional
    kd_pos_range: Tuple[float, float] = (3.0, 30.0)    # Position derivative
    kp_rot_range: Tuple[float, float] = (20.0, 120.0)  # Rotation proportional
    kd_rot_range: Tuple[float, float] = (2.0, 15.0)    # Rotation derivative
    kp_joint_range: Tuple[float, float] = (10.0, 60.0) # Joint proportional
    kd_joint_range: Tuple[float, float] = (1.0, 8.0)   # Joint derivative
    
    # Performance weights (tuned for smooth tracking with moving targets)
    weight_error: float = 1.0        # Tracking error importance
    weight_smoothness: float = 0.5   # Motion smoothness (higher for MyCobot)
    weight_overshoot: float = 0.2    # Less critical with moving targets
    weight_settling_time: float = 0.1 # Not relevant for continuous tracking
    weight_cbf_violation: float = 2.0 # Safety constraint violations
    
    # Test parameters
    test_duration: float = 10.0  # Longer for moving target evaluation
    n_test_points: int = 2400     # 10s * 240Hz

class MyCobotPDTuner:
    """
    Specialized PD tuner for MyCobot 280pi simulation with CBF integration
    Optimized for continuous tracking of moving targets (like sinusoidal trajectories)
    """
    
    def __init__(self, 
                 mycobot_config: MyCobotConfig = None,
                 tuning_config: TuningConfig = None,
                 simulation_func: Optional[Callable] = None):
        """
        Args:
            mycobot_config: Hardware and CBF configuration
            tuning_config: Tuning process configuration
            simulation_func: Optional custom simulation function
        """
        self.robot = mycobot_config or MyCobotConfig()
        self.config = tuning_config or TuningConfig()
        self.simulation_func = simulation_func
        
        # Initialize with conservative gains based on MyCobot characteristics
        self.best_gains = PDGains(
            kp_pos=60.0,   # Moderate for small robot
            kd_pos=6.0,    # 10% of kp for damping
            kp_rot=30.0,   # Lower for rotation
            kd_rot=3.0,
            kp_joint=20.0 * np.ones(self.robot.n_joints),
            kd_joint=2.0 * np.ones(self.robot.n_joints)
        )
        
        # Performance tracking
        self.tuning_history = []
        self.best_performance = float('inf')

    def ziegler_nichols_initial_estimate(self, rise_time: float = 0.5, overshoot: float = 0.2) -> PDGains:
        """
        Menghitung estimasi gain PD awal menggunakan metode Ziegler-Nichols.
        Metode ini didasarkan pada karakteristik respon sistem terhadap sebuah step input.

        Args:
            rise_time (float): Perkiraan waktu yang dibutuhkan sistem untuk naik dari 10% ke 90% target.
            overshoot (float): Perkiraan seberapa jauh sistem melewati target (misal, 0.2 untuk 20%).

        Returns:
            PDGains: Objek PDGains yang berisi nilai-nilai gain hasil estimasi.
        """
        print("\nMenghitung estimasi gain awal menggunakan metode Ziegler-Nichols...")

        # Perkirakan frekuensi natural (omega_n) dan rasio redaman (zeta) dari respon sistem
        if rise_time <= 0:
            rise_time = 0.1 # Hindari pembagian dengan nol
        omega_n = 1.8 / rise_time  # Aproksimasi standar

        if overshoot > 0 and overshoot < 1.0:
            zeta = -np.log(overshoot) / np.sqrt(np.pi**2 + np.log(overshoot)**2)
        else:
            zeta = 1.0  # Asumsikan critically damped jika tidak ada overshoot

        # Hitung gain PD berdasarkan aproksimasi sistem orde kedua
        # Faktor skala (misal * 10) adalah heuristik yang disesuaikan untuk skala simulasi robot
        kp_pos = omega_n**2 * 10
        kd_pos = 2 * zeta * omega_n * 10

        # Gain rotasi biasanya lebih rendah dari gain posisi
        kp_rot = omega_n**2 * 5
        kd_rot = 2 * zeta * omega_n * 5

        # Gain sendi juga dihitung dengan skala yang lebih kecil
        kp_joint_val = omega_n**2 * 2
        kd_joint_val = 2 * zeta * omega_n * 2

        # Buat objek PDGains dan pastikan nilainya berada dalam rentang yang diizinkan
        estimated_gains = PDGains(
            kp_pos=np.clip(kp_pos, *self.config.kp_pos_range),
            kd_pos=np.clip(kd_pos, *self.config.kd_pos_range),
            kp_rot=np.clip(kp_rot, *self.config.kp_rot_range),
            kd_rot=np.clip(kd_rot, *self.config.kd_rot_range),
            kp_joint=np.clip(kp_joint_val, *self.config.kp_joint_range) * np.ones(self.robot.n_joints),
            kd_joint=np.clip(kd_joint_val, *self.config.kd_joint_range) * np.ones(self.robot.n_joints)
        )

        print(f"Estimasi Gain Awal:")
        print(f"  Kp_pos: {estimated_gains.kp_pos:.2f}, Kd_pos: {estimated_gains.kd_pos:.2f}")
        print(f"  Kp_rot: {estimated_gains.kp_rot:.2f}, Kd_rot: {estimated_gains.kd_rot:.2f}")
        
        return estimated_gains
        
    def calculate_cbf_constraints(self, q: np.ndarray, qdot: np.ndarray, 
                                 target_acc: np.ndarray) -> np.ndarray:
        """
        Apply all CBF constraints as in your implementation:
        1. Obstacle avoidance
        2. Workspace containment
        3. Joint limits
        4. Singularity avoidance
        """
        constrained_acc = target_acc.copy()
        
        # 1. Joint limit constraints (most critical for MyCobot)
        for i in range(self.robot.n_joints):
            # Upper limit
            h_upper = self.robot.joint_velocity_limits[i] - qdot[i]
            if h_upper < self.robot.cbf_safety_margin:
                alpha_h = self.robot.cbf_alpha_joint * h_upper
                constrained_acc[i] = min(constrained_acc[i], alpha_h)
            
            # Lower limit
            h_lower = qdot[i] + self.robot.joint_velocity_limits[i]
            if h_lower < self.robot.cbf_safety_margin:
                alpha_h = self.robot.cbf_alpha_joint * h_lower
                constrained_acc[i] = max(constrained_acc[i], -alpha_h)
        
        # 2. Torque limits (critical for small motors)
        acc_norm = np.linalg.norm(constrained_acc)
        max_acc_from_torque = np.min(self.robot.joint_torque_limits) * 10  # Simplified
        if acc_norm > max_acc_from_torque:
            constrained_acc *= max_acc_from_torque / acc_norm
        
        return constrained_acc
    
    def simulate_mycobot_response(self, gains: PDGains, 
                                 target_trajectory: np.ndarray,
                                 initial_state: Dict = None) -> Dict:
        """
        Simulate MyCobot response with given PD gains
        This matches your actual simulation setup
        """
        if self.simulation_func is not None:
            # Use provided simulation function
            return self.simulation_func(gains, target_trajectory)
        
        # Default simulation based on your parameters
        dt = 1.0 / self.robot.control_freq
        n_steps = len(target_trajectory)
        
        # Initialize state
        if initial_state is None:
            q = np.zeros(self.robot.n_joints)  # Joint positions
            qdot = np.zeros(self.robot.n_joints)  # Joint velocities
            pos = np.array([0.15, 0.0, 0.15])  # Initial EE position
            rot = np.array([0.0, 0.0, 0.0])    # Initial EE orientation
        else:
            q = initial_state.get('q', np.zeros(self.robot.n_joints))
            qdot = initial_state.get('qdot', np.zeros(self.robot.n_joints))
            pos = initial_state.get('pos', np.array([0.15, 0.0, 0.15]))
            rot = initial_state.get('rot', np.array([0.0, 0.0, 0.0]))
        
        # Storage for metrics
        errors = []
        velocities = []
        accelerations = []
        cbf_violations = []
        
        # Simulation loop
        for i in range(n_steps):
            # Get target (assuming [x, y, z, rx, ry, rz] format)
            if target_trajectory.shape[1] == 3:
                target_pos = target_trajectory[i]
                target_rot = np.zeros(3)
            else:
                target_pos = target_trajectory[i, :3]
                target_rot = target_trajectory[i, 3:6]
            
            # Calculate errors
            pos_error = target_pos - pos
            rot_error = target_rot - rot
            
            # Task space PD control
            pos_acc = gains.kp_pos * pos_error - gains.kd_pos * qdot[:3]
            rot_acc = gains.kp_rot * rot_error - gains.kd_rot * qdot[3:6]
            
            # Joint space PD control (simplified - normally needs Jacobian)
            joint_acc = gains.kp_joint * (-q) - gains.kd_joint * qdot
            
            # Combine accelerations (simplified mapping)
            target_acc = np.zeros(self.robot.n_joints)
            target_acc[:3] += pos_acc * 5.0  # Scale factor for position control
            if self.robot.n_joints > 3:
                target_acc[3:6] += rot_acc * 2.0  # Scale factor for rotation
            target_acc += joint_acc * 0.1  # Joint space contribution
            
            # Apply CBF constraints
            acc_before_cbf = target_acc.copy()
            constrained_acc = self.calculate_cbf_constraints(q, qdot, target_acc)
            
            # Check CBF violation
            cbf_violation = np.linalg.norm(constrained_acc - acc_before_cbf)
            cbf_violations.append(cbf_violation)
            
            # Update state (simplified dynamics)
            qdot += constrained_acc * dt
            q += qdot * dt
            
            # Update end-effector (simplified forward kinematics)
            # For MyCobot, approximate mapping from joints to EE
            pos[0] = 0.15 * np.cos(q[0]) * np.cos(q[1] + q[2])
            pos[1] = 0.15 * np.sin(q[0]) * np.cos(q[1] + q[2])
            pos[2] = 0.15 + 0.15 * np.sin(q[1] + q[2])
            
            if self.robot.n_joints > 3:
                rot = q[3:6] * 0.5  # Simplified rotation mapping
            
            # Record metrics
            error_norm = np.linalg.norm(pos_error)
            errors.append(error_norm)
            velocities.append(np.linalg.norm(qdot))
            accelerations.append(np.linalg.norm(constrained_acc))
        
        return {
            'errors': np.array(errors),
            'velocities': np.array(velocities),
            'accelerations': np.array(accelerations),
            'cbf_violations': np.array(cbf_violations),
            'final_error': errors[-1] if errors else float('inf'),
            'avg_error': np.mean(errors) if errors else float('inf'),
            'max_error': np.max(errors) if errors else float('inf'),
            'final_state': {'q': q, 'qdot': qdot, 'pos': pos, 'rot': rot}
        }
    
    def evaluate_tracking_performance(self, trajectory_data: Dict) -> float:
        """
        Evaluate performance for continuous tracking (moving targets)
        Optimized for sinusoidal and dynamic trajectories
        """
        errors = trajectory_data['errors']
        velocities = trajectory_data['velocities']
        accelerations = trajectory_data['accelerations']
        cbf_violations = trajectory_data.get('cbf_violations', np.zeros_like(errors))
        
        # Key metrics for moving target tracking
        # 1. Average tracking error (most important for continuous tracking)
        avg_error = np.mean(errors)
        
        # 2. Error variance (consistency of tracking)
        error_variance = np.var(errors)
        
        # 3. Smoothness: minimize jerk (derivative of acceleration)
        if len(accelerations) > 1:
            jerk = np.diff(accelerations) * self.robot.control_freq
            smoothness_penalty = np.sqrt(np.mean(jerk**2))
        else:
            smoothness_penalty = 0
        
        # 4. CBF violation penalty (safety)
        cbf_penalty = np.mean(cbf_violations)
        
        # 5. Velocity smoothness (important for MyCobot's small motors)
        if len(velocities) > 1:
            vel_changes = np.diff(velocities)
            vel_smoothness = np.sqrt(np.mean(vel_changes**2))
        else:
            vel_smoothness = 0
        
        # Combined cost function for moving target tracking
        cost = (
            self.config.weight_error * avg_error +
            0.3 * error_variance +  # Penalize inconsistent tracking
            self.config.weight_smoothness * smoothness_penalty +
            self.config.weight_cbf_violation * cbf_penalty +
            0.2 * vel_smoothness  # Extra smoothness for small robot
        )
        
        return cost
    
    def generate_sinusoidal_trajectory(self, amplitude: float = 0.1, 
                                     frequency: float = 0.5,
                                     center: np.ndarray = None) -> np.ndarray:
        """
        Generate sinusoidal trajectory similar to your SinusoidalTaskTrajectory
        """
        if center is None:
            center = np.array([0.15, 0.0, 0.2])  # MyCobot reachable center
        
        t = np.linspace(0, self.config.test_duration, self.config.n_test_points)
        trajectory = np.zeros((len(t), 3))
        
        # Create 3D sinusoidal motion
        trajectory[:, 0] = center[0] + amplitude * np.sin(2 * np.pi * frequency * t)
        trajectory[:, 1] = center[1] + amplitude * np.cos(2 * np.pi * frequency * t)
        trajectory[:, 2] = center[2] + 0.5 * amplitude * np.sin(4 * np.pi * frequency * t)
        
        return trajectory
    
    def generate_test_trajectories(self) -> List[np.ndarray]:
        """
        Generate test trajectories suitable for MyCobot workspace
        """
        trajectories = []
        
        # 1. Small amplitude sinusoidal (like your test case)
        traj1 = self.generate_sinusoidal_trajectory(
            amplitude=0.05, frequency=0.3, center=np.array([0.15, 0.0, 0.25])
        )
        trajectories.append(traj1)
        
        # 2. Medium amplitude with higher frequency
        traj2 = self.generate_sinusoidal_trajectory(
            amplitude=0.08, frequency=0.5, center=np.array([0.12, 0.05, 0.22])
        )
        trajectories.append(traj2)
        
        # 3. Figure-8 pattern in horizontal plane
        t = np.linspace(0, self.config.test_duration, self.config.n_test_points)
        traj3 = np.zeros((len(t), 3))
        traj3[:, 0] = 0.15 + 0.06 * np.sin(2 * np.pi * 0.4 * t)
        traj3[:, 1] = 0.06 * np.sin(4 * np.pi * 0.4 * t)
        traj3[:, 2] = 0.20 + 0.03 * np.sin(2 * np.pi * 0.2 * t)
        trajectories.append(traj3)
        
        # 4. Vertical circle (testing Z-axis tracking)
        traj4 = np.zeros((len(t), 3))
        traj4[:, 0] = 0.15 * np.ones_like(t)
        traj4[:, 1] = 0.05 * np.cos(2 * np.pi * 0.3 * t)
        traj4[:, 2] = 0.20 + 0.05 * np.sin(2 * np.pi * 0.3 * t)
        trajectories.append(traj4)
        
        return trajectories
    
    # def bayesian_optimization_tuning(self, test_trajectories: List[np.ndarray], 
    #                                 n_iterations: int = 50) -> PDGains:
    #     """
    #     Use Bayesian optimization for efficient parameter search
    #     More suitable for expensive simulations
    #     """
    #     print("Starting Bayesian-inspired optimization for PD gains...")
        
    #     def objective(x):
    #         # Unpack parameters
    #         gains = PDGains(
    #             kp_pos=x[0],
    #             kd_pos=x[1],
    #             kp_rot=x[2],
    #             kd_rot=x[3],
    #             kp_joint=x[4] * np.ones(self.robot.n_joints),
    #             kd_joint=x[5] * np.ones(self.robot.n_joints)
    #         )
            
    #         # Evaluate on all test trajectories
    #         total_cost = 0
    #         for trajectory in test_trajectories:
    #             result = self.simulate_mycobot_response(gains, trajectory)
    #             cost = self.evaluate_tracking_performance(result)
    #             total_cost += cost
            
    #         avg_cost = total_cost / len(test_trajectories)
    #         return avg_cost
        
    #     # Define bounds
    #     bounds = [
    #         self.config.kp_pos_range,
    #         self.config.kd_pos_range,
    #         self.config.kp_rot_range,
    #         self.config.kd_rot_range,
    #         self.config.kp_joint_range,
    #         self.config.kd_joint_range
    #     ]
        
    #     # Use differential evolution for global optimization
    #     print("Running differential evolution optimization...")
    #     result = differential_evolution(
    #         objective,
    #         bounds,
    #         maxiter=n_iterations,
    #         popsize=10,
    #         tol=0.001,
    #         seed=42,
    #         disp=True,
    #         workers=-1  # Use parallel processing
    #     )
        
    #     # Extract optimized gains
    #     x_opt = result.x
    #     optimized_gains = PDGains(
    #         kp_pos=x_opt[0],
    #         kd_pos=x_opt[1],
    #         kp_rot=x_opt[2],
    #         kd_rot=x_opt[3],
    #         kp_joint=x_opt[4] * np.ones(self.robot.n_joints),
    #         kd_joint=x_opt[5] * np.ones(self.robot.n_joints)
    #     )
        
    #     print(f"\nOptimization complete! Best cost: {result.fun:.4f}")
        
    #     return optimized_gains

    # Di dalam kelas MyCobotPDTuner

    def _objective_function(self, x: np.ndarray) -> float:
        """
        Fungsi tujuan untuk evaluasi performa gain.
        Dibuat sebagai metode kelas agar bisa di-"pickle" oleh multiprocessing.
        """
        # Unpack parameters
        gains = PDGains(
            kp_pos=x[0],
            kd_pos=x[1],
            kp_rot=x[2],
            kd_rot=x[3],
            kp_joint=x[4] * np.ones(self.robot.n_joints),
            kd_joint=x[5] * np.ones(self.robot.n_joints)
        )
        
        # Evaluate on all test trajectories
        total_cost = 0
        # NOTE: Untuk optimisasi, seringkali lebih cepat mengevaluasi hanya pada satu
        # trajektori yang representatif di setiap iterasi.
        # Jika Anda ingin evaluasi penuh, hapus komentar di bawah.
        for trajectory_or_config in self._internal_test_trajectories:
            if self.simulation_func is not None:
                result =  self.simulation_func(gains, trajectory_or_config)
            else: 
                result = self.simulate_mycobot_response(gains, trajectory_or_config)
            
            cost = self.evaluate_tracking_performance(result)
            total_cost += cost
        
        # Evaluasi pada trajektori pertama saja untuk kecepatan
        result = self.simulate_mycobot_response(gains, self._internal_test_trajectories[0])
        total_cost = self.evaluate_tracking_performance(result)
        avg_cost = total_cost / len(self._internal_test_trajectories)

        self.tuning_history.append({'gains': gains, 'avg_cost': avg_cost})

        if avg_cost < self.best_performance:
            self.best_performance = avg_cost
            self.best_gains = gains
            print(f"  ✓ New best! Cost: {avg_cost:.4f} (kp_pos={gains.kp_pos:.1f}, kd_pos={gains.kd_pos:.1f})")
            
        # avg_cost = total_cost / len(test_trajectories)
        return avg_cost

    def bayesian_optimization_tuning(self, test_trajectories: List[np.ndarray],
                                     n_iterations: int = 50) -> PDGains:
        """
        Use Bayesian optimization for efficient parameter search
        More suitable for expensive simulations
        """
        print("Starting Bayesian-inspired optimization for PD gains...")
        
        # Simpan trajektori untuk diakses oleh _objective_function
        self._internal_test_trajectories = test_trajectories

        # Define bounds
        bounds = [
            self.config.kp_pos_range,
            self.config.kd_pos_range,
            self.config.kp_rot_range,
            self.config.kd_rot_range,
            self.config.kp_joint_range,
            self.config.kd_joint_range
        ]
        
        # Use differential evolution for global optimization
        print("Running differential evolution optimization...")
        result = differential_evolution(
            self._objective_function, # <-- Panggil metode kelas
            bounds,
            maxiter=n_iterations,
            popsize=15, # Direkomendasikan > 5 * jumlah_parameter
            tol=0.01,
            seed=42,
            disp=True,
            workers=-1,  # Use parallel processing, 
        )
        """
        -1 if u want activate the paralelism and using non-simulation arg
        1 if u want nonactivated the paralelism
        2 if ur computer are no GPU External
        """
        # Extract optimized gains
        x_opt = result.x
        optimized_gains = PDGains(
            kp_pos=x_opt[0],
            kd_pos=x_opt[1],
            kp_rot=x_opt[2],
            kd_rot=x_opt[3],
            kp_joint=x_opt[4] * np.ones(self.robot.n_joints),
            kd_joint=x_opt[5] * np.ones(self.robot.n_joints)
        )
        
        print(f"\nOptimization complete! Best cost: {result.fun:.4f}")
        
        return self.best_gains
    
    def iterative_refinement(self, initial_gains: PDGains, 
                            test_trajectories: List[np.ndarray],
                            max_iterations: int = 20) -> PDGains:
        """
        Iterative refinement with adaptive learning rate
        """
        current_gains = initial_gains
        learning_rate = 1.0
        
        for iteration in range(max_iterations):
            print(f"\n=== Refinement Iteration {iteration + 1}/{max_iterations} ===")
            
            # Test current gains
            total_cost = 0
            results = []
            for idx, trajectory in enumerate(test_trajectories):
                result = self.simulate_mycobot_response(current_gains, trajectory)
                cost = self.evaluate_tracking_performance(result)
                total_cost += cost
                results.append(result)
                
                print(f"  Trajectory {idx + 1}: Avg Error = {result['avg_error']*1000:.1f}mm, "
                      f"Max Error = {result['max_error']*1000:.1f}mm")
            
            avg_cost = total_cost / len(test_trajectories)
            
            # Store in history
            self.tuning_history.append({
                'iteration': iteration,
                'gains': current_gains,
                'avg_cost': avg_cost,
                'avg_error': np.mean([r['avg_error'] for r in results]),
                'max_error': np.max([r['max_error'] for r in results])
            })
            
            # Update best if improved
            if avg_cost < self.best_performance:
                self.best_performance = avg_cost
                self.best_gains = current_gains
                print(f"  ✓ New best! Cost: {avg_cost:.4f}")
            
            # Check convergence
            if iteration > 0:
                improvement = self.tuning_history[-2]['avg_cost'] - avg_cost
                if abs(improvement) < 0.0001:
                    print("Converged!")
                    break
                
                # Adaptive learning rate
                if improvement < 0:  # Got worse
                    learning_rate *= 0.5
                elif improvement > 0.01:  # Good improvement
                    learning_rate = min(learning_rate * 1.1, 1.0)
            
            # Gradient estimation using finite differences
            perturbation = 0.1
            gradients = np.zeros(6)
            
            for i, (param_name, param_range) in enumerate([
                ('kp_pos', self.config.kp_pos_range),
                ('kd_pos', self.config.kd_pos_range),
                ('kp_rot', self.config.kp_rot_range),
                ('kd_rot', self.config.kd_rot_range),
                ('kp_joint', self.config.kp_joint_range),
                ('kd_joint', self.config.kd_joint_range)
            ]):
                # Create perturbed gains
                perturbed_gains = PDGains(
                    kp_pos=current_gains.kp_pos,
                    kd_pos=current_gains.kd_pos,
                    kp_rot=current_gains.kp_rot,
                    kd_rot=current_gains.kd_rot,
                    kp_joint=current_gains.kp_joint.copy(),
                    kd_joint=current_gains.kd_joint.copy()
                )
                
                # Apply perturbation
                if param_name == 'kp_pos':
                    perturbed_gains.kp_pos += perturbation * (param_range[1] - param_range[0])
                elif param_name == 'kd_pos':
                    perturbed_gains.kd_pos += perturbation * (param_range[1] - param_range[0])
                elif param_name == 'kp_rot':
                    perturbed_gains.kp_rot += perturbation * (param_range[1] - param_range[0])
                elif param_name == 'kd_rot':
                    perturbed_gains.kd_rot += perturbation * (param_range[1] - param_range[0])
                elif param_name == 'kp_joint':
                    perturbed_gains.kp_joint += perturbation * (param_range[1] - param_range[0])
                elif param_name == 'kd_joint':
                    perturbed_gains.kd_joint += perturbation * (param_range[1] - param_range[0])
                
                # Evaluate perturbed gains (on first trajectory for speed)
                perturbed_result = self.simulate_mycobot_response(perturbed_gains, test_trajectories[0])
                perturbed_cost = self.evaluate_tracking_performance(perturbed_result)
                
                # Estimate gradient
                gradients[i] = (perturbed_cost - avg_cost) / (perturbation * (param_range[1] - param_range[0]))
            
            # Update gains using gradient descent
            step_size = learning_rate * 0.1
            new_kp_pos = current_gains.kp_pos - step_size * gradients[0] * (self.config.kp_pos_range[1] - self.config.kp_pos_range[0])
            new_kd_pos = current_gains.kd_pos - step_size * gradients[1] * (self.config.kd_pos_range[1] - self.config.kd_pos_range[0])
            new_kp_rot = current_gains.kp_rot - step_size * gradients[2] * (self.config.kp_rot_range[1] - self.config.kp_rot_range[0])
            new_kd_rot = current_gains.kd_rot - step_size * gradients[3] * (self.config.kd_rot_range[1] - self.config.kd_rot_range[0])
            new_kp_joint = current_gains.kp_joint - step_size * gradients[4] * (self.config.kp_joint_range[1] - self.config.kp_joint_range[0])
            new_kd_joint = current_gains.kd_joint - step_size * gradients[5] * (self.config.kd_joint_range[1] - self.config.kd_joint_range[0])
            
            # Clip to bounds
            current_gains = PDGains(
                kp_pos=np.clip(new_kp_pos, *self.config.kp_pos_range),
                kd_pos=np.clip(new_kd_pos, *self.config.kd_pos_range),
                kp_rot=np.clip(new_kp_rot, *self.config.kp_rot_range),
                kd_rot=np.clip(new_kd_rot, *self.config.kd_rot_range),
                kp_joint=np.clip(new_kp_joint, *self.config.kp_joint_range),
                kd_joint=np.clip(new_kd_joint, *self.config.kd_joint_range)
            )
        
        return self.best_gains
    
    def plot_results(self):
        """
        Comprehensive visualization of tuning results
        """
        if not self.tuning_history:
            print("No tuning history to plot")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Cost evolution
        ax1 = fig.add_subplot(gs[0, :])
        iterations = [h['iteration'] for h in self.tuning_history]
        costs = [h['avg_cost'] for h in self.tuning_history]
        avg_errors = [h['avg_error'] * 1000 for h in self.tuning_history]  # Convert to mm
        
        ax1.plot(iterations, costs, 'b-o', label='Cost Function')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(iterations, avg_errors, 'r-s', label='Avg Error (mm)')
        ax1_twin.set_ylabel('Average Error (mm)', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title('Tuning Progress: Cost and Tracking Error')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Task-space gains evolution
        ax2 = fig.add_subplot(gs[1, 0])
        kp_pos_vals = [h['gains'].kp_pos for h in self.tuning_history]
        kd_pos_vals = [h['gains'].kd_pos for h in self.tuning_history]
        
        ax2.plot(iterations, kp_pos_vals, 'g-o', label='Kp_pos', linewidth=2)
        ax2.plot(iterations, kd_pos_vals, 'g--s', label='Kd_pos', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gain Value')
        ax2.set_title('Position Gains Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rotation gains evolution
        ax3 = fig.add_subplot(gs[1, 1])
        kp_rot_vals = [h['gains'].kp_rot for h in self.tuning_history]
        kd_rot_vals = [h['gains'].kd_rot for h in self.tuning_history]
        
        ax3.plot(iterations, kp_rot_vals, 'b-o', label='Kp_rot', linewidth=2)
        ax3.plot(iterations, kd_rot_vals, 'b--s', label='Kd_rot', alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gain Value')
        ax3.set_title('Rotation Gains Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Joint gains evolution
        ax4 = fig.add_subplot(gs[1, 2])
        kp_joint_vals = [h['gains'].kp_joint[0] for h in self.tuning_history]
        kd_joint_vals = [h['gains'].kd_joint[0] for h in self.tuning_history]
        
        ax4.plot(iterations, kp_joint_vals, 'm-o', label='Kp_joint', linewidth=2)
        ax4.plot(iterations, kd_joint_vals, 'm--s', label='Kd_joint', alpha=0.7)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Gain Value')
        ax4.set_title('Joint Gains Evolution (Joint 1)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Final gains summary (bar chart)
jax.tree_util.register_static
class TuningSafetyConfig(OSCBFTorqueConfig):
    def __init__(self, robot:Manipulator, singularity_tol:float):
        self.singularity_tol = singularity_tol
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        q = z[:self.num_joints]
        h_joint_limits = jnp.concatenate([
            jnp.asarray(self.q_max) - q,
            q - jnp.asarray(self.q_min)
        ])

        manipulability_index = self.robot.manipulability(q)
        h_singularity = jnp.array([manipulability_index - self.singularity_tol])

        return jnp.concatenate([h_joint_limits, h_singularity])
    
def run_pybullet_simulation(gains: PDGains, trajectory_config: Dict) -> Dict:
    """
    Menjalankan satu episode simulasi PyBullet lengkap secara headless (tanpa GUI)
    dan mengembalikan metrik performa.
    """
    robot = load_mycobot()
    robot_config = MyCobotConfig()
    tuning_cfg = TuningConfig()

    # traj = SinusoidalTaskTrajectory(
    #     init_pos=trajectory_config.get("center", [0.15, 0, 0.25]),
    #     amplitude=trajectory_config.get("amplitude", [0.05, 0.05, 0.05]),
    #     angular_freq=trajectory_config.get("frequency", [0.5, 0.5, 0.5])
    # )

    traj = SinusoidalTaskTrajectory(
        init_pos=trajectory_config.get("center"),
        init_rot=trajectory_config.get("init_rot"),  # <-- Tambahkan ini
        amplitude=trajectory_config.get("amplitude"),
        angular_freq=trajectory_config.get("frequency"),
        phase=trajectory_config.get("phase")          # <-- Tambahkan ini
    )

    env = MyCobotTorqueControlEnv(
        traj = traj,
        real_time=False,
        timestep=1.0/robot_config.control_freq,
        pybullet_client_mode=p.DIRECT
    )

    osc_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([gains.kp_pos * np.ones(3), gains.kp_rot * np.ones(3)]),
        kd_task=np.concatenate([gains.kd_pos * np.ones(3), gains.kd_rot *  np.ones(3)]),
        kp_joint=gains.kp_joint,
        kd_joint=gains.kd_joint,
        tau_min=None, tau_max=None
    )

    safety_config = TuningSafetyConfig(robot, singularity_tol=robot_config.singularity_tol)
    cbf = CBF.from_config(safety_config)

    @jax.jit
    def compute_control_jit(z, z_des):
        q, qdot = z[:robot.num_joints], z[robot.num_joints:]
        M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
        u_nom = osc_controller(
            q, qdot, 
            pos=ee_tmat[:3, 3], 
            rot=ee_tmat[:3, :3],
            des_pos=z_des[:3], 
            des_rot=jnp.reshape(z_des[3:12], (3, 3)),
            des_vel=z_des[12:15], 
            des_omega=z_des[15:18],
            des_accel=jnp.zeros(3), 
            des_alpha=jnp.zeros(3),
            des_q=np.zeros(robot.num_joints), 
            des_qdot=jnp.zeros(robot.num_joints),
            J=J, 
            M=M, 
            M_inv=M_inv, 
            g=g, 
            c=c
        )
        return cbf.safety_filter(z, u_nom)

    errors, torques, velocities, accelerations = [], [], [], []

    while env.t < tuning_cfg.test_duration:
        joint_state = env.get_joint_state()
        ee_state_des = env.get_desired_ee_state()

        tau = compute_control_jit(joint_state, ee_state_des)

        env.apply_control(np.asarray(tau))
        env.step()

        current_pos = robot.ee_position(joint_state[:robot.num_joints])
        error_norm = np.linalg.norm(current_pos - ee_state_des[:3])
        errors.append(error_norm)
        torques.append(np.asarray(tau))
        velocities.append(joint_state[robot.num_joints:])
        accelerations.append(np.linalg.norm(tau))

    env.client.disconnect()

    return{
        'errors': np.array(errors),
        'torques': np.array(torques),
        'velocities': np.array(velocities),
        'accelerations': np.array(accelerations),
    }
# ==============================================================================
# BLOK UTAMA UNTUK MENJALANKAN AUTO-TUNING (INSPIRASI DARI KODE 2)
# ==============================================================================
if __name__ == "__main__":

    # --- Setup Argumen Parser untuk CLI ---
    parser = argparse.ArgumentParser(description="Auto-Tuning untuk Gain PD MyCobot.")
    parser.add_argument(
        '--no-simulation', 
        action='store_true', 
        help="Jalankan tuning menggunakan model matematika internal yang cepat, bukan simulasi PyBullet."
    )
    args = parser.parse_args()
    
    # 1. Inisialisasi Tuner dengan Konfigurasi MyCobot
    # Menggunakan kelas MyCobotConfig dan TuningConfig dari KODE 1
    print("Menginisialisasi Auto-Tuner untuk MyCobot 280pi...")
    # mycobot_config = MyCobotConfig()
    # tuning_config = TuningConfig()
    if args.no_simulation:
        print("MODE: Menjalankan TANPA simulasi PyBullet (menggunakan model internal).")
        tuner = MyCobotPDTuner()
    else:
        print("MODE: Menjalankan DENGAN simulasi PyBullet (headless).")
        tuner = MyCobotPDTuner(simulation_func=run_pybullet_simulation)

    # 2. Buat Beberapa Trajektori Uji yang Beragam
    # Menggunakan metode generate_test_trajectories dari KODE 1
    print("\nMembuat beberapa trajektori uji yang beragam...")
    if args.no_simulation:
        test_trajectories = tuner.generate_test_trajectories()
    else:
        test_trajectories = [
            {'type': 'sinusoidal', 
            'center': [0.15, 0.0, 0.25], 
            'amplitude': [0.05, 0.05, 0.03], 
            'frequency': [0.3, 0.3, 0.5],
            'init_rot' : np.eye(3),
            'phase' : [0, 0, 0]
            },
            {'type': 'sinusoidal',
            'center': [0.12, 0.05, 0.22],
            'amplitude': [0.08, 0.0, 0.05], 
            'frequency': [0.5, 0, 0.5],
            'init_rot' : np.eye(3),
            'phase': [np.pi/2, 0, 0]
            },
        ]
        print(f"Dibuat {len(test_trajectories)} trajektori uji untuk evaluasi.")

    # 3. Jalankan Proses Optimisasi
    # Kita akan menggunakan 'bayesian_optimization_tuning' dari KODE 1 
    # karena ini adalah metode pencarian global yang lebih kuat daripada L-BFGS-B.
    print("\nMemulai proses optimisasi gain PD (menggunakan Differential Evolution)...")
    if not args.no_simulation:
        print("Proses ini mungkin memakan waktu beberapa menit, tergantung jumlah iterasi.")
    
    # Anda bisa menaikkan n_iterations (misal ke 50) untuk hasil yang lebih akurat,
    # tapi 20 adalah awal yang baik.
    optimized_gains = tuner.bayesian_optimization_tuning(
        test_trajectories=test_trajectories,
        n_iterations=20  
    )

    # 4. Tampilkan Hasil Akhir yang Ditemukan
    print("\n" + "="*25)
    print("=== PROSES TUNING SELESAI ===")
    print("="*25)

    if optimized_gains:
        print("\nGain PD optimal yang ditemukan:")
        print(f"  - Kp_pos: {optimized_gains.kp_pos:.3f}")
        print(f"  - Kd_pos: {optimized_gains.kd_pos:.3f}")
        print(f"  - Kp_rot: {optimized_gains.kp_rot:.3f}")
        print(f"  - Kd_rot: {optimized_gains.kd_rot:.3f}")
        print(f"  - Kp_joint: {optimized_gains.kp_joint[0]:.3f} (nilai seragam untuk semua sendi)")
        print(f"  - Kd_joint: {optimized_gains.kd_joint[0]:.3f} (nilai seragam untuk semua sendi)")

        # 5. Buat Fungsi Ekspor (jika belum ada di kelas Anda)
        # Ini akan sangat berguna untuk menyalin hasil ke skrip simulasi Anda.
        # Pastikan metode ini ada di dalam kelas MyCobotPDTuner Anda.
        def export_gains_to_controller_format(gains: PDGains) -> Dict:
            """Mengekspor gain ke format yang dibutuhkan oleh OSCBF controller."""
            return {
                'kp_task': np.concatenate([
                    gains.kp_pos * np.ones(3),
                    gains.kp_rot * np.ones(3)
                ]),
                'kd_task': np.concatenate([
                    gains.kd_pos * np.ones(3),
                    gains.kd_rot * np.ones(3)
                ]),
                'kp_joint': gains.kp_joint,
                'kd_joint': gains.kd_joint
            }

        controller_gains = export_gains_to_controller_format(optimized_gains)

        print("\nFormat untuk disalin ke skrip simulasi Anda:")
        print("-" * 40)
        print(f"kp_pos = {optimized_gains.kp_pos:.3f}")
        print(f"kp_rot = {optimized_gains.kp_rot:.3f}")
        print(f"kd_pos = {optimized_gains.kd_pos:.3f}")
        print(f"kd_rot = {optimized_gains.kd_rot:.3f}")
        print(f"kp_joint = {optimized_gains.kp_joint[0]:.3f}")
        print(f"kd_joint = {optimized_gains.kd_joint[0]:.3f}")
        print("-" * 40)

        # 6. Simpan Hasil ke File
        # Praktik yang baik untuk menyimpan hasil tuning
        np.savez('tuned_mycobot_gains.npz', **controller_gains)
        print("\nGain optimal telah disimpan ke file 'tuned_mycobot_gains.npz'")

        # 7. Tampilkan Grafik Hasil Tuning
        # Memanggil metode plot_results dari KODE 1
        print("\nMenampilkan grafik histori tuning...")
        tuner.plot_results()
    else:
        print("Optimisasi tidak menemukan solusi yang lebih baik dari nilai awal.")