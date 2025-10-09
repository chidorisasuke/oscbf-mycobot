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

class EnhancedMyCobotPDTuner:
    """
    Enhanced PD tuner with comprehensive collision avoidance and CBF integration
    """
    
    def __init__(self, 
                 mycobot_config: MyCobotConfig = None,
                 tuning_config: TuningConfig = None,
                 simulation_func: Optional[Callable] = None,
                 verbose: bool = True):
        
        self.robot = mycobot_config or MyCobotConfig()
        self.config = tuning_config or TuningConfig()
        self.simulation_func = simulation_func
        self.verbose = verbose
        
        # Add default obstacles if none specified
        if not self.robot.obstacles:
            self.robot.obstacles = [
                ObstacleConfig(position=np.array([0.1, 0.1, 0.15]), radius=0.05),
                ObstacleConfig(position=np.array([0.2, -0.1, 0.2]), radius=0.04),
            ]
        
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
            kp_pos=np.clip(kp_pos * 10, *self.config.kp_pos_range),
            kd_pos=np.clip(kd_pos * 10, *self.config.kd_pos_range),
            kp_rot=np.clip(kp_pos * 5, *self.config.kp_rot_range),
            kd_rot=np.clip(kd_pos * 5, *self.config.kd_rot_range),
            kp_joint=np.clip(kp_pos * 2, *self.config.kp_joint_range) * np.ones(self.robot.n_joints),
            kd_joint=np.clip(kd_pos * 2, *self.config.kd_joint_range) * np.ones(self.robot.n_joints)
        )
    
    def calculate_obstacle_cbf(self, ee_pos: np.ndarray, ee_vel: np.ndarray,
                               jacobian: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate CBF constraints for all obstacles
        Returns: (constraint_direction, min_distance)
        """
        min_distance = float('inf')
        critical_constraint = None
        
        for obstacle in self.robot.obstacles:
            # Distance from end-effector to obstacle
            distance_vec = ee_pos - obstacle.position
            distance = np.linalg.norm(distance_vec)
            safe_distance = obstacle.radius + self.robot.obstacle_safety_margin
            
            if distance < min_distance:
                min_distance = distance
            
            # CBF: h(x) = ||p_ee - p_obs||^2 - r_safe^2
            h = distance**2 - safe_distance**2
            
            if h < self.robot.cbf_safety_margin:
                # Gradient of h w.r.t. end-effector position
                grad_h = 2 * distance_vec
                
                # CBF constraint: dh/dt >= -alpha * h
                # dh/dt = grad_h^T * J * qdot
                h_dot_min = -self.robot.cbf_alpha_obstacle * h
                
                if critical_constraint is None or h < critical_constraint[1]:
                    critical_constraint = (grad_h @ jacobian, h)
        
        return critical_constraint, min_distance
    
    def calculate_link_collision_cbf(self, q: np.ndarray, qdot: np.ndarray,
                                     robot_model: Optional[Manipulator] = None) -> float:
        """
        Check collision for all robot links, not just end-effector
        """
        if robot_model is None:
            return 0.0
        
        total_violation = 0.0
        
        # Simplified link collision checking
        # In practice, you'd compute forward kinematics for each link
        for link_idx in range(self.robot.n_joints):
            # Approximate link position (simplified)
            link_pos = self._approximate_link_position(q, link_idx)
            
            for obstacle in self.robot.obstacles:
                distance = np.linalg.norm(link_pos - obstacle.position)
                safe_distance = obstacle.radius + self.robot.link_radii[link_idx] + 0.02
                
                if distance < safe_distance:
                    violation = safe_distance - distance
                    total_violation += violation * self.robot.cbf_alpha_obstacle
        
        return total_violation
    
    def _approximate_link_position(self, q: np.ndarray, link_idx: int) -> np.ndarray:
        """Approximate position of robot link (simplified)"""
        # This is a simplified approximation
        # In practice, use proper forward kinematics
        base_height = 0.1
        link_length = 0.08
        
        x = link_length * link_idx * np.cos(q[0]) * np.cos(np.sum(q[1:link_idx+1]))
        y = link_length * link_idx * np.sin(q[0]) * np.cos(np.sum(q[1:link_idx+1]))
        z = base_height + link_length * link_idx * np.sin(np.sum(q[1:link_idx+1]))
        
        return np.array([x, y, z])
    
    def simulate_with_full_cbf(self, gains: PDGains, 
                               target_trajectory: np.ndarray,
                               initial_state: Dict = None) -> Dict:
        """
        Enhanced simulation with complete CBF implementation
        """
        dt = 1.0 / self.robot.control_freq
        n_steps = len(target_trajectory)
        
        # Initialize state
        if initial_state is None:
            q = np.zeros(self.robot.n_joints)
            qdot = np.zeros(self.robot.n_joints)
            pos = np.array([0.15, 0.0, 0.15])
            rot = np.array([0.0, 0.0, 0.0])
        else:
            q = initial_state['q']
            qdot = initial_state['qdot']
            pos = initial_state['pos']
            rot = initial_state['rot']
        
        # Storage for comprehensive metrics
        errors = []
        velocities = []
        accelerations = []
        cbf_violations = []
        collision_proximities = []
        energy_consumption = []
        jerk_values = []
        
        prev_acc = np.zeros(self.robot.n_joints)
        
        for i in range(n_steps):
            # Get target
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
            
            # Joint space PD control
            joint_acc = gains.kp_joint * (-q) - gains.kd_joint * qdot
            
            # Combine accelerations with improved mapping
            target_acc = np.zeros(self.robot.n_joints)
            
            # Use pseudo-inverse of Jacobian for better mapping (simplified)
            jacobian = self._compute_jacobian(q)
            if jacobian is not None:
                task_acc = np.concatenate([pos_acc, rot_acc[:min(3, self.robot.n_joints-3)]])
                target_acc = np.linalg.pinv(jacobian) @ task_acc
            else:
                target_acc[:3] += pos_acc * 5.0
                if self.robot.n_joints > 3:
                    target_acc[3:6] += rot_acc * 2.0
            
            target_acc += joint_acc * 0.1
            
            # Apply comprehensive CBF constraints
            acc_before_cbf = target_acc.copy()
            
            # 1. Joint limit constraints
            constrained_acc = self._apply_joint_limit_cbf(q, qdot, target_acc)
            
            # 2. Obstacle avoidance constraints
            if jacobian is not None:
                obstacle_constraint, min_dist = self.calculate_obstacle_cbf(pos, qdot[:3], jacobian[:3, :])
                if obstacle_constraint is not None:
                    # Apply obstacle constraint
                    constraint_vec, h_value = obstacle_constraint
                    if h_value < 0:
                        # Project acceleration to satisfy constraint
                        violation = constraint_vec @ constrained_acc
                        if violation < -self.robot.cbf_alpha_obstacle * h_value:
                            # Need to modify acceleration
                            correction = (-self.robot.cbf_alpha_obstacle * h_value - violation)
                            constrained_acc -= correction * constraint_vec / (np.linalg.norm(constraint_vec)**2 + 1e-6)
                
                collision_proximities.append(min_dist)
            else:
                collision_proximities.append(float('inf'))
            
            # 3. Singularity avoidance
            constrained_acc = self._apply_singularity_cbf(q, jacobian, constrained_acc)
            
            # 4. Torque limits
            constrained_acc = self._apply_torque_limit_cbf(constrained_acc)
            
            # Calculate metrics
            cbf_violation = np.linalg.norm(constrained_acc - acc_before_cbf)
            cbf_violations.append(cbf_violation)
            
            # Jerk (smoothness metric)
            jerk = np.linalg.norm(constrained_acc - prev_acc) * self.robot.control_freq
            jerk_values.append(jerk)
            prev_acc = constrained_acc.copy()
            
            # Energy (simplified as torque squared)
            energy = np.sum(constrained_acc**2) * dt
            energy_consumption.append(energy)
            
            # Update state
            qdot += constrained_acc * dt
            q += qdot * dt
            
            # Update end-effector (improved forward kinematics)
            pos, rot = self._forward_kinematics(q)
            
            # Record metrics
            error_norm = np.linalg.norm(pos_error)
            errors.append(error_norm)
            velocities.append(np.linalg.norm(qdot))
            accelerations.append(np.linalg.norm(constrained_acc))
        
        # Calculate comprehensive performance metrics
        metrics = {
            'errors': np.array(errors),
            'velocities': np.array(velocities),
            'accelerations': np.array(accelerations),
            'cbf_violations': np.array(cbf_violations),
            'collision_proximities': np.array(collision_proximities),
            'energy_consumption': np.array(energy_consumption),
            'jerk_values': np.array(jerk_values),
            'final_error': errors[-1] if errors else float('inf'),
            'avg_error': np.mean(errors) if errors else float('inf'),
            'max_error': np.max(errors) if errors else float('inf'),
            'error_std': np.std(errors) if errors else float('inf'),
            'avg_cbf_violation': np.mean(cbf_violations),
            'min_obstacle_distance': np.min(collision_proximities) if collision_proximities else 0,
            'total_energy': np.sum(energy_consumption),
            'avg_jerk': np.mean(jerk_values),
            'final_state': {'q': q, 'qdot': qdot, 'pos': pos, 'rot': rot}
        }
        
        return metrics
    
    def _apply_joint_limit_cbf(self, q: np.ndarray, qdot: np.ndarray, 
                               target_acc: np.ndarray) -> np.ndarray:
        """Apply joint limit CBF constraints"""
        constrained_acc = target_acc.copy()
        
        for i in range(self.robot.n_joints):
            # Position limits (simplified, should use actual joint limits)
            q_max = np.pi  # Approximate max joint angle
            q_min = -np.pi
            
            # Upper position limit
            h_pos_upper = q_max - q[i] - 0.1  # 0.1 rad safety margin
            if h_pos_upper < self.robot.cbf_safety_margin:
                max_acc = self.robot.cbf_alpha_joint * h_pos_upper - qdot[i]
                constrained_acc[i] = min(constrained_acc[i], max_acc)
            
            # Lower position limit
            h_pos_lower = q[i] - q_min - 0.1
            if h_pos_lower < self.robot.cbf_safety_margin:
                min_acc = -self.robot.cbf_alpha_joint * h_pos_lower - qdot[i]
                constrained_acc[i] = max(constrained_acc[i], min_acc)
            
            # Velocity limits
            h_vel_upper = self.robot.joint_velocity_limits[i] - qdot[i]
            if h_vel_upper < self.robot.cbf_safety_margin:
                constrained_acc[i] = min(constrained_acc[i], 
                                        self.robot.cbf_alpha_joint * h_vel_upper)
            
            h_vel_lower = qdot[i] + self.robot.joint_velocity_limits[i]
            if h_vel_lower < self.robot.cbf_safety_margin:
                constrained_acc[i] = max(constrained_acc[i], 
                                        -self.robot.cbf_alpha_joint * h_vel_lower)
        
        return constrained_acc
    
    def _apply_singularity_cbf(self, q: np.ndarray, jacobian: Optional[np.ndarray],
                               target_acc: np.ndarray) -> np.ndarray:
        """Apply singularity avoidance CBF"""
        if jacobian is None:
            return target_acc
        
        # Manipulability measure
        manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
        
        if manipulability < self.robot.singularity_tol:
            # Near singularity - reduce acceleration
            scale_factor = max(0.1, manipulability / self.robot.singularity_tol)
            return target_acc * scale_factor
        
        return target_acc
    
    def _apply_torque_limit_cbf(self, target_acc: np.ndarray) -> np.ndarray:
        """Apply torque limit constraints"""
        # Simplified inertia estimate
        inertia_estimate = 0.01  # kg*m^2
        
        constrained_acc = target_acc.copy()
        for i in range(self.robot.n_joints):
            max_acc = self.robot.joint_torque_limits[i] / inertia_estimate
            constrained_acc[i] = np.clip(constrained_acc[i], -max_acc, max_acc)
        
        return constrained_acc
    
    def _compute_jacobian(self, q: np.ndarray) -> Optional[np.ndarray]:
        """Compute robot Jacobian (simplified)"""
        # This is a simplified Jacobian computation
        # In practice, use the actual robot kinematics
        try:
            J = np.zeros((6, self.robot.n_joints))
            
            # Simplified Jacobian for 6-DOF robot
            for i in range(min(3, self.robot.n_joints)):
                J[i, i] = 0.1 * np.cos(q[i])  # Position Jacobian
            for i in range(3, min(6, self.robot.n_joints)):
                J[i, i] = 0.5  # Orientation Jacobian
            
            return J
        except:
            return None
    
    def _forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics (simplified)"""
        # Simplified FK for MyCobot
        l1, l2, l3 = 0.1, 0.08, 0.07  # Link lengths
        
        # Position
        x = (l1 * np.cos(q[1]) + l2 * np.cos(q[1] + q[2]) + 
             l3 * np.cos(q[1] + q[2] + q[3] if self.robot.n_joints > 3 else 0)) * np.cos(q[0])
        y = (l1 * np.cos(q[1]) + l2 * np.cos(q[1] + q[2]) + 
             l3 * np.cos(q[1] + q[2] + q[3] if self.robot.n_joints > 3 else 0)) * np.sin(q[0])
        z = 0.1 + l1 * np.sin(q[1]) + l2 * np.sin(q[1] + q[2]) + \
            l3 * np.sin(q[1] + q[2] + q[3] if self.robot.n_joints > 3 else 0)
        
        pos = np.array([x, y, z])
        
        # Orientation (simplified)
        if self.robot.n_joints >= 6:
            rot = q[3:6] * 0.5
        else:
            rot = np.zeros(3)
        
        return pos, rot
    
    def evaluate_collision_aware_performance(self, trajectory_data: Dict) -> float:
        """
        Enhanced performance evaluation with collision awareness
        """
        # Extract metrics
        errors = trajectory_data['errors']
        cbf_violations = trajectory_data['cbf_violations']
        collision_proximities = trajectory_data['collision_proximities']
        energy = trajectory_data['energy_consumption']
        jerk = trajectory_data['jerk_values']
        
        # Adaptive weight calculation
        if self.config.use_adaptive_weights:
            # Increase collision weight if we're getting too close to obstacles
            min_distance = trajectory_data['min_obstacle_distance']
            if min_distance < self.robot.obstacle_safety_margin * 2:
                collision_weight = self.config.weight_collision_proximity * 2
            else:
                collision_weight = self.config.weight_collision_proximity
        else:
            collision_weight = self.config.weight_collision_proximity
        
        # Calculate collision penalty (exponential increase as we get closer)
        collision_penalty = 0
        for dist in collision_proximities:
            if dist < self.robot.obstacle_safety_margin * 3:
                collision_penalty += np.exp(-dist / self.robot.obstacle_safety_margin)
        collision_penalty /= len(collision_proximities)
        
        # Combined cost function
        cost = (
            self.config.weight_error * trajectory_data['avg_error'] +
            self.config.weight_error * 0.2 * trajectory_data['error_std'] +  # Consistency
            self.config.weight_smoothness * np.mean(jerk) +
            self.config.weight_cbf_violation * trajectory_data['avg_cbf_violation'] +
            collision_weight * collision_penalty +
            self.config.weight_energy * trajectory_data['total_energy'] / len(errors)
        )
        
        # Add heavy penalty for actual collisions
        if trajectory_data['min_obstacle_distance'] < self.robot.obstacle_safety_margin:
            cost *= 10  # Heavy penalty for violation
        
        return cost
    
    def generate_collision_test_trajectories(self) -> List[np.ndarray]:
        """
        Generate test trajectories that specifically test collision avoidance
        """
        trajectories = []
        t = np.linspace(0, self.config.test_duration, self.config.n_test_points)
        
        # 1. Trajectory passing near first obstacle
        traj1 = np.zeros((len(t), 3))
        obstacle1 = self.robot.obstacles[0]
        # Create path that would collide without avoidance
        traj1[:, 0] = 0.15 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
        traj1[:, 1] = 0.1 * np.cos(2 * np.pi * 0.3 * t)  # Pass near obstacle
        traj1[:, 2] = 0.2 + 0.05 * np.sin(2 * np.pi * 0.2 * t)
        trajectories.append(traj1)
        
        # 2. Weaving between obstacles
        traj2 = np.zeros((len(t), 3))
        traj2[:, 0] = 0.15 + 0.08 * np.sin(2 * np.pi * 0.4 * t)
        traj2[:, 1] = 0.1 * np.sin(2 * np.pi * 0.3 * t)  # Weave motion
        traj2[:, 2] = 0.18 + 0.04 * np.cos(2 * np.pi * 0.25 * t)
        trajectories.append(traj2)
        
        # 3. Vertical motion near obstacle
        traj3 = np.zeros((len(t), 3))
        traj3[:, 0] = obstacle1.position[0] + 0.1  # Fixed X near obstacle
        traj3[:, 1] = obstacle1.position[1]  # Fixed Y at obstacle
        traj3[:, 2] = 0.1 + 0.15 * (1 + np.sin(2 * np.pi * 0.3 * t))  # Vertical motion
        trajectories.append(traj3)
        
        # 4. Safe trajectory (baseline)
        traj4 = np.zeros((len(t), 3))
        traj4[:, 0] = 0.2
        traj4[:, 1] = 0.0
        traj4[:, 2] = 0.25 + 0.03 * np.sin(2 * np.pi * 0.5 * t)
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
            self.config.kp_pos_range,
            self.config.kd_pos_range,
            self.config.kp_rot_range,
            self.config.kd_rot_range,
            self.config.kp_joint_range,
            self.config.kd_joint_range
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
            tol=self.config.convergence_tol,
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
            kp_joint=x[4] * np.ones(self.robot.n_joints),
            kd_joint=x[5] * np.ones(self.robot.n_joints)
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
            if result.get('min_obstacle_distance', float('inf')) < self.robot.obstacle_safety_margin:
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
        if self.no_improvement_count >= self.config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {self.config.early_stopping_patience} iterations without improvement")
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
            print(f"  Avg CBF Violations: {result['avg_cbf_violation']:.4f}")
            print(f"  Energy Consumption: {result['total_energy']:.2f}")
        
        # Aggregate statistics
        avg_metrics = {
            'mean_error': np.mean([r['avg_error'] for r in validation_results]) * 1000,
            'max_error_overall': np.max([r['max_error'] for r in validation_results]) * 1000,
            'min_distance_overall': np.min([r['min_obstacle_distance'] for r in validation_results]) * 100,
            'collision_free': all(r['min_obstacle_distance'] > self.robot.obstacle_safety_margin 
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
        kp_pos_normalized = [(kp - self.config.kp_pos_range[0]) / 
                             (self.config.kp_pos_range[1] - self.config.kp_pos_range[0]) 
                             for kp in kp_pos]
        kd_pos_normalized = [(kd - self.config.kd_pos_range[0]) / 
                             (self.config.kd_pos_range[1] - self.config.kd_pos_range[0]) 
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
            filename = f"mycobot_tuning_results_{timestamp}"
        
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
                'n_joints': self.robot.n_joints,
                'control_freq': self.robot.control_freq,
                'cbf_alpha_obstacle': self.robot.cbf_alpha_obstacle,
                'cbf_alpha_joint': self.robot.cbf_alpha_joint,
                'obstacle_safety_margin': self.robot.obstacle_safety_margin
            },
            'tuning_config': {
                'weight_error': self.config.weight_error,
                'weight_smoothness': self.config.weight_smoothness,
                'weight_cbf_violation': self.config.weight_cbf_violation,
                'weight_collision_proximity': self.config.weight_collision_proximity
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
    parser = argparse.ArgumentParser(description="Enhanced Auto-Tuning untuk Gain PD MyCobot dengan Collision Avoidance.")
    parser.add_argument(
        '--use-pybullet', 
        action='store_true', 
        help="Gunakan simulasi PyBullet eksternal (membutuhkan wrapper function)."
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=25,
        help="Jumlah iterasi untuk proses optimisasi."
    )
    args = parser.parse_args()

    # --- Langkah 1: Konfigurasi ---
    print("1. Mengkonfigurasi lingkungan dan target tuning...")
    
    # Konfigurasi robot dan rintangan
    mycobot_config = MyCobotConfig(
        obstacles=[
            ObstacleConfig(position=np.array([0.20, 0.1, 0.2]), radius=0.05),
            ObstacleConfig(position=np.array([0.20, -0.1, 0.25]), radius=0.04),
        ]
    )
    
    # Konfigurasi proses tuning
    tuning_config = TuningConfig()

    # --- Langkah 2: Inisialisasi Tuner Sesuai Mode ---
    print("\n2. Menginisialisasi Auto-Tuner...")
    
    tuner = None
    if args.use_pybullet:
        print("   MODE: Menjalankan DENGAN simulasi PyBullet (pastikan wrapper 'run_pybullet_simulation' ada).")
        # Catatan: Baris ini akan error jika Anda belum mendefinisikan 'run_pybullet_simulation'
        # Untuk saat ini, fokus pada mode internal
        try:
            tuner = EnhancedMyCobotPDTuner(
                mycobot_config=mycobot_config, 
                tuning_config=tuning_config,
                simulation_func=run_pybullet_simulation # Membutuhkan fungsi wrapper dari diskusi sebelumnya
            )
        except NameError:
            print("\n[ERROR] Fungsi 'run_pybullet_simulation' tidak ditemukan.")
            print("        Jalankan tanpa flag '--use-pybullet' untuk menggunakan simulasi internal.")
            exit()
    else:
        print("   MODE: Menjalankan TANPA simulasi PyBullet (menggunakan simulasi internal yang canggih).")
        # Menggunakan simulasi internal 'simulate_with_full_cbf'
        tuner = EnhancedMyCobotPDTuner(
            mycobot_config=mycobot_config, 
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