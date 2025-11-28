"""Testing the performance of OSCBF during dynamic motions with containment constraints

The end-effector (EE) of the MyCobot robot must stay within a containment space (green box)
while chasing a moving red ball. Even if the red ball moves outside the box, the robot's
EE must remain contained within the safe space.
"""

import argparse
from functools import partial
import time

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_mycobot
from oscbf.core.manipulation_env import MyCobotTorqueControlEnv, MyCobotVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
)

DATA_DIR = "oscbf/experiments/data/"
SAVE_DATA = False
PAUSE_FOR_PICTURES = False
RECORD_VIDEO = False
PICTURE_IDXS = [1000, 1250, 1600, 1900, 2200]


@jax.tree_util.register_static
class EESafeSetTorqueConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        compensate_centrifugal_coriolis: bool,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.singularity_tol = 1e-4
        super().__init__(
            robot, compensate_centrifugal_coriolis=compensate_centrifugal_coriolis
        )

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        manipulability_index = self.robot.manipulability(q)
        h_singularity = jnp.array([manipulability_index - self.singularity_tol])
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min, h_singularity])

    def alpha(self, h):
        return 50.0 * h

    def alpha_2(self, h_2):
        return 50.0 * h_2


@jax.tree_util.register_static
class EESafeSetVelocityConfig(OSCBFVelocityConfig):

    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2, 3))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    compensate_centrifugal_coriolis: bool,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)

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
    return cbf.safety_filter(z, u_nom)


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


def main(control_method="torque"):
    assert control_method in ["torque", "velocity"]

    robot = load_mycobot()
    
    # Define the containment box (green box) for the end-effector
    pos_min = (0.15, -0.2, 0.3)
    pos_max = (0.45, 0.2, 0.55)

    # Buat variabel q_init dengan 6 elemen
    mycobot_q_init = (0, -0.446, -0.071, -0.041, 0, 0) # Ganti dengan posisi awal yang aman

    time_log = []
    h_log = []

    # NOTE: This term has a noticeable impact on the performance for this demo.
    # It's often neglected due to computational demands and model error
    compensate_centrifugal_coriolis = False

    torque_config = EESafeSetTorqueConfig(
        robot,
        pos_min,
        pos_max,
        compensate_centrifugal_coriolis=compensate_centrifugal_coriolis,
    )
    torque_cbf = CBF.from_config(torque_config)
    velocity_config = EESafeSetVelocityConfig(robot, pos_min, pos_max)
    velocity_cbf = CBF.from_config(velocity_config)
    
    timestep = 1 / 240  # Use 240Hz for better performance
    bg_color = (1, 1, 1)
    if control_method == "torque":
        env = MyCobotTorqueControlEnv(
            torque_config.pos_min,
            torque_config.pos_max,
            q_init=mycobot_q_init,
            # No trajectory - we'll define our own moving red ball
            real_time=False,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )
    else:
        env = MyCobotVelocityControlEnv(
            velocity_config.pos_min,
            velocity_config.pos_max,
            # No trajectory - we'll define our own moving red ball
            real_time=False,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.00,
        cameraYaw=12,
        cameraPitch=-2.6,
        cameraTargetPosition=(0.44, 0.16, 0.28),
    )

    # Create the containment box visualization (green box)
    box_center = [(pos_min[0] + pos_max[0]) / 2, (pos_min[1] + pos_max[1]) / 2, (pos_min[2] + pos_max[2]) / 2]
    box_half_extents = [(pos_max[0] - pos_min[0]) / 2, (pos_max[1] - pos_min[1]) / 2, (pos_max[2] - pos_min[2]) / 2]
    
    box_visual_shape = env.client.createVisualShape(
        shapeType=env.client.GEOM_BOX,
        halfExtents=box_half_extents,
        rgbaColor=[0, 1, 0, 0.1],  # Green with transparency
        specularColor=[0, 1, 0]
    )
    
    box_body = env.client.createMultiBody(
        baseVisualShapeIndex=box_visual_shape,
        baseCollisionShapeIndex=-1,  # No collision
        basePosition=box_center
    )
    
    # Set up the moving red ball that the robot will chase
    red_ball_radius = 0.02
    red_ball_visual_shape = env.client.createVisualShape(
        shapeType=env.client.GEOM_SPHERE,
        radius=red_ball_radius,
        rgbaColor=[1, 0, 0, 1],  # Red
        specularColor=[1, 0, 0]
    )
    
    red_ball_body = env.client.createMultiBody(
        baseVisualShapeIndex=red_ball_visual_shape,
        baseCollisionShapeIndex=-1,  # No collision
        basePosition=[0.3, 0.0, 0.4]  # Start position
    )
    
    # PD gains from tuning results
    kp_pos = 20.91
    kp_rot = 22.76
    kd_pos = 77.76
    kd_rot = 8.21
    kp_joint = 30.22
    kd_joint = 1.64
    
    osc_torque_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
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
        kp_joint=kp_joint,
        # Note: velocity limits will be enforced via the QP
        qdot_min=None,
        qdot_max=None,
    )

    @jax.jit
    def compute_torque_control_jit(z, z_ee_des):
        return compute_torque_control(
            robot,
            osc_torque_controller,
            torque_cbf,
            compensate_centrifugal_coriolis,
            z,
            z_ee_des,
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

    # Performance monitoring
    max_simulation_time = 20.0  # 20 seconds for testing
    last_print_time = 0
    last_ball_update = 0
    ball_update_interval = 0.05  # Update ball position every 50ms
    
    # Ball movement parameters
    ball_center = np.array([0.3, 0.0, 0.4])  # Center of ball movement
    ball_amplitude = np.array([0.15, 0.15, 0.1])  # Amplitude of movement (x, y, z)
    ball_frequency = 0.5  # Frequency of movement in Hz
    
    try:
        # Initialize the desired state variable before the loop
        z_ee_des = np.zeros(18)  # 18 elements for pose + velocity
        new_ball_pos = ball_center  # Initialize with center position
        
        while env.t < max_simulation_time:
            current_time = time.time()
            
            # Update the red ball position in a periodic motion
            if (env.t - last_ball_update) >= ball_update_interval:
                # Calculate new ball position using sinusoidal motion
                ball_time = env.t * ball_frequency * 2 * np.pi
                ball_offset = ball_amplitude * np.array([
                    np.sin(ball_time),
                    np.sin(ball_time * 0.7),  # Different frequency for y
                    np.sin(ball_time * 1.3)   # Different frequency for z
                ])
                new_ball_pos = ball_center + ball_offset
                
                # Set the new position of the red ball
                env.client.resetBasePositionAndOrientation(
                    red_ball_body, 
                    new_ball_pos, 
                    [0, 0, 0, 1]  # No rotation
                )
                
                # Update the desired EE state to follow the red ball
                # This is the position the robot should try to reach (but will be constrained)
                z_ee_des = np.zeros(18)  # 18 elements for pose + velocity
                z_ee_des[:3] = new_ball_pos  # Set desired position to follow the ball
                
                # Set desired rotation (identity matrix flattened)
                z_ee_des[3:12] = np.eye(3).flatten()
                
                # Set desired velocities (zero for now, can be computed from ball movement)
                z_ee_des[12:15] = ball_amplitude * np.array([
                    ball_frequency * 2 * np.pi * np.cos(ball_time),
                    ball_frequency * 2 * np.pi * 0.7 * np.cos(ball_time * 0.7),
                    ball_frequency * 2 * np.pi * 1.3 * np.cos(ball_time * 1.3)
                ])
                
                # Set desired angular velocities (zero)
                z_ee_des[15:18] = np.zeros(3)
                
                last_ball_update = env.t

            q_qdot = env.get_joint_state()
            
            # Data Collection for Plot - done at intervals for performance
            if (env.t - last_print_time) >= 0.1:  # Log every 100ms
                if control_method == "torque":
                    h_values = torque_config.h_2(q_qdot)
                else:
                    h_values = velocity_config.h_1(q_qdot)

                time_log.append(env.t)
                h_log.append(h_values)
                
                # Print status
                q = q_qdot[:robot.num_joints]
                ee_pos = robot.ee_position(q)
                ball_pos = new_ball_pos  # Use the current ball position
                
                print(f"Time: {env.t:.2f}s | EE: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}) | Ball: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f})")
                
                # Check if EE is within bounds
                if not (pos_min[0] <= ee_pos[0] <= pos_max[0] and 
                        pos_min[1] <= ee_pos[1] <= pos_max[1] and 
                        pos_min[2] <= ee_pos[2] <= pos_max[2]):
                    print(f"  WARNING: EE is OUTSIDE bounds! Constraint violated!")
                else:
                    print(f"  OK: EE is within bounds.")
                    
                last_print_time = env.t

            # Compute control
            tau = compute_control(q_qdot, z_ee_des)
            
            # Apply torque limits for safety
            max_torque = 0.2  # Conservative torque limit (Nm) based on MyCobot specs
            tau = np.clip(tau, -max_torque, max_torque)
            
            env.apply_control(tau)
            env.step()
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    finally:
        print("Simulation finished. Plotting data...")
        if h_log:
            h_log = np.array(h_log)
            constraint_labels = [
            "X max", "Y max", "Z max", # Sesuai urutan pos_max - ee_pos
            "X min", "Y min", "Z min"  # Sesuai urutan ee_pos - pos_min
            ]

            plt.figure(figsize=(10, 6))
            for i, label in enumerate(constraint_labels):
                plt.plot(time_log, h_log[:, i], label=label)

            plt.axhline(0, color='r', linestyle='--', label='Batas Aman (h=0)')
            plt.xlabel("Waktu (s)")
            plt.ylabel("Nilai h(z)")
            plt.title("Evolusi Batasan Keamanan (Constraint Evolution) - EE Containment")
            plt.grid(True)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-effector containment experiment with moving target."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="torque",
        help="Control method to use (default: torque)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method)
