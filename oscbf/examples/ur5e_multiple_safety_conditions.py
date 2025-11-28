
"""Testing the performance of OSCBF for UR5e under many different safety constraints, namely:

1. End-effector set containment
2. Joint limit avoidance
3. Singularity avoidance
4. Collision avoidance
5. Whole-body set containment
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from cbfpy import CBF

from oscbf.core.manipulator import Manipulator, load_ur5e
from oscbf.core.manipulation_env import UR5eTorqueControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController


@jax.tree_util.register_static
class CombinedConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        assert len(collision_positions) == len(collision_radii)
        self.num_collision_bodies = len(collision_positions)
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # EE Set Containment
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole-body Set Containment
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()
        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate(
            [
                h_ee_safe_set,
                h_joint_limits,
                h_singularity,
                h_collision,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    # Set nullspace desired joint position
    nullspace_posture_goal = jnp.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])

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


def main():
    robot = load_ur5e()
    ee_pos_min = np.array([0.15, -0.25, 0.25])
    ee_pos_max = np.array([0.75, 0.25, 0.75])
    wb_pos_min = np.array([-0.5, -0.5, 0.0])
    wb_pos_max = np.array([0.75, 0.5, 1.0])
    collision_pos = np.array([[0.5, 0.5, 0.5]])
    collision_radii = np.array([0.3])
    collision_data = {"positions": collision_pos, "radii": collision_radii}
    config = CombinedConfig(
        robot,
        ee_pos_min,
        ee_pos_max,
        collision_pos,
        collision_radii,
        wb_pos_min,
        wb_pos_max,
    )
    cbf = CBF.from_config(config)
    env = UR5eTorqueControlEnv(
        xyz_min=config.pos_min,
        xyz_max=config.pos_max,
        collision_data=collision_data,
        wb_xyz_min=wb_pos_min,
        wb_xyz_max=wb_pos_max,
        load_floor=False,
        bg_color=(1, 1, 1),
        real_time=True,
        q_init = (0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0), # Set natural start pose
    )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraPitch=-27.80,
        cameraYaw=36.80,
        cameraTargetPosition=(0.08, 0.49, -0.04),
    )

    # UR5e gains from cluttered tabletop
    kp_pos = 20.91
    kp_rot = 22.76
    kd_pos = 77.76
    kd_rot = 8.21
    kp_joint = 30.22
    kd_joint = 1.64
    
    osc_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        tau_min=None,
        tau_max=None,
    )

    @jax.jit
    def compute_control_jit(z, z_des):
        return compute_control(robot, osc_controller, cbf, z, z_des)

    times, q_data, qdot_data, tau_data, ee_pos_data, ee_pos_des_data, h_data = [], [], [], [], [], [], []

    try:
        print("Starting simulation...")
        while env.t < 10.0:
            joint_state = env.get_joint_state()
            ee_state_des = env.get_desired_ee_state()
            tau = compute_control_jit(joint_state, ee_state_des)
            env.apply_control(tau)
            env.step()
            
            # Store data
            times.append(env.t)
            q_data.append(joint_state[:robot.num_joints])
            qdot_data.append(joint_state[robot.num_joints:])
            tau_data.append(tau)
            ee_pos_data.append(robot.ee_position(joint_state[:robot.num_joints]))
            ee_pos_des_data.append(ee_state_des[:3])
            h_data.append(config.h_2(joint_state))

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    finally:
        print("Simulation finished. Plotting data...")
        import matplotlib.pyplot as plt

        times = np.array(times)
        q_data = np.array(q_data)
        tau_data = np.array(tau_data)
        ee_pos_data = np.array(ee_pos_data)
        ee_pos_des_data = np.array(ee_pos_des_data)
        h_data = np.array(h_data)

        plt.figure(figsize=(15, 10))
        plt.suptitle('UR5e Simulation Results', fontsize=16)

        # Plot EE position tracking X
        plt.subplot(3, 2, 1)
        plt.plot(times, ee_pos_data[:, 0], label='Actual X')
        plt.plot(times, ee_pos_des_data[:, 0], label='Desired X', linestyle='--')
        plt.title('End-Effector Position (X)')
        plt.ylabel('X [m]')
        plt.grid(True)
        plt.legend()

        # Plot joint positions
        plt.subplot(3, 2, 2)
        for j in range(robot.num_joints):
            plt.plot(times, q_data[:, j], label=f'Joint {j+1}')
        plt.title('Joint Positions')
        plt.ylabel('Angle [rad]')
        plt.grid(True)
        plt.legend()

        # Plot EE position tracking Y
        plt.subplot(3, 2, 3)
        plt.plot(times, ee_pos_data[:, 1], label='Actual Y')
        plt.plot(times, ee_pos_des_data[:, 1], label='Desired Y', linestyle='--')
        plt.title('End-Effector Position (Y)')
        plt.ylabel('Y [m]')
        plt.grid(True)
        plt.legend()

        # Plot torques
        plt.subplot(3, 2, 4)
        for j in range(robot.num_joints):
            plt.plot(times, tau_data[:, j], label=f'Joint {j+1}')
        plt.title('Control Torques')
        plt.ylabel('Torque [Nm]')
        plt.grid(True)
        plt.legend()

        # Plot EE position tracking Z
        plt.subplot(3, 2, 5)
        plt.plot(times, ee_pos_data[:, 2], label='Actual Z')
        plt.plot(times, ee_pos_des_data[:, 2], label='Desired Z', linestyle='--')
        plt.title('End-Effector Position (Z)')
        plt.xlabel('Time [s]')
        plt.ylabel('Z [m]')
        plt.grid(True)
        plt.legend()
        
        # Plot CBF values
        plt.subplot(3, 2, 6)
        if h_data.size > 0:
            plt.plot(times, np.min(h_data, axis=1), label='Min h(x)')
            plt.hlines(0.0, 0, times[-1], colors='r', linestyles='--', label='Safety Boundary')
        plt.title('Minimum CBF Value (h)')
        plt.xlabel('Time [s]')
        plt.ylabel('h_min')
        plt.grid(True)
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    main()
