"""
Collision model for the Universal Robots UR5e, approximated using spheres.
Positions are relative to the link's origin frame defined in the URDF.
Radii are in meters.

NOTE: These are ESTIMATES and may need refinement for high-accuracy collision checking.
"""

import numpy as np

# Define collision spheres for each link
# Format: ([x, y, z], radius) relative to the link's origin frame

# Link 1: shoulder_link
shoulder_spheres_pos = (
    [0.0, 0.0, -0.05],  # Near base connection
    [0.0, 0.0, 0.08],   # Upper part of shoulder
)
shoulder_spheres_rad = (0.10, 0.10)

# Link 2: upper_arm_link (Length ~0.425m along its X-axis)
upper_arm_spheres_pos = (
    [-0.1, 0.0, 0.138],   # Near shoulder joint
    [-0.25, 0.0, 0.138],  # Middle of upper arm
    [-0.4, 0.0, 0.138],   # Near elbow joint
)
upper_arm_spheres_rad = (0.07, 0.07, 0.07)

# Link 3: forearm_link (Length ~0.392m along its X-axis)
forearm_spheres_pos = (
    [-0.1, 0.0, 0.007],   # Near elbow joint
    [-0.25, 0.0, 0.007],  # Middle of forearm
    [-0.35, 0.0, 0.007],  # Near wrist 1 joint
)
forearm_spheres_rad = (0.06, 0.06, 0.06)

# Link 4: wrist_1_link
wrist_1_spheres_pos = (
    [0.0, -0.05, -0.127],   # Centered around wrist 1 body
)
wrist_1_spheres_rad = (0.06,)

# Link 5: wrist_2_link
wrist_2_spheres_pos = (
    [0.0, 0.0, -0.0997],   # Centered around wrist 2 body
)
wrist_2_spheres_rad = (0.06,)

# Link 6: wrist_3_link
wrist_3_spheres_pos = (
    [0.0, 0.0, -0.05],   # Near the tool flange, adjusted Z
)
wrist_3_spheres_rad = (0.06,)

# Combine into the final data structure
# NOTE: The order MUST match the order of links in the parsed URDF (usually base -> end-effector)
# We might skip the base_link_inertia as it's often fixed or part of the environment.
position_list = (
    shoulder_spheres_pos, 
    upper_arm_spheres_pos,
    forearm_spheres_pos,
    wrist_1_spheres_pos,
    wrist_2_spheres_pos,
    wrist_3_spheres_pos,
    )

radii_list = (
    shoulder_spheres_rad,
    upper_arm_spheres_rad,
    forearm_spheres_rad,
    wrist_1_spheres_rad,
    wrist_2_spheres_rad,
    wrist_3_spheres_rad,
)

ur5e_collision_data = {
    "positions": position_list,
    "radii": radii_list,
}

# Optional: Add data for flange or tool0 if needed for finer collision checking at TCP
# flange_spheres_pos = ([0.0, 0.0, 0.0],)
# flange_spheres_rad = (0.05,)
# ur5e_collision_data["positions"] += (flange_spheres_pos,)
# ur5e_collision_data["radii"] += (flange_spheres_rad,)