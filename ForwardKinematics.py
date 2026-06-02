import numpy as np
from math import cos, sin, pi

def get_modified_dh_matrix(alpha, a, d, theta):
    """
    Calculates the transformation matrix using Craig's Modified DH convention.

    Parameters:
    alpha (float): Twist angle (alpha_{i-1})
    a (float): Link length (a_{i-1})
    d (float): Link offset (d_i)
    theta (float): Joint angle (theta_i)

    Returns:
    np.array: 4x4 Homogeneous Transformation Matrix
    """
    c_th = cos(theta)
    s_th = sin(theta)
    c_alp = cos(alpha)
    s_alp = sin(alpha)

    # Formula from Craig's "Introduction to Robotics", Eq. 3.6
    T = np.array([
        [c_th, -s_th, 0, a],
        [s_th * c_alp, c_th * c_alp, -s_alp, -d * s_alp],
        [s_th * s_alp, c_th * s_alp, c_alp, d * c_alp],
        [0, 0, 0, 1]
    ])
    return T


def forward_kinematics(joints):
    """
    Computes the End-Effector position and orientation based on joint angles.

    Parameters:
    joints (list): List of 6 joint angles in radians [q1, q2, q3, q4, q5, q6]

    Returns:
    np.array: Final 4x4 transformation matrix (Base to End-Effector)
    """

    # Define the DH Parameters
    # Format: [alpha_(i-1), a_(i-1), d_i]
    dh_table = [
        # i=1: alpha=0, a=0, d=0
        [0, 0, 0],

        # i=2: alpha=90, a=0.040 (Shoulder Offset), d=0
        [pi / 2, 0.0435, 0],

        # i=3: alpha=0, a=0.300 (Upper Arm), d=0
        [0, 0.300, 0],

        # i=4: alpha=-90, a=0.000, d=0.334
        [pi / 2, 0.000, 0.389],

        # i=5: alpha=90, a=0, d=0
        [-pi / 2, 0, 0],

        # i=6: alpha=-90, a=0, d=0
        [pi / 2, 0, 0.203]
    ]

    # Initialize the total transformation as an Identity Matrix (4x4)
    T_total = np.eye(4)

    # Loop through each joint and multiply matrices
    for i, params in enumerate(dh_table):
        alpha = params[0]
        a = params[1]
        d = params[2]
        theta = joints[i]  # The variable input
        # Convert from physical motor angle q to mathematical DH angle theta:
        if i == 0:
            theta = -theta
        elif i == 1:
            theta = -theta + pi/2
        elif i == 2:
            theta = theta + pi/2

        # Calculate transformation for current link
        T_i = get_modified_dh_matrix(alpha, a, d, theta)

        # Multiply to the chain
        T_total = np.dot(T_total, T_i)

        # Debug: Print position of each joint frame
        # x, y, z are the first 3 rows of the 4th column
        #pos = T_total[:3, 3]
        #print(f"Frame {i + 1} pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    return T_total

def rotation_matrix_to_fixed_angles(T):
    """
    Converts a 4x4 Transformation Matrix to Roll, Pitch, Yaw (RPY).
    Returns angles in degrees.
    """

    # Check for singularity by calculating sin(y)
    sy = np.sqrt(T[0, 0] * T[0, 0] + T[1, 0] * T[1, 0])

    singular = sy < 1e-6  # If sy is close to 0, we have Gimbal Lock

    if not singular:
        # Standard formula
        gamma = np.atan2(T[2, 1], T[2, 2])
        beta = np.atan2(-T[2, 0], sy)
        alpha = np.atan2(T[1, 0], T[0, 0])
    else:
        # Special case (Gimbal Lock) - if the robot points straight up/down
        gamma = np.atan2(-T[1, 2], T[1, 1])
        beta = np.atan2(-T[2, 0], sy)
        alpha = 0

    # Convert to degrees for readability
    return np.degrees([gamma, beta, alpha])

# --- TESTING ---
if __name__ == "__main__":
    # Test 1: Zero Position (Home)
    test_joints = [-0.786, 0.304, -0.961, -0.002, -1.876, 0.000]

    final_transform = forward_kinematics(test_joints)

    print("-" * 30)
    print("Final End-Effector Matrix:")
    print(np.round(final_transform, 4))

    x = final_transform[0, 3]
    y = final_transform[1, 3]
    z = final_transform[2, 3]

    rpy = rotation_matrix_to_fixed_angles(final_transform)
    print(f"Position (XYZ): [{x:.3f}, {y:.3f}, {z:.3f}] meters")
    print(f"Orientation (RPY): [Gamma/Roll: {rpy[0]:.1f}°, Beta/Pitch: {rpy[1]:.1f}°, Alpha/Yaw: {rpy[2]:.1f}°]")