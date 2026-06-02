import numpy as np
import matplotlib.pyplot as plt
from InverseKinematics import inverse_kinematics

# DH parameters:
A2 = 0.300
D4 = 0.389
D6 = 0.203

# The Z-height of the ground relative to the robot's base frame (meters)
GROUND_Z = -0.5

# Maximum continuous/stall torque limits for the motors (Nm).
TAU_MAX = {
    2: 17.4, # Joint 2 (Shoulder)
    3: 10.8, # Joint 3 (Elbow)
    4: 2.45, # Joint 4 (Wrist 1)
    5: 2.45  # Joint 5 (Wrist 2)
}

# Physical mechanical limits for each joint [Min Radian, Max Radian]
JOINT_LIMITS = {
    1: (-3.08, 3.08),   # Joint 1 (Base)
    2: (-0.87, 1.95),   # Joint 2 (Shoulder)
    3: (-1.83, 1.83),     # Joint 3 (Elbow)
    4: (-2.09, 2.09),   # Joint 4 (Wrist 1)
    5: (-2.0, 2.0),     # Joint 5 (Wrist 2)
}

def is_within_limits(q):
    """
    Checks if the calculated inverse kinematics joint angles fall
    within the safe physical mechanical limits of the robot.
    """
    for i in range(5):
        min_limit, max_limit = JOINT_LIMITS[i + 1]
        # q[i] corresponds to joint i+1
        if not (min_limit <= q[i] <= max_limit):
            return False
    return True

def calculate_pull_forces(q):
    """
    Calculates the maximum possible upward force (Fz) allowed by each motor,
    as well as the overall combined maximum force (the weakest link).
    Returns a tuple: (dict_of_individual_forces, overall_max_force)
    """
    q1, q2, q3, q4, q5, q6 = q

    # Pre-compute the required trigonometric functions
    s2 = np.sin(q2)
    c2 = np.cos(q2)
    s23 = np.sin(q2 - q3)
    c23 = np.cos(q2 - q3)
    s4 = np.sin(q4)
    c4 = np.cos(q4)
    s5 = np.sin(q5)
    c5 = np.cos(q5)

    # Extract the elements from the 3rd row of the liner Jacobian (J_v)
    # which corresponds to the Z-axis linear velocity/force
    j_32 = - A2 * s2 - D4 * s23 + D6 * (s5 * c4 * c23 - s23 * c5)
    j_33 = D4 * s23 - D6 * (s5 * c4 * c23 - s23 * c5)
    j_34 = - D6 * s4 * s5 * s23
    j_35 = - D6 * (s5 * c23 - s23 * c4 * c5)

    # Calculate max force limited by each individual motor: Fz_max = Tau_max / |j|
    # A small epsilon (1e-6) is added to prevent division by zero
    f_max = {
        2: TAU_MAX[2] / (abs(j_32) + 1e-6),
        3: TAU_MAX[3] / (abs(j_33) + 1e-6),
        4: TAU_MAX[4] / (abs(j_34) + 1e-6),
        5: TAU_MAX[5] / (abs(j_35) + 1e-6)
    }
    # The maximum physical pull force is dictated by the weakest link
    overall_max = min(f_max[2], f_max[3], f_max[4], f_max[5])

    return f_max, overall_max

def generate_all_force_heatmaps():
    """
    Generates two separate figures:
    1. The combined overall Force Manipulability Map.
    2. A 1x2 grid showing the decoupled limits of the Shoulder (J2) and Elbow (J3).
    """
    # Workspace dimensions
    grid_size = 300
    x_vals = np.linspace(-0.7, 0.7, grid_size)
    y_vals = np.linspace(-0.7, 0.7, grid_size)

    X, Y = np.meshgrid(x_vals, y_vals)

    # Data storage maps
    Combined_Map = np.zeros_like(X)
    Decoupled_Maps = {
        2: np.zeros_like(X),
        3: np.zeros_like(X)
    }

    cap_limit = 300

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            target_x = X[i, j]
            target_y = Y[i, j]

            # Define the target TCP pose
            position = np.array([target_x, target_y, GROUND_Z])
            orientation = np.array([np.pi, 0, np.arctan2(target_y, target_x)])

            # Get joint angles from IK solver
            q = inverse_kinematics(position, orientation)

            # Ensure IK returned a valid result and respects physical joint limits
            if q is not None and len(q) == 6 and is_within_limits(q):
                indiv_forces, overall_force = calculate_pull_forces(q)

                # Cap the maximum forces for a clean visual color scale
                Combined_Map[i, j] = min(overall_force, cap_limit)

                # We only store Joint 2 and 3, as Joint 4 and 5 are functionally infinite in Z-pull
                Decoupled_Maps[2][i, j] = min(indiv_forces[2], cap_limit)
                Decoupled_Maps[3][i, j] = min(indiv_forces[3], cap_limit)
            else:
                # Mark unreachable workspace with NaNs
                Combined_Map[i, j] = np.nan
                Decoupled_Maps[2][i, j] = np.nan
                Decoupled_Maps[3][i, j] = np.nan

    # Figure 1: The combined overall heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    # Set background color to light gray for unreachable areas
    ax1.set_facecolor('#e0e0e0')

    contour1 = ax1.contourf(X, Y, Combined_Map, levels=30, cmap='viridis', vmin=0, vmax=cap_limit)
    cbar1 = fig1.colorbar(contour1, ax=ax1)
    cbar1.set_label('Maximum Z-Axis Pull Force (Newtons)', fontsize=12)

    ax1.set_title('Overall Force Manipulability: Maximum Vertical Extraction Capacity', fontsize=14)
    ax1.set_xlabel('Robot Base X-Axis (meters)', fontsize=12)
    ax1.set_ylabel('Robot Base Y-Axis (meters)', fontsize=12)

    ax1.plot(0, 0, marker='X', color='red', markersize=12, label='Robot Base (0,0)')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.axis('equal')

    # Figure 2: 1x2 Decoupled heatmap grid (shoulder vs elbow)
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Decoupled Force Limits: Shoulder (Joint 2) vs. Elbow (Joint 3)', fontsize=16, y=0.98)

    joint_names = {
        2: "Joint 2 (Shoulder) Capacity",
        3: "Joint 3 (Elbow) Capacity"
    }

    # Iterate through the 2 plots
    for ax, joint_id in zip(axes, [2, 3]):
        # Set background color to light gray for unreachable areas
        ax.set_facecolor('#e0e0e0')

        contour2 = ax.contourf(X, Y, Decoupled_Maps[joint_id], levels=30, cmap='viridis', vmin=0, vmax=cap_limit)

        ax.set_title(joint_names[joint_id], fontsize=14)
        ax.set_xlabel('Robot Base X-Axis (meters)')
        ax.set_ylabel('Robot Base Y-Axis (meters)')

        ax.plot(0, 0, marker='X', color='red', markersize=10, label='Robot Base (0,0)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axis('equal')

    # Add a single shared colorbar for the 1x2 grid
    cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    fig2.colorbar(contour2, cax=cbar_ax, label='Maximum Supported Z-Pull Force (Newtons)')

    plt.subplots_adjust(wspace=0.3, right=0.9)

    # Determining workspace radii and maximum pull force
    mask = ~np.isnan(Combined_Map)
    if np.any(mask):
        # Operating Boundaries
        R = np.sqrt(X ** 2 + Y ** 2)
        valid_radii = R[mask]
        inner_r = np.min(valid_radii)
        outer_r = np.max(valid_radii)

        # 2. Maximum Force Location
        max_idx = np.nanargmax(Combined_Map)
        r_idx, c_idx = np.unravel_index(max_idx, Combined_Map.shape)

        print("\n=== WORKSPACE ANALYSIS ===")
        print(f"Inner Operating Boundary (Radius): {inner_r:.3f} m")
        print(f"Outer Operating Boundary (Radius): {outer_r:.3f} m")
        print(f"Maximum Z-Pull Force found:      {Combined_Map[r_idx, c_idx]:.2f} N")
        print(f"  -> Located at X: {X[r_idx, c_idx]:.3f} m, Y: {Y[r_idx, c_idx]:.3f} m")
        print(f"  -> Located at Radius: {np.sqrt(X[r_idx, c_idx] ** 2 + Y[r_idx, c_idx] ** 2):.3f} m")
        print("==========================\n")

    import json

    # Udflad data og filtrér NaN fra
    mask = ~np.isnan(Combined_Map)
    x_export = X[mask].tolist()
    y_export = Y[mask].tolist()

    # Gem data for alle tre plots i én JSON-fil
    with open("heatmap_data_lite.json", "w") as f:
        json.dump({
            "x": x_export,
            "y": y_export,
            "z_combined": Combined_Map[mask].tolist(),
            "z_j2": Decoupled_Maps[2][mask].tolist(),
            "z_j3": Decoupled_Maps[3][mask].tolist()
        }, f)

    # Show both figures
    plt.show()

if __name__ == "__main__":
    generate_all_force_heatmaps()