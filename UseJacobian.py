import numpy as np
from ValidateJacobian import get_analytical_jacobian

def compute_torques_from_forces(q, F_cartesian):
    """
    Computes the required joint torques given an external force applied at the end-effector.

    Parameters:
    q (list or array): The 6 joint angles in radians.
    F_cartesian (list or array): The 3D force vector [Fx, Fy, Fz] in Newtons.

    Returns:
    numpy.ndarray: The 6D array of joint torques [tau1, tau2, tau3, tau4, tau5, tau6] in Nm.
    """
    # Convert input to numpy array to ensure matrix math works
    F = np.array(F_cartesian)

    # Get the 3x6 analytical Jacobian
    J = get_analytical_jacobian(q)

    # Calculate torques: tau = J^T * F
    tau = J.T @ F

    return tau


def compute_forces_from_torques(q, tau_joints):
    """
    Estimates the Cartesian force applied at the end-effector based on measured joint torques.

    Parameters:
    q (list or array): The 6 joint angles in radians.
    tau_joints (list or array): The 6 joint torques [tau1, tau2, tau3, tau4, tau5, tau6] in Nm.

    Returns:
    numpy.ndarray: The 3D array of estimated forces [Fx, Fy, Fz] in Newtons.
    """
    # Convert input to numpy array
    tau = np.array(tau_joints)

    # Get the 3x6 analytical Jacobian
    J = get_analytical_jacobian(q)

    # Calculate the Moore-Penrose pseudo-inverse of the transposed Jacobian
    J_T_pinv = np.linalg.pinv(J.T)

    # Calculate forces: F = (J^T)^+ * tau
    F_est = J_T_pinv @ tau

    return F_est


def main():
    # 1. Define a test configuration
    test_q = [-0.000, 1.555, -0.632, -0.000, -0.954, 0.0]

    # 2. Define a known force
    applied_force = [0.0, 0.0, -12.3]

    print(f"--- Testing at joint configuration: {test_q} ---")
    print(f"Applied External Force (N): {applied_force}")

    # --- TEST 1: Force to Torque ---
    calculated_torques = compute_torques_from_forces(test_q, applied_force)
    print("\n[Calculated Joint Torques (Nm)]")
    print(np.round(calculated_torques, 4))

    # --- TEST 2: Torque to Force ---
    # We feed the calculated torques back into the reverse function to see if we get our 20N back.
    estimated_force = compute_forces_from_torques(test_q, calculated_torques)
    print("\n[Estimated Forces from Torques (N)]")
    print(np.round(estimated_force, 4))

    # Validate the round-trip calculation
    if np.allclose(applied_force, estimated_force):
        print("\nSUCCESS: The mathematical round-trip (Force -> Torque -> Force) matches perfectly.")


if __name__ == "__main__":
    main()