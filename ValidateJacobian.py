import numpy as np
from ForwardKinematics import forward_kinematics

# DH parameters:
A1 = 0.0435
A2 = 0.300
D4 = 0.389
D6 = 0.203

def get_analytical_jacobian(q):
    """
    Calculates the 3x6 analytical Linear Jacobian using the exact formulas derived from SymPy.
    """
    q1, q2, q3, q4, q5, q6 = q

    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s4, c4 = np.sin(q4), np.cos(q4)
    s5, c5 = np.sin(q5), np.cos(q5)
    s23, c23 = np.sin(q2 - q3), np.cos(q2 - q3)

    J = np.zeros((3, 6))

    # Row 1 (X-axis derivatives)
    J[0, 0] = -A1 * s1 - A2 * s1 * s2 - D4 * s1 * s23 + D6 * ((s1 * c4 * c23 - s4 * c1) * s5 - s1 * s23 * c5)
    J[0, 1] = (A2 * c2 + D4 * c23 + D6 * (s5 * s23 * c4 + c5 * c23)) * c1
    J[0, 2] = -(D4 * c23 + D6 * s5 * s23 * c4 + D6 * c5 * c23) * c1
    J[0, 3] = -D6 * (s1 * c4 - s4 * c1 * c23) * s5
    J[0, 4] = D6 * (-(s1 * s4 + c1 * c4 * c23) * c5 - s5 * s23 * c1)
    J[0, 5] = 0.0

    # Row 2 (Y-axis derivatives)
    J[1, 0] = -A1 * c1 - A2 * s2 * c1 - D4 * s23 * c1 + D6 * ((s1 * s4 + c1 * c4 * c23) * s5 - s23 * c1 * c5)
    J[1, 1] = -(A2 * c2 + D4 * c23 + D6 * s5 * s23 * c4 + D6 * c5 * c23) * s1
    J[1, 2] = (D4 * c23 + D6 * (s5 * s23 * c4 + c5 * c23)) * s1
    J[1, 3] = -D6 * (s1 * s4 * c23 + c1 * c4) * s5
    J[1, 4] = D6 * ((s1 * c4 * c23 - s4 * c1) * c5 + s1 * s5 * s23)
    J[1, 5] = 0.0

    # Row 3 (Z-axis derivatives)
    J[2, 0] = 0.0
    J[2, 1] = -A2 * s2 - D4 * s23 + D6 * (s5 * c4 * c23 - s23 * c5)
    J[2, 2] = D4 * s23 - D6 * (s5 * c4 * c23 - s23 * c5)
    J[2, 3] = -D6 * s4 * s5 * s23
    J[2, 4] = -D6 * (s5 * c23 - s23 * c4 * c5)
    J[2, 5] = 0.0
    return J

def get_numerical_jacobian(q, delta=1e-5):
    """
    Calculates the Linear Jacobian numerically using the Finite Difference Method
    based on the validated Forward Kinematics solver.
    """
    J_num = np.zeros((3, 6))

    # Calculate the base Position Vector (X, Y, Z)
    base_matrix = forward_kinematics(q)
    P_base = base_matrix[:3, 3]

    # Perturb each joint by a tiny amount (delta) and see how the position changes
    for i in range(6):
        q_test = list(q)    # Copy the joint array
        q_test[i] += delta  # Move joint 'i' slightly

        test_matrix = forward_kinematics(q_test)
        P_test = test_matrix[:3, 3]

        # Derivative = (Change in Position) / (Change in Angle)
        J_num[:, i] = (P_test - P_base) / delta
    return J_num

def main():
    # A random, non-singular configuration to test
    test_joints = [0.099, 1.018, -1.547, -0.00, -0.577, 0.00]

    print(f"--- Validating Jacobian at q = {test_joints} ---")

    # Calculate Analytical
    J_ana = get_analytical_jacobian(test_joints)
    print("\n[Analytical Jacobian (from SymPy)]")
    print(np.round(J_ana, 4))

    # Calculate Numerical
    J_num = get_numerical_jacobian(test_joints)
    print("\n[Numerical Jacobian (from ForwardKinematics)]")
    print(np.round(J_num, 4))

    # Compare them mathematically
    # We check if the matrices are identical within a tiny tolerance
    is_valid = np.allclose(J_ana, J_num, atol=1e-4)

    print("\n--- RESULT ---")
    if is_valid:
        print("Validation successful. The analytical equations perfectly match the physical kinematics model.")
    else:
        print("Validation failed. There is a mismatch.")

if __name__ == "__main__":
    main()