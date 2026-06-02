import sympy as sp

def get_symbolic_dh_matrix(alpha, a, d, theta):
    """
    Calculates the transformation matrix using Craig's Modified DH convention.
    """
    c_th = sp.cos(theta)
    s_th = sp.sin(theta)
    c_alp = sp.cos(alpha)
    s_alp = sp.sin(alpha)

    # Formula from Craig's "Introduction to Robotics", Eq. 3.6
    T = sp.Matrix([
        [c_th, -s_th, 0, a],
        [s_th * c_alp, c_th * c_alp, -s_alp, -d * s_alp],
        [s_th * s_alp, c_th * s_alp, c_alp, d * c_alp],
        [0, 0, 0, 1]
    ])
    return T

def main():
    # Define symbolic joint variables (q1 to q6)
    q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

    # Define symbolic physical dimensions
    a1, a2, d4, d6 = sp.symbols('a1 a2 d4 d6')

    # Define the DH table parameters [alpha_(i-1), a_(i-1), d_i, theta_i]
    dh_table = [
        [0, 0, 0, -q1],
        [sp.pi / 2, a1, 0, -q2 + sp.pi / 2],
        [0, a2, 0, q3 + sp.pi / 2],
        [sp.pi / 2, 0, d4, q4],
        [-sp.pi / 2, 0, 0, q5],
        [sp.pi / 2, 0, d6, q6]
    ]

    T_total = sp.eye(4)  # Initialize as 4x4 Identity Matrix for the complete transformation matrix

    for params in dh_table:
        alpha, a, d, theta = params
        T_total = T_total * get_symbolic_dh_matrix(alpha, a, d, theta)

    # Extract the translation vector (X, Y, Z) from the final transformation matrix
    P = sp.Matrix([T_total[0, 3], T_total[1, 3], T_total[2, 3]])

    # Define the vector of joint variables to differentiate against
    Q = sp.Matrix([q1, q2, q3, q4, q5, q6])

    # Calculate the Linear Jacobian Matrix (3x6) using partial derivatives
    J_v = P.jacobian(Q)

    # Simplify the trigonometric equations
    J_v_simplified = sp.simplify(J_v)

    # Print the results
    print("Linear Jacobian Matrix:")
    print(J_v_simplified)

if __name__ == "__main__":
    main()