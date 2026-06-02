# Static Force Analysis and Manipulability

This repository contains the mathematical modeling, kinematic verification, and force manipulability mapping software developed for a 6-axis agricultural robotic manipulator. This codebase serves as an appendix for our Bachelor Thesis (2026) at Aarhus University.

The primary objective of this software framework is to analyze and map the manipulator's pose-dependent ability to exert vertical extraction forces ($f_z$) required to remove weeds from soil without stalling the joint actuators.

## Repository Structure & Workflow

The codebase is organized chronologically following the analytical derivation to physical validation pipeline:

1. **`DetermineJacobian.py`**
   * Uses symbolic math (`sympy`) to derive the exact analytical equations for the $3 \times 6$ Linear Jacobian matrix based on Craig's Modified Denavit-Hartenberg (DH) convention.
2. **`ForwardKinematics.py`**
   * Computes the final $4 \times 4$ homogeneous transformation matrix mapping the robot's base frame to its Tool Center Point (TCP) using joint positions.
3. **`ValidateJacobian.py`**
   * Validates the analytical formulas by comparing them against a numerical Jacobian calculated using the Finite Difference Method on the forward kinematics model.
4. **`InverseKinematics.py`**
   * Geometric-analytical decoupling script that solves for the required joint angles given a target 3D spatial position and tool head orientation.
5. **`Heatmap.py`**
   * Sweeps across the 2D workspace plane at a specified ground target ($Z = -0.5\text{ m}$) to compute the maximum pull force allowed by individual actuator stall-torque limits, outputting the total force manipulability envelope.
6. **`UseJacobian.py`**
   * Maps Cartesian end-effector forces directly to joint space torques ($\tau = J^T F$) and estimates external forces via the Moore-Penrose pseudo-inverse ($J^{\dagger}$); utilized during physical hardware tests.

## Software Requirements

The scripts are implemented in Python 3 and depend on standard scientific computing libraries. You can install them via:

```bash
pip install numpy matplotlib sympy
