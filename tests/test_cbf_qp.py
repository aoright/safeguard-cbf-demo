"""
Unit tests for SafeGuard CBF-QP Safety Filter and Trajectory Constraints.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import safeguard_cbf_demo as cbf_demo
from safeguard_cbf_demo import (
    Config,
    forward_kinematics,
    jacobian,
    cbf_constraint,
    cbf_qp_filter,
    generate_vla_trajectory,
    run_simulation,
)


class TestKinematicsAndJacobian(unittest.TestCase):
    def test_forward_kinematics(self):
        # Angle [0, 0] -> arm fully extended along x-axis
        q = np.array([0.0, 0.0])
        ee = forward_kinematics(q)
        expected_x = Config.LINK_LENGTHS[0] + Config.LINK_LENGTHS[1]
        expected_y = 0.0
        np.testing.assert_allclose(ee, [expected_x, expected_y], atol=1e-5)

    def test_jacobian_finite_difference(self):
        q = np.array([np.pi / 4, np.pi / 3])
        J = jacobian(q)
        
        eps = 1e-6
        J_num = np.zeros((2, 2))
        for i in range(2):
            q_plus = q.copy()
            q_plus[i] += eps
            q_minus = q.copy()
            q_minus[i] -= eps
            ee_plus = forward_kinematics(q_plus)
            ee_minus = forward_kinematics(q_minus)
            J_num[:, i] = (ee_plus - ee_minus) / (2 * eps)
            
        np.testing.assert_allclose(J, J_num, atol=1e-4)


class TestQPSolverBoundaryConditions(unittest.TestCase):
    def test_inactive_constraints_passthrough(self):
        # Position far away from any obstacle
        q = np.array([-np.pi / 2, 0.0])  # End-effector pointing downwards, far from body parts
        dq_vla = np.array([1.0, -0.5])
        
        dq_safe, corrections = cbf_qp_filter(q, dq_vla)
        
        # When no constraints are active, dq_safe should be identical to dq_vla
        np.testing.assert_allclose(dq_safe, dq_vla, atol=1e-5)
        self.assertEqual(len(corrections), 0)

    def test_active_constraint_modification(self):
        # Position near the head obstacle where dq_vla moves directly into the head
        q = np.array([0.8, 0.5])
        ee = forward_kinematics(q)
        head_pos = Config.BODY_PARTS['head']['pos']
        head_radius = Config.BODY_PARTS['head']['radius']
        
        # Unsafe direction (moving towards head)
        J = jacobian(q)
        diff = head_pos - ee
        dq_unsafe = J.T @ diff  # velocity towards head
        
        dq_safe, corrections = cbf_qp_filter(q, dq_unsafe)
        
        # Verify that dq_safe was modified to satisfy the CBF inequality constraint
        h, constraint_val = cbf_constraint(q, dq_safe, head_pos, head_radius)
        self.assertGreaterEqual(constraint_val, -1e-4)

    def test_velocity_limit_bounds(self):
        q = np.array([0.1, 0.1])
        dq_extreme = np.array([10.0, -10.0])  # Exceeds [-5.0, 5.0] bounds
        
        dq_safe, _ = cbf_qp_filter(q, dq_extreme)
        
        self.assertTrue(np.all(dq_safe >= -5.0 - 1e-5))
        self.assertTrue(np.all(dq_safe <= 5.0 + 1e-5))

    def test_solver_failure_fallback(self):
        q = np.array([0.5, 0.5])
        dq_vla = np.array([2.0, -1.0])
        
        # Mock scipy minimize to simulate solver failure
        fake_result = MagicMock()
        fake_result.success = False
        
        with patch('safeguard_cbf_demo.minimize', return_value=fake_result):
            dq_safe, _ = cbf_qp_filter(q, dq_vla)
            # Fallback policy reduces speed by 50%
            np.testing.assert_allclose(dq_safe, dq_vla * 0.5, atol=1e-5)


class TestTrajectoryConstraintViolations(unittest.TestCase):
    def test_cbf_constraint_calculation(self):
        q = np.array([0.5, 0.5])
        dq = np.array([0.1, 0.1])
        body_pos = np.array([0.5, 0.5])
        body_radius = 0.1
        
        h, constraint_val = cbf_constraint(q, dq, body_pos, body_radius)
        ee = forward_kinematics(q)
        expected_h = np.dot(ee - body_pos, ee - body_pos) - (body_radius + Config.D_SAFE) ** 2
        
        self.assertAlmostEqual(h, expected_h, places=5)

    def test_vla_trajectory_generator_modes(self):
        q_start = np.array([np.pi/4, np.pi/6])
        target = np.array([0.55, 0.40])
        n_steps = 100
        
        for mode in ['drift', 'sudden', 'oscillate']:
            dq_seq = generate_vla_trajectory(
                q_start, target, n_steps, hallucinate=True, hallucination_type=mode
            )
            self.assertEqual(dq_seq.shape, (n_steps, 2))
            self.assertTrue(np.all(dq_seq >= -5.0))
            self.assertTrue(np.all(dq_seq <= 5.0))

    def test_simulation_safety_enforcement(self):
        # Test drift scenario simulation results
        results = run_simulation(hallucinate=True, hallucination_type='drift')
        
        # Unsafe trajectory should violate safety margins
        ee_unsafe_final = results['traj_unsafe'][-1]
        
        # Check that CBF safety filter kept distance above critical thresholds
        for name, h_vals in results['h_history'].items():
            if h_vals:
                min_h = min(h_vals)
                # Minimum h value with CBF safety filter should remain near or above 0
                self.assertGreater(min_h, -0.005, f"Constraint for {name} severely violated: {min_h}")

    def test_correction_effort_recorded(self):
        results = run_simulation(hallucinate=True, hallucination_type='sudden')
        self.assertEqual(len(results['correction_norms']), int(Config.T_TOTAL / Config.DT))
        self.assertTrue(np.any(results['correction_norms'] > 0))


if __name__ == '__main__':
    unittest.main()
