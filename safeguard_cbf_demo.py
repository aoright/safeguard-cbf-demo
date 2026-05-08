"""
SafeGuard CBF-QP Demo: Real-time Safety Filtering for VLA Models
================================================================

Demonstrates how Control Barrier Functions (CBF) with Quadratic Programming (QP)
can intercept and minimally correct unsafe VLA trajectories before they reach hardware.

Key concepts:
- CBF safety constraint: h(x) >= 0 ensures system stays in safe set
- QP finds minimum correction: min ||u - u_vla||^2 s.t. safety constraints
- ISO 15066 human body part risk mapping with differentiated force limits

Author: WBBot Safety Lab
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

class Config:
    """Simulation parameters."""
    DT = 0.001             # 1kHz control loop
    T_TOTAL = 3.0          # Total simulation time (seconds)
    ROBOT_DOF = 2          # 2D planar robot for visualization
    LINK_LENGTHS = [0.4, 0.35]  # Link lengths (meters)
    
    # CBF parameters
    CBF_ALPHA = 5.0        # CBF class-K function coefficient (higher = more conservative)
    
    # ISO 15066 body part force limits (N) - simplified subset
    # Real implementation uses 29 body parts
    BODY_PARTS = {
        'head':      {'pos': np.array([0.55, 0.65]), 'radius': 0.10, 'F_max': 65,  'color': '#E53935'},
        'neck':      {'pos': np.array([0.55, 0.53]), 'radius': 0.06, 'F_max': 35,  'color': '#D32F2F'},
        'chest':     {'pos': np.array([0.5, 0.35]), 'radius': 0.12, 'F_max': 140, 'color': '#FF9800'},
        'upper_arm': {'pos': np.array([0.65, 0.4]), 'radius': 0.06, 'F_max': 150, 'color': '#FFC107'},
        'hand':      {'pos': np.array([0.72, 0.25]),'radius': 0.05, 'F_max': 190, 'color': '#4CAF50'},
    }
    
    # Safety distances (meters)
    D_SAFE = 0.08          # Minimum safe distance to body part surface
    D_INFLUENCE = 0.25     # CBF starts influencing at this distance


# ============================================================
# Robot Kinematics (2-link planar)
# ============================================================

def forward_kinematics(q, link_lengths=Config.LINK_LENGTHS):
    """Compute end-effector position from joint angles."""
    x = link_lengths[0] * np.cos(q[0]) + link_lengths[1] * np.cos(q[0] + q[1])
    y = link_lengths[0] * np.sin(q[0]) + link_lengths[1] * np.sin(q[0] + q[1])
    return np.array([x, y])

def jacobian(q, link_lengths=Config.LINK_LENGTHS):
    """Compute 2x2 Jacobian matrix."""
    l1, l2 = link_lengths
    s1 = np.sin(q[0])
    c1 = np.cos(q[0])
    s12 = np.sin(q[0] + q[1])
    c12 = np.cos(q[0] + q[1])
    
    J = np.array([
        [-l1*s1 - l2*s12, -l2*s12],
        [ l1*c1 + l2*c12,  l2*c12]
    ])
    return J

def get_link_points(q, link_lengths=Config.LINK_LENGTHS):
    """Get all link endpoint positions for visualization."""
    base = np.array([0.0, 0.0])
    elbow = np.array([
        link_lengths[0] * np.cos(q[0]),
        link_lengths[0] * np.sin(q[0])
    ])
    ee = forward_kinematics(q, link_lengths)
    return base, elbow, ee


# ============================================================
# VLA Trajectory Generator (simulated)
# ============================================================

def generate_vla_trajectory(q_start, target_pos, n_steps, hallucinate=False, 
                             hallucination_type='drift'):
    """
    Simulate VLA model output: a sequence of joint velocity commands.
    
    Args:
        q_start: Initial joint angles
        target_pos: Target end-effector position
        n_steps: Number of timesteps
        hallucinate: Whether to inject hallucination
        hallucination_type: 'drift' (gradual), 'sudden' (abrupt), 'oscillate'
    
    Returns:
        dq_sequence: (n_steps, 2) array of joint velocity commands
    """
    dq_sequence = []
    q = q_start.copy()
    
    for i in range(n_steps):
        t = i / n_steps  # Normalized time [0, 1]
        
        # Basic inverse kinematics controller (what a correct VLA would do)
        ee = forward_kinematics(q)
        J = jacobian(q)
        error = target_pos - ee
        
        # Damped least-squares IK
        J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(2))
        dq_nominal = J_pinv @ (3.0 * error)  # Proportional gain
        
        if hallucinate:
            if hallucination_type == 'drift':
                # Gradual drift: VLA slowly diverges from correct trajectory
                # Simulates autoregressive error accumulation
                drift = np.array([0.8 * t, 0.5 * t]) * 2.0
                dq_nominal += drift
                
            elif hallucination_type == 'sudden':
                # Sudden jump: cross-modal misalignment at step 40%
                if 0.35 < t < 0.55:
                    dq_nominal += np.array([4.0, -3.0])  # Large sudden deviation
                    
            elif hallucination_type == 'oscillate':
                # Oscillating: model uncertainty causes jittery output
                dq_nominal += 2.0 * np.array([
                    np.sin(20 * np.pi * t),
                    np.cos(15 * np.pi * t)
                ])
        
        # Clamp to reasonable joint velocity limits
        dq_clamped = np.clip(dq_nominal, -5.0, 5.0)
        dq_sequence.append(dq_clamped)
        
        # Integrate for next step
        q = q + dq_clamped * Config.DT
    
    return np.array(dq_sequence)


# ============================================================
# CBF Safety Constraints
# ============================================================

def cbf_constraint(q, dq, body_part_pos, body_part_radius, d_safe=Config.D_SAFE):
    """
    Compute CBF constraint value for a single body part.
    
    Safety function: h(x) = ||p_ee - p_body||^2 - (r_body + d_safe)^2
    h(x) >= 0 means safe (outside minimum distance)
    
    CBF condition: dh/dt + alpha * h >= 0
    
    Returns:
        h: Current safety margin
        constraint_value: dh + alpha * h (must be >= 0 for safety)
    """
    ee = forward_kinematics(q)
    J = jacobian(q)
    
    diff = ee - body_part_pos
    dist_sq = np.dot(diff, diff)
    min_dist_sq = (body_part_radius + d_safe) ** 2
    
    # h(x) = ||p_ee - p_body||^2 - (r + d_safe)^2
    h = dist_sq - min_dist_sq
    
    # dh/dt = 2 * (p_ee - p_body)^T * J * dq
    dh = 2 * diff @ J @ dq
    
    # CBF condition: dh + alpha * h >= 0
    alpha = Config.CBF_ALPHA
    constraint_value = dh + alpha * h
    
    return h, constraint_value


def cbf_qp_filter(q, dq_vla):
    """
    CBF-QP Safety Filter: Find minimum correction to VLA command.
    
    Solves:  min ||u - u_vla||^2
             s.t.  dh_i/dt + alpha * h_i >= 0  for all body parts
    
    Args:
        q: Current joint angles
        dq_vla: VLA's desired joint velocities
    
    Returns:
        dq_safe: Safe joint velocities (minimally modified)
        corrections: Dict of per-body-part safety info
    """
    ee = forward_kinematics(q)
    J = jacobian(q)
    
    # Collect active constraints (only body parts within influence distance)
    active_constraints = []
    corrections = {}
    
    for name, bp in Config.BODY_PARTS.items():
        diff = ee - bp['pos']
        dist = np.linalg.norm(diff)
        surface_dist = dist - bp['radius']
        
        if surface_dist < Config.D_INFLUENCE:
            min_dist_sq = (bp['radius'] + Config.D_SAFE) ** 2
            h = np.dot(diff, diff) - min_dist_sq
            
            # Gradient of h w.r.t. dq: dh/dq = 2 * diff^T * J
            grad_h = 2 * diff @ J
            
            active_constraints.append({
                'name': name,
                'h': h,
                'grad_h': grad_h,
                'F_max': bp['F_max']
            })
            
            corrections[name] = {
                'h': h,
                'dist': surface_dist,
                'active': True
            }
    
    if not active_constraints:
        # No constraints active, pass through VLA command
        return dq_vla.copy(), corrections
    
    # Solve QP using scipy (for simplicity; production uses OSQP/qpOASES)
    def objective(u):
        return 0.5 * np.sum((u - dq_vla) ** 2)
    
    def objective_grad(u):
        return u - dq_vla
    
    constraints = []
    for c in active_constraints:
        alpha = Config.CBF_ALPHA
        # Scale alpha inversely by ISO 15066 force limit (lower limit = MORE conservative)
        alpha_scaled = alpha * (190.0 / max(c['F_max'], 1.0))  # Head gets ~3x more aggressive
        
        def make_constraint(grad_h, h, alpha_s):
            return {
                'type': 'ineq',
                'fun': lambda u, g=grad_h, hv=h, a=alpha_s: g @ u + a * hv,
                'jac': lambda u, g=grad_h: g
            }
        
        constraints.append(make_constraint(c['grad_h'], c['h'], alpha_scaled))
    
    # Joint velocity limits
    bounds = [(-5.0, 5.0)] * 2
    
    result = minimize(
        objective, dq_vla, jac=objective_grad,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 50}
    )
    
    dq_safe = result.x if result.success else dq_vla * 0.5  # Fallback: reduce speed
    
    return dq_safe, corrections


# ============================================================
# Simulation Engine
# ============================================================

def run_simulation(hallucinate=True, hallucination_type='drift'):
    """Run a complete simulation with and without safety filter."""
    
    n_steps = int(Config.T_TOTAL / Config.DT)
    q_start = np.array([np.pi/4, np.pi/6])   # Initial joint config
    target = np.array([0.55, 0.40])            # Target near human chest
    
    # Generate VLA commands
    dq_vla_cmds = generate_vla_trajectory(
        q_start, target, n_steps, 
        hallucinate=hallucinate, 
        hallucination_type=hallucination_type
    )
    
    # --- Run WITHOUT safety filter ---
    q_unsafe = q_start.copy()
    traj_unsafe = [forward_kinematics(q_unsafe)]
    q_hist_unsafe = [q_unsafe.copy()]
    
    for i in range(n_steps):
        q_unsafe = q_unsafe + dq_vla_cmds[i] * Config.DT
        traj_unsafe.append(forward_kinematics(q_unsafe))
        q_hist_unsafe.append(q_unsafe.copy())
    
    traj_unsafe = np.array(traj_unsafe)
    
    # --- Run WITH CBF-QP safety filter ---
    q_safe = q_start.copy()
    traj_safe = [forward_kinematics(q_safe)]
    q_hist_safe = [q_safe.copy()]
    h_history = {name: [] for name in Config.BODY_PARTS}
    correction_norms = []
    
    for i in range(n_steps):
        dq_safe, corrections = cbf_qp_filter(q_safe, dq_vla_cmds[i])
        correction_norm = np.linalg.norm(dq_safe - dq_vla_cmds[i])
        correction_norms.append(correction_norm)
        
        q_safe = q_safe + dq_safe * Config.DT
        traj_safe.append(forward_kinematics(q_safe))
        q_hist_safe.append(q_safe.copy())
        
        # Record safety margins
        for name, bp in Config.BODY_PARTS.items():
            ee = forward_kinematics(q_safe)
            diff = ee - bp['pos']
            dist = np.linalg.norm(diff) - bp['radius']
            h_history[name].append(dist - Config.D_SAFE)
    
    traj_safe = np.array(traj_safe)
    time_axis = np.linspace(0, Config.T_TOTAL, n_steps)
    
    return {
        'traj_unsafe': traj_unsafe,
        'traj_safe': traj_safe,
        'q_hist_unsafe': np.array(q_hist_unsafe),
        'q_hist_safe': np.array(q_hist_safe),
        'h_history': h_history,
        'correction_norms': np.array(correction_norms),
        'time': time_axis,
        'target': target,
        'q_start': q_start,
    }


# ============================================================
# Visualization
# ============================================================

def plot_results(results, hallucination_type='drift'):
    """Generate publication-quality figures."""
    
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Color scheme
    UNSAFE_COLOR = '#FF5252'
    SAFE_COLOR = '#69F0AE'
    BG_COLOR = '#1a1a2e'
    TEXT_COLOR = '#E0E0E0'
    GRID_COLOR = '#333355'
    
    # ---- Figure 1: Trajectory Comparison ----
    ax1 = fig.add_subplot(221)
    ax1.set_facecolor(BG_COLOR)
    ax1.set_title('Trajectory: VLA Hallucination vs CBF-QP Corrected', 
                   color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=12)
    
    # Draw human body parts (ISO 15066 zones)
    for name, bp in Config.BODY_PARTS.items():
        # Danger zone (body part + safety margin)
        danger_circle = plt.Circle(bp['pos'], bp['radius'] + Config.D_SAFE, 
                                    color=bp['color'], alpha=0.15, linestyle='--', fill=True)
        ax1.add_patch(danger_circle)
        # Body part
        body_circle = plt.Circle(bp['pos'], bp['radius'], 
                                  color=bp['color'], alpha=0.4, fill=True)
        ax1.add_patch(body_circle)
        ax1.annotate(f'{name}\nF≤{bp["F_max"]}N', bp['pos'], 
                     color=TEXT_COLOR, fontsize=7, ha='center', va='center',
                     fontweight='bold')
    
    # Draw trajectories
    traj_u = results['traj_unsafe']
    traj_s = results['traj_safe']
    
    # Unsafe trajectory
    ax1.plot(traj_u[:, 0], traj_u[:, 1], color=UNSAFE_COLOR, alpha=0.8, 
             linewidth=2, label='❌ VLA (hallucinated)', linestyle='--')
    ax1.scatter(traj_u[-1, 0], traj_u[-1, 1], color=UNSAFE_COLOR, s=100, 
                zorder=5, marker='x', linewidths=3)
    
    # Safe trajectory
    ax1.plot(traj_s[:, 0], traj_s[:, 1], color=SAFE_COLOR, alpha=0.9, 
             linewidth=2.5, label='✅ CBF-QP corrected')
    ax1.scatter(traj_s[-1, 0], traj_s[-1, 1], color=SAFE_COLOR, s=100, 
                zorder=5, marker='o')
    
    # Draw robot arm at final safe position
    q_final = results['q_hist_safe'][-1]
    base, elbow, ee = get_link_points(q_final)
    ax1.plot([base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]], 
             color='#90CAF9', linewidth=4, alpha=0.7, solid_capstyle='round')
    ax1.scatter([base[0], elbow[0]], [base[1], elbow[1]], 
                color='#64B5F6', s=60, zorder=5)
    
    # Start and target
    start_pos = forward_kinematics(results['q_start'])
    ax1.scatter(*start_pos, color='#FFF176', s=120, zorder=5, marker='*', label='Start')
    ax1.scatter(*results['target'], color='#CE93D8', s=120, zorder=5, marker='D', label='Target')
    
    ax1.set_xlim(-0.1, 0.9)
    ax1.set_ylim(-0.1, 0.8)
    ax1.set_xlabel('X (m)', color=TEXT_COLOR)
    ax1.set_ylabel('Y (m)', color=TEXT_COLOR)
    ax1.legend(loc='lower left', fontsize=9, facecolor='#2a2a4e', edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR)
    ax1.grid(True, alpha=0.2, color=GRID_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    
    # ---- Figure 2: Safety Margin h(x) over time ----
    ax2 = fig.add_subplot(222)
    ax2.set_facecolor(BG_COLOR)
    ax2.set_title('Safety Margin h(x) Over Time', 
                   color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=12)
    
    time = results['time']
    colors = ['#E53935', '#D32F2F', '#FF9800', '#FFC107', '#4CAF50']
    for i, (name, h_vals) in enumerate(results['h_history'].items()):
        if len(h_vals) > 0:
            ax2.plot(time[:len(h_vals)], h_vals, label=name, 
                     color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
    
    ax2.axhline(y=0, color=UNSAFE_COLOR, linestyle='--', alpha=0.5, label='Safety boundary (h=0)')
    ax2.fill_between(time, -0.15, 0, color=UNSAFE_COLOR, alpha=0.08)
    ax2.annotate('UNSAFE ZONE', xy=(Config.T_TOTAL*0.5, -0.07), 
                 color=UNSAFE_COLOR, fontsize=10, ha='center', alpha=0.5, fontweight='bold')
    
    ax2.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax2.set_ylabel('h(x) = dist - d_safe (m)', color=TEXT_COLOR)
    ax2.legend(fontsize=8, facecolor='#2a2a4e', edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax2.grid(True, alpha=0.2, color=GRID_COLOR)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.set_ylim(-0.15, max(0.5, max(max(v) for v in results['h_history'].values() if v) + 0.1))
    
    # ---- Figure 3: Correction Effort ----
    ax3 = fig.add_subplot(223)
    ax3.set_facecolor(BG_COLOR)
    ax3.set_title('CBF-QP Correction Effort ||u_safe - u_vla||', 
                   color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=12)
    
    ax3.fill_between(time, 0, results['correction_norms'], 
                      color='#7C4DFF', alpha=0.3)
    ax3.plot(time, results['correction_norms'], color='#B388FF', linewidth=1.5)
    
    # Highlight peak correction
    peak_idx = np.argmax(results['correction_norms'])
    peak_val = results['correction_norms'][peak_idx]
    ax3.annotate(f'Peak: {peak_val:.2f} rad/s\n@ t={time[peak_idx]:.2f}s',
                 xy=(time[peak_idx], peak_val),
                 xytext=(time[peak_idx]+0.3, peak_val*0.8),
                 color='#E0E0E0', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#B388FF', lw=1.5))
    
    ax3.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax3.set_ylabel('Correction magnitude (rad/s)', color=TEXT_COLOR)
    ax3.grid(True, alpha=0.2, color=GRID_COLOR)
    ax3.tick_params(colors=TEXT_COLOR)
    
    # ---- Figure 4: ISO 15066 Risk Map ----
    ax4 = fig.add_subplot(224)
    ax4.set_facecolor(BG_COLOR)
    ax4.set_title('ISO 15066 Body Part Force Limits (Simplified)', 
                   color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=12)
    
    body_names = list(Config.BODY_PARTS.keys())
    f_limits = [Config.BODY_PARTS[n]['F_max'] for n in body_names]
    bar_colors = [Config.BODY_PARTS[n]['color'] for n in body_names]
    
    bars = ax4.barh(body_names, f_limits, color=bar_colors, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    for bar, val in zip(bars, f_limits):
        ax4.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                 f'{val} N', va='center', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Maximum Permissible Force (N)', color=TEXT_COLOR)
    ax4.set_xlim(0, 220)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.2, color=GRID_COLOR, axis='x')
    ax4.tick_params(colors=TEXT_COLOR)
    
    # Add note
    ax4.text(110, 4.7, 'Head/Neck: 5x stricter than Hand',
             color='#FFAB91', fontsize=9, ha='center', style='italic')
    
    plt.tight_layout(pad=2.0)
    
    # Save
    output_path = 'safeguard_cbf_demo_results.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), 
                bbox_inches='tight', pad_inches=0.3)
    print(f'\n✅ Results saved to: {output_path}')
    
    plt.show()
    return fig


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  SafeGuard CBF-QP Demo")
    print("  VLA Safety Filter with ISO 15066 Body Part Mapping")
    print("=" * 60)
    
    # Run three scenarios
    scenarios = [
        ('drift', 'Autoregressive drift hallucination'),
        ('sudden', 'Cross-modal sudden misalignment'),
        ('oscillate', 'Model uncertainty oscillation'),
    ]
    
    for h_type, desc in scenarios:
        print(f'\n--- Scenario: {desc} ---')
        
        results = run_simulation(hallucinate=True, hallucination_type=h_type)
        
        # Compute summary stats
        min_h = {name: min(vals) if vals else float('inf') 
                 for name, vals in results['h_history'].items()}
        
        violations_unsafe = 0
        q = results['q_start'].copy()
        n_steps = len(results['correction_norms'])
        
        print(f'  Timesteps:           {n_steps} ({n_steps * Config.DT:.1f}s @ 1kHz)')
        print(f'  Mean correction:     {results["correction_norms"].mean():.4f} rad/s')
        print(f'  Peak correction:     {results["correction_norms"].max():.4f} rad/s')
        print(f'  Min safety margin:')
        for name, val in min_h.items():
            status = '✅ SAFE' if val >= 0 else '❌ VIOLATED'
            print(f'    {name:12s}: h_min = {val:+.4f}m  {status}')
    
    # Plot the last scenario
    print('\n📊 Generating visualization for last scenario...')
    plot_results(results, hallucination_type=scenarios[-1][0])
    
    print('\n' + '=' * 60)
    print('  Demo complete!')
    print('  Full SafeGuard SDK: contact us for enterprise access')
    print('=' * 60)


if __name__ == '__main__':
    main()
