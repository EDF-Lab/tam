import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

output_folder = "slides/output"
os.makedirs(output_folder, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

c_primary = '#2980b9'
c_dark = '#2c3e50'
c_accent = '#3498db'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The PID Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Autoregressive Control Dynamics & Physics-Informed Inductive Biases", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.85, 0.41, 0.10, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.91, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.87, "pid(y, w=7, d_pen=10.0, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Physical Momentum over AR(p)", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Standard AR(p) treats lags as free parameters, overfitting high-frequency noise." + "\n" +
        r"• TAM projects lags into a Proportional-Integral-Derivative (PID) control space." + "\n" +
        r"  Forces the hypothesis to explicitly capture physical momentum and inertial dynamics.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Primal Mapping Function ($\Phi$) & Hyperparameters", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• w: Rolling mean window (Integral) | d_pen: Derivative stiffness multiplier", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.38, 0.40, 0.22, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.56, "Discrete Dynamic Components:", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.51, r"Proportional (P): $y_{t-1} \quad$ Integral (I): $\frac{1}{w} \sum_{k=1}^w y_{t-k} \quad$ Derivative (D): $y_{t-1} - y_{t-2}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.46, r"Explicit Primal Feature Mapping ($\Phi$):", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.41, r"$\phi_{pid}(y_{t-1}) = \left[ y_{t-1}, \ \frac{1}{w} \sum_{k=1}^w y_{t-k}, \ y_{t-1} - y_{t-2} \right]^T$", fontsize=10, color=c_dark)

ax_ins_signal = ax.inset_axes([x_left + 0.02, 0.06, 0.40, 0.30])
t_plot = np.linspace(0, 10, 200)
y_signal = np.sin(t_plot) + 0.15 * np.sin(5 * t_plot)
y_integral = np.convolve(y_signal, np.ones(20)/20, mode='same')
y_derivative = np.gradient(y_signal) * 3

ax_ins_signal.plot(t_plot, y_signal, color=c_primary, lw=1.5, label='P (Immediate State)')
ax_ins_signal.plot(t_plot, y_integral, color=c_dark, lw=2.5, linestyle='--', label='I (Rolling Inertia)')
ax_ins_signal.plot(t_plot, y_derivative, color=c_accent, lw=1, alpha=0.8, label='D (Trajectory/Velocity)')
ax_ins_signal.set_title("Decomposed Target Dynamics", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_signal.legend(loc='upper right', fontsize=7.5, frameon=True)
ax_ins_signal.tick_params(labelsize=8)
ax_ins_signal.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Anisotropic Ridge Regularization ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{pid} = \lambda \cdot \mathrm{diag}(1, 1, d_{mult})$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.61, 
        r"• Standard AR(p) treats lags equally, making Ordinary Least Squares (OLS)" + "\n" +
        r"  highly vulnerable to high-frequency noise." + "\n" +
        r"• TAM isolates the rate-of-change (Derivative) and applies a severe stiffness multiplier ($d_{mult}$)." + "\n" +
        r"• This statistical low-pass filter aggressively shrinks noisy gradient extrapolations" + "\n" +
        r"  while allowing the long-term integral momentum to drive the generalized forecast.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_zplane = ax.inset_axes([x_right + 0.04, 0.35, 0.36, 0.21])
theta = np.linspace(0, 2*np.pi, 100)
ax_ins_zplane.plot(np.cos(theta), np.sin(theta), color=c_dark, linestyle='--', label='Unit Circle ($|z|=1$)')
ax_ins_zplane.axhline(0, color='gray', lw=0.5)
ax_ins_zplane.axvline(0, color='gray', lw=0.5)
poles_x = [0.8, 0.4, 0.4, -0.1]
poles_y = [0.0, 0.5, -0.5, 0.2]
ax_ins_zplane.scatter(poles_x, poles_y, marker='x', color=c_penalty, s=80, lw=2, label='AR Characteristic Roots')
ax_ins_zplane.set_xlim(-1.2, 1.2)
ax_ins_zplane.set_ylim(-1.2, 1.2)
ax_ins_zplane.set_aspect('equal')
ax_ins_zplane.set_title("AR Characteristic Polynomial Roots (Stationarity)", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_zplane.legend(loc='upper left', fontsize=7.5, frameon=True)
ax_ins_zplane.tick_params(labelsize=8)
ax_ins_zplane.grid(False)

ax.text(x_right + 0.02, 0.26, "4. AR Characteristic Roots & Stationarity", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Because PID is an endogenous constraint, it forms a closed-loop AR(w) dynamical system." + "\n" +
        r"• We extract the roots of its characteristic polynomial to analyze its stationarity." + "\n" +
        r"• If all poles lie strictly inside the unit circle, the machine learning model is" + "\n" +
        r"  guaranteed to be BIBO stable, preventing runaway out-of-sample extrapolations.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Isolating autoregressive memory into a penalized PID space separates physical inertia from exogenous drivers, ensuring stability via constraint."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "pid_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()