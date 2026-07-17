# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

#: <diagnostics_imports>
import torch
import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional
from tam.common.utils import TORCH_DEVICE
#: </diagnostics_imports>

#: <diagnostics_class>
class ControlDiagnostics:
    """
    Diagnostic suite for evaluating the structural stability and frequency 
    response of autoregressive components within StaticTAM models.
    """
    
    def __init__(self, model):
        self.model = model
#: </diagnostics_class>

#: <plot_bode_api>
    def plot_bode(
        self, 
        pid_feature: str, 
        target_group: Optional[str] = None, 
        cond_feature: Optional[str] = None, 
        cond_value: Optional[float] = None
    ):
        """
        Generates a Control Theory Bode Plot for an autoregressive PID component.
        
        Automatically handles both static PID constraints and Gain-Scheduled 
        (Tensor Product) PID controllers based on the provided arguments. The 
        integral window size is extracted dynamically from the fitted model.

        Args:
            pid_feature (str): The name of the autoregressive target lag feature.
            target_group (str, optional): The specific group (e.g., time of day) to analyze.
            cond_feature (str, optional): The conditional feature name (e.g., 'temperature') 
                                          if analyzing a Gain-Scheduled (Tensor Product) PID.
            cond_value (float, optional): The physical value of the condition to evaluate.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if getattr(self.model, 'coefficients_', None) is None:
            raise RuntimeError("Model must be fitted before running Control Diagnostics.")

        group_idx, group_key, target_group = self._resolve_group_routing(target_group)

        if cond_feature is not None and cond_value is not None:
            Kp, Ki, Kd, window = self._extract_dynamic_weights(
                pid_feature, cond_feature, cond_value, group_idx, group_key
            )
            title = f"Local Filter Response: {pid_feature} | {cond_feature}={cond_value} (tod={target_group})"
        else:
            Kp, Ki, Kd, window = self._extract_static_weights(
                pid_feature, group_idx, group_key
            )
            title = f"Autoregressive Filter Response: {pid_feature} (tod={target_group})"

        print(f"\n--- Dynamics for Group (tod): {target_group} ---")
        print(f"Physical Control Weights -> Kp: {Kp:.4f} | Ki: {Ki:.4f} | Kd: {Kd:.4f} | Window: {window}")

        self._render_bode_plot(Kp, Ki, Kd, window, title)
#: </plot_bode_api>

    # ---------------------------------------------------------
    # Private Helper Methods
    # ---------------------------------------------------------

#: <helpers_routing>
    def _resolve_group_routing(self, target_group: str) -> Tuple[int, str, str]:
        """
        Resolves the group index and handles type-safe key matching for the norm_params dictionary.
        """
        groups = self.model.unique_groups_
        if target_group is None:
            target_group = groups[0]
            print(f"No group specified. Defaulting to first group: '{target_group}'")
            
        try:
            group_idx = list(groups).index(target_group)
        except ValueError:
            raise ValueError(f"Group '{target_group}' not found in the fitted model.")

        group_key = target_group
        if group_key not in self.model.norm_params_:
            for k in self.model.norm_params_.keys():
                if str(k) == str(target_group):
                    group_key = k
                    break
                    
        return group_idx, group_key, target_group

    def _get_feature_scale(self, feature_name: str, group_key: str, return_center: bool = False):
        """
        Retrieves the physical scaling factor (half-amplitude) used during data normalization.
        """
        scale, center = 1.0, 0.0
        try:
            group_stats = self.model.norm_params_[group_key]
            f_max = group_stats['max'][feature_name]
            f_min = group_stats['min'][feature_name]
            
            scale = (f_max - f_min) / 2.0
            center = (f_max + f_min) / 2.0
            if scale == 0.0: 
                scale = 1.0
        except KeyError as e:
            print(f"Warning: Could not extract normalization scale for {feature_name}. Error: {e}")

        return (scale, center) if return_center else scale
#: </helpers_routing>

#: <helpers_extraction>
    def _extract_static_weights(self, pid_feature: str, group_idx: int, group_key: str) -> Tuple[float, float, float, int]:
        """
        Extracts and unscales physical coefficients for a standard PID effect.
        Returns Kp, Ki, Kd, and the integral window size.
        """
        coeff_idx = 0
        pid_effect = None
        for effect in self.model.effects_list_:
            if getattr(effect, 'effect_type', None) == "pid" and effect.feature_name == pid_feature:
                pid_effect = effect
                break
            coeff_idx += effect.get_n_coeffs()
        
        if pid_effect is None:
            raise ValueError(f"No standalone PIDEffect found for feature: {pid_feature}")

        group_coeffs = self.model.coefficients_[group_idx] 
        Kp_raw = group_coeffs[coeff_idx, 0].item()
        Ki_raw = group_coeffs[coeff_idx + 1, 0].item()
        Kd_raw = group_coeffs[coeff_idx + 2, 0].item()

        window = getattr(pid_effect, 'window', 1)
        scale = self._get_feature_scale(pid_feature, group_key)
        
        return Kp_raw / scale, Ki_raw / scale, Kd_raw / scale, window

    def _extract_dynamic_weights(
        self, pid_feature: str, cond_feature: str, cond_value: float, group_idx: int, group_key: str
    ) -> Tuple[float, float, float, int]:
        """
        Extracts, evaluates, and unscales coefficients for a Gain-Scheduled (Tensor Product) PID.
        Returns Kp, Ki, Kd, and the integral window size.
        """
        coeff_idx = 0
        target_te_effect = None
        
        for effect in self.model.effects_list_:
            if getattr(effect, 'effect_type', None) == "tensor_product":
                sub_names = [e.feature_name for e in effect.effects]
                if pid_feature in sub_names and cond_feature in sub_names:
                    target_te_effect = effect
                    break
            coeff_idx += effect.get_n_coeffs()
            
        if not target_te_effect:
            raise ValueError(f"No Tensor Product found linking {pid_feature} and {cond_feature}.")

        pid_eff, cond_eff = target_te_effect.effects[0], target_te_effect.effects[1]
        if getattr(pid_eff, 'effect_type', None) != "pid":
            raise ValueError("The PID effect must be the FIRST argument in the tensor product formula.")

        group_coeffs = self.model.coefficients_[group_idx]
        te_coeffs_raw = group_coeffs[coeff_idx : coeff_idx + target_te_effect.get_n_coeffs(), 0]
        te_coeffs_2d = te_coeffs_raw.view(3, cond_eff.get_n_coeffs())

        # 1. Dynamically extract the device where the model's coefficients reside
        target_device = te_coeffs_2d.device

        scale_cond, center_cond = self._get_feature_scale(cond_feature, group_key, return_center=True)
        cond_norm = (cond_value - center_cond) / scale_cond
        
        # 2. Force the condition tensor to spawn on that exact same device
        cond_tensor = torch.tensor([cond_norm], dtype=torch.get_default_dtype(), device=target_device)
        
        phi_cond = cond_eff.build_feature_map(cond_tensor)
        local_raw_pid = torch.matmul(te_coeffs_2d, phi_cond.squeeze())
        
        window = getattr(pid_eff, 'window', 1)
        scale_pid = self._get_feature_scale(pid_feature, group_key)
        
        return local_raw_pid[0].item() / scale_pid, local_raw_pid[1].item() / scale_pid, local_raw_pid[2].item() / scale_pid, window
#: </helpers_extraction>

#: <render_bode>
    def _render_bode_plot(self, Kp: float, Ki: float, Kd: float, window: int, title: str):
        """
        Builds the discrete digital filter transfer function and renders the Bode diagram.
        """
        import matplotlib.pyplot as plt
        Ki_w = Ki / max(1, window)
        a_1 = Kp + Ki_w + Kd
        a_2 = Ki_w - Kd
        
        num = [1.0] + [0.0] * window
        den = [1.0, -a_1, -a_2]
        for _ in range(3, window + 1):
            den.append(-Ki_w)

        sys = signal.TransferFunction(num, den, dt=1.0)
        w_freq, mag, phase = signal.dbode(sys)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        ax1.semilogx(w_freq, mag, color='#1f77b4', linewidth=2.5)
        ax1.set_ylabel('Gain Magnitude (dB)', fontweight='bold')
        ax1.grid(True, which="both", ls="--", alpha=0.6)
        
        ax2.semilogx(w_freq, phase, color='#ff7f0e', linewidth=2.5)
        ax2.set_ylabel('Phase (degrees)', fontweight='bold')
        ax2.set_xlabel('Frequency (rad/hour)', fontweight='bold')
        ax2.grid(True, which="both", ls="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

        max_pole = max(abs(p) for p in sys.poles)
        print("-" * 50)
        print(f"Maximum Pole Magnitude: {max_pole:.4f}")
        if max_pole >= 1.0:
            print("CONTROL WARNING: System is UNSTABLE (Poles outside unit circle).")
        else:
            print("CONTROL VERIFIED: System is strictly stable.")
        print("-" * 50)
#: </render_bode>