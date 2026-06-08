"""
Hardware Abstraction Layer (HAL) for the TAM framework.

This module provides a unified interface for device management, memory queries,
and safe execution routing across diverse hardware backends (CUDA, MPS, XPU, CPU).
It dynamically probes hardware capabilities at runtime to prevent crashes during
heavy linear algebra operations.
"""

import warnings
import psutil
import torch
from typing import Tuple


class HardwareManager:
    """
    Singleton manager for cross-platform hardware operations.
    
    Dynamically routes tensor operations and memory management based on available
    compute backends, ensuring safe fallbacks for unsupported operations.
    """

    def __init__(self):
        self.device, self.backend = self._detect_hardware()
        
        # Capability flags determined at runtime
        self.supports_float64 = True
        self.supports_linalg = True
        
        self._run_smoke_tests()
        self._print_hardware_status()

    def _print_hardware_status(self) -> None:
        """Prints a one-time summary of the detected hardware configuration."""
        sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        print("\n" + "="*50)
        print("TAM Hardware Initialization")
        print("="*50)
        
        if self.backend == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            print("Compute Backend : CUDA (GPU)")
            print(f"Device          : {gpu_name}")
            print(f"Dedicated VRAM  : {vram_gb:.1f} GB")
        elif self.backend == "mps":
            print("Compute Backend : Apple Metal (MPS)")
            print("Device          : Apple Silicon")
            print(f"Unified Memory  : {sys_ram_gb:.1f} GB (Shared)")
        elif self.backend == "xpu":
            gpu_name = torch.xpu.get_device_name(self.device)
            vram_gb = torch.xpu.get_device_properties(self.device).total_memory / (1024 ** 3)
            print("Compute Backend : Intel XPU (GPU)")
            print(f"Device          : {gpu_name}")
            print(f"Dedicated VRAM  : {vram_gb:.1f} GB")
        else:
            print("Compute Backend : CPU (No GPU detected/supported)")
            print("Device          : Host CPU")

        print(f"System RAM      : {sys_ram_gb:.1f} GB")
        print(f"Float64 Native  : {'Yes' if self.supports_float64 else 'No (Fallback to Float32)'}")
        print("="*50 + "\n")

    def _detect_hardware(self) -> Tuple[torch.device, str]:
        """
        Detects and returns the most capable available compute backend.

        This method probes the system for hardware accelerators in the following order:
        1. NVIDIA CUDA (GPU)
        2. Apple Metal Performance Shaders (MPS)
        3. Intel XPU (Supports both legacy `intel_extension_for_pytorch` and PyTorch 2.5+ native XPU)
        4. CPU (Ultimate fallback)

        Returns:
            Tuple[torch.device, str]: A tuple containing the PyTorch device object 
            and its string identifier (e.g., 'cuda', 'mps', 'xpu', or 'cpu').
        """
        #  NVIDIA CUDA
        if torch.cuda.is_available():
            return torch.device("cuda:0"), "cuda"
            
        #  Apple Metal (MPS)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
            
        #  Intel GPU (XPU) - Supports both the Legacy Extension and Native Mode
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            pass  # Extension is unavailable, but native support might still exist
            
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_name = torch.xpu.get_device_name(0)
            
            # --- INTELLIGENT CPU FALLBACK ---
            # Integrated graphics lack the hardware kernels for complex tensor math.
            # We intercept them here and route to the highly-optimized CPU MKL instead.
            if "UHD" in device_name or "Iris" in device_name or "Integrated" in device_name:
                print(f"Notice: Unsupported Integrated GPU detected ({device_name}).")
                print("Routing compute to CPU for maximum stability and MKL acceleration.")
                return torch.device("cpu"), "cpu"
                
            # If it passes the check, it's a dedicated Arc/Max GPU. Safe to proceed.
            return torch.device("xpu:0"), "xpu"
            
        #  Ultimate Fallback
        return torch.device("cpu"), "cpu"

    def _run_smoke_tests(self) -> None:
        """
        Probes the backend for specific data type and operation support.
        Sets capability flags to guide safe execution routing.
        """
        if self.backend == "cpu":
            return  # CPU reference implementation supports all operations

        #  Probe Float64 matrix multiplication support
        try:
            a = torch.ones((2, 2), dtype=torch.float64, device=self.device)
            _ = a @ a
        except (RuntimeError, TypeError):
            self.supports_float64 = False
            warnings.warn(
                f"Backend '{self.backend}' lacks native float64 support. "
                "Framework will fallback to float32 or CPU routing."
            )

        #  Probe Linear System Solver support
        try:
            # We test with float64 as it is the target standard for the framework
            test_dtype = torch.float64 if self.supports_float64 else torch.float32
            A = torch.eye(2, dtype=test_dtype, device=self.device)
            B = torch.ones((2, 1), dtype=test_dtype, device=self.device)
            _ = torch.linalg.solve(A, B)
        except (RuntimeError, NotImplementedError):
            self.supports_linalg = False
            warnings.warn(
                f"Backend '{self.backend}' cannot solve linear systems for the "
                "required dtype natively. Routing linalg operations to CPU."
            )

    def get_available_memory(self) -> int:
        """
        Queries the host or device for available memory in bytes.
        
        Returns:
            int: Safely allocatable memory in bytes.
        """
        if self.backend == "cuda":
            torch.cuda.empty_cache()
            free_memory, _ = torch.cuda.mem_get_info(self.device)
            return free_memory
        elif self.backend == "xpu":
            torch.xpu.empty_cache()
            free_memory, _ = torch.xpu.mem_get_info(self.device)
            return free_memory
        else:
            # For CPU and MPS (Unified Memory), we rely on system RAM
            return psutil.virtual_memory().available

    def empty_cache(self) -> None:
        """Clears the computational backend's memory cache."""
        if self.backend == "cuda":
            torch.cuda.empty_cache()
        elif self.backend == "mps":
            torch.mps.empty_cache()
        elif self.backend == "xpu":
            torch.xpu.empty_cache()

    def safe_solve(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A hardware-agnostic wrapper for solving linear systems (AX = B).
        
        Safely routes the computation to the CPU if the native backend lacks 
        support for the specific operation or dtype, preventing runtime crashes.
        """
        if self.supports_linalg:
            try:
                return torch.linalg.solve(A, B)
            except (RuntimeError, NotImplementedError):
                # Catch-all for unforeseen backend limitations on specific matrices
                pass
                
        # Fallback routing: Transfer to CPU, solve, and return to original device
        A_cpu = A.cpu()
        B_cpu = B.cpu()
        result_cpu = torch.linalg.solve(A_cpu, B_cpu)
        return result_cpu.to(self.device)

    def handle_oom(
        self, 
        current_batch: int, 
        context: str, 
        allow_cpu_fallback: bool = False
    ) -> Tuple[int, torch.device]:
        """
        Handles Out-Of-Memory events by clearing cache and reducing batch sizes.
        
        Args:
            current_batch: The batch size that triggered the OOM.
            context: String describing the operation that failed.
            allow_cpu_fallback: Whether to attempt execution on CPU if batch hits 1.
            
        Returns:
            Tuple[int, torch.device]: The new safe batch size and compute device.
        """
        self.empty_cache()

        if current_batch <= 1:
            if allow_cpu_fallback and self.backend != "cpu":
                warnings.warn(
                    f"[OOM] VRAM exhausted during {context}. "
                    "Falling back to CPU execution."
                )
                return current_batch, torch.device("cpu")
            else:
                raise MemoryError(
                    f"Memory exhausted during {context} on {self.backend}. "
                    "Batch size 1 is too large to process."
                )

        new_batch = max(1, current_batch // 2)
        warnings.warn(
            f"[OOM] Alert during {context} on {self.backend}: "
            f"Reducing batch from {current_batch} to {new_batch}."
        )
        
        return new_batch, self.device


# Global instantiation for the framework
hw = HardwareManager()