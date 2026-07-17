# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

import pytest
import torch
from tam.common.hardware import hw
from tam.model._memory import get_safe_chunk_size

def test_hardware_detection():
    """Ensures the Hardware Abstraction Layer selects a valid device."""
    device = hw.device
    assert isinstance(device, torch.device), "Hardware manager did not return a valid torch.device"

    # It should fallback to CPU if no GPU/MPS is available on the CI runner
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        assert device.type == 'cpu'

def test_chunking_logic():
    """Ensures the anti-OOM chunk calculation returns valid batch sizes."""
    safe_batch = get_safe_chunk_size(
        total_samples=1000,
        total_d=500,
        device=hw.device
    )

    assert isinstance(safe_batch, int), "Batch size must be an integer."
    assert 1 <= safe_batch <= 1000, "Safe batch size is out of logical bounds."
