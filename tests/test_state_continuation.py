# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import pandas as pd
import torch
from tam import StaticTAM


def test_static_tam_predict_continue_state_default_identical():
    df = pd.DataFrame({"x": torch.linspace(-1, 1, 40).numpy(), "y": torch.linspace(0, 2, 40).numpy()})
    m = StaticTAM("y ~ s(x, k=5)").fit(df.iloc[:30])
    p0 = m.predict(df.iloc[30:])
    p1 = m.predict(df.iloc[30:], continue_state=True)
    assert torch.allclose(torch.tensor(p0["Estimatedy"].to_numpy()), torch.tensor(p1["Estimatedy"].to_numpy()))
