# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Bounded island rewards (opt-in): the bandit that draws the islands is rewarded with 1 / (1 + scaled error), in (0, 1], instead of
1 / scaled error, which is unbounded and lets the best island dominate.
"""

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.knowledge_graph import KnowledgeGraph
from tam.model.autotam.population_nodes import LinearIsland

GRIDS = {"l": {"ap": [-9.0]}}
def test_the_bounded_reward_is_in_zero_one_and_the_default_reward_is_not():
    from tam.model.autotam.drag_tam import DragTAM
    assert DragTAM("y").ucb_E == 2.0 and DragTAM("y", bounded_rewards=True).ucb_E == 0.1
    rewards = {}

    class Recorder(KnowledgeGraph):
        def update_island_reward(self, island_name, reward):
            rewards.setdefault(self.flag, []).append(reward)
            super().update_island_reward(island_name, reward)

    for flag in (False, True):
        engine = DragTAM("y", population_size=6, n_generations=2, bounded_rewards=flag)
        engine.kg = Recorder()
        engine.kg.flag = flag
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"x": rng.normal(size=300)})
        frame["y"] = 3 * frame["x"] + rng.normal(scale=0.05, size=300)
        engine.optimize(cv_folds=[(frame.iloc[:200], frame.iloc[200:])], islands=[LinearIsland()], search_space={"x": {
            "eligible_effects": ["l"], "grids": GRIDS, "max_active_effects": 2, "topology": "continuous", "profile": {"linear": True}}})
    assert max(r for r in rewards[True] if r is not None) <= 1.0
    assert max(r for r in rewards[False] if r is not None) > 5.0


def test_the_argument_reaches_the_engine(tmp_path):
    assert AutoTAM("y ~ AutoPipe(x1)", export_dir=str(tmp_path)).discoverer.bounded_rewards is False
    assert AutoTAM("y ~ AutoPipe(x1)", bounded_rewards=True, export_dir=str(tmp_path)).discoverer.bounded_rewards is True
