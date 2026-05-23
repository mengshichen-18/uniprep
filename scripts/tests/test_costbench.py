from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import costbench  # noqa: E402


class CostBenchTests(unittest.TestCase):
    def test_select_anchor_rows_reproducible(self) -> None:
        rows = [
            {"ltable_name": "a.csv", "l_id": 0, "_source_row_index": 0},
            {"ltable_name": "a.csv", "l_id": 0, "_source_row_index": 1},
            {"ltable_name": "b.csv", "l_id": 1, "_source_row_index": 2},
            {"ltable_name": "c.csv", "l_id": 2, "_source_row_index": 3},
            {"ltable_name": "c.csv", "l_id": 2, "_source_row_index": 4},
        ]

        picked1 = costbench.select_anchor_rows(rows, pairs_target=3, seed=42)
        picked2 = costbench.select_anchor_rows(rows, pairs_target=3, seed=42)

        idx1 = [r["_source_row_index"] for r in picked1]
        idx2 = [r["_source_row_index"] for r in picked2]
        self.assertEqual(idx1, idx2)
        self.assertGreaterEqual(len(picked1), 3)

    def test_usd_formula(self) -> None:
        snap = costbench.PriceSnapshot(
            model="gpt-4o-mini",
            prompt_per_1m=0.15,
            completion_per_1m=0.60,
            snapshot_date="2026-04-19",
        )
        usd = costbench.usd_from_tokens(prompt_tokens=1000, completion_tokens=100, price=snap)
        expected = (1000 * 0.15 + 100 * 0.60) / 1_000_000
        self.assertAlmostEqual(usd, expected, places=12)

    def test_call_volume_shape(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "dataset": "magellan",
                    "source_row_index": 0,
                    "pair_id": "magellan:0",
                    "anchor_id": "a::1",
                    "pair_order_in_anchor": 0,
                    "left_record_text": "L1",
                    "right_record_text": "R1",
                },
                {
                    "dataset": "magellan",
                    "source_row_index": 1,
                    "pair_id": "magellan:1",
                    "anchor_id": "a::1",
                    "pair_order_in_anchor": 1,
                    "left_record_text": "L1",
                    "right_record_text": "R2",
                },
                {
                    "dataset": "magellan",
                    "source_row_index": 2,
                    "pair_id": "magellan:2",
                    "anchor_id": "b::2",
                    "pair_order_in_anchor": 0,
                    "left_record_text": "L2",
                    "right_record_text": "R3",
                },
            ]
        )

        c_calls = costbench._build_comem_calls(df, topk=4)
        m_calls = costbench._build_matchgpt_calls(df)

        self.assertEqual(len(c_calls), 2)  # anchors
        self.assertEqual(len(m_calls), 3)  # pairs


if __name__ == "__main__":
    unittest.main()
