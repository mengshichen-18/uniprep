from __future__ import annotations

import unittest

import numpy as np

from symbolic_feature import SymbolicFeatureExecutor, SymbolicSpecError, validate_symbolic_feature_spec


class SymbolicFeatureV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed = ["f1", "f2", "f3", "f4"]

    def test_v1_threshold_range_validation(self) -> None:
        doc = {
            "spec_version": "v1",
            "task": "entity_matching",
            "feature_pool_used": ["f1", "f2"],
            "expression": "avg(f1,f2)",
            "decision": {"threshold": 1.2, "positive_if": ">="},
        }
        with self.assertRaises(SymbolicSpecError):
            validate_symbolic_feature_spec(doc, expected_task="entity_matching", allowed_features=self.allowed)

    def test_v1_min_feature_validation(self) -> None:
        doc = {
            "spec_version": "v1",
            "task": "entity_matching",
            "feature_pool_used": ["f1"],
            "expression": "sigmoid(f1)",
            "decision": {"threshold": 0.5, "positive_if": ">="},
        }
        with self.assertRaises(SymbolicSpecError):
            validate_symbolic_feature_spec(doc, expected_task="entity_matching", allowed_features=self.allowed)

    def test_v2_duplicate_role_validation(self) -> None:
        doc = {
            "spec_version": "v2",
            "task": "entity_matching",
            "feature_pool_used": ["f1", "f2", "f3"],
            "channels": [
                {
                    "name": "c1",
                    "role": "lexical",
                    "expression": "avg(f1,f2)",
                },
                {
                    "name": "c2",
                    "role": "lexical",
                    "expression": "avg(f2,f3)",
                },
            ],
            "aggregation": {
                "method": "weighted_sum",
                "weights": [0.5, 0.5],
                "bias": 0.0,
                "postprocess": "sigmoid",
            },
            "decision": {"threshold": 0.5, "positive_if": ">="},
        }
        with self.assertRaises(SymbolicSpecError):
            validate_symbolic_feature_spec(doc, expected_task="entity_matching", allowed_features=self.allowed)

    def test_v2_duplicate_feature_signature_validation(self) -> None:
        doc = {
            "spec_version": "v2",
            "task": "entity_matching",
            "feature_pool_used": ["f1", "f2", "f3"],
            "channels": [
                {
                    "name": "c1",
                    "role": "lexical",
                    "expression": "avg(f1,f2)",
                },
                {
                    "name": "c2",
                    "role": "semantic",
                    "expression": "safe_div(f2,f1)",
                },
            ],
            "aggregation": {
                "method": "weighted_sum",
                "weights": [0.5, 0.5],
                "bias": 0.0,
                "postprocess": "sigmoid",
            },
            "decision": {"threshold": 0.5, "positive_if": ">="},
        }
        with self.assertRaises(SymbolicSpecError):
            validate_symbolic_feature_spec(doc, expected_task="entity_matching", allowed_features=self.allowed)

    def test_v2_executor_outputs(self) -> None:
        doc = {
            "spec_version": "v2",
            "task": "entity_matching",
            "feature_pool_used": ["f1", "f2", "f3"],
            "channels": [
                {
                    "name": "c1",
                    "role": "lexical",
                    "expression": "avg(f1,f2)",
                    "output_range_hint": [0.0, 1.0],
                },
                {
                    "name": "c2",
                    "role": "semantic",
                    "expression": "clip(sigmoid(avg(f2,f3)),0.0,1.0)",
                    "output_range_hint": [0.0, 1.0],
                },
            ],
            "aggregation": {
                "method": "weighted_sum",
                "weights": [0.5, 0.5],
                "bias": 0.0,
                "postprocess": "sigmoid",
            },
            "output_range_hint": [0.0, 1.0],
            "decision": {"threshold": 0.5, "positive_if": ">="},
        }

        spec = validate_symbolic_feature_spec(
            doc,
            expected_task="entity_matching",
            allowed_features=self.allowed,
        )
        exe = SymbolicFeatureExecutor(spec=spec, strict=True)

        fmap = {
            "f1": np.asarray([0.2, 0.6, 0.9], dtype=np.float64),
            "f2": np.asarray([0.4, 0.5, 0.2], dtype=np.float64),
            "f3": np.asarray([0.7, 0.3, 0.8], dtype=np.float64),
        }
        channels = exe.run_channels(fmap)
        scores = exe.run_score(fmap)

        self.assertEqual(channels.shape, (3, 2))
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all(np.isfinite(channels)))
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
