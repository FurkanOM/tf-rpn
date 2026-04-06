"""Unit tests for training helper utilities."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from tests.test_support import import_project_module


class TrainUtilsTests(unittest.TestCase):
    """Cover hyper-parameter and batching helpers."""

    def setUp(self) -> None:
        """Import the training helpers and restore mutable defaults after each test."""
        self.module, _ = import_project_module(
            "utils.train_utils",
            clear_modules=("utils.train_utils", "utils.bbox_utils"),
        )
        original_rpn = copy.deepcopy(self.module.RPN)

        def restore_rpn() -> None:
            self.module.RPN.clear()
            self.module.RPN.update(copy.deepcopy(original_rpn))

        self.addCleanup(restore_rpn)

    def test_get_hyper_params_populates_defaults_and_anchor_count(self) -> None:
        """Add runtime defaults without requiring callers to pass every value."""
        params = self.module.get_hyper_params("vgg16")

        self.assertEqual(params["img_size"], 500)
        self.assertEqual(params["test_nms_topn"], 300)
        self.assertEqual(params["total_pos_bboxes"], 128)
        self.assertEqual(params["total_neg_bboxes"], 128)
        self.assertEqual(params["variances"], [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(params["anchor_count"], 9)

    def test_get_hyper_params_applies_truthy_overrides(self) -> None:
        """Allow callers to override selected defaults for experiments."""
        params = self.module.get_hyper_params(
            "mobilenet_v2",
            img_size=320,
            total_pos_bboxes=64,
        )

        self.assertEqual(params["img_size"], 320)
        self.assertEqual(params["total_pos_bboxes"], 64)
        self.assertEqual(params["anchor_count"], 9)

    def test_get_hyper_params_returns_a_copy_of_backbone_defaults(self) -> None:
        """Avoid mutating the global backbone presets when callers tweak a result."""
        params = self.module.get_hyper_params("vgg16")

        params["img_size"] = 320

        self.assertEqual(self.module.RPN["vgg16"]["img_size"], 500)

    def test_get_step_size_rounds_up_for_partial_batches(self) -> None:
        """Cover the ceil-based batching behavior used by the training loop."""
        self.assertEqual(self.module.get_step_size(10, 4), 3)
        self.assertEqual(self.module.get_step_size(8, 4), 2)

    def test_rpn_generator_yields_images_and_targets(self) -> None:
        """Produce the tuple structure expected by `Model.fit`."""
        dataset = [("images", "gt_boxes", "gt_labels")]

        with mock.patch.object(
            self.module,
            "calculate_rpn_actual_outputs",
            return_value=("bbox_deltas", "bbox_labels"),
        ) as target_builder:
            generator = self.module.rpn_generator(dataset, "anchors", {"feature_map_shape": 32})
            batch = next(generator)

        target_builder.assert_called_once_with("anchors", "gt_boxes", "gt_labels", {"feature_map_shape": 32})
        self.assertEqual(batch, ("images", ("bbox_deltas", "bbox_labels")))


if __name__ == "__main__":
    unittest.main()
