"""Unit tests for dataset metadata and local file helpers."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from typing import Any, Callable, Optional, Tuple
from unittest import mock

from tests.test_support import import_project_module, make_tf_stub


class FakeDataset:
    """Minimal dataset double that records mapping and batching requests."""

    def __init__(self) -> None:
        """Initialize the fake dataset recorder."""
        self.map_fn: Optional[Callable[[Any], Any]] = None
        self.padded_batch_args: Optional[Tuple[Any, Any, Any]] = None

    def map(self, map_fn: Callable[[Any], Any]) -> "FakeDataset":
        """Record the mapping function and return the dataset itself."""
        self.map_fn = map_fn
        return self

    def padded_batch(self, batch_size: Any, padded_shapes: Any, padding_values: Any) -> str:
        """Record batch arguments and return a sentinel dataset object."""
        self.padded_batch_args = (batch_size, padded_shapes, padding_values)
        return "batched-dataset"


class DataUtilsTests(unittest.TestCase):
    """Verify helper functions that do not require TensorFlow datasets at runtime."""

    def setUp(self) -> None:
        """Import the module under test with lightweight dependency stubs."""
        pil_module = types.ModuleType("PIL")
        pil_module.Image = types.SimpleNamespace(LANCZOS="LANCZOS")
        numpy_module = types.ModuleType("numpy")
        tfds_module = types.ModuleType("tensorflow_datasets")
        self.tf_stub = make_tf_stub()
        self.module, _ = import_project_module(
            "utils.data_utils",
            extra_modules={
                "PIL": pil_module,
                "numpy": numpy_module,
                "tensorflow_datasets": tfds_module,
            },
            clear_modules=("utils.data_utils",),
            tensorflow_stub=self.tf_stub,
        )

    def test_get_total_item_size_supports_combined_split(self) -> None:
        """Sum train and validation counts when the combined split is requested."""
        info = types.SimpleNamespace(
            splits={
                "train": types.SimpleNamespace(num_examples=8),
                "validation": types.SimpleNamespace(num_examples=3),
                "test": types.SimpleNamespace(num_examples=5),
            }
        )

        self.assertEqual(self.module.get_total_item_size(info, "train+validation"), 11)
        self.assertEqual(self.module.get_total_item_size(info, "test"), 5)

    def test_get_labels_returns_dataset_label_names(self) -> None:
        """Read label names from the dataset metadata structure."""
        info = types.SimpleNamespace(
            features={"labels": types.SimpleNamespace(names=["person", "car", "dog"])}
        )

        self.assertEqual(self.module.get_labels(info), ["person", "car", "dog"])

    def test_get_custom_imgs_returns_top_level_files_only(self) -> None:
        """Ignore nested directories because the production helper stops at depth one."""
        with tempfile.TemporaryDirectory() as tempdir:
            top_level_files = [
                os.path.join(tempdir, "first.jpg"),
                os.path.join(tempdir, "second.png"),
            ]
            nested_dir = os.path.join(tempdir, "nested")
            os.makedirs(nested_dir)
            nested_file = os.path.join(nested_dir, "ignored.jpg")

            for path in top_level_files + [nested_file]:
                with open(path, "w", encoding="utf-8") as file_obj:
                    file_obj.write("placeholder")

            img_paths = self.module.get_custom_imgs(tempdir)

        self.assertCountEqual(img_paths, top_level_files)

    def test_build_dataset_maps_preprocessing_and_applies_padding(self) -> None:
        """Pipe preprocessing through `map` and finalize the dataset with padding."""
        fake_dataset = FakeDataset()

        with mock.patch.object(
            self.module,
            "preprocessing",
            return_value=("img", "boxes", "labels"),
        ) as preprocessing_mock:
            result = self.module.build_dataset(
                fake_dataset,
                320,
                320,
                4,
                apply_augmentation=True,
                evaluate=True,
            )
            mapped_output = fake_dataset.map_fn("raw-sample")

        self.assertEqual(mapped_output, ("img", "boxes", "labels"))
        preprocessing_mock.assert_called_once_with(
            "raw-sample",
            320,
            320,
            apply_augmentation=True,
            evaluate=True,
        )
        self.assertEqual(
            fake_dataset.padded_batch_args,
            (4, self.module.get_data_shapes(), self.module.get_padding_values()),
        )
        self.assertEqual(result, "batched-dataset")

    def test_build_custom_dataset_uses_project_generator_contract(self) -> None:
        """Construct the custom-image dataset with the repo's declared shapes and types."""
        captured: dict = {}

        def from_generator(generator: Callable[[], Any], output_types: Any, output_shapes: Any) -> str:
            captured["generator"] = generator
            captured["output_types"] = output_types
            captured["output_shapes"] = output_shapes
            return "custom-dataset"

        self.tf_stub.data = types.SimpleNamespace(
            Dataset=types.SimpleNamespace(from_generator=from_generator)
        )

        with mock.patch.object(
            self.module,
            "custom_data_generator",
            return_value=iter([("img", "boxes", "labels")]),
        ) as generator_mock, mock.patch.object(
            self.module,
            "get_data_types",
            return_value="types",
        ) as get_types_mock, mock.patch.object(
            self.module,
            "get_data_shapes",
            return_value="shapes",
        ) as get_shapes_mock:
            result = self.module.build_custom_dataset(["sample.jpg"], 128, 256)
            generated_items = list(captured["generator"]())

        generator_mock.assert_called_once_with(["sample.jpg"], 128, 256)
        get_types_mock.assert_called_once_with()
        get_shapes_mock.assert_called_once_with()
        self.assertEqual(captured["output_types"], "types")
        self.assertEqual(captured["output_shapes"], "shapes")
        self.assertEqual(generated_items, [("img", "boxes", "labels")])
        self.assertEqual(result, "custom-dataset")


if __name__ == "__main__":
    unittest.main()
