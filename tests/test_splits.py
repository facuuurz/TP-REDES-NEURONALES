"""Check the two data boundaries and exported inference contract."""

import math
import unittest

import numpy as np

from src.data import cnn_data, temporal_splits
from src.models import build_cnn


class SplitTests(unittest.TestCase):
    def test_cnn_groups_do_not_cross_holdout(self):
        images, labels, groups, development, test = cnn_data()
        self.assertEqual(len(images), len(labels))
        self.assertEqual(len(set(development) & set(test)), 0)
        self.assertFalse(set(groups[development]) & set(groups[test]))
        self.assertTrue(np.isfinite(images).all())

    def test_temporal_windows_are_isolated_and_scaled_on_training(self):
        splits, scalers = temporal_splits()
        bounds = {"train": (0, 960), "validation": (960, 1280), "test": (1280, 1600)}
        for name, (_, _, times) in splits.items():
            self.assertGreaterEqual(times.min() - 36, bounds[name][0])
            self.assertLess(times.max(), bounds[name][1])
        self.assertAlmostEqual(float(splits["train"][0].mean(axis=(0, 1))[0]), 0, places=5)
        self.assertEqual(scalers["feature_mean"].shape, (4,))


class ArchitectureTests(unittest.TestCase):
    def test_cnn_spatial_sizes_follow_same_padding_formula(self):
        model = build_cnn()
        size = 32
        for layer in model.layers:
            if layer.__class__.__name__ in ("Conv2D", "MaxPooling2D"):
                stride = layer.strides[0]
                size = math.ceil(size / stride) if layer.padding == "same" else (size - layer.pool_size[0]) // stride + 1
                self.assertEqual(layer.output.shape[1], size)
        self.assertEqual(size, 4)
        self.assertEqual(model.layers[-1].output.shape[-1], 10)


if __name__ == "__main__":
    unittest.main()
