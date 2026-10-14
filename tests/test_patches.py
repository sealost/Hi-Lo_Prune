import unittest

from PIL import Image

from hilo_prune.patches import image_to_merged_patch_features


class PatchFeatureTests(unittest.TestCase):
    def test_aligns_with_merged_grid(self) -> None:
        image = Image.new("RGB", (80, 40), color=(10, 20, 30))
        features = image_to_merged_patch_features(
            image, (1, 6, 8), patch_size=16, merge_size=2
        )
        self.assertEqual(features.shape, (12, 3 * 32 * 32))

    def test_rejects_video_grid(self) -> None:
        image = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(ValueError, "single still image"):
            image_to_merged_patch_features(image, (2, 4, 4))


if __name__ == "__main__":
    unittest.main()
