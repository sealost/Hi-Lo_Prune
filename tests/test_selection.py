import unittest

import torch

from hilo_prune.selection import hierarchical_token_selection


class HierarchicalSelectionTests(unittest.TestCase):
    def test_partitions_candidate_pool(self) -> None:
        patch_features = torch.eye(8)
        token_features = torch.flip(torch.eye(8), dims=(0,))
        result = hierarchical_token_selection(
            patch_features, token_features, keep_count=2, candidate_count=5
        )

        self.assertEqual(result.keep.shape, (2,))
        self.assertEqual(result.fusion.shape, (3,))
        self.assertEqual(result.candidates.shape, (5,))
        self.assertTrue(set(result.keep.tolist()).isdisjoint(result.fusion.tolist()))
        self.assertEqual(
            set(result.keep.tolist()) | set(result.fusion.tolist()),
            set(result.candidates.tolist()),
        )

    def test_is_deterministic(self) -> None:
        generator = torch.Generator().manual_seed(7)
        patches = torch.randn(12, 6, generator=generator)
        tokens = torch.randn(12, 10, generator=generator)
        first = hierarchical_token_selection(
            patches, tokens, keep_count=3, candidate_count=7
        )
        second = hierarchical_token_selection(
            patches, tokens, keep_count=3, candidate_count=7
        )

        self.assertTrue(torch.equal(first.keep, second.keep))
        self.assertTrue(torch.equal(first.fusion, second.fusion))

    def test_rejects_invalid_counts(self) -> None:
        features = torch.randn(8, 4)
        for keep_count, candidate_count in [(0, 3), (4, 3), (3, 9)]:
            with self.subTest(keep_count=keep_count, candidate_count=candidate_count):
                with self.assertRaises(ValueError):
                    hierarchical_token_selection(
                        features,
                        features,
                        keep_count=keep_count,
                        candidate_count=candidate_count,
                    )


if __name__ == "__main__":
    unittest.main()
