import unittest

try:
    import torch
    from headroom.attention import full_attention, tiled_attention, library_attention
    from scripts.benchmark_attention import validate
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class AttentionTests(unittest.TestCase):
    def test_tiled_and_library_match_unscaled_baseline(self):
        for n, m, d in ((3, 5, 8), (65, 129, 16)):
            generator = torch.Generator().manual_seed(n)
            q = torch.randn(n, d, generator=generator)
            k = torch.randn(m, d, generator=generator)
            v = torch.randn(m, d, generator=generator)
            expected = full_attention(q, k, v)
            self.assertTrue(torch.allclose(tiled_attention(q, k, v), expected, rtol=1e-4, atol=1e-5))
            self.assertTrue(torch.allclose(library_attention(q, k, v), expected, rtol=1e-4, atol=1e-5))

    def test_validation_covers_asymmetric_and_tile_edge_shapes(self):
        checks = validate()
        self.assertEqual(len(checks), 9)
        self.assertTrue(all(check["passed"] and check["library_passed"] for check in checks))


if __name__ == "__main__":
    unittest.main()
