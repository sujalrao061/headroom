import unittest

try:
    import torch
    from headroom.engine import _check_contract
    from headroom.workloads import tiled_nearest, nearest_baseline, tiled_radius_count, radius_baseline
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class ContractTests(unittest.TestCase):
    def test_nearest_preserves_output(self):
        self.assertTrue(all(c["passed"] for c in _check_contract("nearest", tiled_nearest)))

    def test_full_matrix_rejects_same_candidate(self):
        self.assertFalse(all(c["passed"] for c in _check_contract("matrix", tiled_nearest)))

    def test_tie_keeps_first_index(self):
        q = torch.tensor([[0., 0., 0.]])
        r = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
        self.assertTrue(torch.equal(nearest_baseline(q, r), tiled_nearest(q, r, tile=1)))

    def test_radius_count_preserves_output(self):
        self.assertTrue(all(c["passed"] for c in _check_contract("radius", tiled_radius_count)))

    def test_nearest_repair_does_not_satisfy_radius_contract(self):
        self.assertFalse(all(c["passed"] for c in _check_contract("radius", tiled_nearest)))

    def test_radius_boundary_is_inclusive_across_tiles(self):
        q = torch.tensor([[0., 0., 0.]])
        r = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [2., 0., 0.]])
        self.assertTrue(torch.equal(radius_baseline(q, r), tiled_radius_count(q, r, tile=1)))
        self.assertEqual(tiled_radius_count(q, r, tile=1).item(), 2)

    def test_default_tile_edges_are_part_of_independent_validation(self):
        for case_id, candidate in (("nearest", tiled_nearest), ("radius", tiled_radius_count)):
            checks = _check_contract(case_id, candidate)
            edge = next(check for check in checks if check["shape"] == "tile edges: 130×131")
            self.assertTrue(edge["passed"])
            self.assertEqual(len(checks), 5)

    def test_tile_edge_check_catches_a_broken_nearest_repair(self):
        def wrong_first_tile_only(query, reference):
            return nearest_baseline(query, reference[:128])

        checks = _check_contract("nearest", wrong_first_tile_only)
        edge = next(check for check in checks if check["shape"] == "tile edges: 130×131")
        self.assertFalse(edge["passed"])


if __name__ == "__main__":
    unittest.main()
