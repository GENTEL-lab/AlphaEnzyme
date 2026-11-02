import unittest

from alphaenzyme.utils.sharding import partition_indices


class PartitionIndicesTests(unittest.TestCase):
    def test_single_process_gpu_id_is_not_used_as_rank(self):
        self.assertEqual(partition_indices(7, gpu_id=1, world_size=1), (0, 0, 7))

    def test_multi_rank_partitions_cover_each_sample_once(self):
        for sample_count, world_size in ((11, 3), (2, 4), (0, 1)):
            indices = []
            for gpu_id in range(world_size):
                rank, start, end = partition_indices(sample_count, gpu_id, world_size)
                self.assertEqual(rank, gpu_id)
                indices.extend(range(start, end))
            self.assertEqual(indices, list(range(sample_count)))

    def test_explicit_rank_overrides_gpu_id_and_invalid_rank_fails(self):
        self.assertEqual(partition_indices(8, gpu_id=3, world_size=4, rank=1), (1, 2, 4))
        with self.assertRaises(ValueError):
            partition_indices(8, gpu_id=0, world_size=4, rank=4)


if __name__ == "__main__":
    unittest.main()
