import numpy as np

from muon_tomo.postprocess import fill_empty_with_six_neighbor_mean


def test_empty_voxel_uses_six_face_neighbors_without_changing_measured_voxels():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    center = (2, 2, 2)
    neighbors = [
        (1, 2, 2), (3, 2, 2),
        (2, 1, 2), (2, 3, 2),
        (2, 2, 1), (2, 2, 3),
    ]
    for value, index in enumerate(neighbors, start=1):
        volume[index] = value
    empty = volume == 0

    filled = fill_empty_with_six_neighbor_mean(volume, empty)

    assert filled[center] == 3.5
    assert np.array_equal(filled[~empty], volume[~empty])


def test_fill_is_single_pass_and_uses_available_neighbors_at_boundaries():
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1, 0, 0] = 3.0
    empty = volume == 0

    filled = fill_empty_with_six_neighbor_mean(volume, empty)

    assert filled[0, 0, 0] == 1.0
    assert filled[2, 0, 0] == 0.75
    assert filled[3, 0, 0] == 0.0
