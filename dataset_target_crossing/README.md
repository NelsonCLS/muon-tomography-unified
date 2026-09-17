# Target-crossing dataset copies

Nonzero cube path: entry-PoCA-exit when PoCA is inside, otherwise straight segment between fitted z-face endpoints. Includes side crossings. This is an inferred path, not stored transport truth.

Positions and normalized directions only. No truth, momentum magnitude, RPC coincidence, or top-bottom-only cut.

Originals are read-only inputs. All samples remain in their original order, including empty samples. Every retained x row preserves every original value and column. Serialized y and extra feature payloads are unchanged. Outputs use GZIP compression.

| Dataset | Samples | Mean retained | Min | Max | Retained (%) |
|---|---:|---:|---:|---:|---:|
| custom_object.tfrecord | 1 | 14,852.00 | 14,852 | 14,852 | 33.53 |
| voxels_prediction_funny.tfrecord | 2 | 14,921.00 | 14,899 | 14,943 | 33.70 |
| voxels_prediction_test.tfrecord | 1,608 | 14,839.07 | 9,818 | 15,319 | 33.60 |
| voxels_prediction_val.tfrecord | 1,608 | 14,864.95 | 13,832 | 15,341 | 33.60 |
| voxels_prediction.tfrecord | 20,000 | 14,865.40 | 13,647 | 15,399 | 33.60 |

All output files were fully reread and checked against serialized record hashes, truth hashes, retained geometry, and row provenance. Source size, modification time and inode were unchanged.

Each original_indices.u32.gz decompresses to repeated little-endian uint64 record index, uint64 retained count, then uint32 original row indices[count]. Indices are zero-based and ascending. Per-sample counts CSVs record input/output counts and hashes.

Dose semantics change: 8192 rows drawn from these files are target-crossing tracks when a sample contains enough. No cut on truth object material is applied; paths through air inside the cube still qualify.

Completed 5 / 5 datasets.
